"""Gateway lifecycle ledger — durable termination-reason evidence.

Graceful shutdowns leave forensics (``gateway-exit-diag.log``); an unclean
death (SIGKILL, kernel OOM, VM death) runs no handler.  A sentinel at
``<HERMES_HOME>/state/gateway.lifecycle.json`` closes the gap:
:func:`record_startup` finds ``phase == "running"`` from the previous life →
unclean death, appended to the exit-diag log as ``gateway.previous_unclean_exit``
and logged at WARNING; :func:`mark_exited` rewrites ``phase=exited`` on every
clean exit path.  Best-effort: forensics must never affect the lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore task overrides)."""
    from hermes_constants import get_hermes_home, get_process_hermes_home

    # get_process_hermes_home expands ``~``/``$VAR`` (python -m gateway.run skips the CLI normalizer).
    return get_process_hermes_home() if os.environ.get("HERMES_HOME", "").strip() else get_hermes_home()


def _home_path(home: Optional[Path], *relative: str) -> Path:
    return (_process_hermes_home() if home is None else home).joinpath(*relative)


def get_lifecycle_sentinel_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.lifecycle.json``."""
    return _home_path(home, "state", "gateway.lifecycle.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proc_fields(path: str, wanted: Dict[str, str]) -> Dict[str, int]:
    """``{dst: int}`` for each ``src: dst`` key found in a ``Key: value`` /proc file."""
    found: Dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in wanted:
                    found[wanted[key]] = int(line.split()[1])
                    if len(found) == len(wanted):
                        break
    except (OSError, ValueError, IndexError):
        return {}
    return found


def sample_memory() -> Dict[str, Any]:
    """Cheap /proc snapshot (KiB): own RSS + MemTotal/MemAvailable + swap used.  Linux-only
    (``{}`` elsewhere), never raises; the 30s heartbeat embeds it so OOM cycles are classifiable."""
    sample = _proc_fields("/proc/self/status", {"VmRSS": "rss_kib"})
    mem = _proc_fields("/proc/meminfo", {"MemTotal": "mem_total_kib", "MemAvailable": "mem_available_kib",
                                         "SwapTotal": "SwapTotal", "SwapFree": "SwapFree"})
    swap_total, swap_free = mem.pop("SwapTotal", None), mem.pop("SwapFree", None)
    sample.update(mem)
    if swap_total is not None and swap_free is not None:
        sample["swap_used_kib"] = swap_total - swap_free
    return sample


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_sentinel(payload: Dict[str, Any], home: Optional[Path]) -> None:
    try:
        from utils import atomic_json_write

        path = get_lifecycle_sentinel_path(home)
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(path.parent)
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write lifecycle sentinel", exc_info=True)


def _append_exit_diag(record: Dict[str, Any], home: Optional[Path]) -> None:
    """Append a JSON line to gateway-exit-diag.log (same format as the CLI's ``_exit_diag``)."""
    try:
        path = _home_path(home, "logs", "gateway-exit-diag.log")
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(path.parent)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.debug("Failed to append unclean-exit record", exc_info=True)


def _pid_is_sentinel_owner(pid: Any, start_time: Any, create_time: Any) -> bool:
    """True when ``pid`` is a live process that is the sentinel's incarnation — guards the
    ``--replace`` race: a live matching owner mid-teardown is a handover, not a death.

    Identity is the psutil ``create_time`` the sentinel stamps at claim (epoch seconds, same
    producer on both sides). The sentinel's ``start_time`` is the ledger claim time and is NOT
    comparable with ``gateway.status.get_process_start_time`` (proc ticks on Linux, centiseconds
    elsewhere) — the old comparison never matched, so every ``--replace`` handover read as an
    unclean death. A pre-stamp sentinel (no ``create_time``) still has ``start_time`` in epoch
    seconds: the owner was born BEFORE it claimed, a PID reuser AFTER the owner died, so a birth
    later than the claim is a reuser."""
    try:
        pid_int = int(pid)
        # NOT os.kill(pid, 0): on Windows that sends CTRL_C_EVENT to the target's console group.
        from gateway.status import _pid_exists

        if pid_int <= 0 or not _pid_exists(pid_int):
            return False
    except Exception:
        return False
    from hermes_cli.process_identity import _process_create_time

    actual = _process_create_time(pid_int)
    if actual is None:
        return True  # can't disambiguate PID reuse
    if type(create_time) in (int, float):
        return abs(actual - float(create_time)) <= 2.0
    if type(start_time) in (int, float):
        return actual <= float(start_time) + 2.0
    return True


def _suspected_oom(mem: Dict[str, Any]) -> bool:
    """Heuristic only (classification stays with the reader); thresholds are
    memory_status' "critical" tier so a live warning and a post-mortem verdict agree."""
    from gateway.memory_status import _CRITICAL_AVAILABLE_FRACTION, _CRITICAL_AVAILABLE_KIB

    total, avail = mem.get("mem_total_kib"), mem.get("mem_available_kib")
    if not isinstance(avail, int):
        return False
    return avail < _CRITICAL_AVAILABLE_KIB or (
        isinstance(total, int) and total > 0 and avail / total < _CRITICAL_AVAILABLE_FRACTION
    )


def detect_unclean_exit(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Evidence dict when the previous life died uncleanly, else ``None``. Read-only."""
    sentinel = _read_json(get_lifecycle_sentinel_path(home))
    if not sentinel or sentinel.get("phase") != "running":
        return None
    if _pid_is_sentinel_owner(sentinel.get("pid"), sentinel.get("start_time"), sentinel.get("create_time")):
        return None  # live owner — planned takeover in flight, not a death
    evidence: Dict[str, Any] = {
        "prior_pid": sentinel.get("pid"), "prior_started_at": sentinel.get("started_at"),
        "prior_start_time": sentinel.get("start_time"),
    }
    # Enrich with the last heartbeat: last proven liveness and memory at that moment.
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        hb = _read_json(get_loop_heartbeat_path(home)) or {}
    except Exception:
        hb = {}
    if hb:
        evidence["last_heartbeat_at"] = hb.get("updated_at")
    mem = hb.get("mem")
    if isinstance(mem, dict):
        evidence["last_heartbeat_mem"] = mem
        if _suspected_oom(mem):
            evidence["suspected_oom"] = True
    return evidence


# Startup-watchdog lease held while the unclean-exit integrity check runs (#115542):
# ``PRAGMA quick_check(1)`` is one monolithic SQLite operation and per-call leases
# clamp at 900s, so a single entry lease cannot cover a multi-thousand-second healthy
# check on a huge store — without renewal the watchdog kills the attempt with exit 75
# and the restart loop never reaches the cron ticker. The progress handler below renews
# a phase-owned lease for as long as SQLite is making progress, and always returns 0
# so it never aborts the check; the ok/absent/first-complaint verdicts stay fail-closed.
_INTEGRITY_CHECK_LEASE_PHASE = "state_db_unclean_integrity_check"
_INTEGRITY_CHECK_LEASE_S = 900.0
# VM instructions between progress-handler invocations; each invocation is a
# monotonic-clock compare, renewing at most every _INTEGRITY_CHECK_LEASE_RENEW_S.
_INTEGRITY_CHECK_PROGRESS_OPS = 100_000
_INTEGRITY_CHECK_LEASE_RENEW_S = 60.0


def _install_integrity_check_lease(conn: sqlite3.Connection) -> None:
    """Claim the integrity-check lease and renew it from a SQLite progress handler.

    ``report_startup_progress`` is a no-op when no watchdog is armed and never raises.
    The handler always returns 0 -- it must never abort the verdict PRAGMA.  Synchronous
    by design: no checker worker thread can outlive the check.
    """
    from hermes_startup_watchdog import report_startup_progress

    report_startup_progress(_INTEGRITY_CHECK_LEASE_S, phase=_INTEGRITY_CHECK_LEASE_PHASE)
    last_renew = time.monotonic()

    def _on_progress() -> int:
        nonlocal last_renew
        now = time.monotonic()
        if now - last_renew >= _INTEGRITY_CHECK_LEASE_RENEW_S:
            last_renew = now
            report_startup_progress(_INTEGRITY_CHECK_LEASE_S, phase=_INTEGRITY_CHECK_LEASE_PHASE)
        return 0

    conn.set_progress_handler(_on_progress, _INTEGRITY_CHECK_PROGRESS_OPS)


def check_state_db_integrity(home: Optional[Path] = None) -> str:
    """``"ok"``, ``"absent"``, or the first ``quick_check`` complaint.  Never raises.

    Only after an unclean death — SIGKILL mid-WAL-checkpoint can leave half-written
    b-tree pages.  ``quick_check(1)`` stops at the first problem (~2s on a healthy
    500MB store): cheap once per unclean boot, too costly every boot.  Opened
    normally: a WAL store needs its -shm sidecar for read-only, and the PRAGMA writes nothing.

    Holds a phase-owned startup-watchdog lease for the check duration
    (renewed from a SQLite progress handler, #115542) so a long healthy check
    on a huge store is not mistaken for a parked deadlock.
    """
    path = _home_path(home, "state.db")
    if not path.exists():
        return "absent"
    try:
        with closing(sqlite3.connect(str(path))) as conn:
            _install_integrity_check_lease(conn)
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
    except Exception as exc:
        return f"check-failed: {exc}"
    return "check-failed: no result" if not row or row[0] is None else str(row[0])


def _report_unclean_exit(evidence: Dict[str, Any], home: Optional[Path]) -> None:
    """Integrity-check the store, persist the exit-diag record, log at WARNING."""
    # The death may have torn the store; this is the only moment we know to look.
    verdict = evidence["state_db_integrity"] = check_state_db_integrity(home=home)
    if verdict not in ("ok", "absent"):
        logger.error(
            "state.db FAILED integrity check after an unclean gateway exit: %s — sessions may read as "
            "missing until it is repaired. Run `hermes doctor`.",
            verdict,
        )
    _append_exit_diag({"ts": _now_iso(), "tag": "gateway.previous_unclean_exit", "pid": os.getpid(), **evidence}, home)
    logger.warning(
        "Previous gateway life (pid=%s, started_at=%s) exited UNCLEANLY (no exit path ran — SIGKILL / OOM / "
        "VM death, or a process kill issued by the agent or one of its descendants, e.g. a pkill/taskkill of "
        "the host interpreter image; see #113667). last_heartbeat_at=%s last_mem=%s suspected_oom=%s",
        evidence.get("prior_pid"), evidence.get("prior_started_at"), evidence.get("last_heartbeat_at"),
        evidence.get("last_heartbeat_mem"), evidence.get("suspected_oom", False),
    )


def record_startup(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Boot entry point: report any unclean previous exit (evidence dict, also persisted
    to ``gateway-exit-diag.log`` and logged at WARNING) then claim the sentinel.  Never raises."""
    evidence: Optional[Dict[str, Any]] = None
    try:
        evidence = detect_unclean_exit(home)
        if evidence is not None:
            _report_unclean_exit(evidence, home)
    except Exception:
        logger.debug("Unclean-exit detection failed", exc_info=True)
    try:
        claim: Dict[str, Any] = {"phase": "running", "pid": os.getpid(), "start_time": time.time(), "started_at": _now_iso()}
        # Process birth (psutil), distinct from ``start_time`` (the ledger claim, seconds later once
        # imports finish): the Windows start attestation binds PIDs to birth time (#110020 review).
        from hermes_cli.process_identity import _process_create_time

        create_time = _process_create_time(os.getpid())
        if create_time is not None:
            claim["create_time"] = create_time
        # Carry the verdict on the PREVIOUS life on the new sentinel: it is the only
        # machine-readable copy (/api/status reads it to report an OOM restart).
        # Scoped to this life — the next clean exit or boot rewrites the sentinel.
        if evidence is not None:
            claim["prior_unclean_exit"] = True
            if evidence.get("suspected_oom"):
                claim["prior_suspected_oom"] = True
        _write_sentinel(claim, home)
    except Exception:
        logger.debug("Failed to claim lifecycle sentinel", exc_info=True)
    return evidence


def mark_exited(exit_code: Optional[int] = None, reason: str = "graceful_shutdown", home: Optional[Path] = None) -> None:
    """Mark the current life as cleanly exited.  Idempotent, never raises.

    Only rewrites a sentinel provably owned by this process: during ``--replace``
    the replacement claims it before the old process finishes teardown and must not
    be clobbered.  ``pid=None`` / malformed sentinels have unknown ownership → left alone.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is not None and sentinel.get("pid") != os.getpid():
            return
        exited: Dict[str, Any] = {"phase": "exited", "pid": os.getpid(), "exit_code": exit_code, "exit_reason": reason,
                                  "exited_at": _now_iso()}
        # Carry the incarnation identity: the Windows start attestation matches a clean exit by
        # PID *and* start time so a reused PID's exit cannot vouch for a different life (#110020).
        for key in ("start_time", "create_time"):
            if sentinel is not None and sentinel.get(key) is not None:
                exited[key] = sentinel[key]
        _write_sentinel(exited, home)
    except Exception:
        logger.debug("Failed to mark lifecycle sentinel exited", exc_info=True)


def read_prior_exit_label(profile_home: Path) -> str:
    """``clean``/``unclean``/``unknown`` for the profile's last gateway life; annotates
    ``container-boot.log``.  ``running`` is unclean: the old PID namespace is gone at container boot."""
    try:
        phase = (_read_json(get_lifecycle_sentinel_path(profile_home)) or {}).get("phase")
        return {"exited": "clean", "running": "unclean"}.get(phase, "unknown")
    except Exception:
        return "unknown"
# Specialist discovery uses the Kanban SQLite ledger rather than this process
# sentinel.  The read model lives here because it has the same crucial
# property: it is reconstructed from append-only durable evidence at startup,
# rather than replaying a model/advisory operation after a restart.
class SpecialistLifecycleRow(NamedTuple):
    """One sanitized, durable specialist-discovery lifecycle observation."""

    stage: str
    status: str
    receipt_hash: Optional[str] = None
    expires_at: Optional[int] = None


class SpecialistDiscoveryRecovery(NamedTuple):
    """Read-only recovery/status result for one candidate request."""

    candidate_id: str
    rows: tuple[SpecialistLifecycleRow, ...]
    recovery_action: str


def _specialist_rows_query(conn: Any, candidate_id: str, now: int) -> tuple[SpecialistLifecycleRow, ...]:
    """Read only the immutable receipts that belong to a candidate.

    The function deliberately has no callback parameter.  A restart must
    never cause an advisory, benchmark, sandbox, or canary to run again.
    """
    candidate = conn.execute(
        """
        SELECT request_hash, lifecycle_status FROM candidate_profile_requests
        WHERE request_id = ?
        """,
        (candidate_id,),
    ).fetchone()
    if candidate is None:
        return (SpecialistLifecycleRow("candidate_request", "missing"),)
    latest_candidate = conn.execute(
        """
        SELECT lifecycle_status FROM candidate_profile_requests
        WHERE request_hash = ? ORDER BY id DESC LIMIT 1
        """,
        (candidate["request_hash"],),
    ).fetchone()
    rows: list[SpecialistLifecycleRow] = [
        SpecialistLifecycleRow(
            "candidate_request",
            latest_candidate["lifecycle_status"] if latest_candidate is not None else candidate["lifecycle_status"],
            candidate_id,
        )
    ]
    source_rows = conn.execute(
        "SELECT body, status FROM tasks WHERE body LIKE ? ORDER BY created_at DESC",
        (f'%"candidate_request_id": "{candidate_id}"%',),
    ).fetchall()
    source_status = "not_found"
    for source in source_rows:
        try:
            payload = json.loads(source["body"] or "")
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("candidate_request_id") == candidate_id:
            source_status = str(source["status"])
            break
    rows.insert(0, SpecialistLifecycleRow("source_task", source_status))
    latest_benchmark = conn.execute(
        """
        SELECT result_hash, status, expires_at FROM specialist_benchmark_receipts
        WHERE candidate_id = ? ORDER BY id DESC LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if latest_benchmark is not None:
        benchmark_status = latest_benchmark["status"]
        if now >= latest_benchmark["expires_at"]:
            benchmark_status = "expired"
        rows.append(
            SpecialistLifecycleRow(
                "benchmark", benchmark_status, latest_benchmark["result_hash"], latest_benchmark["expires_at"]
            )
        )
    sandbox = conn.execute(
        """
        SELECT benchmark_result_hash, disposable, task_count FROM specialist_sandbox_runs
        WHERE candidate_id = ? ORDER BY id DESC LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if sandbox is not None:
        status = "verified_shape" if sandbox["disposable"] == 1 and sandbox["task_count"] == 1 else "rejected"
        rows.append(SpecialistLifecycleRow("sandbox", status, sandbox["benchmark_result_hash"]))
    verification = conn.execute(
        """
        SELECT result_hash, status, expires_at FROM specialist_verification_receipts
        WHERE candidate_id = ? ORDER BY id DESC LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if verification is not None:
        verification_status = verification["status"]
        if now >= verification["expires_at"]:
            verification_status = "expired"
        rows.append(
            SpecialistLifecycleRow(
                "verification", verification_status, verification["result_hash"], verification["expires_at"]
            )
        )
    canary = conn.execute(
        """
        SELECT result_hash, status FROM specialist_canary_receipts
        WHERE candidate_id = ? ORDER BY id DESC LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    rows.append(
        SpecialistLifecycleRow("canary", canary["status"] if canary is not None else "not_run", canary["result_hash"] if canary else None)
    )
    approvals = conn.execute(
        """
        SELECT target_state, approval_hash FROM specialist_operator_approvals
        WHERE candidate_id = ? AND approved = 1 ORDER BY id
        """,
        (candidate_id,),
    ).fetchall()
    approved_targets = {row["target_state"] for row in approvals}
    rows.append(SpecialistLifecycleRow("staged_approval", "approved" if "staged" in approved_targets else "missing"))
    rows.append(SpecialistLifecycleRow("active_approval", "approved" if "active" in approved_targets else "missing"))
    promotion = conn.execute(
        """
        SELECT profile_id, signature_hash, permissions_hash FROM specialist_promotion_proofs
        WHERE candidate_id = ? AND target_state = 'active' ORDER BY id DESC LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    profile = None
    if promotion is not None and promotion["profile_id"]:
        profile = conn.execute(
            """
            SELECT id, profile_id, signature_hash, permissions_hash, expires_at
            FROM capability_profiles
            WHERE profile_id = ? AND signature_hash = ? AND permissions_hash = ?
              AND status = 'active'
            ORDER BY id DESC LIMIT 1
            """,
            (promotion["profile_id"], promotion["signature_hash"], promotion["permissions_hash"]),
        ).fetchone()
    if profile is not None:
        revoked = conn.execute(
            """
            SELECT revocation_hash FROM specialist_profile_revocations
            WHERE capability_profile_id = ? AND profile_id = ?
              AND signature_hash = ? AND permissions_hash = ? ORDER BY id DESC LIMIT 1
            """,
            (profile["id"], profile["profile_id"], profile["signature_hash"], profile["permissions_hash"]),
        ).fetchone()
        if revoked is not None:
            rows.append(SpecialistLifecycleRow("rollback", "revoked", revoked["revocation_hash"]))
            rows.append(SpecialistLifecycleRow("active_profile", "revoked"))
        elif profile["expires_at"] is not None and now >= profile["expires_at"]:
            rows.append(SpecialistLifecycleRow("expiry", "expired", expires_at=profile["expires_at"]))
            rows.append(SpecialistLifecycleRow("active_profile", "expired", expires_at=profile["expires_at"]))
        else:
            rows.append(SpecialistLifecycleRow("active_profile", "active"))
    else:
        rows.append(SpecialistLifecycleRow("active_profile", "not_active"))
    return tuple(rows)


def recover_specialist_discovery(
    candidate_id: str, *, db_path: Optional[Path] = None, board: Optional[str] = None, now: Optional[int] = None
) -> SpecialistDiscoveryRecovery:
    """Reconcile one candidate exclusively from append-only local evidence.

    Recovery never retries advisory/benchmark work and never dispatches or
    activates a profile.  Expired/revoked results explicitly direct the caller
    to the normal task-orchestrator/triage fallback.
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    if now is None:
        now = int(time.time())
    if isinstance(now, bool) or not isinstance(now, int):
        raise ValueError("now must be an integer timestamp")
    try:
        from hermes_cli import kanban_db

        path = Path(db_path) if db_path is not None else kanban_db.kanban_db_path(board=board)
        # `mode=ro` is critical here: the normal Kanban connect helper can
        # initialize/migrate a DB and create WAL sidecars. Recovery must refuse
        # a missing/uninitialized ledger rather than making evidence appear.
        if not path.is_file():
            raise FileNotFoundError(path)
        # Always use a fresh ``mode=ro`` snapshot for a mutable Hermes ledger.
        # Choosing a mode from a prior WAL-path observation races a writer that
        # commits a revocation immediately afterwards. SQLite may attach or
        # create a ``-shm`` mapping while observing WAL state, but this reader
        # never writes the main database or WAL. Any read failure is normal
        # triage rather than a stale active resolution.
        uri = f"{path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = _specialist_rows_query(conn, candidate_id, now)
        finally:
            conn.close()
    except Exception as exc:
        return SpecialistDiscoveryRecovery(
            candidate_id,
            (SpecialistLifecycleRow("ledger", f"unavailable:{type(exc).__name__}"),),
            "normal_triage",
        )
    active = next((row for row in rows if row.stage == "active_profile"), None)
    action = "active_resolution" if active and active.status == "active" else "task_orchestrator"
    return SpecialistDiscoveryRecovery(candidate_id, rows, action)
