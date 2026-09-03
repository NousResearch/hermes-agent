"""Gateway lifecycle ledger — durable termination-reason evidence (NS-608).

The gateway already has *graceful* shutdown forensics
(:mod:`gateway.shutdown_forensics` — who sent the SIGTERM) and an exit-path
diagnostic log (``gateway-exit-diag.log`` — every way ``asyncio.run`` can
return).  What it does NOT have is any record of an **unclean death**: a
SIGKILL, a kernel OOM kill, or the whole VM dying takes the process out
before any handler runs, so the next boot has no idea the previous life
ended violently — support tickets like NS-608 then require manually
cross-correlating four log files and two external APIs to answer "what
killed the gateway?".

This module closes that gap with a tiny state machine persisted to
``<HERMES_HOME>/state/gateway.lifecycle.json``:

* On startup, :func:`record_startup` reads the sentinel left by the
  previous life.  ``phase == "running"`` means that life never reached any
  exit path → it died uncleanly.  The finding — including the last
  heartbeat's memory sample, which is the closest thing to a pre-death
  telemetry snapshot — is appended to ``gateway-exit-diag.log`` as a
  ``gateway.previous_unclean_exit`` record and logged at WARNING.  The
  sentinel is then rewritten as ``phase=running`` for the new life.
* On every clean exit path, :func:`mark_exited` rewrites the sentinel as
  ``phase=exited`` with the exit code and a reason string.  Wired into
  ``_exit_after_graceful_shutdown`` (the single funnel for all graceful
  exits, #53107) and the two watchdog ``os._exit`` sites in
  :mod:`gateway.shutdown_watchdog`.

:func:`sample_memory` provides the cheap (<1ms, pure /proc reads) memory
snapshot that :func:`gateway.shutdown_watchdog.write_loop_heartbeat`
embeds in the 30s heartbeat — giving every unclean-death report a
"memory available N seconds before death" data point so OOM crash cycles
are classifiable from the volume alone (no Prometheus retention races).

Everything here is best-effort: a forensics failure must never affect the
gateway lifecycle it is observing.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)

_LIFECYCLE_RELATIVE = ("state", "gateway.lifecycle.json")
_EXIT_DIAG_RELATIVE = ("logs", "gateway-exit-diag.log")
_STATE_DB_RELATIVE = ("state.db",)

# Heuristic OOM-suspicion thresholds applied to the last heartbeat's memory
# sample.  Deliberately conservative: this only annotates the report with a
# hint; classification stays with the human reading the evidence.
_LOW_MEM_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_LOW_MEM_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal available


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore task overrides)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def get_lifecycle_sentinel_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.lifecycle.json``."""
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(*_LIFECYCLE_RELATIVE)


def sample_memory() -> Dict[str, Any]:
    """Cheap memory snapshot: own RSS + system availability + swap.

    Pure ``/proc`` reads, Linux-only (returns ``{}`` elsewhere), never
    raises.  Values in KiB to match the kernel's units.
    """
    sample: Dict[str, Any] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    sample["rss_kib"] = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        pass
    try:
        meminfo: Dict[str, int] = {}
        wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in wanted:
                    meminfo[key] = int(line.split()[1])
                    if len(meminfo) == len(wanted):
                        break
        if "MemTotal" in meminfo:
            sample["mem_total_kib"] = meminfo["MemTotal"]
        if "MemAvailable" in meminfo:
            sample["mem_available_kib"] = meminfo["MemAvailable"]
        if "SwapTotal" in meminfo and "SwapFree" in meminfo:
            sample["swap_used_kib"] = meminfo["SwapTotal"] - meminfo["SwapFree"]
    except (OSError, ValueError, IndexError):
        pass
    return sample


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_sentinel(payload: Dict[str, Any], home: Optional[Path]) -> None:
    path = get_lifecycle_sentinel_path(home)
    try:
        from utils import atomic_json_write

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write lifecycle sentinel", exc_info=True)


def _append_exit_diag(record: Dict[str, Any], home: Optional[Path]) -> None:
    """Append a JSON line to gateway-exit-diag.log (same format as the CLI's
    ``_exit_diag`` records so existing tooling greps both)."""
    base = home if home is not None else _process_hermes_home()
    path = base.joinpath(*_EXIT_DIAG_RELATIVE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.debug("Failed to append unclean-exit record", exc_info=True)


def _pid_alive_with_start_time(pid: Any, start_time: Any) -> bool:
    """True when ``pid`` is a live process matching ``start_time`` (±2s).

    Guards the takeover race: during ``--replace`` the old gateway can still
    be mid-teardown when the new one boots — a live matching owner is a
    planned handover, not an unclean death.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        # NOT os.kill(pid, 0): on Windows that sends CTRL_C_EVENT to the
        # target's console group (bpo-14484). _pid_exists is the repo's
        # canonical no-kill cross-platform probe (psutil-backed).
        from gateway.status import _pid_exists

        if not _pid_exists(pid_int):
            return False
    except Exception:
        return False
    if start_time is None:
        return True  # alive; can't disambiguate PID reuse — err on "alive"
    try:
        from gateway.status import get_process_start_time

        actual = get_process_start_time(pid_int)
        if actual is None:
            return True
        return abs(float(actual) - float(start_time)) <= 2.0
    except Exception:
        return True


def detect_unclean_exit(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Inspect the previous life's sentinel; return an evidence dict when it
    died uncleanly, else ``None``.  Read-only — does not rewrite the sentinel.
    """
    sentinel = _read_json(get_lifecycle_sentinel_path(home))
    if not sentinel or sentinel.get("phase") != "running":
        return None
    if _pid_alive_with_start_time(sentinel.get("pid"), sentinel.get("start_time")):
        return None  # live owner — planned takeover in flight, not a death

    evidence: Dict[str, Any] = {
        "prior_pid": sentinel.get("pid"),
        "prior_started_at": sentinel.get("started_at"),
        "prior_start_time": sentinel.get("start_time"),
    }

    # Enrich with the last heartbeat: when did the loop last prove liveness,
    # and what did memory look like at that moment?
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        hb = _read_json(get_loop_heartbeat_path(home))
    except Exception:
        hb = None
    if hb:
        evidence["last_heartbeat_at"] = hb.get("updated_at")
        mem = hb.get("mem")
        if isinstance(mem, dict):
            evidence["last_heartbeat_mem"] = mem
            total = mem.get("mem_total_kib")
            avail = mem.get("mem_available_kib")
            if isinstance(avail, int) and (
                avail < _LOW_MEM_AVAILABLE_KIB
                or (
                    isinstance(total, int)
                    and total > 0
                    and avail / total < _LOW_MEM_AVAILABLE_FRACTION
                )
            ):
                evidence["suspected_oom"] = True
    return evidence


def check_state_db_integrity(home: Optional[Path] = None) -> str:
    """Return ``"ok"``, ``"absent"``, or the first ``quick_check`` complaint.

    Called only after an unclean death, because that is when the store may
    have been torn: a SIGKILL landing on a gateway mid-WAL-checkpoint can
    leave half-written b-tree pages behind (see
    ``_enforce_macos_synchronous_full`` in :mod:`hermes_state` — macOS
    ``fsync`` guarantees neither data-on-platter nor write ordering).

    ``quick_check(1)`` stops at the first problem, so this costs ~2s on a
    healthy 500MB store and less on a damaged one — cheap enough for a path
    that runs at most once per unclean boot, far too expensive for every
    boot. Corruption used to sit undetected for days (2026-08-26 → 08-30)
    because nothing ever looked.

    The connection is opened normally, not read-only: a WAL store needs its
    -shm sidecar for a read-only open. The PRAGMA itself writes nothing.
    Never raises — this is forensics, not lifecycle.
    """
    base = home if home is not None else _process_hermes_home()
    path = base.joinpath(*_STATE_DB_RELATIVE)
    if not path.exists():
        return "absent"
    try:
        # A WAL store cannot be opened read-only without its -shm sidecar, so
        # open normally; quick_check itself writes nothing.
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
        finally:
            conn.close()
    except Exception as exc:  # sqlite3.Error, OSError, anything
        return f"check-failed: {exc}"
    if not row or row[0] is None:
        return "check-failed: no result"
    return str(row[0])


def record_startup(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Boot-time entry point: report any unclean previous exit, then claim
    the sentinel for the current life.

    Returns the unclean-exit evidence dict (also persisted to
    ``gateway-exit-diag.log`` and logged at WARNING) or ``None``.  Never
    raises.
    """
    evidence: Optional[Dict[str, Any]] = None
    try:
        evidence = detect_unclean_exit(home)
        if evidence is not None:
            # The death may have torn the store. This is the only moment
            # we know to look, and looking is what turns a 3.5-day silent
            # corruption into a startup warning.
            verdict = check_state_db_integrity(home=home)
            evidence["state_db_integrity"] = verdict
            if verdict not in ("ok", "absent"):
                logger.error(
                    "state.db FAILED integrity check after an unclean gateway "
                    "exit: %s — sessions may read as missing until it is "
                    "repaired. Run `hermes doctor`.",
                    verdict,
                )
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": "gateway.previous_unclean_exit",
                "pid": os.getpid(),
                **evidence,
            }
            _append_exit_diag(record, home)
            logger.warning(
                "Previous gateway life (pid=%s, started_at=%s) exited UNCLEANLY "
                "(no exit path ran — SIGKILL / OOM / VM death). "
                "last_heartbeat_at=%s last_mem=%s suspected_oom=%s",
                evidence.get("prior_pid"),
                evidence.get("prior_started_at"),
                evidence.get("last_heartbeat_at"),
                evidence.get("last_heartbeat_mem"),
                evidence.get("suspected_oom", False),
            )
    except Exception:
        logger.debug("Unclean-exit detection failed", exc_info=True)

    try:
        claim: Dict[str, Any] = {
            "phase": "running",
            "pid": os.getpid(),
            "start_time": time.time(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        # Carry the verdict on the PREVIOUS life forward on the new
        # sentinel: it is the only place the finding survives in
        # machine-readable form (the exit-diag log is append-only prose
        # for humans), and /api/status reads it to tell the user "your
        # agent restarted after (suspected) running out of memory"
        # (NS-656).  Scoped to this life only — the next clean exit or
        # boot rewrites the sentinel and the flags age out with it.
        if evidence is not None:
            claim["prior_unclean_exit"] = True
            if evidence.get("suspected_oom"):
                claim["prior_suspected_oom"] = True
        _write_sentinel(claim, home)
    except Exception:
        logger.debug("Failed to claim lifecycle sentinel", exc_info=True)
    return evidence


def mark_exited(
    exit_code: Optional[int] = None,
    reason: str = "graceful_shutdown",
    home: Optional[Path] = None,
) -> None:
    """Mark the current life as cleanly exited.  Idempotent, never raises.

    Only rewrites the sentinel when it is provably owned by this process —
    during a ``--replace`` takeover the replacement claims the sentinel
    before the old process finishes teardown, and the old life must not
    clobber the new owner's ``running`` phase on its way out.  A sentinel
    with ``pid=None`` (or a malformed pid) has *unknown* ownership and is
    likewise left alone: we must not overwrite evidence we cannot prove is
    ours with a ``clean exit`` claim.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is not None and sentinel.get("pid") != os.getpid():
            return
        _write_sentinel(
            {
                "phase": "exited",
                "pid": os.getpid(),
                "exit_code": exit_code,
                "exit_reason": reason,
                "exited_at": datetime.now(timezone.utc).isoformat(),
            },
            home,
        )
    except Exception:
        logger.debug("Failed to mark lifecycle sentinel exited", exc_info=True)


def read_prior_exit_label(profile_home: Path) -> str:
    """Container-boot helper: one-word summary of how the profile's last
    gateway life ended.  ``clean`` / ``unclean`` / ``unknown`` (no sentinel
    or never ran).  Read-only and exception-free — used by
    ``hermes_cli.container_boot`` to annotate ``container-boot.log``.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(profile_home))
        if not sentinel:
            return "unknown"
        phase = sentinel.get("phase")
        if phase == "exited":
            return "clean"
        if phase == "running":
            # At container boot the old PID namespace is gone — any
            # "running" sentinel is from a life that never exited cleanly.
            return "unclean"
    except Exception:
        pass
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
    profile = conn.execute(
        """
        SELECT profile_id, signature_hash, expires_at FROM capability_profiles
        WHERE model_receipt_hash IN (
            SELECT result_hash FROM specialist_benchmark_receipts WHERE candidate_id = ?
        ) AND verification_receipt_hash IN (
            SELECT result_hash FROM specialist_verification_receipts WHERE candidate_id = ?
        ) ORDER BY id DESC LIMIT 1
        """,
        (candidate_id, candidate_id),
    ).fetchone()
    if profile is not None:
        revoked = conn.execute(
            """
            SELECT revocation_hash FROM specialist_profile_revocations
            WHERE profile_id = ? AND signature_hash = ? ORDER BY id DESC LIMIT 1
            """,
            (profile["profile_id"], profile["signature_hash"]),
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
