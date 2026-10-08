"""Structured update receipts + post-update fleet version verification.

Phase 1 of the fleet-update reliability plan (#91277): the updater must
*prove* its outcome instead of assuming it.

Two additive capabilities, both designed so a failure inside them can never
break an update (every public entry point is exception-swallowing):

1. **Update receipt** — a machine-readable JSON record of what one
   ``hermes update`` run discovered, did, skipped (and why), written to
   the ROOT home's ``logs/update_receipts/`` (``get_default_hermes_root()``,
   never a sticky profile's ``HERMES_HOME``). Silent-failure classes this
   makes visible: #88848 (helper died after "success" printed), #74973
   (restart silently skipped), #85753 (restart phase never ran), #81193
   (desktop shows failure for a successful update).

2. **Fleet version verification** — after the restart phase, read every
   profile's ``gateway_state.json``, compare each live gateway's stamped
   ``code_sha`` (written by ``gateway/status.py`` on every runtime-status
   write) against the freshly-updated checkout's HEAD, and print a fleet
   version matrix. Mixed-version fleets (#88654, #69754, #77553, #56717)
   become a loud, actionable report instead of a latent state.

Deployment-kind awareness (docker/image-managed installs) rides on
``hermes_cli.version_info.get_code_identity()``: a packaged build reports
its install-stamp provenance (``source="docker"``/``"nix"``/…) and the
receipt records that the install is not in-place updatable.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RECEIPT_KEEP = 20  # keep the last N receipts in the root home's receipt directory
#: Terminal records finalized by THIS process, by update id (see ``finalized_receipt``).
_FINALIZED: dict[str, dict[str, Any]] = {}
COMMAND_BOUNDARY_STOP_REASON = "completed at command boundary"

# Receipt state is per-CONTEXT, not a module global: a nested
# ``hermes update`` receipt (or one in another thread) must never clobber
# the outer one, and the boundary finalize must see exactly its own
# process's receipt. Same pattern as pm.receipt's ContextVars — no
# manager object.
_current: contextvars.ContextVar[Optional["UpdateReceipt"]] = contextvars.ContextVar(
    "update_receipt_current", default=None
)
#: The terminal outcome of the run the enclosing command scope finalized or adopted (from a
#: completion child): ``committed_success`` answers the command boundary's gateway status from it.
_scope_terminal: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "update_receipt_scope_terminal", default=None
)


@contextmanager
def update_receipt_scope():
    """Keep the command's finalization guard away from an enclosing update."""
    token = _current.set(None)
    terminal = _scope_terminal.set({})
    try:
        yield
    finally:
        _scope_terminal.reset(terminal)
        _current.reset(token)


def adopt_terminal_receipt(data: Any) -> None:
    """A completion child finalized this command's run: its terminal outcome is this scope's too."""
    cell = _scope_terminal.get()
    if cell is not None and isinstance(data, dict) and data.get("finished_at"):
        cell["outcome"] = data.get("outcome")


def committed_success() -> bool:
    """True when this command's run is closed and its receipt says ``success`` (C3).

    The gateway ``/update`` status written at the command boundary follows the receipt: an
    interrupt or error that escapes after the run closed as a success never reports it failed.
    """
    cell = _scope_terminal.get()
    return _current.get() is None and bool(cell) and cell.get("outcome") == "success"


def current_correlation_id() -> Optional[str]:
    """The update correlation id in force in this context, or None.

    Derived from the OPEN update receipt itself — one source of truth, no
    duplicate id variable: a nested update's begin replaces the current
    receipt (so syncs begun under it capture the nested id), and its
    finalize RESTORES the outer receipt via the ContextVar token (so the
    outer id comes back for the rest of the outer update)."""
    current = _current.get()
    return current.correlation_id if current is not None else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_identity(refresh: bool = False) -> dict[str, Any]:
    """Running-code identity, or ``{}`` when the probe fails."""
    with suppress(Exception):
        from hermes_cli.version_info import get_code_identity

        return get_code_identity(refresh=refresh) or {}
    return {}


def _str_records(entries: Any, keys: tuple[str, ...], *, pid: bool = False) -> list[dict[str, Any]]:
    """Dict entries reduced to stringified ``keys`` (plus an int ``pid`` first when requested)."""
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record: dict[str, Any] = {"pid": int(entry.get("pid", 0) or 0)} if pid else {}
        record.update({key: str(entry.get(key, "")) for key in keys})
        records.append(record)
    return records


def _launcher_correlation_id() -> Optional[str]:
    """Correlation id handed in by an external launcher, or None.

    Desktop's managed SSH update exports ``HERMES_UPDATE_CORRELATION_ID`` and
    only accepts a receipt whose ``correlation_id`` equals it. This is separate
    from ``update_id``, which stays the CLI's own id for pm sync receipts."""
    return os.environ.get("HERMES_UPDATE_CORRELATION_ID", "").strip() or None


class UpdateReceipt:
    """Collects the observable facts of one ``hermes update`` run."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "schema": 1, "started_at": _utc_now_iso(), "finished_at": None,
            "argv": list(sys.argv), "pid": os.getpid(),
            "outcome": "running",  # running | success | partial | failed
            "pre_update": _code_identity(), "post_update": {},
            "steps": [], "skips": [], "gateway_restart": {}, "fleet": [],
        }
        # The id binds this update to the pm sync receipts begun under it
        # (pm.receipt captures it via the _correlation ContextVar).
        self.correlation_id = uuid.uuid4().hex
        self.data["update_id"] = self.correlation_id
        self.data["correlation_id"] = _launcher_correlation_id()
        # The dashboard action that spawned this run: the receipt store is root-wide, so the
        # action-status route certifies an outcome only from a receipt naming ITS action.
        self.data["action_id"] = os.environ.get("HERMES_ACTION_ID", "").strip() or None

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.data["steps"].append({"name": name, "ok": bool(ok), "detail": detail, "at": _utc_now_iso()})

    def skip(self, name: str, reason: str) -> None:
        self.data["skips"].append({"name": name, "reason": reason, "at": _utc_now_iso()})

    def stage(self, name: str, outcome: str, **facts: str) -> None:
        # Stage END marks: a stage's duration is the gap since the previous mark (or started_at).
        self.data.setdefault("stages", []).append({"name": name, "outcome": outcome, "at": _utc_now_iso(), **facts})

    def fact(self, key: str, value: Any) -> None:
        self.data[key] = value

    def followup(self, step: str, reason: str) -> None:
        self.data.setdefault("followups", []).append({"step": step, "reason": reason, "at": _utc_now_iso()})

    def gateway_restart_result(
        self, *, restarted_services: list | None = None, relaunched_profiles: list | None = None,
        externally_supervised_profiles: list | None = None, killed_pids: list | None = None,
        failed_units: list | None = None, incomplete: bool = False, phase_error: str = "",
        fresh_recovery: dict[str, Any] | None = None,
    ) -> None:
        result: dict[str, Any] = {
            "restarted_services": list(restarted_services or []),
            "relaunched_profiles": list(relaunched_profiles or []),
            "externally_supervised_profiles": list(externally_supervised_profiles or []),
            "killed_pids": [int(p) for p in (killed_pids or [])],
            "failed_units": [str(u) for u in (failed_units or [])],
            "incomplete": bool(incomplete),
            "phase_error": phase_error,
        }
        if fresh_recovery is not None:
            # Conservative outcome vocabulary: "verified" is the only bucket allowed to claim
            # supervisor coverage; "relaunch_attempted" means the relaunch exited 0 without
            # independent supervisor observation. "skipped" preserves runtimes (manual gateways,
            # serve/dashboard entries) the pass deliberately did not touch.
            persisted: dict[str, Any] = {
                key: [str(profile) for profile in fresh_recovery.get(key, [])]
                for key in ("requested", "verified", "relaunch_attempted", "failed")
            }
            persisted["skipped"] = _str_records(
                fresh_recovery.get("skipped", []), ("profile", "kind", "supervisor", "reason")
            )
            # ``hermes serve`` hosts tui_gateway and is not a gateway profile, so neither the
            # per-profile buckets above nor the fleet-version matrix can describe it. Persist its
            # unit outcomes and any process that survived on the pre-update generation, or the
            # receipt keeps claiming a clean recovery the operator's box contradicts.
            serve_units = fresh_recovery.get("serve_units") or {}
            persisted["serve_units"] = {
                key: [str(unit) for unit in (serve_units.get(key) or [])] for key in ("verified", "failed")
            }
            persisted["stale_runtimes"] = _str_records(
                fresh_recovery.get("stale_runtimes", []), ("kind", "profile", "supervisor"), pid=True
            )
            result["fresh_recovery"] = persisted
        self.data["gateway_restart"] = result

    def finalize(self, outcome: str) -> None:
        if outcome == "success" and self.data.get("user_action"):
            outcome = "partial"  # committed, but the user still has to act (record_user_action)
        self.data["outcome"] = outcome
        self.data["finished_at"] = _utc_now_iso()
        self.data["post_update"] = _code_identity(refresh=True)


def _receipt_dir() -> Path:
    # ``hermes_constants`` (stdlib-only), never ``hermes_cli.config``: the receipt must be
    # writable from the refused/failed paths where config loading itself may be what broke
    # (#112465, #112558). The ROOT home, never a sticky profile's: an update mutates the
    # checkout every profile shares, and the Desktop and the hand-off scripts read the root.
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "logs" / "update_receipts"


def _receipt_dirs() -> list[Path]:
    """The root store, then the profile's own when a named profile makes them differ. ``pm/``
    writes its sync receipts to ``get_hermes_home()`` and receipts from before the root move live
    there too, so readers take both and writers mirror ``latest.json`` into it (``hermes -p <name>
    pm status`` reads only that one)."""
    from hermes_constants import get_hermes_home

    root, profile = _receipt_dir(), get_hermes_home() / "logs" / "update_receipts"
    return [root] if profile.resolve() == root.resolve() else [root, profile]


def _write_latest(payload: bytes) -> None:
    """Replace ``latest.json`` in every store a reader of this home consults."""
    from hermes_cli.runtime_state import _atomic_bytes

    for directory in _receipt_dirs():
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(directory / "latest.json", payload)


def _run_file(directory: Path, data: dict[str, Any]) -> Path:
    """One archive file per run, named at begin so the running and terminal records coincide."""
    stamp = re.sub(r"[^0-9]", "", str(data.get("started_at") or ""))[:14] or time.strftime("%Y%m%d%H%M%S")
    return directory / f"update_{stamp[:8]}_{stamp[8:]}_{data.get('pid') or os.getpid()}_{data.get('update_id')}.json"


def read_run_record(update_id: str) -> Optional[tuple[Path, dict[str, Any]]]:
    """The run's own archive ``(path, record)`` in the root store, or None. Never raises.

    The ONE lookup of a run's archive (named at begin, so its running and terminal records share
    one file); never ``latest.json``, which another profile or run may have replaced since. A file
    that cannot be read, is not a JSON object, or names another run is not this run's record:
    callers treat all of those exactly like a missing archive.
    """
    with suppress(OSError):
        for path in _receipt_dir().glob(f"update_*_{update_id}.json"):
            with suppress(OSError, ValueError):
                record = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(record, dict) and record.get("update_id") == update_id:
                    return path, record
    return None


def _persist_running(data: dict[str, Any]) -> None:
    """Write the open run to disk: a killed update still leaves its own record. Never raises."""
    with suppress(Exception):
        from hermes_cli.runtime_state import _atomic_bytes

        directory = _receipt_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = _run_file(directory, data)
        # A completion child can finalize while its parent still holds the pre-child
        # context. That stale running snapshot must never undo the terminal receipt.
        stored = read_run_record(str(data.get("update_id")))
        if stored is not None and stored[1].get("finished_at"):
            return
        from hermes_cli.process_identity import _process_create_time

        payload = (json.dumps({**data, "writer_pid": os.getpid(),
                               "writer_create_time": _process_create_time(os.getpid())},
                              indent=2, default=str) + "\n").encode("utf-8")
        _atomic_bytes(path, payload)
        _write_latest(payload)


def _owner_alive(record: dict[str, Any]) -> bool:
    from hermes_cli.process_identity import _pid_alive_matches

    identities = {}
    for key, time_key in (("pid", "pid_create_time"), ("writer_pid", "writer_create_time")):
        pid = record.get(key)
        if isinstance(pid, int) and pid > 0:
            # Older receipts recorded no writer incarnation. Do not let a duplicate
            # bare PID override the original owner's proven creation-time mismatch.
            create_time = record.get(time_key)
            if pid not in identities or identities[pid] is None:
                identities[pid] = create_time
    return any(_pid_alive_matches(pid, create_time) is not False
               for pid, create_time in identities.items())


def reconcile_interrupted_runs() -> list[dict[str, Any]]:
    """Mark ``running`` records whose processes are gone as ``interrupted``; returns them.

    The start-of-run reclaimer for the durable running receipt: the next update says plainly
    that the previous one was killed and at which stage, instead of reporting its outcome as
    whatever ran before it. Never raises.
    """
    interrupted: list[dict[str, Any]] = []
    with suppress(Exception):
        from hermes_cli.runtime_state import _atomic_bytes

        directory = _receipt_dir()
        latest = read_latest_receipt() or {}
        for path in sorted(directory.glob("update_*.json"), key=lambda p: p.stat().st_mtime)[-_RECEIPT_KEEP:]:
            with suppress(Exception):
                record = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(record, dict) or record.get("outcome") != "running" or _owner_alive(record):
                    continue
                stages = record.get("stages") or []
                last = stages[-1].get("name") if stages and isinstance(stages[-1], dict) else None
                record.update(outcome="interrupted", interrupted_detected_at=_utc_now_iso(),
                              stop_reason=f"process exited during the update (last stage: {last or 'start'})")
                payload = (json.dumps(record, indent=2, default=str) + "\n").encode("utf-8")
                _atomic_bytes(path, payload)
                if latest.get("update_id") == record.get("update_id"):
                    _write_latest(payload)
                interrupted.append(record)
    for record in interrupted:
        print(f"⚠ The previous update ({record.get('started_at')}) was interrupted before it finished "
              f"({record.get('stop_reason')}); this update completes the work it still owed.")
    return interrupted


def begin_update_receipt(*, previous: dict | None = None, correlation_id: str | None = None) -> None:
    """Start recording a new update receipt, durably ``running`` until it finalizes.

    Nested updates are safe: the previous receipt (if any) is preserved
    behind the ContextVar token and comes back when this one finalizes —
    a nested begin/finalize never drops the outer update's receipt or
    correlation. Never raises."""
    try:
        receipt = UpdateReceipt()
        if previous:
            receipt.data.update(copy.deepcopy(previous))
        receipt.correlation_id = correlation_id or receipt.correlation_id
        receipt.data.update(update_id=receipt.correlation_id, outcome="running", finished_at=None)
        # A handoff receipt from an older interpreter may predate the field.
        receipt.data["correlation_id"] = receipt.data.get("correlation_id") or _launcher_correlation_id()
        with suppress(Exception):
            from hermes_cli.process_identity import _process_create_time

            receipt.data.setdefault("pid_create_time", _process_create_time(receipt.data["pid"]))
        if not previous:
            reconcile_interrupted_runs()
        # The running record is about to replace latest.json: snapshot the previous record's
        # manual serve rows (raw, no side effects) for finalize's carry-forward. A handed-off
        # receipt from an older interpreter predates the field, so it needs the same snapshot.
        if not previous or "carried_manual_serves" not in previous:
            with suppress(Exception):
                prior = read_latest_receipt() or {}
                rows = list((prior.get("plan") or {}).get("runtimes") or [])
                rows += list(prior.get("pending_manual_serves") or [])
                rows += list(prior.get("carried_manual_serves") or [])  # a run killed before finalize
                receipt.data["carried_manual_serves"] = [
                    row for row in rows if isinstance(row, dict) and row.get("supervisor") == "manual-serve"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not start update receipt: %s", exc)
        return
    receipt.current_token = _current.set(receipt)
    _persist_running(receipt.data)


def persist_running_receipt() -> None:
    """Re-assert the open run on disk (the completion child calls this after resuming it)."""
    current = _current.get()
    if current is not None and current.data.get("outcome") == "running":
        _persist_running(current.data)


#: Follow-up steps whose failure leaves the source-update tail owed (``source-completion-pending``).
TAIL_FOLLOWUPS = frozenset({"dependencies", "launchers", "build", "maintenance", "profile_sync",
                            "config_migration", "completion"})

#: Follow-ups that mean the BUILD stage did not succeed (C3: the receipt names what actually failed):
#: the products or the launchers the build publishes failed, or the tail raised before proving them.
#: Owed dependencies, config migration or maintenance keep the tail armed but say nothing about the
#: build; they are reported as their own follow-ups.
BUILD_FOLLOWUPS = frozenset({"launchers", "build", "completion"})


def record_build_stage(followups) -> None:
    """Mark the build stage from the owed ``(step, reason)`` follow-ups: failed only for a build one."""
    record_stage("build", "failed" if any(step in BUILD_FOLLOWUPS for step, _ in followups) else "success")


def record_followup(step: str, reason: str, *, retry: str = "the next launch or `hermes update` retries it") -> None:
    """A post-commit step failed: print ⚠, keep the run a success, and say what is still owed.

    The code already committed, so the update does not fail; the step's own obligation (the
    source tail, the fleet restart, the launch-time bytecode sweep) stays armed and the next
    launch or ``hermes update`` retries it. Never raises. The printed line is a protocol: the
    Desktop hand-offs (scripts/desktop-update/posix.sh, windows.ps1) parse ``Update follow-up
    '<step>' did not finish:`` to report every owed step, so keep it one whole line.
    """
    reason = " ".join(str(reason).split())[:500] or "failed"
    print(f"  ⚠ Update follow-up '{step}' did not finish: {reason} ({retry})", flush=True)
    current = _current.get()
    if current is None:
        return
    _record("followup", f"update followup {step}", step, reason)
    current = _current.get()
    if current is not None:
        _persist_running(current.data)


def record_user_action(step: str, reason: str) -> None:
    """The code committed, but something only the user can do is still owed. Never raises.

    Unlike a follow-up nothing retries it (a stash whose restore conflicted stays parked until the
    user re-applies it), so the run can never be a plain success: it finalizes ``partial`` and
    ``hermes update`` exits 1, as #122557 established for an unrestored autostash.
    """
    _record("fact", f"update user action {step}", "user_action", {"step": step, "reason": " ".join(str(reason).split())[:500]})
    persist_running_receipt()


def amend_terminal_followup(update_id: str, step: str, reason: str) -> None:
    """A follow-up that failed after the run's receipt was finalized lands on that receipt. Never raises."""
    found = read_run_record(update_id)
    if found is None:
        return
    path, record = found
    record.setdefault("followups", []).append({"step": step, "reason": reason, "at": _utc_now_iso()})
    payload = (json.dumps(record, indent=2, default=str) + "\n").encode("utf-8")
    with suppress(Exception):  # an unwritable store: the printed follow-up line still names it
        from hermes_cli.runtime_state import _atomic_bytes

        _atomic_bytes(path, payload)
        if (read_latest_receipt() or {}).get("update_id") == update_id:
            _write_latest(payload)


def owe_followup(update_id: Optional[str], step: str, reason: str, **retry: str) -> None:
    """A post-commit step failed: record it on the run while its receipt is open, else amend the
    run's closed receipt. Amending an OPEN run would append a second copy beside the one
    ``record_followup`` just persisted. Never raises."""
    record_followup(step, reason, **retry)
    if _current.get() is None and update_id:
        amend_terminal_followup(update_id, step, reason)


def _record(method: str, what: str, *args: Any, **kwargs: Any) -> None:
    """Invoke ``method`` on the active receipt; no-op when none, never raises.

    Copy-on-write, same rule as pm.receipt: a copied context (copy_context,
    asyncio.to_thread) inherits the SAME receipt object — mutate a clone
    and re-set it in THIS context only, so a child's records never leak
    into (or corrupt) the parent's receipt.
    """
    try:
        import copy

        current = _current.get()
        if current is None:
            return
        clone = copy.copy(current)
        clone.data = copy.deepcopy(current.data)
        getattr(clone, method)(*args, **kwargs)
        _current.set(clone)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record %s: %s", what, exc)


def record_step(name: str, ok: bool, detail: str = "") -> None:
    """Record one update step outcome. No-op when no receipt is active."""
    _record("step", f"update step {name}", name, ok, detail)


def record_skip(name: str, reason: str) -> None:
    """Record a skipped step WITH the reason it was skipped."""
    _record("skip", f"update skip {name}", name, reason)


def record_stage(name: str, outcome: str, **facts: str) -> None:
    """Mark the END of a pipeline stage (``success``/``failed``/``skipped``) with a timestamp."""
    _record("stage", f"update stage {name}", name, outcome, **facts)
    # Each stage boundary refreshes the durable running record (a kill names its last stage).
    persist_running_receipt()


def record_fact(key: str, value: Any) -> None:
    """Set one top-level receipt field (e.g. ``initiator``)."""
    _record("fact", f"update fact {key}", key, value)


def record_gateway_restart(**kwargs: Any) -> None:
    """Record the gateway restart phase outcome (see UpdateReceipt)."""
    _record("gateway_restart_result", "gateway restart result", **kwargs)


def record_stop_reason(reason: str) -> None:
    """Name the exit that is about to stop this run, as one closed token (``stop_class``).

    Called on the line before the ``sys.exit``/raise it names, so the last one recorded is the
    exit that fired. The token must be one of ``shared_metrics_contract.UPDATE_STOP_CLASSES``
    (anything else is ignored by the metric); the human ``stop_reason`` and printed text are
    untouched. No-op when no receipt is open; never raises.
    """
    _record("fact", f"update stop {reason}", "stop_class", reason)


#: git could not write a file it needs (``index.lock``'s directory, a tracked file): an OS permission
#: error, not a lock another git holds. SSH's ``Permission denied (publickey)`` is an auth failure.
_GIT_FILE_PERMISSION = re.compile(r"Permission denied(?! *\()|\bEACCES\b")


def git_error_stop_class(exc: BaseException) -> Optional[str]:
    """The closed stop token for a git ``CalledProcessError`` that is about to end the run, or None.

    Reads only the failed argv and fixed git/Hermes phrases in its output, never stores either.
    """
    if not isinstance(exc, subprocess.CalledProcessError):
        return None
    output = exc.stderr or exc.output or ""
    output = output.decode("utf-8", "replace") if isinstance(output, bytes) else str(output)
    argv = [str(part) for part in exc.cmd] if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd or "").split()
    return git_output_stop_class(output, argv, exc.returncode)


def git_output_stop_class(output: str, argv: list[str], returncode: Optional[int] = None) -> Optional[str]:
    """:func:`git_error_stop_class` for a git call that returned instead of raising (``_git_run``)."""
    if "index.lock" in output and "File exists" in output:
        return "git_index_locked"  # another git (or a killed one, #132089) holds the index lock
    if _GIT_FILE_PERMISSION.search(output):
        return "permission_denied"  # git may not write a file it needs (index.lock's dir, a tracked file)
    if returncode == 124 and "timed out after" in output:
        return "git_timeout"  # update_cmd._git_run's own timeout text
    if "No space left on device" in output:
        return "disk_full"
    if "stash" in argv:
        return "local_changes_blocked"  # update_cmd_stash._push_stash saved nothing
    if {"checkout", "merge", "reset"} & set(argv):
        return "checkout_move_failed"
    return None


def _started_by_handoff_partner() -> bool:
    """True when the update marker names the Desktop hand-off this process runs under.

    For an exit before the receipt opens, where this process may NOT hold the lock: at the
    update-lock refusal the marker belongs to whichever update holds it, so "a pid other than
    ours" (update_cmd._record_update_initiator, which runs under the lock) would call every
    refusal Desktop-started. The claim is ours only when it names us as its delegate (the posix and
    Windows hand-offs write that line before the update runs) or names our hand-off partner on
    line 1 or the delegate line (``HERMES_UPDATE_HANDOFF_PID``, the Tauri updater; our parent, a
    launcher between the hand-off and us). The marker is read raw: no liveness probe and no
    cleanup (those are lock policy, not a metrics side effect).
    """
    from hermes_cli.update_lock import _handoff_pid, _parse_marker, update_marker_path

    marker = _parse_marker(update_marker_path().read_bytes())
    if marker.started_at is None:
        return False
    pid = os.getpid()
    if marker.delegate_pid == pid:
        return True
    partners = {_handoff_pid(), os.getppid()} - {None, 0, 1, pid}
    return bool(partners & {marker.pid, marker.delegate_pid})


def record_stop_without_receipt(reason: str, outcome: str) -> None:
    """Count an exit that fires before this run's receipt opens, in shared metrics only.

    No receipt is written, so the exit behaves exactly as before: at the update-lock refusal
    ``latest.json`` and the running record belong to the update holding the lock (opening one
    would replace its pointer and reconcile its archive), and the Git-operation refusal runs before
    the receipt, plan and snapshot by design. The row is derived from the same fields a final
    receipt carries (``outcome`` is ``refused`` or ``failed``), including the initiator, so a
    Desktop-started refusal reads ``kind=desktop`` like the same run's receipt would. Never raises.
    """
    with suppress(Exception):
        now = _utc_now_iso()
        data = {
            "schema": 1, "update_id": uuid.uuid4().hex, "started_at": now, "finished_at": now,
            "outcome": "refused" if outcome == "refused" else "failed",
            "stop_class": reason, "pre_update": {}, "stages": [], "steps": [], "fleet": [],
        }
        with suppress(Exception):  # no marker, or an unreadable one: the CLI default
            if _started_by_handoff_partner():
                data["initiator"] = "desktop"
        from hermes_cli.observability.shared_metrics_update import record_update_receipt

        record_update_receipt(data)


def finalize_update_receipt(outcome: str, fleet: list | None = None, stop_reason: str = "") -> Optional[Path]:
    """Finalize + persist the receipt (``success``/``partial``/``failed``/``refused``); path or None.

    Exactly-once by construction: the context's receipt is popped first, so a second call (e.g. the
    command-boundary safety net after an inner path already finalized) is a no-op returning None.
    The receipt is popped via its OWN begin token, so a nested update's finalize RESTORES the outer
    update's open receipt and correlation instead of discarding them.
    """
    current = _current.get()
    if current is None:
        return None
    receipt = copy.copy(current)
    receipt.data = copy.deepcopy(current.data)
    token = getattr(receipt, "current_token", None)
    try:
        if token is not None:
            _current.reset(token)
        else:  # pragma: no cover - receipts begun before token binding
            _current.set(None)
    except ValueError:
        # Token from another context (finalize ran in a copied context) —
        # pop THIS context only, so exactly-once still holds.
        _current.set(None)
    try:
        receipt.finalize(outcome)
        if stop_reason:
            receipt.data["stop_reason"] = stop_reason
        # The terminal facts exist before the store is touched; a store that refuses the write
        # (after the commit point) must not erase them for this process's own caller.
        _FINALIZED[str(receipt.data.get("update_id"))] = receipt.data
        adopt_terminal_receipt(receipt.data)
        if fleet is not None:
            receipt.data["fleet"] = fleet
        # Manual serve restart obligations outlive one receipt rotation: carry the previous
        # receipt's still-pending rows forward so the startup warning survives (see
        # update_serve_obligations).
        from hermes_cli.update_serve_obligations import retain_receipt_manual_serves
        if "carried_manual_serves" in receipt.data:
            prior = {"pending_manual_serves": receipt.data.pop("carried_manual_serves") or []}
        else:  # a receipt begun before the running record existed
            prior = read_latest_receipt() or {}
        pending = retain_receipt_manual_serves(prior)
        if pending:
            receipt.data["pending_manual_serves"] = pending
        # EMBED the pm sync sections (the settled receipts contract): the
        # update's rebuild/bisect ran through pm's own sync receipt, which
        # finalizes before this one. Fold in only the completion filed
        # under THIS update's correlation id (pm.receipt.last_for_update,
        # per-context, never latest.json) — a sync from before this update
        # began, a standalone sync, or one in a concurrent thread cannot
        # be misattributed, and a nested update's sync cannot displace
        # this update's own. ONE file still carries the whole story
        # (desktop reads a single latest.json).
        try:
            from pm import receipt as pm_receipt

            sync = pm_receipt.last_for_update(receipt.correlation_id, consume=True)
            if isinstance(sync, dict):
                for key in ("venv_rebuild", "plugin_bisect", "feature_list", "steps", "exit_code"):
                    if sync.get(key) is not None:
                        receipt.data[f"pm_{key}"] = sync[key]
                if sync.get("outcome") is not None:
                    receipt.data["pm_sync_outcome"] = sync.get("outcome")
                if sync.get("warnings"):
                    receipt.data["pm_warnings"] = sync["warnings"]
                if sync.get("refusal") is not None:
                    receipt.data["pm_refusal"] = sync["refusal"]
        except Exception as exc:  # pragma: no cover — embedding is additive
            logger.debug("pm sync-section embed skipped: %s", exc)
        directory = _receipt_dir()
        directory.mkdir(parents=True, exist_ok=True)
        # Unique name: stamp+pid collides for nested/concurrent receipts in
        # the same process+second — the correlation id makes the name unique
        # per update run. Atomic write for BOTH the stamped receipt and the
        # latest.json pointer (no torn readers).
        from hermes_cli.runtime_state import _atomic_bytes

        # The run's own file (named at begin): the durable running record becomes terminal.
        path = _run_file(directory, receipt.data)
        payload = (json.dumps(receipt.data, indent=2, default=str) + "\n").encode("utf-8")
        _atomic_bytes(path, payload)
        with suppress(Exception):  # stable pointer for the dashboard/desktop
            _write_latest(payload)
        _prune_old_receipts(directory)
        _publish_shared_metrics(receipt.data)
        return path
    except Exception as exc:
        # Visible, not debug: a run that pulled code and left no receipt is exactly the run
        # operators need to post-mortem, and INFO-level logs discard debug (#112465, #112558).
        logger.warning("Could not write update receipt (%s): %s", outcome, exc)
        print(f"  ⚠ Update receipt not written: {exc}")
        return None


def _collection_enabled_now() -> Optional[bool]:
    """Shared-metrics consent via the ALREADY-LOADED config module (never an import: this interpreter
    predates the checkout swap). None when it cannot tell (not loaded, unreadable config)."""
    reader = getattr(sys.modules.get("hermes_cli.config"), "read_raw_config_readonly", None)
    if reader is None:
        return None
    try:
        config: Any = reader()
    except Exception:
        return None
    if type(config) is not dict:  # FailedConfigRead: a fallback, not what the user chose
        return None
    for key in ("telemetry", "shared_metrics"):
        config = config.get(key) if isinstance(config, dict) else None
    return isinstance(config, dict) and config.get("enabled") is True


#: The leading label of a stop reason the parked copy may keep, exactly what the classifier reads:
#: an exception type name (plus its errno token) or one of the fixed phrases Hermes writes
#: (shared_metrics_update._STOP_REASON_PREFIX_CLASSES); never the message after it.
_PARKED_REASON_LABEL = re.compile(
    r"(?:[A-Z][A-Za-z0-9_]{0,63}:(?: \[(?:Errno|WinError) \d+\])?"
    r"|historical takeover preparation failed|Windows gateway recovery failed)")
_STOP_CLASS_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,39}")


def _metric_receipt(data: dict[str, Any]) -> dict[str, Any]:
    """Only what shared_metrics_update.update_receipt_fields reads; never argv or step text.

    Every field the classifier reads is kept in closed form so a parked run classifies exactly as
    the same run finalized in-process: the exit's ``stop_class`` token, the exit code, the stop
    reason's leading label (``-`` for any other text), and flags for the restart/user-action facts.
    """
    raw_pre, raw_restart, raw_reason = (data.get(key) for key in ("pre_update", "gateway_restart", "stop_reason"))
    pre: dict[str, Any] = raw_pre if isinstance(raw_pre, dict) else {}
    restart: dict[str, Any] = raw_restart if isinstance(raw_restart, dict) else {}
    reason: str = raw_reason if isinstance(raw_reason, str) else ""
    label = _PARKED_REASON_LABEL.match(reason)
    stop_class = data.get("stop_class")
    exit_code = data.get("exit_code")
    return {
        "schema": data.get("schema"),
        "update_id": data.get("update_id"), "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"), "outcome": data.get("outcome"),
        "initiator": "desktop" if data.get("initiator") == "desktop" else None,
        "pre_update": {"commit_date": pre.get("commit_date")},
        "stages": [
            {key: mark[key] for key in ("name", "outcome", "at", "mode") if key in mark}
            for mark in data.get("stages") or () if isinstance(mark, dict)
        ],
        "steps": [
            {"name": "admission", "ok": bool(step.get("ok"))}
            for step in data.get("steps") or () if isinstance(step, dict) and step.get("name") == "admission"
        ],
        "fleet": [{"state": row.get("state")} for row in data.get("fleet") or () if isinstance(row, dict)],
        "stop_class": stop_class if isinstance(stop_class, str) and _STOP_CLASS_TOKEN.fullmatch(stop_class) else None,
        "exit_code": exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
        "stop_reason": label.group(0) if label else ("-" if reason else ""),
        "user_action": bool(data.get("user_action")),
        "gateway_restart": {key: bool(restart.get(key)) for key in ("incomplete", "phase_error", "failed_units")},
        "runtime_outcomes": [
            {"outcome": "failed"} for row in data.get("runtime_outcomes") or ()
            if isinstance(row, dict) and row.get("outcome") == "failed"
        ],
    }


def _consent_reader_importable() -> bool:
    """Whether this interpreter can load the config reader the recorder's consent gate uses.

    False in the bare ``-I -S`` bootstrap interpreter that finalizes an update whose dependency
    preparation failed (update_completion._settle_after_commit): it has the new tree but no
    third-party packages, so ``hermes_cli.config`` dies on ``ruamel``. Only asked of an interpreter
    that runs the new tree (never the pre-pull one, which must import nothing); the recorder's
    pre-gate imports the same module next, so a working interpreter loads nothing extra."""
    try:
        import hermes_cli.config  # noqa: F401
    except ImportError:  # ruamel and every other third-party package are missing here
        return False
    return True


def _bare_collection_enabled() -> Optional[bool]:
    """Consent as far as an interpreter without the config reader can tell: the loaded module's
    answer if one is loaded, False when the profile has no config.yaml (the shipped default is off),
    else None (unknowable here; the next normal start decides)."""
    enabled = _collection_enabled_now()
    if enabled is not None:
        return enabled
    from hermes_constants import get_hermes_home

    try:
        (get_hermes_home() / "config.yaml").stat()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return None


def _park_metric_receipt(data: dict[str, Any], enabled: Optional[bool]) -> None:
    """Keep the bounded fields for the next Hermes start (stdlib only): ``report_pending_updates``
    records them when collection is on, ``begin_process`` purges them when it is off. Collection
    known off: park nothing and purge what an earlier run parked."""
    from hermes_constants import get_hermes_home
    from hermes_cli.runtime_state import _atomic_bytes

    pending = get_hermes_home() / "telemetry" / "shared_metrics" / "pending_updates"  # = PENDING_DIRNAME
    if enabled is False:
        import shutil

        shutil.rmtree(pending, ignore_errors=True)  # opted out: nothing parked may be counted later
        return
    pending.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(pending / f"{data.get('update_id')}.json", json.dumps(_metric_receipt(data), default=str).encode())


def _publish_shared_metrics(data: dict[str, Any]) -> None:
    """hermes.update.run/stage from this FINAL receipt; must never fail or slow the update.

    Rows are emitted only by an interpreter that can read consent with the real config reader.
    Any other interpreter parks the bounded fields instead, and the next normal start applies the
    collection gate (records them, or purges them when collection is off)."""
    with suppress(Exception):
        pre, post = data.get("pre_update") or {}, data.get("post_update") or {}
        if data.get("pid") == os.getpid() and not (pre.get("sha") and pre.get("sha") == post.get("sha")):
            # This interpreter began the run before the checkout swap: importing now would load
            # pulled code into it. Consent comes from the config module it already has loaded.
            _park_metric_receipt(data, _collection_enabled_now())
            return
        if not _consent_reader_importable():
            # The bare bootstrap interpreter (dependency preparation failed): no config reader, so
            # it can neither read consent nor emit. Park, unless consent is knowably off.
            _park_metric_receipt(data, _bare_collection_enabled())
            return
        from hermes_cli.observability.shared_metrics_update import record_update_receipt

        record_update_receipt(data)


def finalized_receipt(update_id: str) -> Optional[dict[str, Any]]:
    """The terminal record this process finalized for ``update_id``, published or not.

    The completion child answers its parent from this when the receipt store refused the
    terminal write: the run is still correlated and terminal, only its archive is missing
    (already reported by ``⚠ Update receipt not written``).
    """
    data = _FINALIZED.get(str(update_id))
    return copy.deepcopy(data) if data is not None else None


def finalize_pending_update_receipt(exit_code: Optional[int] = None, stop_reason: str = "") -> Optional[Path]:
    """Command-boundary safety net: persist a still-open receipt, if any. Never raises.

    ``hermes update`` has many early ``sys.exit`` paths (preflight refusals, venv-holder refusal,
    fetch failure) predating the inner finalize calls; finalizing here means refused/failed runs —
    where a receipt matters most — leave a record. Exit 0/None → ``success``, exit 2 → ``refused``
    (preflight convention), else → ``failed`` (``partial`` when the run committed and only owes a
    user action, see ``record_user_action``).

    No-op when no receipt is open (the inner paths already finalized — exactly-once via the popped
    per-context receipt) or when recording was never started. See #91283.
    """
    current = _current.get()
    if current is None:
        return None
    outcome = ("success" if exit_code in (0, None) else "refused" if exit_code == 2
               else "partial" if current.data.get("user_action") else "failed")
    if exit_code is not None:
        with suppress(Exception):
            clone = copy.copy(current)
            clone.data = copy.deepcopy(current.data)
            clone.data["exit_code"] = int(exit_code)
            _current.set(clone)
    return finalize_update_receipt(outcome, stop_reason=stop_reason)


def resume_run_record() -> Optional[Path]:
    """Carry the open run forward from its own archive when another process took it further.

    The completion child resumes this process's snapshot and persists stages (or closes the run)
    the snapshot never saw; closing the run from that stale snapshot would erase them. Returns the
    archive when the run is already closed there (the open receipt is dropped: nothing is left to
    finalize), else None. Never raises.
    """
    current = _current.get()
    found = read_run_record(str(current.data.get("update_id"))) if current is not None else None
    if found is None:
        return None
    path, record = found
    if record.get("finished_at"):
        try:
            _current.reset(current.current_token)
        except ValueError:  # token from another context: pop THIS context only
            _current.set(None)
        return path
    record.pop("writer_pid", None)
    clone = copy.copy(current)
    clone.data = record
    _current.set(clone)
    return None


def finalize_interrupted_update_receipt(stop_reason: str, *, exit_code: int = 130) -> Optional[Path]:
    """Close a run the operator interrupted AFTER the commit point as ``interrupted``. Never raises.

    Not ``failed``: the code already moved, so "still on the previous version" would be false; the
    armed obligations finish the rest. The run's own on-disk record is preferred when it is further
    along (the completion child persisted stages this process never saw).
    """
    print("⚠ Interrupted after the code was updated: the new code is in place; its remaining steps are owed "
          "and the next launch or `hermes update` finishes them.", flush=True)
    if _current.get() is None:
        return None
    closed = resume_run_record()
    if closed is not None:  # the completion child already closed the run
        return closed
    with suppress(Exception):
        clone = copy.copy(_current.get())
        clone.data = copy.deepcopy(clone.data)
        clone.data["exit_code"] = int(exit_code)
        _current.set(clone)
    return finalize_update_receipt("interrupted", stop_reason=stop_reason)


def _prune_old_receipts(directory: Path) -> None:
    with suppress(Exception):
        receipts = (p for p in directory.glob("update_*.json") if p.is_file())
        for stale in sorted(receipts, key=lambda p: p.stat().st_mtime, reverse=True)[_RECEIPT_KEEP:]:
            with suppress(OSError):
                stale.unlink()


def settle_latest_receipt_fleet(fleet: list[dict[str, Any]], *, discharges) -> bool:
    """Record on ``latest.json`` that the fleet it still reports as owed now serves the checkout.

    A failed receipt whose plan rows cannot be matched to a live gateway (unknown identity,
    pre-pull SHAs) keeps ``hermes update`` exiting 1 and every CLI start warning about mixed
    modules, long after the operator's ``hermes gateway restart`` fixed the fleet (#117051). The
    caller has just verified every live row is current at the checkout SHA; persisting that
    matrix as the receipt's post-restart ``fleet`` (and un-flagging ``gateway_restart``) is what
    lets the stale-runtime readers see the recovery. ``discharges(settled_receipt)`` decides on
    the in-memory copy; ``latest.json`` is rewritten only when it answers True, so a catch-up
    that still owes the restart leaves the receipt byte-identical. Only the ``latest.json`` pointer is
    rewritten; the archived per-run file keeps the original outcome. Never raises.
    """
    try:
        path = _receipt_dir() / "latest.json"
        receipt = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(receipt, dict):
            return False
        receipt["fleet"] = list(fleet)
        gateway_restart = receipt.get("gateway_restart")
        if not isinstance(gateway_restart, dict):
            gateway_restart = {}
        gateway_restart.update({"incomplete": False, "phase_error": ""})
        gateway_restart["settled_from_live_fleet_at"] = _utc_now_iso()
        receipt["gateway_restart"] = gateway_restart
        if not discharges(receipt):
            return False
        _write_latest((json.dumps(receipt, indent=2, default=str) + "\n").encode("utf-8"))
        return True
    except Exception as exc:
        logger.debug("Could not settle latest update receipt from the live fleet: %s", exc)
        return False


def read_latest_receipt() -> Optional[dict[str, Any]]:
    """Read the most recent update receipt, or None. Never raises."""
    with suppress(Exception):
        # The newest pointer wins (the root on a tie): the last writer of either store, exactly
        # as when updater and pm shared one folder.
        points = [d / "latest.json" for d in _receipt_dirs() if (d / "latest.json").is_file()]
        if not points:
            return None
        path = max(points, key=lambda p: p.stat().st_mtime)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else None
    return None


def read_receipt_for_action(action_id: str) -> Optional[dict[str, Any]]:
    """Newest receipt written by the dashboard action ``action_id`` (``latest.json`` first, then
    the retained archive), or None. Never raises."""
    latest = read_latest_receipt()
    if latest and latest.get("action_id") == action_id:
        return latest
    with suppress(Exception):
        archives = (p for d in _receipt_dirs() for p in d.glob("update_*.json"))
        for path in sorted(archives, key=lambda p: p.stat().st_mtime, reverse=True):
            with suppress(Exception):
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict) and payload.get("action_id") == action_id:
                    return payload
    return None


def _profile_homes() -> list[tuple[str, Path]]:
    """``(profile, home)`` for the default home plus every valid named profile dir, sorted."""
    from hermes_cli.profiles import _get_default_hermes_home, _get_profiles_root, _PROFILE_ID_RE

    homes: list[tuple[str, Path]] = []
    default_home = _get_default_hermes_home()
    if default_home.is_dir():
        homes.append(("default", default_home))
    root = _get_profiles_root()
    if root.is_dir():
        homes.extend(
            (entry.name, entry)
            for entry in sorted(root.iterdir())
            if entry.is_dir() and entry.name != "default" and _PROFILE_ID_RE.match(entry.name)
        )
    return homes


def _socket_identity(home: Path) -> Optional[tuple[int, dict]]:
    """``(pid, identity)`` declared by the gateway owning ``home``'s control socket, else None.

    A live ``identify`` answer is authoritative — no PID-reuse or stale-file heuristics. Callers
    fall back to ``gateway_state.json`` for gateways that predate the socket or whose socket
    didn't bind.
    """
    try:
        # Prefer the gateway-owned control socket (#92091): identity declared by the process itself,
        # including its own supervisor provenance — no argv/PID inference. Scan fallback below.
        from gateway.control_socket import identify_gateway

        identity = identify_gateway(home)
        return (int(identity.get("pid")), identity) if identity else None
    except Exception:  # probe failure, no gateway, or an unparseable pid
        return None


_CODE_ROOT_MAX_DEPTH = 8


def _code_root_for_path(raw: Any) -> Optional[Path]:
    """Return the Hermes checkout containing an absolute process path."""
    if not isinstance(raw, str) or not raw:
        return None
    with suppress(Exception):
        candidate = Path(raw)
        if not candidate.is_absolute():
            return None
        for parent in [candidate, *candidate.parents][:_CODE_ROOT_MAX_DEPTH]:
            if (parent / "hermes_cli" / "main.py").is_file():
                return parent.resolve()
    return None


def _updater_code_root() -> Optional[Path]:
    return _code_root_for_path(str(Path(__file__).resolve()))


def _gateway_code_root(pid: int, home: Path) -> Optional[Path]:
    """Resolve the checkout served by a verified live gateway when possible."""
    # Older gateways cannot publish a new identity field, but their pid-guarded
    # status record already carries sys.argv (whose first item is the resolved
    # module path for ``python -m`` launches).
    with suppress(Exception):
        from gateway.status import read_runtime_status

        record = read_runtime_status(home / "gateway_state.json") or {}
        if int(record.get("pid")) == pid:
            for probe in record.get("argv") or []:
                root = _code_root_for_path(probe)
                if root:
                    return root

    with suppress(Exception):
        import psutil  # type: ignore

        process = psutil.Process(pid)
        with suppress(Exception):
            root = _code_root_for_path((process.environ() or {}).get("VIRTUAL_ENV"))
            if root:
                return root
        probes: list[Any] = []
        with suppress(Exception):
            probes.append(process.exe())
        with suppress(Exception):
            probes.extend(process.cmdline() or [])
        for probe in probes:
            root = _code_root_for_path(probe)
            if root:
                return root
    return None


EXTERNAL_STATE = "external"
# A gateway this updater runs INSIDE that accepted a self-restart request: it is still on the
# pre-update code by construction and restarts once the updater exits (#100179 / #119597).
RESTART_PENDING_STATE = "restart_pending"


def row_is_external(row: Any) -> bool:
    """Whether a fleet row serves a checkout this update did not touch."""
    return isinstance(row, dict) and row.get("state") == EXTERNAL_STATE


def _fleet_row(
    profile: str, pid: int, code_sha: Any, code_version: Any, expected_sha: Any,
    state: str = "unknown", code_root: Optional[Path] = None,
    expected_root: Optional[Path] = None, served_profiles: Any = None,
    self_restart_pending: Optional[set] = None,
) -> dict[str, Any]:
    if state == "unknown" and code_root and expected_root and code_root != expected_root:
        state = EXTERNAL_STATE
    if state == "unknown" and code_sha and expected_sha:
        state = "current" if str(code_sha) == str(expected_sha) else "stale"
    if state == "stale" and self_restart_pending and pid in self_restart_pending:
        state = RESTART_PENDING_STATE
    row = {
        "profile": profile, "pid": pid, "code_sha": str(code_sha) if code_sha else None,
        "code_version": code_version, "state": state,
        "code_root": str(code_root) if code_root else None,
    }
    # A live, identity-verified multiplexer represents every profile in this
    # list. Keep the field only when its shape is usable: callers use it to
    # discharge per-profile restart obligations, so corrupt status must not
    # widen coverage.
    if isinstance(served_profiles, list) and served_profiles and all(
        isinstance(name, str) and name for name in served_profiles
    ):
        row["served_profiles"] = list(dict.fromkeys(served_profiles))
    return row


# Runtime-status states that do not describe a gateway that should be running now — no down row.
_NOT_EXPECTED_STATES = {"stopped", "startup_failed"}


def collect_fleet_versions(
    *, pre_restart_pids: Optional[list[int]] = None, self_restart_pending: Optional[set] = None,
) -> list[dict[str, Any]]:
    """Snapshot every profile's gateway code identity vs. the current tree.

    ``self_restart_pending`` — pids of gateways that are ancestors of this updater and accepted a
    self-restart request (cron update inside the gateway tree, #100179). They can only restart after
    this process exits, so their pre-update ``code_sha`` is expected: such a row is
    ``restart_pending`` instead of ``stale`` and does not fail the matrix (#119597). Every other
    live gateway on the old sha keeps its ``stale`` verdict.

    Rollout safety: ``down`` requires membership in ``pre_restart_pids`` — a stale state file from a
    long-dead gateway (machine reboot, manual kill weeks ago) must NOT fail every future update.
    Without a pre-restart snapshot (``None``/empty) dead PIDs are skipped (historical behavior).

    ``stale``   — gateway stamped a code_sha that differs from the updated checkout's HEAD (it is still
    serving pre-update modules). ``unknown`` — gateway predates the code-identity stamp (started before this
    feature landed), identity could not be resolved, or the state file's live PID is not the
    verified gateway for that home (``live_gateway_pid_for_home``): ``write_runtime_status`` re-stamps
    ``pid``/``code_sha`` for whatever process writes it, so a foreign writer must never read as
    ``current`` (#110420, sibling of #109680). ``down``    — the gateway was ALIVE when this update
    started (``pre_restart_pids``), its runtime status still says running, but the PID is dead and no
    successor rewrote the record: the restart phase stopped it and nothing came back. Without this row a
    killed-and-never-replaced gateway produced NO entry at all and the matrix passed silently (Phase-1
    verification gap, #88848/#74973 class).
    """
    _pre_restart = {int(p) for p in (pre_restart_pids or []) if isinstance(p, int)}
    _pending = {int(p) for p in (self_restart_pending or ()) if isinstance(p, int)}
    results: list[dict[str, Any]] = []
    expected_sha = _code_identity(refresh=True).get("sha")
    expected_root = _updater_code_root()
    try:
        from gateway.status import (
            live_gateway_pid_for_home,
            read_runtime_status,
            runtime_status_pid_is_live,
        )

        for profile, home in _profile_homes():
            sock = _socket_identity(home)
            if sock is not None:
                pid, identity = sock
                row = _fleet_row(
                    profile, pid, identity.get("code_sha"), identity.get("code_version"), expected_sha,
                    served_profiles=identity.get("served_profiles"),
                    code_root=_gateway_code_root(pid, home), expected_root=expected_root,
                    self_restart_pending=_pending,
                )
                results.append({**row, "source": "socket"})
                continue
            record = read_runtime_status(home / "gateway_state.json")
            if not record:
                continue
            try:
                pid = int(record.get("pid"))
            except (TypeError, ValueError):
                continue
            # A state file is only a fallback claim. Its SHA is evidence about
            # its own PID only when the profile's canonical identity resolver
            # verifies that same live gateway.
            if live_gateway_pid_for_home(home) == pid:
                results.append(
                    _fleet_row(
                        profile, pid, record.get("code_sha"), record.get("code_version"), expected_sha,
                        served_profiles=record.get("served_profiles"),
                        code_root=_gateway_code_root(pid, home), expected_root=expected_root,
                        self_restart_pending=_pending,
                    )
                )
                continue
            # A live non-gateway (or a gateway for another profile) can write a
            # plausible state file. Keep the fail-open visibility row, but never
            # let that file's self-reported SHA or version classify or label
            # the process — both claims have the same trust problem.
            if runtime_status_pid_is_live(record):
                results.append(_fleet_row(profile, pid, None, None, None))
                continue
            # Dead PID (or a live PID recycled by an unrelated process during the update's own
            # churn): a DOWN row only when this exact pid was alive at update start AND the record
            # still claims a running state — "the restart phase stopped it and nothing came back."
            # Everything else (clean stop, startup failure, long-dead stale record) keeps the no-row
            # behavior so the rollout can't false-positive. ``_pre_restart`` is a bare PID set, not
            # (pid, start_time) pairs, so a recycled PID from gateway A landing in B's stale record
            # could still mislabel B as down — inherent to the snapshot's data model.
            # See #93258.
            gw_state = record.get("gateway_state")
            if pid in _pre_restart and isinstance(gw_state, str) and gw_state and gw_state not in _NOT_EXPECTED_STATES:
                results.append(_fleet_row(profile, pid, None, record.get("code_version"), None, state="down"))
    except Exception as exc:
        logger.debug("Fleet version probe failed: %s", exc)
    return results


_FLEET_ROW_LINES = {
    "current": "  ✓ {profile} (pid {pid}) @ {short} — up to date",
    "stale": "  ✗ {profile} (pid {pid}) @ {short} — STALE (pre-update code)",
    "down": "  ✗ {profile} — DOWN (gateway was running before the update; pid {pid} is gone and nothing replaced it)",
    "external": "  ◆ {profile} (pid {pid}) @ {short} — separate checkout, not updated by this run",
    RESTART_PENDING_STATE: (
        "  ↻ {profile} (pid {pid}) @ {short} — restart pending (deferred until this process exits;"
        " the update runs inside this gateway)"
    ),
}
_FLEET_ROW_UNKNOWN = "  ? {profile} (pid {pid}) — version unknown (gateway predates version stamping; restart to enable)"
# A gateway pid the pre-update snapshot did not know that had not published its code identity when
# the settle window closed (#112634): most likely the successor this update relaunched, still
# booting, so "restart to enable" would be wrong — but the poll never observed the restart itself,
# so the copy does not claim one.
_FLEET_ROW_IDENTITY_PENDING = (
    "  ? {profile} (pid {pid}) — new pid since the update, code identity not published yet"
    " — re-check with `hermes gateway status`"
)


def print_fleet_version_matrix(fleet: list[dict[str, Any]]) -> bool:
    """Print the post-update fleet version matrix.

    Returns True when at least one gateway is provably stale (still serving pre-update code) OR
    provably down (killed by the restart phase, nothing came back), so the caller can escalate.
    ``unknown`` entries are reported but do NOT fail the update: gateways started before the
    code-identity stamp existed have no sha to compare, and failing them would be a false-positive
    storm. ``restart_pending`` entries (the gateway this updater runs inside, self-restart
    accepted) are on the old code by construction and do not fail it either (#119597).
    """
    if not fleet:
        return False
    print()
    print("Fleet version check:")
    states = set()
    external_roots: list[str] = []
    for entry in fleet:
        sha = entry.get("code_sha")
        states.add(entry.get("state"))
        if row_is_external(entry) and entry.get("code_root"):
            external_roots.append(f"{entry.get('profile')}: {entry.get('code_root')}")
        fallback = _FLEET_ROW_IDENTITY_PENDING if entry.get("identity_pending") else _FLEET_ROW_UNKNOWN
        print(_FLEET_ROW_LINES.get(entry.get("state"), fallback).format(
            profile=entry.get("profile"), pid=entry.get("pid"), short=sha[:8] if isinstance(sha, str) and sha else "?",
        ))
    if external_roots:
        print()
        print("  ℹ These profiles run their own checkout and are updated separately:")
        for line in external_roots:
            print(f"      {line}")
    if RESTART_PENDING_STATE in states:
        print()
        print("  ℹ A restart-pending gateway has not yet been verified on the new code;")
        print("    check after this update exits with `hermes gateway status`.")
    stale_or_down = sum(1 for entry in fleet if entry.get("state") in ("stale", "down"))
    if stale_or_down:
        print()
        if "stale" in states:
            print("  ⚠ Stale gateways keep serving pre-update code until restarted.")
        if "down" in states:
            print("  ⚠ Down gateways stopped serving messaging entirely.")
        # The code is committed (the update exits 0), but the fleet is not on it yet: say what
        # is still owed after ``✓ Update complete!``; every CLI start repeats it until restarted.
        print()
        print(
            f"⚠ Gateway restart still owed: {stale_or_down} gateway(s) still running the old code (or stopped).")
        print("  Run `hermes gateway restart` (or `hermes -p <profile> gateway restart` for a named")
        print("  profile), then `hermes gateway status` to confirm.")
    return stale_or_down > 0
