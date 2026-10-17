"""Replace a gateway left running by an older Hermes on this install's own home.

After an upgrade the old daemon keeps answering the home's control socket, but its ``identify`` has
no ``runtime_protocol`` (it predates the session-authority runtime). Every client of this version
ensures through :func:`hermes_cli.gateway_runtime.ensure_gateway_runtime` (the TUI bootstrap,
Desktop's ``hermes gateway ensure``, ``hermes chat``), so ensure replaces that owner once, exactly as
``hermes gateway stop`` would stop it (a planned-stop marker in the owner's home, then SIGTERM; a
bounded wait; a start-time-guarded force kill), and then starts the current one the normal way.

Refused, never stopped:
- a gateway that speaks a NEWER runtime protocol than this client (a downgrade must not kill it);
- a gateway of another install: its home resolves outside this install's root (a profile symlinked
  into another HERMES root), or the process serves another home than the socket that answered;
- anything whose ``identify`` cannot be pinned to one live process (pid/start time disagree, our
  own process, not a gateway).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Graceful drain before the guarded force kill; an idle old gateway exits in well under a second.
_GRACEFUL_STOP_SECONDS = 10.0
_FORCED_STOP_SECONDS = 5.0

# One line each, short enough for the TUI's activity row after its "gateway incompatible: " prefix.
REFUSAL_DETAIL = {
    "owner_newer": "a newer Hermes's gateway owns this profile; update Hermes or run: hermes gateway stop",
    "foreign_install": "another Hermes install's gateway owns this profile; stop it there: hermes gateway stop",
    "owner_unverified": "an older Hermes gateway owns this profile and was not verified; run: hermes gateway stop",
    "owner_is_parent": "this process was started by an older Hermes gateway that is still running; it is replaced when that gateway restarts",
    "owner_did_not_exit": "an older Hermes gateway owns this profile and did not stop; run: hermes gateway stop",
    "owner_unreachable": "an older Hermes gateway owns this profile; run: hermes gateway stop",
}


def _install_root() -> Path:
    from hermes_constants import get_default_hermes_root
    return Path(get_default_hermes_root()).resolve()


def _identify(home: Path, timeout: float) -> tuple[Path, dict] | None:
    """``(control_home, identify payload)`` of the process answering for *home*: its own socket, or
    the default multiplexer's for a served named profile (the same order discovery uses)."""
    from hermes_cli.gateway_runtime_discovery import query_identify
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    for control in (home, multiplexer_root_for(home)):
        if control is None:
            continue
        try:
            return control, query_identify(control, timeout=timeout)
        except (FileNotFoundError, ConnectionRefusedError):
            continue
    return None


def _same(a: Any, b: Path) -> bool:
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(b.resolve()))
    except (OSError, TypeError, ValueError):
        return False


def _within_install(home: Path, root: Path) -> bool:
    home = home.resolve()
    return home == root or (home.parent == root / "profiles")


def _verified_owner(control: Path, payload: dict) -> tuple[int, Any] | str:
    from gateway.status import _GATEWAY_KIND, _pid_exists, get_process_start_time, start_time_fingerprints_match
    protocol = payload.get("runtime_protocol")
    if type(protocol) is int and protocol > 1:
        return "owner_newer"
    if not _within_install(control, _install_root()) or not _same(payload.get("hermes_home"), control):
        return "foreign_install"
    pid, recorded = payload.get("pid"), payload.get("start_time")
    if (payload.get("kind") != _GATEWAY_KIND or type(pid) is not int or pid <= 1
            or pid == os.getpid() or recorded is None or not _pid_exists(pid)):
        return "owner_unverified"
    if pid == os.getppid():
        # A child this gateway spawned (a restart-safe cron worker) can run code an update installed
        # ahead of the gateway's own replacement; stopping its parent is never the child's call.
        return "owner_is_parent"
    current = get_process_start_time(pid)
    try:
        if current is None or not start_time_fingerprints_match(recorded, current):
            return "owner_unverified"
    except (TypeError, ValueError):
        return "owner_unverified"
    return pid, current


def _write_planned_stop(control: Path, pid: int, start_time: Any) -> None:
    """The marker ``hermes gateway stop`` writes, in the OWNER's home (it reads its own), so the old
    gateway exits as a planned stop: clean exit, ``stopped`` persisted, no supervisor revival."""
    from gateway.status import _PLANNED_STOP_MARKER_FILENAME, _utc_now_iso, _write_marker
    _write_marker(control / _PLANNED_STOP_MARKER_FILENAME, {
        "target_pid": pid, "target_start_time": start_time,
        "stopper_pid": os.getpid(), "written_at": _utc_now_iso(),
    })


def _wait_gone(pid: int, seconds: float) -> bool:
    from gateway.status import _pid_exists
    end = time.monotonic() + max(seconds, 0.0)
    while _pid_exists(pid):
        if time.monotonic() >= end:
            return False
        time.sleep(0.1)
    return True


def replace_incompatible_owner(home: Path, *, deadline: float) -> str | None:
    """Stop the older-Hermes gateway answering for *home*; None once it is gone, else a refusal code
    (see :data:`REFUSAL_DETAIL`). Never starts anything: the caller's ensure loop does that."""
    from gateway.status import terminate_pid
    budget = deadline - time.monotonic()
    if budget <= 0:
        return "owner_unreachable"
    try:
        found = _identify(home, timeout=min(2.0, budget))
    except (OSError, ValueError, TypeError):  # timeout, refused/denied peer, malformed reply
        return "owner_unreachable"
    if found is None:
        return None  # it is already gone: nothing to replace
    control, payload = found
    verdict = _verified_owner(control, payload)
    if isinstance(verdict, str):
        return verdict
    pid, start_time = verdict
    _write_planned_stop(control, pid, start_time)
    graceful = min(_GRACEFUL_STOP_SECONDS, max(deadline - time.monotonic() - _FORCED_STOP_SECONDS, 1.0))
    try:
        if os.name != "nt":
            # Windows has no SIGTERM for an asyncio loop: the marker watcher IS its graceful stop.
            terminate_pid(pid, expected_start_time=start_time)
        if _wait_gone(pid, graceful):
            return None
        terminate_pid(pid, force=True, expected_start_time=start_time)
    except ProcessLookupError:
        return None
    except (PermissionError, OSError):
        return "owner_did_not_exit" if not _wait_gone(pid, 0) else None
    return None if _wait_gone(pid, _FORCED_STOP_SECONDS) else "owner_did_not_exit"
