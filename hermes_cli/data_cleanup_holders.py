"""Home-scoped runtime coordination for an explicitly confirmed data wipe."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import time


_OWNER_STOP_HINTS = {
    "systemd": "stop its systemd unit with systemctl before retrying",
    "launchd": "stop its LaunchAgent with launchctl before retrying",
    "desktop": "quit the owning desktop app before retrying",
    "external": "stop the gateway through its external supervisor before retrying",
}


def _drain_manual_gateway(home: Path) -> None:
    from gateway.control_socket import identify_gateway, pause_gateway_for_update
    from gateway.status import get_process_start_time, get_running_pid_identity_strict, _pid_exists

    identity = identify_gateway(home)
    if identity is None:
        if get_running_pid_identity_strict(home / "gateway.pid") is not None:
            raise RuntimeError(f"gateway has no usable control socket for {home}; stop it through its owner and retry")
        return
    if not identity.get("hermes_home") or Path(identity["hermes_home"]).resolve() != home:
        raise RuntimeError(f"gateway control identity does not match {home}")
    owner = identity.get("supervisor")
    if owner != "manual":
        hint = _OWNER_STOP_HINTS.get(owner, "stop it through its supervisor before retrying")
        raise RuntimeError(f"{owner or 'unknown'} gateway still owns {home}; {hint}")
    if identity.get("served_profiles"):
        raise RuntimeError("gateway serves multiple profiles; stop it explicitly before deleting one profile's data")
    pid = identity.get("pid")
    start = identity.get("start_time")
    if type(pid) is not int or pid <= 0 or start is None or get_process_start_time(pid) != start:
        raise RuntimeError(f"gateway process identity is unverified for {home}")
    response = pause_gateway_for_update(home)
    if not response or response.get("pid") != identity.get("pid") or not (
        response.get("pausing") or response.get("already_stopping")
    ):
        raise RuntimeError(f"gateway refused to drain for {home}")
    deadline = time.monotonic() + min(120.0, max(5.0, float(response.get("drain_timeout", 30.0)) + 5.0))
    while time.monotonic() < deadline:
        current = get_process_start_time(pid)
        original_gone = not _pid_exists(pid) or (current is not None and current != start)
        if original_gone and identify_gateway(home) is None and get_running_pid_identity_strict(home / "gateway.pid") is None:
            return
        time.sleep(0.05)
    raise RuntimeError(f"gateway did not exit for {home}; no data removed")


def _refuse_backend_writers(home: Path) -> None:
    from hermes_constants import get_default_hermes_root
    from hermes_cli.process_identity import LEDGER_FILENAME, _pid_alive_matches, _read_ledger

    root = get_default_hermes_root(home=home).resolve()
    rows = _read_ledger(root / LEDGER_FILENAME)
    if rows is None:
        raise RuntimeError(f"cannot read backend ownership: {root / LEDGER_FILENAME}")
    for row in rows:
        if row.get("purpose") not in {"serve", "dashboard"}:
            continue
        # The recorded profile is the initial UI selection, not the backend's
        # write scope: it can serve any profile under this machine root.
        pid = row.get("pid")
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("backend ownership contains an invalid PID")
        if _pid_alive_matches(pid, row.get("create_time")) is not False:
            raise RuntimeError(f"backend PID {pid} still owns {home}; close its desktop/dashboard before retrying")


def _refuse_cron_writers(home: Path) -> None:
    from contextlib import closing
    import sqlite3
    from gateway.status import get_process_start_time, _pid_exists

    path = home / "cron" / "executions.db"
    try:
        path.stat()
    except FileNotFoundError:
        return
    # Importing cron initializes scheduler/config defaults. A deletion probe
    # must only read the existing ledger, not recreate the data being removed.
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM executions WHERE status IN ('claimed','running')").fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot inspect cron owners in {path}: {exc}") from exc
    for raw in rows:
        row = dict(raw)
        pid = row["pid"]
        if row.get("handoff_pending"):
            raise RuntimeError(f"cron execution {row['id']} is being handed to a worker; wait for it to finish")
        if not _pid_exists(pid):
            continue
        current = get_process_start_time(pid)
        recorded = row.get("process_started_at")
        if current is None or recorded is None or current == recorded:
            raise RuntimeError(f"cron execution {row['id']} still owns {home} (PID {pid}); stop it before retrying")


def _refuse_multiplexer(home: Path) -> None:
    from hermes_constants import get_default_hermes_root
    from gateway.control_socket import identify_gateway
    from gateway.status import get_running_pid_identity_strict

    root = get_default_hermes_root(home=home).resolve()
    if home == root:
        return
    identity = identify_gateway(root)
    if identity is not None:
        if home.name in identity.get("served_profiles", []):
            raise RuntimeError(f"the default gateway serves {home}; stop the multiplexer explicitly before retrying")
    elif get_running_pid_identity_strict(root / "gateway.pid") is not None:
        raise RuntimeError("the default gateway's profile scope is unverified; stop it explicitly before retrying")


@contextmanager
def quiescent_home(home: Path):
    """Hold existing writer locks through deletion; do not remove their inodes."""
    from gateway.status import _release_file_lock, _try_acquire_file_lock
    from hermes_cli.active_sessions import _FileLock, _prune_dead, _read_entries
    from hermes_cli.runtime_state import _lock
    from tools.checkpoint_pruning import store_lock

    home = home.resolve()
    _refuse_backend_writers(home)
    _refuse_multiplexer(home)
    _drain_manual_gateway(home)
    with ExitStack() as stack:
        stack.enter_context(store_lock(home / "checkpoints"))
        gateway_lock = stack.enter_context((home / "gateway.lock").open("a+", encoding="utf-8"))
        if not _try_acquire_file_lock(gateway_lock):
            raise RuntimeError(f"gateway is still active for {home}")
        stack.callback(_release_file_lock, gateway_lock)
        stack.enter_context(_FileLock(home / "runtime" / "active_sessions.lock"))
        sessions = _prune_dead(_read_entries(home / "runtime" / "active_sessions.json", strict=True), strict=True)
        if sessions:
            pids = sorted({entry["pid"] for entry in sessions})
            raise RuntimeError(f"live chat sessions still own {home} (PIDs {pids}); close them before retrying")
        for path in (home / ".backup.lock", home / "cron" / ".tick.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+b"))
            if not _lock(handle.fileno(), wait=False):
                raise RuntimeError(f"an active writer still owns {path}")
        _refuse_cron_writers(home)
        yield
