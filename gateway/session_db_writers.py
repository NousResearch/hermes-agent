"""Stop same-home SQLite writers around gateway --replace.

Dashboard is an independent s6 service. `hermes gateway run --replace` historically
SIGTERM'd only the gateway PID. Dashboard kept state.db-wal/shm open, the exiting
gateway unlinked that generation, and the new gateway hit DeletedWalGenerationError.

Hold dashboard down first, then replace the gateway, then bring dashboard back only
after the new gateway process is the one that will open SQLite.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

logger = logging.getLogger(__name__)

_DASHBOARD_SERVICE_CANDIDATES = (
    Path("/run/service/dashboard"),
    Path("/run/s6-rc/servicedirs/dashboard"),
)
_S6_SVC_BINS = ("s6-svc", "/command/s6-svc", "/package/admin/s6/command/s6-svc")
_DASHBOARD_PURPOSES = frozenset({"serve", "dashboard"})


def dashboard_service_dirs() -> list[Path]:
    """Return live s6 dashboard supervise dirs (supervise/ present)."""
    found: list[Path] = []
    for path in _DASHBOARD_SERVICE_CANDIDATES:
        if path.is_dir() and (path / "supervise").is_dir():
            found.append(path)
    return found


def _s6_svc(flag: str, service: Path) -> bool:
    last_exc: Exception | None = None
    for binary in _S6_SVC_BINS:
        try:
            result = subprocess.run(
                [binary, flag, str(service)],
                capture_output=True, timeout=5, check=False,
            )
        except FileNotFoundError as exc:
            last_exc = exc
            continue
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("%s %s %s failed: %s", binary, flag, service, exc)
            return False
        if result.returncode != 0:
            logger.warning(
                "%s %s %s exited %s: %s",
                binary, flag, service, result.returncode,
                (result.stderr or result.stdout or b"").decode("utf-8", "replace")[:200],
            )
            return False
        return True
    logger.warning("s6-svc %s %s failed: %s", flag, service, last_exc)
    return False


def hold_supervised_dashboard_down() -> list[Path]:
    """s6-svc -d so supervise will not restart dashboard during gateway replace."""
    held: list[Path] = []
    for service in dashboard_service_dirs():
        if _s6_svc("-d", service):
            held.append(service)
            logger.info("Held dashboard service down at %s for gateway replace", service)
    return held


def release_supervised_dashboard(held: Sequence[Path]) -> None:
    """s6-svc -u after the replacement gateway owns SQLite."""
    for service in held:
        if _s6_svc("-u", service):
            logger.info("Released dashboard service at %s after gateway replace", service)


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _same_home(pid: int, home: str) -> bool:
    from hermes_cli.dashboard_procs import _hermes_home_for_pid

    env_home = _hermes_home_for_pid(pid)
    return env_home is not None and os.path.realpath(env_home) == home


def iter_same_home_dashboard_pids(hermes_home: Path, *, except_pids: Iterable[int] = ()) -> list[int]:
    """Same-home dashboard/serve PIDs via spawn ledger, then canonical subcommand."""
    from hermes_cli.process_identity import ledger_entries
    from hermes_cli.update_cmd import _hermes_holder_subcommand

    home = os.path.realpath(str(hermes_home))
    skip = {int(p) for p in except_pids}
    skip.add(os.getpid())
    found: list[int] = []
    seen: set[int] = set()

    try:
        entries = ledger_entries()
    except Exception:
        entries = []
    for entry in entries:
        if entry.get("purpose") not in _DASHBOARD_PURPOSES:
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int) or pid in skip or pid in seen:
            continue
        if not _same_home(pid, home):
            continue
        seen.add(pid)
        found.append(pid)

    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in skip or pid in seen:
            continue
        if _hermes_holder_subcommand(_cmdline(pid)) not in _DASHBOARD_PURPOSES:
            continue
        if not _same_home(pid, home):
            continue
        seen.add(pid)
        found.append(pid)
    return found


def terminate_pids(pids: Sequence[int], *, force: bool = False) -> None:
    sig = getattr(signal, 'SIGKILL', signal.SIGTERM) if force else signal.SIGTERM  # windows-footgun: ok
    for pid in pids:
        try:
            os.kill(pid, sig)  # windows-footgun: ok — POSIX dashboard PIDs only
        except ProcessLookupError:
            continue
        except PermissionError:
            logger.warning("Permission denied signalling dashboard PID %s", pid)


def _deleted_sidecar_holder_iter() -> Callable | None:
    try:
        from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders
    except ImportError:
        logger.error("iter_deleted_sqlite_sidecar_holders unavailable; aborting replace")
        return None
    return iter_deleted_sqlite_sidecar_holders


def wait_deleted_sidecar_holders_gone(db_path: Path, *, timeout_s: float = 10.0) -> bool:
    """True when no process holds a deleted WAL/SHM inode for db_path."""
    holders_fn = _deleted_sidecar_holder_iter()
    if holders_fn is None:
        return False
    deadline = time.monotonic() + timeout_s
    while True:
        holders = list(holders_fn(db_path))
        if not holders:
            return True
        if time.monotonic() >= deadline:
            logger.error(
                "Deleted WAL/SHM holders still present for %s after %.1fs: %s",
                db_path, timeout_s, holders[:8],
            )
            return False
        time.sleep(0.1)


def fail_closed_if_deleted_holders(hermes_home: Path) -> bool:
    """True if --replace may open SQLite. False leaves dashboard held down."""
    db_path = Path(hermes_home) / "state.db"
    if wait_deleted_sidecar_holders_gone(db_path):
        return True
    logger.error("Aborting gateway --replace; deleted WAL/SHM holders remain for %s", db_path)
    return False


def quiesce_session_db_writers_for_replace(
    hermes_home: Path, *, except_pids: Iterable[int] = (),
) -> list[Path]:
    """Hold dashboard down, SIGTERM leftover dashboard PIDs, return s6 dirs to release later."""
    held = hold_supervised_dashboard_down()
    pids = iter_same_home_dashboard_pids(hermes_home, except_pids=except_pids)
    if pids:
        logger.info("SIGTERM dashboard/serve PIDs before gateway replace: %s", pids)
        terminate_pids(pids, force=False)
    return held
