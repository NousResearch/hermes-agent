"""One backend migration at a time per home: the run lock of ``hermes gateway migrate --backends``.

Every run that may change this home (``--apply``, ``--restore``, the rollback of an interrupted run, and
the background run a ``--detach`` starts) takes ``backend_migration/run.lock`` in the home BEFORE it reads
any state for its preflight, and holds it until the process ends. It is a kernel lock (flock / msvcrt, the
gateway's own helper in ``gateway.status``), so the kernel drops it when the holder dies: a killed run
never leaves a stale lock behind, and the next run's rollback (rule 7) can take it at once.

A run that finds the lock held changes nothing, never touches the manifest, and exits 3 saying how to
watch the run that holds it (``--status``). ``--status`` and ``--apply --dry-run`` never take it.

A detached run holds the lock in the background child itself (the caller returns at once and may be
stopped by the run). Its caller learns whether the child got the lock through a one-shot acknowledgement
file (``backend_migration/detach-ack-TOKEN.json``) the child writes right after trying, so a caller whose
child lost the race reports the refusal (exit 3; the dashboard routes answer 409) instead of "started".
A child that lost the race writes no refusal record for ``--status``: the run holding the lock owns the
status.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import IO, Optional

LOCK_NAME = "run.lock"
STATUS_COMMAND = "hermes gateway migrate --backends --status"
#: How long a ``--detach`` caller waits for its background run to report whether it got the lock.
ACK_WAIT_SECONDS = 60.0

_HELD: Optional[IO[str]] = None  # this process's lock handle, kept open until the process ends


def _dir(home: Path) -> Path:
    path = Path(home) / "backend_migration"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def lock_path(home: Path) -> Path:
    return Path(home) / "backend_migration" / LOCK_NAME


def holder_pid(home: Path) -> Optional[int]:
    """The pid the holder stamped into the lock file (None when unknown)."""
    try:
        return int(lock_path(home).read_text(encoding="utf-8-sig").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def acquire(home: Path) -> bool:
    """Take this home's run lock for the rest of this process (non-blocking). False when another holds it."""
    global _HELD
    if _HELD is not None:
        return True
    from gateway.status import _try_acquire_file_lock
    # Held (open) until this process ends; only ever written through this handle.
    handle = open(_dir(home) / LOCK_NAME, "a+", encoding="utf-8")  # windows-footgun: ok
    if not _try_acquire_file_lock(handle):
        handle.close()
        return False
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        pass  # the pid is only a hint for the loser's message
    _HELD = handle
    return True


def release() -> None:
    """Drop the lock early (tests; a normal run keeps it until the process ends)."""
    global _HELD
    if _HELD is None:
        return
    from gateway.status import _release_file_lock
    _release_file_lock(_HELD)
    _HELD.close()
    _HELD = None


def status_command(home: Path) -> str:
    """``--status`` for this home's profile (``hermes -p NAME ...``)."""
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    return STATUS_COMMAND.replace("hermes", hermes_command(home, powershell=os.name == 'nt'), 1)


def busy_message(home: Path) -> str:
    pid = holder_pid(home)
    who = f" (pid {pid})" if pid else ""
    return (f"A backend migration is already running for this home{who}; nothing changed. "
            f"Watch it with: {status_command(home)}")


def print_busy(home: Path, *, as_json: bool, mode: str) -> None:
    message = busy_message(home)
    if as_json:
        print(json.dumps({"applied": False, "mode": mode, "error": "already_running", "message": message,
                          "holder_pid": holder_pid(home), "status_command": status_command(home)}))
    else:
        print(f"✗ {message}")


# --------------------------------------------------------------------------- detached hand-off


def new_ack_token() -> str:
    return secrets.token_hex(8)


def _ack_path(home: Path, token: str) -> Path:
    return Path(home) / "backend_migration" / f"detach-ack-{token}.json"


def write_ack(home: Path, token: str, *, locked: bool) -> None:
    """The background run tells its caller whether it got the lock (it writes this once, first thing)."""
    if not token.isalnum():
        return
    from utils import atomic_json_write
    _dir(home)
    atomic_json_write(_ack_path(home, token), {"locked": locked, "pid": os.getpid(),
                                               "holder_pid": None if locked else holder_pid(home)}, mode=0o600)


def prune_stale_acks(home: Path, now: Optional[float] = None) -> None:
    """Delete acknowledgements no caller waits for any more: a caller that stopped waiting (exit 7,
    unconfirmed) never reads the one its background run wrote later. Older than the wait: abandoned."""
    cutoff = (time.time() if now is None else now) - ACK_WAIT_SECONDS
    for path in (Path(home) / "backend_migration").glob("detach-ack-*.json"):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < cutoff:
                path.unlink()


def wait_ack(home: Path, token: str, seconds: Optional[float] = None) -> Optional[dict]:
    """The background run's acknowledgement (consumed), or None when it did not answer in time."""
    path = _ack_path(home, token)
    deadline = time.monotonic() + (ACK_WAIT_SECONDS if seconds is None else seconds)
    while True:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            path.unlink(missing_ok=True)
            return data
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)
