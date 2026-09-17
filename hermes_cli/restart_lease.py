"""Cross-process serialization for gateway lifecycle operations.

The lease is shared by updates, CLI/service restarts, safe-restart wrappers, and
boot recovery. It is an OS lock: stale metadata alone can never block recovery.
Nested subprocesses inherit a token and join the owner's lease instead of
self-deadlocking.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from hermes_constants import get_hermes_home

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt  # type: ignore[import-not-found]

LEASE_TOKEN_ENV = "HERMES_RESTART_LEASE_TOKEN"


class RestartLeaseBusy(RuntimeError):
    """Another lifecycle controller owns the restart lease."""


def restart_lease_path(home: Path | None = None) -> Path:
    return Path(home or get_hermes_home()).expanduser().resolve() / "state" / "gateway-restart.lease"


def _try_lock(handle) -> bool:
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:  # pragma: no cover - Windows
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle) -> None:
    with contextlib.suppress(OSError):
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _metadata(handle) -> dict:
    try:
        handle.seek(0)
        value = json.loads(handle.read().decode("utf-8", errors="replace") or "{}")
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


@dataclass(frozen=True)
class RestartLease:
    path: Path
    token: str
    purpose: str
    joined: bool = False

    def child_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env[LEASE_TOKEN_ENV] = self.token
        return env


@contextlib.contextmanager
def restart_lease(
    purpose: str, *, home: Path | None = None, timeout: float = 0.0, poll: float = 0.1
) -> Iterator[RestartLease]:
    """Acquire the restart lease or join an inherited owner token."""
    path = restart_lease_path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    inherited = os.environ.get(LEASE_TOKEN_ENV, "").strip()
    handle = path.open("a+b")
    current = _metadata(handle)
    if inherited and current.get("token") == inherited:
        handle.close()
        yield RestartLease(path, inherited, purpose, joined=True)
        return

    deadline = time.monotonic() + max(0.0, float(timeout))
    while not _try_lock(handle):
        if time.monotonic() >= deadline:
            owner = _metadata(handle)
            handle.close()
            raise RestartLeaseBusy(
                f"gateway restart already owned by PID {owner.get('pid') or 'unknown'} "
                f"({owner.get('purpose') or 'unknown'})"
            )
        time.sleep(max(0.01, float(poll)))

    token = secrets.token_hex(16)
    payload = {"pid": os.getpid(), "purpose": str(purpose), "token": token, "acquired_at": time.time()}
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    handle.seek(0)
    handle.truncate()
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
    prior = os.environ.get(LEASE_TOKEN_ENV)
    os.environ[LEASE_TOKEN_ENV] = token
    try:
        yield RestartLease(path, token, purpose)
    finally:
        if prior is None:
            os.environ.pop(LEASE_TOKEN_ENV, None)
        else:
            os.environ[LEASE_TOKEN_ENV] = prior
        _unlock(handle)
        handle.close()
