"""Serialized read/modify/write transactions for Bot Marketplace profile metadata."""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

_T = TypeVar("_T")
_METADATA_LOCK = threading.RLock()
_METADATA_LOCK_STATE = threading.local()
_LOCK_TIMEOUT_SECONDS = 10.0


def metadata_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "profile.yaml"


def _kernel_lock(handle, acquire: bool) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def bot_metadata_lock():
    """Lock one profile's metadata transaction in this process and across backend processes."""
    with _METADATA_LOCK:
        depth = getattr(_METADATA_LOCK_STATE, "depth", 0)
        if depth:
            _METADATA_LOCK_STATE.depth = depth + 1
            try:
                yield
            finally:
                _METADATA_LOCK_STATE.depth -= 1
            return

        path = metadata_path()
        lock_path = path.with_name(".profile.yaml.lock")
        # A deleted profile must not be resurrected by a late status poll.
        if not lock_path.parent.is_dir():
            raise FileNotFoundError("Bot profile no longer exists")
        if msvcrt is not None and (not lock_path.exists() or lock_path.stat().st_size == 0):  # pragma: no cover - Windows
            with contextlib.suppress(OSError, PermissionError):
                lock_path.write_text(" ", encoding="utf-8")
        handle = lock_path.open("r+" if msvcrt is not None else "a+", encoding="utf-8")
        acquired = fcntl is None and msvcrt is None
        try:
            if not acquired:
                deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        _kernel_lock(handle, True)
                        acquired = True
                        break
                    except (BlockingIOError, OSError, PermissionError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out waiting for bot profile metadata lock")
                        time.sleep(0.05)
            _METADATA_LOCK_STATE.depth = 1
            yield
        finally:
            _METADATA_LOCK_STATE.depth = 0
            if acquired and (fcntl is not None or msvcrt is not None):
                with contextlib.suppress(OSError, IOError):
                    _kernel_lock(handle, False)
            handle.close()


def _read_unlocked() -> dict[str, Any]:
    path = metadata_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return raw if isinstance(raw, dict) else {}


def read_bot_metadata() -> dict[str, Any]:
    with bot_metadata_lock():
        return _read_unlocked()


def mutate_bot_metadata(mutator: Callable[[dict[str, Any]], _T]) -> tuple[dict[str, Any], _T]:
    """Apply ``mutator`` to the latest metadata and atomically persist its complete result."""
    from utils import atomic_yaml_write

    with bot_metadata_lock():
        metadata = _read_unlocked()
        result = mutator(metadata)
        atomic_yaml_write(metadata_path(), metadata, sort_keys=False)
        return metadata, result
