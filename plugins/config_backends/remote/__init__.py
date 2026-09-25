"""Remote config backend: ``HERMES_CONFIG_BACKEND=remote`` serves each profile's config from the
config plane (config-config) instead of ``config.yaml``. See ``backend.py``."""
from __future__ import annotations

import threading
from typing import Optional

from .backend import RemoteBackend

_BACKEND: Optional[RemoteBackend] = None
_LOCK = threading.Lock()


def get_remote_backend() -> RemoteBackend:
    """The process-wide instance (one poller and one in-memory layer per profile home)."""
    global _BACKEND
    if _BACKEND is None:
        with _LOCK:
            if _BACKEND is None:
                _BACKEND = RemoteBackend()
    return _BACKEND


def _reset_for_tests() -> None:
    global _BACKEND
    with _LOCK:
        if _BACKEND is not None:
            _BACKEND._stop.set()
        _BACKEND = None


__all__ = ["RemoteBackend", "get_remote_backend"]
