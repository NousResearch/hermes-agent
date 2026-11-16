"""Remote secret store: ``HERMES_SECRET_STORE=remote`` keeps every profile secret on the
connector's secrets plane instead of the profile's ``.env`` and ``auth.json``. See ``backend.py``.

A SecretStoreProvider, discovered by ``plugins.secret_store`` and selected only by the
environment (``hermes_secret_routing``); never loaded as a general plugin."""
from __future__ import annotations

import threading
from typing import Optional

from .backend import RemoteSecretStoreProvider

_PROVIDER: Optional[RemoteSecretStoreProvider] = None
_LOCK = threading.Lock()


def get_provider() -> RemoteSecretStoreProvider:
    """The process-wide instance (one poller, and one view per profile home loaded at start)."""
    global _PROVIDER
    if _PROVIDER is None:
        with _LOCK:
            if _PROVIDER is None:
                _PROVIDER = RemoteSecretStoreProvider()
    return _PROVIDER


def register(ctx) -> None:
    ctx.register_secret_store(get_provider())


def _reset_for_tests() -> None:
    global _PROVIDER
    with _LOCK:
        if _PROVIDER is not None:
            _PROVIDER._stop.set()
        _PROVIDER = None


__all__ = ["RemoteSecretStoreProvider", "get_provider", "register"]
