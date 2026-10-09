"""Profile-scoped plugin credential names excluded from child environments.

Names only: values remain with Hermes's secret sources. Independent ownership
tokens keep overlapping declarations safe across unload/reload. This is not a
sandbox against a trusted local shell reading the user's files.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import re
import threading
from typing import Callable

_LOCK = threading.RLock()
_REGISTRATIONS: dict[object, tuple[str, frozenset[str]]] = {}
_RETAINED: ContextVar[tuple[tuple[str, frozenset[str]], ...]] = ContextVar(
    "hermes_retained_private_child_env", default=(),
)


def register_private_env_keys(scope: str, names: list[str]) -> Callable[[], None]:
    """Return idempotent cleanup for a declaration owned by the plugin ledger."""
    if (not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 128
            or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", n)
                   for n in names)):
        raise ValueError("private environment keys must be 1..128 valid environment names")
    from hermes_constants import hermes_home_key
    token = object()
    with _LOCK:
        _REGISTRATIONS[token] = (hermes_home_key(scope), frozenset(n.upper() for n in names))

    def release():
        with _LOCK:
            _REGISTRATIONS.pop(token, None)
    return release


def private_env_keys() -> frozenset[str]:
    """Current profile's declared names, case-folded on every platform."""
    retained = _RETAINED.get()
    with _LOCK:
        if not _REGISTRATIONS and not retained:
            return frozenset()
    from hermes_constants import hermes_home_key
    scope = hermes_home_key()
    with _LOCK:
        groups = tuple(_REGISTRATIONS.values()) + retained
        return frozenset(name for owner, names in groups if owner == scope for name in names)


@contextmanager
def retained_middleware_callbacks(manager, kind: str):
    """Capture callbacks and private names together, then run without registry locks.

    Discovery/unload uses the manager lock; individual key-handle disposal uses
    the registry lock. An in-flight callback keeps its profile's names even if
    its plugin is unloaded before the approved child is actually dispatched.
    This snapshot contains names only and never becomes middleware payload.
    """
    if not manager._middleware.get(kind):
        yield ()
        return
    from hermes_cli.plugins_loader import in_plugin_load_worker
    if in_plugin_load_worker():
        # The discovery owner waits for this worker while holding its lock.
        raise RuntimeError("Authorized tool middleware cannot execute during plugin registration")
    from hermes_constants import hermes_home_key
    with manager._discovery_lock, _LOCK:
        callbacks = tuple(manager._middleware.get(kind, ()))
        if callbacks:
            scope = hermes_home_key(manager.scope_key)
            names = frozenset(name for owner, keys in _REGISTRATIONS.values()
                              if owner == scope for name in keys)
            retained = ((scope, names),)
        else:
            retained = ()
    token = _RETAINED.set(_RETAINED.get() + retained)
    try:
        yield callbacks
    finally:
        _RETAINED.reset(token)
