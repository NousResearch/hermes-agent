"""Profile-scoped plugin credential names excluded from child environments.

Names only: values remain with Hermes's secret sources. Independent ownership
tokens keep overlapping declarations safe across unload/reload. This is not a
sandbox against a trusted local shell reading the user's files.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

_LOCK = threading.RLock()
_REGISTRATIONS: dict[object, tuple[str, frozenset[str]]] = {}


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
    with _LOCK:
        if not _REGISTRATIONS:
            return frozenset()
    from hermes_constants import hermes_home_key
    scope = hermes_home_key()
    with _LOCK:
        return frozenset(name for owner, names in _REGISTRATIONS.values() if owner == scope for name in names)
