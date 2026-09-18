"""Profile-scoped registry for plugin credential-vault backends.

Built-in backends remain owned by :mod:`agent.vault_backends.base`; this registry
holds only third-party classes registered through ``PluginContext``.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

_BUILTIN_NAMES = frozenset({"local", "onepassword", "bitwarden"})
_BUILTIN_PREFIXES = frozenset({"vault_", "op:", "bw:"})
_backends: Dict[str, Type] = {}
_scoped_backends: Dict[str, Dict[str, Type]] = {}
_lock = threading.RLock()


def _validate(backend: Type) -> Optional[str]:
    from agent.vault_backends.base import LoginBackend

    if not isinstance(backend, type) or not issubclass(backend, LoginBackend):
        return "does not inherit from LoginBackend"
    name = getattr(backend, "name", "")
    prefix = getattr(backend, "prefix", "")
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum() or name != name.lower():
        return f"has invalid name {name!r}"
    if not isinstance(prefix, str) or not prefix:
        return "has an empty prefix"
    if name in _BUILTIN_NAMES:
        return f"name {name!r} shadows a built-in backend"
    if prefix in _BUILTIN_PREFIXES:
        return f"prefix {prefix!r} shadows a built-in backend"
    return None


def register_backend(backend: Type, *, scope: Optional[str] = None) -> bool:
    """Register one plugin backend class; return False for invalid collisions."""
    problem = _validate(backend)
    if problem:
        logger.warning("Ignoring login backend %r: %s", backend, problem)
        return False
    name, prefix = backend.name, backend.prefix
    with _lock:
        target = _backends if scope is None else _scoped_backends.setdefault(scope, {})
        effective = dict(_backends)
        effective.update(_scoped_backends.get(scope or hermes_home_key(), {}))
        existing = target.get(name)
        if existing is not None and existing is not backend:
            logger.warning("Ignoring login backend '%s': name is already registered", name)
            return False
        owner = next((registered.name for registered in effective.values()
                      if registered.name != name and registered.prefix == prefix), None)
        if owner:
            logger.warning("Ignoring login backend '%s': prefix %r is already owned by '%s'", name, prefix, owner)
            return False
        target[name] = backend
    return True


def list_backends(*, scope: Optional[str] = None) -> List[Type]:
    """Return registered classes, with the active profile overlaying global entries."""
    with _lock:
        merged = dict(_backends)
        merged.update(_scoped_backends.get(scope or hermes_home_key(), {}))
        return list(merged.values())


def snapshot_registration(name: str, *, scope: Optional[str] = None) -> Optional[Type]:
    with _lock:
        target = _backends if scope is None else _scoped_backends.get(scope, {})
        return target.get(name)


def restore_registration(name: str, current: Type, previous: Optional[Type], *, scope: Optional[str] = None) -> bool:
    """Restore a plugin lifecycle slot only if it is still owned by ``current``."""
    with _lock:
        target = _backends if scope is None else _scoped_backends.setdefault(scope, {})
        if target.get(name) is not current:
            return False
        if previous is None:
            target.pop(name, None)
        else:
            target[name] = previous
        if scope is not None and not target:
            _scoped_backends.pop(scope, None)
    return True


def _reset_for_tests() -> None:
    with _lock:
        _backends.clear()
        _scoped_backends.clear()
