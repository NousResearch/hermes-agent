"""Profile-scoped registrations for external browser-login backends."""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

_BACKENDS: Dict[str, Type] = {}
_SCOPED_BACKENDS: Dict[str, Dict[str, Type]] = {}
_REGISTRY_LOCK = threading.RLock()


def _builtin_classes() -> tuple[Type, ...]:
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.local import LocalLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    return (LocalLoginBackend, OnePasswordLoginBackend, BitwardenLoginBackend)


def _merged(scope: Optional[str]) -> Dict[str, Type]:
    merged = dict(_BACKENDS)
    merged.update(_SCOPED_BACKENDS.get(scope or hermes_home_key(), {}))
    return merged


def _known_classes(scope: Optional[str]) -> List[Type]:
    return [*_builtin_classes(), *_merged(scope).values()]


def register_backend(backend_cls: Type, *, scope: Optional[str] = None) -> bool:
    """Register a ``LoginBackend`` class without allowing name or prefix shadowing."""
    from agent.vault_backends.base import LoginBackend

    if not isinstance(backend_cls, type) or not issubclass(backend_cls, LoginBackend):
        logger.warning("Ignoring login backend %r: does not inherit from LoginBackend", backend_cls)
        return False
    name = getattr(backend_cls, "name", "")
    prefix = getattr(backend_cls, "prefix", "")
    if not name or not name.replace("_", "").isalnum() or name != name.lower():
        logger.warning("Ignoring login backend with invalid name %r", name)
        return False
    if not isinstance(prefix, str) or not prefix:
        logger.warning("Ignoring login backend '%s': prefix must be a non-empty string", name)
        return False

    with _REGISTRY_LOCK:
        known = _known_classes(scope)
        if any(cls.name == name for cls in known):
            logger.warning("Login backend '%s' already registered; ignoring duplicate", name)
            return False
        owner = next(
            (
                cls
                for cls in known
                if prefix.startswith(cls.prefix) or cls.prefix.startswith(prefix)
            ),
            None,
        )
        if owner is not None:
            logger.warning(
                "Ignoring login backend '%s': handle prefix %r conflicts with backend '%s'",
                name,
                prefix,
                owner.name,
            )
            return False
        target = _BACKENDS if scope is None else _SCOPED_BACKENDS.setdefault(scope, {})
        target[name] = backend_cls
    return True


def list_backend_classes(*, scope: Optional[str] = None) -> List[Type]:
    """Registered plugin classes in deterministic registration order."""
    with _REGISTRY_LOCK:
        return list(_merged(scope).values())


def snapshot_registration(name: str, *, scope: Optional[str] = None) -> Optional[Type]:
    with _REGISTRY_LOCK:
        return (_BACKENDS if scope is None else _SCOPED_BACKENDS.get(scope, {})).get(name)


def restore_registration(
    name: str,
    current: Type,
    previous: Optional[Type],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore one host-owned registration if it is still current."""
    with _REGISTRY_LOCK:
        target = _BACKENDS if scope is None else _SCOPED_BACKENDS.setdefault(scope, {})
        if target.get(name) is not current:
            return False
        if previous is None:
            target.pop(name, None)
        else:
            target[name] = previous
        if scope is not None and not target:
            _SCOPED_BACKENDS.pop(scope, None)
    return True


def _reset_for_tests() -> None:
    with _REGISTRY_LOCK:
        _BACKENDS.clear()
        _SCOPED_BACKENDS.clear()