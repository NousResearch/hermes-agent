"""Profile-scoped registrations for external browser-login backends.

There is no process-global fallback: a login provider belongs to one profile.
The plugin manager owns registration lifetime through its existing scoped ledger.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from typing import TYPE_CHECKING

from hermes_constants import hermes_home_key

if TYPE_CHECKING:
    from agent.vault_backends.base import LoginBackend

logger = logging.getLogger(__name__)
_SCOPED_BACKENDS: dict[str, dict[str, type[LoginBackend]]] = {}
_REGISTRY_LOCK = threading.RLock()


def _builtin_classes() -> tuple[type[LoginBackend], ...]:
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.local import LocalLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    return (LocalLoginBackend, OnePasswordLoginBackend, BitwardenLoginBackend)


def register_backend(backend_cls: type[LoginBackend], *, scope: str | None = None) -> bool:
    """Reject invalid classes and ambiguous namespaces; retain the existing owner."""
    from agent.vault_backends.base import LoginBackend

    if not isinstance(backend_cls, type) or not issubclass(backend_cls, LoginBackend) or inspect.isabstract(backend_cls):
        logger.warning("Ignoring login backend: expected a concrete LoginBackend subclass")
        return False
    name = getattr(backend_cls, "name", None)
    prefix = getattr(backend_cls, "prefix", None)
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        logger.warning("Ignoring login backend: invalid name")
        return False
    if not isinstance(prefix, str) or not prefix or any(char.isspace() for char in prefix):
        logger.warning("Ignoring login backend '%s': invalid handle prefix", name)
        return False

    scope = hermes_home_key() if scope is None else scope
    with _REGISTRY_LOCK:
        known = (*_builtin_classes(), *_SCOPED_BACKENDS.get(scope, {}).values())
        if any(cls.name == name for cls in known):
            logger.warning("Login backend '%s' already registered; ignoring duplicate", name)
            return False
        if any(prefix.startswith(cls.prefix) or cls.prefix.startswith(prefix) for cls in known):
            logger.warning("Ignoring login backend '%s': overlapping handle prefix", name)
            return False
        _SCOPED_BACKENDS.setdefault(scope, {})[name] = backend_cls
    return True


def list_backend_classes(*, scope: str | None = None) -> tuple[type[LoginBackend], ...]:
    """Registered classes for this profile, in registration order."""
    scope = hermes_home_key() if scope is None else scope
    with _REGISTRY_LOCK:
        return tuple(_SCOPED_BACKENDS.get(scope, {}).values())


def snapshot_registration(name: str, *, scope: str | None = None) -> type[LoginBackend] | None:
    scope = hermes_home_key() if scope is None else scope
    with _REGISTRY_LOCK:
        return _SCOPED_BACKENDS.get(scope, {}).get(name)


def restore_registration(
    name: str,
    current: type[LoginBackend],
    previous: type[LoginBackend] | None,
    *,
    scope: str | None = None,
) -> bool:
    """Release a host-owned slot only if the disposing generation still owns it."""
    scope = hermes_home_key() if scope is None else scope
    with _REGISTRY_LOCK:
        target = _SCOPED_BACKENDS.get(scope, {})
        if target.get(name) is not current:
            return False
        if previous is None:
            target.pop(name, None)
        else:
            target[name] = previous
        if not target:
            _SCOPED_BACKENDS.pop(scope, None)
    return True


def _reset_for_tests() -> None:
    with _REGISTRY_LOCK:
        _SCOPED_BACKENDS.clear()
