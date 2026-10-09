"""Profile attribution for callbacks defined by loaded plugins.

The platform registry needs this small runtime primitive without importing the tool registry.
Attribution is intentionally durable: unloading a plugin's current override policy must not
make a delayed callback lose the profile that originally loaded its module.
"""

from __future__ import annotations

import functools
import threading
from typing import Callable, Optional, Set

from hermes_constants import hermes_home_key


_lock = threading.RLock()
_module_scopes: dict[str, Set[Optional[str]]] = {}


def _scope_key(scope: Optional[str]) -> Optional[str]:
    """Canonicalize profile keys at the attribution boundary."""
    return hermes_home_key(scope) if scope is not None else None


def record_plugin_module_scope(module_namespace: str, scope: Optional[str]) -> None:
    """Remember the original registry scope spelling used for this plugin module."""
    with _lock:
        _module_scopes.setdefault(module_namespace, set()).add(scope)


def _plugin_namespace_of_module(module_namespace: str) -> Optional[str]:
    """Resolve a module/submodule to its recorded plugin namespace."""
    with _lock:
        matches = [
            namespace
            for namespace in _module_scopes
            if module_namespace == namespace or module_namespace.startswith(f"{namespace}.")
        ]
    if matches:
        return max(matches, key=len)
    if module_namespace.startswith("hermes_plugins."):
        return ".".join(module_namespace.split(".")[:2])
    return None


def plugin_scope_for_module(module_namespace: str) -> Optional[str]:
    """Return the durable profile scope for a loaded plugin module."""
    owner = _plugin_namespace_of_module(module_namespace)
    if owner is None:
        return None
    with _lock:
        scopes = _module_scopes.get(owner)
        if not scopes:
            return None
        active_scope = hermes_home_key()
        active_matches = [scope for scope in scopes if _scope_key(scope) == active_scope]
        if active_matches:
            return active_matches[0]
        if len(scopes) == 1:
            return next(iter(scopes))
        raise PermissionError(
            f"Plugin module {module_namespace!r} is active in multiple profiles and cannot "
            "register outside one of those scopes."
        )


def _callable_module(callback: Callable) -> str:
    """Resolve a defining module through wrappers, partials, and callable objects."""
    current = callback
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        globals_dict = getattr(current, "__globals__", None)
        if isinstance(current, functools.partial):
            current = current.func
        elif getattr(current, "__func__", None) is not None:
            current = current.__func__
        elif isinstance(globals_dict, dict) and globals_dict.get("__name__", ""):
            return str(globals_dict["__name__"])
        elif getattr(current, "__wrapped__", None) is not None:
            current = current.__wrapped__
        else:
            break
    module_name = getattr(current, "__module__", "")
    return str(module_name or getattr(type(current), "__module__", "") or "")


def plugin_scope_for_callable(callback: Callable) -> Optional[str]:
    """Return the durable plugin scope for any supported callable shape."""
    module_name = _callable_module(callback)
    return plugin_scope_for_module(module_name) if module_name else None


def plugin_namespace_for_module(module_namespace: str) -> Optional[str]:
    """Return the owning plugin namespace for a module, if it is known."""
    return _plugin_namespace_of_module(module_namespace)
