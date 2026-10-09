"""Late-bound host integrations used by the lower plugin runtime.

The runtime owns plugin lifecycle and contracts; host surfaces register the few callbacks that
must remain above this boundary.  Bindings are deliberately process-global and replaceable so
CLI, gateway, dashboard, and tests can install their own host implementation without importing
host modules from :mod:`plugin_runtime`.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable


_lock = RLock()
_bindings: dict[str, Callable[..., Any]] = {}


def bind_plugin_host(**callbacks: Callable[..., Any] | None) -> None:
    """Register host-owned callbacks; ``None`` leaves an existing callback unchanged."""
    with _lock:
        for name, callback in callbacks.items():
            if callback is not None:
                _bindings[name] = callback


def get_plugin_host_callback(name: str) -> Callable[..., Any] | None:
    """Return one callback, or ``None`` when its host surface is not loaded."""
    with _lock:
        return _bindings.get(name)


def clear_plugin_host_bindings() -> None:
    """Clear process-global bindings for isolated tests."""
    with _lock:
        _bindings.clear()


__all__ = ["bind_plugin_host", "get_plugin_host_callback", "clear_plugin_host_bindings"]
