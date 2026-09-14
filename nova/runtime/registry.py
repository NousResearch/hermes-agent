"""Runtime adapter registry.

Adapters register by name so callers select one from configuration rather than by
importing it. This is what lets a future adapter be added without touching any caller —
and what keeps the Hermes import out of NOVA's import graph until a Hermes runtime is
actually requested.
"""

from __future__ import annotations

from typing import Callable, Dict

from nova.errors import RuntimeAdapterError
from nova.runtime.base import AgentRuntime

#: name -> zero-argument factory. Factories are lazy so importing ``nova.runtime`` does
#: not import every adapter's dependencies.
_FACTORIES: Dict[str, Callable[..., AgentRuntime]] = {}


def register_runtime(name: str, factory: Callable[..., AgentRuntime]) -> None:
    """Register an adapter factory under ``name`` (last registration wins)."""
    _FACTORIES[name] = factory


def available_runtimes() -> list[str]:
    _ensure_builtins()
    return sorted(_FACTORIES)


def get_runtime(name: str, **kwargs) -> AgentRuntime:
    """Construct the adapter registered as ``name``."""
    _ensure_builtins()
    factory = _FACTORIES.get(name)
    if factory is None:
        known = ", ".join(sorted(_FACTORIES)) or "(none registered)"
        raise RuntimeAdapterError(f"unknown runtime {name!r}; available: {known}")
    return factory(**kwargs)


def _ensure_builtins() -> None:
    """Import bundled adapters on first use."""
    if "hermes" not in _FACTORIES:
        from nova.runtime.hermes import HermesRuntime

        register_runtime("hermes", HermesRuntime)
