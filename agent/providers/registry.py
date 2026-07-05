"""Provider registry for Phase 4b LLM Execution Engine.

Read-only at runtime: register_provider() is called at module-load
time only. There is no unregister or set_provider API.

Each adapter module (agent/providers/*.py) calls register_provider()
when it's imported. The Engine calls get_provider() at runtime to
instantiate the chosen adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.provider_adapter import ProviderAdapter


# Internal registry. Module-level so it's shared across imports.
_PROVIDER_REGISTRY: dict[str, type["ProviderAdapter"]] = {}


class UnknownProviderError(KeyError):
    """Raised when get_provider() is called with a name not in the registry.

    The Engine catches this exception ONLY in UnknownProviderError
    contexts; all other KeyErrors propagate.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unknown provider {name!r}")


def register_provider(name: str, adapter_class: type["ProviderAdapter"]) -> None:
    """Register an adapter class under a stable, lowercase name.

    Idempotency: re-registration is allowed (last writer wins). This
    enables test suites to swap adapters without polluting the global
    registry permanently.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"provider name must be a non-empty string, got {name!r}")
    _PROVIDER_REGISTRY[name] = adapter_class


def unregister_provider(name: str) -> None:
    """Remove a provider from the registry. Used only by tests.

    NOT exposed in the production Engine path. Production code does
    not call this.
    """
    _PROVIDER_REGISTRY.pop(name, None)


def get_provider(name: str) -> "ProviderAdapter":
    """Instantiate the adapter registered under name.

    Raises UnknownProviderError if name is not in the registry.
    """
    if name not in _PROVIDER_REGISTRY:
        raise UnknownProviderError(name)
    cls = _PROVIDER_REGISTRY[name]
    return cls()


def list_providers() -> tuple[str, ...]:
    """Return the sorted tuple of registered provider names."""
    return tuple(sorted(_PROVIDER_REGISTRY.keys()))


def is_registered(name: str) -> bool:
    """Return True if a provider is registered under name."""
    return name in _PROVIDER_REGISTRY


def reset_registry() -> None:
    """Clear the registry. Used only by tests.

    NOT exposed in the production Engine path.
    """
    _PROVIDER_REGISTRY.clear()