"""Register pluggable coordinator implementations."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class CoordinatorFactory(Protocol[T_co]):
    def __call__(self, **kwargs: Any) -> T_co: ...


class OrchestratorRegistry:
    """Maps coordinator names to factories (dependency injection hook)."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("coordinator name must be non-empty")
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(name.strip().lower(), None)

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        return self._factories.get(name.strip().lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories.keys()))
