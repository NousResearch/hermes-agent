"""Thread-safe lazy singletons for plugin authors (stdlib-only).

The ``if _client is None: _client = Expensive()`` footgun: two threads pass the guard, both
build, the second leaks the first's connections. :func:`lazy_singleton` decorates a zero-arg
accessor; :class:`SingletonSlot` is the manual slot when the instance depends on an argument.
"""

from __future__ import annotations

import functools
import threading
from typing import Callable, Generic, Optional, TypeVar

__all__ = ["lazy_singleton", "SingletonSlot"]

T = TypeVar("T")


class SingletonSlot(Generic[T]):
    """Thread-safe lazy slot: caches the first successfully-built instance ("first config wins").
    The factory runs at most once under concurrent first calls; if it raises, nothing is cached
    and the next call retries. ``_slot.get(lambda: Honcho(**resolve(config)))``."""

    __slots__ = ("_lock", "_value", "_set")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._set = False

    def get(self, factory: Callable[[], T]) -> T:
        # Fast path without the lock: a bool + ref read is atomic under the GIL.
        if self._set:
            return self._value  # type: ignore[return-value]
        with self._lock:
            if self._set:
                return self._value  # type: ignore[return-value]
            value = factory()
            self._value = value
            self._set = True
            return value

    def peek(self) -> Optional[T]:
        """Return the cached instance without building it (None if unset)."""
        return self._value if self._set else None

    def reset(self) -> None:
        """Drop the cached instance so the next ``get()`` rebuilds it."""
        with self._lock:
            self._value = None
            self._set = False


def lazy_singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """Wrap a zero-argument factory into a thread-safe lazy singleton accessor (factory runs once
    even under concurrent first calls; on raise the next call retries). ``.reset()`` drops it."""
    slot: SingletonSlot[T] = SingletonSlot()

    @functools.wraps(factory)
    def accessor() -> T:
        return slot.get(factory)

    accessor.reset = slot.reset  # type: ignore[attr-defined]
    return accessor
