"""Shared concurrency helpers for plugin authors.

The most common plugin footgun is the lazy process-wide singleton:

    _client = None

    def get_client():
        global _client
        if _client is not None:
            return _client
        _client = ExpensiveClient(...)   # <-- TOCTOU: two threads both run this
        return _client

When two threads call ``get_client()`` before the singleton is set, both pass
the ``is not None`` guard, both run the expensive initialization, and the
second write clobbers the first — leaking whatever resource the first client
opened (connections, file handles, background threads).

Multi-threaded agent sessions share one process (delegated tool calls,
background workers, the self-improvement fork), so this race is reachable in
practice. Rather than make every plugin author remember to hand-roll
double-checked locking, this module gives them two thread-safe primitives:

* :func:`lazy_singleton` — decorator for the zero-arg accessor case.
* :class:`SingletonSlot` — manual slot for accessors that build different
  instances depending on a config/key argument.

Both are import-light (stdlib ``threading`` only) so any plugin can import
them without dragging in heavyweight host modules.
"""

from __future__ import annotations

import functools
import threading
from typing import Callable, Generic, Optional, TypeVar

__all__ = ["lazy_singleton", "SingletonSlot"]

T = TypeVar("T")


def lazy_singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """Wrap a zero-argument factory into a thread-safe lazy singleton accessor.

    The wrapped callable returns the same instance on every call; the factory
    runs exactly once even under concurrent first calls, using double-checked
    locking. A ``.reset()`` attribute is attached for tests/teardown.

    Example::

        @lazy_singleton
        def get_client():
            return ExpensiveClient(load_config())

        client = get_client()   # built once, safe across threads
        get_client.reset()      # drop the instance (next call rebuilds)

    Note: if the factory raises, no instance is cached and the next call
    retries (the lock is released either way).
    """
    lock = threading.Lock()
    box: list = []  # one-element [instance]; empty == not yet built

    @functools.wraps(factory)
    def accessor() -> T:
        if box:
            return box[0]
        with lock:
            if box:  # re-check inside the lock
                return box[0]
            instance = factory()
            box.append(instance)
            return instance

    def reset() -> None:
        with lock:
            box.clear()

    accessor.reset = reset  # type: ignore[attr-defined]
    return accessor


class SingletonSlot(Generic[T]):
    """Thread-safe lazy slot for accessors that take a build argument.

    Use this when the cached instance depends on a config/key passed to the
    accessor (so a bare zero-arg :func:`lazy_singleton` doesn't fit). The slot
    caches the first successfully-built instance and ignores the argument on
    subsequent calls — matching the established "first config wins" singleton
    semantics most plugins already rely on.

    Example::

        _slot: SingletonSlot[Honcho] = SingletonSlot()

        def get_honcho_client(config=None):
            return _slot.get(lambda: Honcho(**resolve(config)))

        def reset_honcho_client():
            _slot.reset()

    The factory runs at most once even under concurrent first calls. If the
    factory raises, nothing is cached and the next call retries.
    """

    __slots__ = ("_lock", "_value", "_set")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._set = False

    def get(self, factory: Callable[[], T]) -> T:
        # Fast path: already built, no lock needed (a set bool + ref read is
        # atomic under CPython's GIL).
        if self._set:
            return self._value  # type: ignore[return-value]
        with self._lock:
            if self._set:  # re-check inside the lock
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
