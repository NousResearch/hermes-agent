"""Process-wide state a shared (hosted) process may hold: per-agent caches and declared
tenant-neutral values.

A hosted process serves many tenants' agents at once, so a module-level ``dict``/``list``/``set``,
an ``lru_cache`` or a class attribute filled at runtime is shared by all of them: one agent's
entry is visible to the next. The hosted-path lint (rule HX015 in ``scripts/code_health``)
refuses such state in every module the hosted entry point can import, unless it is one of:

- a :class:`ScopedCache`, keyed by the ``(tenant, agent)`` of the bound scope, so each agent sees
  only its own slot;
- declared tenant-neutral with :func:`tenant_neutral` and a reason: data that is the same for
  every tenant by construction (a model catalog, tool schemas).

Import-safe and stdlib-only beside :mod:`hermes_scope`.
"""

from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar

from hermes_scope import current_scope

V = TypeVar("V")
T = TypeVar("T")

_MISSING = object()


class UnscopedStateError(RuntimeError):
    """A :class:`ScopedCache` was used with no scope bound.

    There is no shared fallback slot on purpose: a read or write outside any scope would land in
    state every agent can see. Bind the activity's scope at its entry point
    (``agent.runtime_scope.bind_profile_scope`` / ``bind_state``)."""


class ScopedCache(Generic[V]):
    """One value per ``(tenant, agent)``, created on first use by *factory*.

    ``cache.slot()`` returns the bound agent's value (a fresh ``factory()`` the first time);
    ``cache.drop()`` forgets it, for eviction. Both raise :class:`UnscopedStateError` when no
    scope is bound. The tenant and agent always come from :func:`hermes_scope.current_scope`,
    never from a caller-supplied key, so a caller cannot reach another agent's slot."""

    def __init__(self, name: str, factory: Callable[[], V] = dict) -> None:  # type: ignore[assignment]
        if not isinstance(name, str) or not name:
            raise ValueError("ScopedCache needs a non-empty name")
        self.name = name
        self._factory = factory
        self._slots: dict[tuple[str, str], V] = {}
        self._lock = threading.Lock()

    def _key(self) -> tuple[str, str]:
        scope = current_scope()
        if scope is None:
            raise UnscopedStateError(
                f"ScopedCache {self.name!r} used with no scope bound; bind the activity's scope "
                "at its entry point (agent.runtime_scope.bind_profile_scope / bind_state)")
        return scope.tenant, scope.agent

    def slot(self) -> V:
        key = self._key()
        with self._lock:
            if key not in self._slots:
                self._slots[key] = self._factory()
            return self._slots[key]

    def drop(self) -> None:
        key = self._key()
        with self._lock:
            self._slots.pop(key, None)


def tenant_neutral(value: T = _MISSING, *, reason: str) -> T:  # type: ignore[assignment]
    """Declare module state the same for every tenant; returns *value* unchanged.

    ``_CATALOG = tenant_neutral({}, reason="model catalog: identical for every tenant")``. On a
    function, as a decorator above its cache: ``@tenant_neutral(reason="...")`` then
    ``@functools.lru_cache``. The lint accepts only a non-empty literal *reason*; reviewers check
    it, as they check a ``# health: allow`` comment."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("tenant_neutral needs a non-empty reason")
    if value is _MISSING:
        return lambda fn: fn  # type: ignore[return-value]
    return value
