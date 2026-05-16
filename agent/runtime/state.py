"""Typed run state with an explicit mutation audit log.

The legacy ``state: dict[str, Any]`` allowed callbacks to mutate anything
unobserved. ``RunState`` replaces that: every mutation goes through a typed
method, every mutation records a ``StateMutation`` with a required reason,
and the mutation log is part of the run's audit trail.

Callbacks receive a ``RunState`` instance, not a dict. Attempting to set
attributes outside the typed API fails. Once a run completes the state is
frozen — any later mutation raises ``StateFrozenError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .interfaces import Clock


MutationKind = Literal["set", "append", "increment", "delete"]


class StateFrozenError(RuntimeError):
    """Raised when code tries to mutate a frozen ``RunState``."""


@dataclass(frozen=True, slots=True)
class StateMutation:
    at: float
    kind: MutationKind
    key: str
    value: Any
    reason: str


class RunState:
    """Append-only audited key/value store for a single run.

    Callbacks read via ``get()`` / ``snapshot()``; they write via ``set()``,
    ``append()``, ``increment()``, or ``delete()`` — each requires a
    ``reason`` so the audit log is meaningful.
    """

    __slots__ = ("_values", "_mutations", "_clock", "_frozen")

    def __init__(self, clock: Clock) -> None:
        self._values: dict[str, Any] = {}
        self._mutations: list[StateMutation] = []
        self._clock = clock
        self._frozen = False

    # ---- read access --------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def snapshot(self) -> dict[str, Any]:
        """Shallow copy suitable for handing to a governance context."""
        return dict(self._values)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._values.keys())

    @property
    def mutations(self) -> tuple[StateMutation, ...]:
        return tuple(self._mutations)

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ---- mutation -----------------------------------------------------------

    def set(self, key: str, value: Any, *, reason: str) -> None:
        self._guard(reason)
        self._record("set", key, value, reason)
        self._values[key] = value

    def append(self, key: str, value: Any, *, reason: str) -> None:
        self._guard(reason)
        bucket = self._values.setdefault(key, [])
        if not isinstance(bucket, list):
            raise TypeError(f"cannot append to non-list state key {key!r} (type={type(bucket).__name__})")
        bucket.append(value)
        self._record("append", key, value, reason)

    def increment(self, key: str, amount: int = 1, *, reason: str) -> int:
        self._guard(reason)
        current = self._values.get(key, 0)
        if not isinstance(current, (int, float)):
            raise TypeError(f"cannot increment non-numeric state key {key!r} (type={type(current).__name__})")
        new_value = current + amount
        self._values[key] = new_value
        # Audit log records the post-mutation value so all kinds (set /
        # append / increment) carry the same semantics — ``value`` is what
        # the key holds (or holds at end of) after the mutation.
        self._record("increment", key, new_value, reason)
        return new_value

    def delete(self, key: str, *, reason: str) -> None:
        self._guard(reason)
        if key in self._values:
            self._record("delete", key, self._values[key], reason)
            del self._values[key]

    # ---- lifecycle ----------------------------------------------------------

    def freeze(self) -> None:
        """Make the state immutable. Called when the run terminates."""
        self._frozen = True

    # ---- internals ----------------------------------------------------------

    def _guard(self, reason: str) -> None:
        if self._frozen:
            raise StateFrozenError("RunState is frozen; the run has already terminated")
        if not reason:
            raise ValueError("RunState mutations require a non-empty reason")

    def _record(self, kind: MutationKind, key: str, value: Any, reason: str) -> None:
        self._mutations.append(
            StateMutation(at=self._clock.now(), kind=kind, key=key, value=value, reason=reason)
        )
