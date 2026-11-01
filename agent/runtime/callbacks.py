"""Step-callback dispatch.

Callbacks are the single extension point of ``MultiStepLoop``. Hermes-specific
behavior — context compression, redaction, budget enforcement, telemetry —
lands here rather than as inline branches inside the loop.

A callback receives the step *after* it has been appended to memory but
before the loop's next iteration begins. Callbacks may inspect ``RunState``
and mutate it via the typed API (``set`` / ``append`` / ``increment`` /
``delete`` with a required reason). Setting ``state.set("__interrupt__",
True, reason=...)`` terminates the run cooperatively.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Protocol, runtime_checkable

from .state import RunState
from .steps import MemoryStep


@runtime_checkable
class StepCallback(Protocol):
    """Any callable matching this shape qualifies."""

    def __call__(self, step: MemoryStep, state: RunState) -> None: ...


_AnyCallback = Callable[[MemoryStep, RunState], None]


class CallbackRegistry:
    """Dispatch callbacks by step type with ``isinstance`` semantics.

    Registering against ``MemoryStep`` catches every step type. Registering
    against a subclass scopes the callback. Callbacks fire in registration
    order within a type bucket.
    """

    def __init__(self) -> None:
        self._by_type: dict[type[MemoryStep], list[_AnyCallback]] = defaultdict(list)

    def register(self, step_type: type[MemoryStep], callback: _AnyCallback) -> None:
        if not isinstance(step_type, type) or not issubclass(step_type, MemoryStep):
            raise TypeError(f"step_type must be a MemoryStep subclass, got {step_type!r}")
        self._by_type[step_type].append(callback)

    def dispatch(self, step: MemoryStep, state: RunState) -> None:
        for step_type, callbacks in self._by_type.items():
            if isinstance(step, step_type):
                for callback in callbacks:
                    callback(step, state)

    def clear(self) -> None:
        self._by_type.clear()

    def __len__(self) -> int:
        return sum(len(callbacks) for callbacks in self._by_type.values())
