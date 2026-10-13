"""Dispatch-start accounting and ordered worker gates."""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class _ToolExecutionState:
    """Atomically decide whether a call started before it was abandoned."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._abandoned = False
        self._function_args: dict[str, Any] | None = None
        self._middleware_trace: list[dict[str, Any]] = []

    def try_start(
        self,
        function_args: dict[str, Any],
        middleware_trace: list[dict[str, Any]],
    ) -> bool:
        """Mark the real callback boundary unless the executor abandoned it."""
        with self._lock:
            if self._abandoned or self._started:
                return False
            self._function_args = copy.deepcopy(function_args)
            self._middleware_trace = copy.deepcopy(middleware_trace)
            self._started = True
            return True

    def abandon(self) -> bool:
        """Prevent a future start and return whether execution already began."""
        with self._lock:
            self._abandoned = True
            return self._started

    def apply_to(self, ref) -> bool:
        started, args, trace = self.started_metadata()
        if started:
            ref.args, ref.trace = args, trace
        return started

    def has_started(self) -> bool:
        with self._lock:
            return self._started

    def started_metadata(
        self,
    ) -> tuple[bool, dict[str, Any] | None, list[dict[str, Any]]]:
        """Return callback-boundary metadata captured atomically at dispatch."""
        with self._lock:
            args = (
                copy.deepcopy(self._function_args)
                if self._function_args is not None
                else None
            )
            return self._started, args, copy.deepcopy(self._middleware_trace)


class _BatchAbandoned(BaseException):
    """Raised inside a worker when the batch was abandoned before dispatch; a BaseException
    so ``except Exception`` handlers in the middleware chain can't swallow it."""


class _StartOrderGate:
    """Serialize worker dispatch by submit order (prompts appear in call order); ``abandon()``
    releases every parked worker so none dispatches a tool the turn already gave up on."""

    def __init__(self, timeout: float) -> None:
        self._condition = threading.Condition()
        self._next_order = 0
        self._timeout = timeout
        self.abandoned = threading.Event()

    def abandon(self) -> None:
        self.abandoned.set()
        with self._condition:
            self._condition.notify_all()

    def begin_in_order(self, order: int, callback=None, *, tool_name: str = "") -> bool:
        """Wait for ``order``, run ``callback``, advance. Returns False if abandoned."""
        with self._condition:
            # Bounded wait so one wedged dispatch can't starve later-ordered workers; on
            # expiry proceed out of order (interleaved prompts beat starvation). ``>=`` (not
            # ``==``) releases every skipped worker at once; abandoned short-circuits.
            in_order = self._condition.wait_for(
                lambda: self._next_order >= order or self.abandoned.is_set(), timeout=self._timeout,
            )
            if self.abandoned.is_set():
                return False  # the turn already synthesized this result; don't advance
            if not in_order:
                logger.warning(
                    "start-order gate timed out for %s (order=%d next=%d); proceeding out of order",
                    tool_name or "tool", order, self._next_order,
                )
            try:
                if callback is not None:
                    callback()
            finally:
                self._next_order = max(self._next_order, order + 1)
                self._condition.notify_all()
        return True


class _WorkerStartOnce:
    """One worker's handle on the start-order gate: advances at most once, raising
    ``_BatchAbandoned`` (instead of dispatching late) when the batch was abandoned."""

    def __init__(self, gate: _StartOrderGate, order: int, tool_name: str) -> None:
        self._gate, self._order, self._tool_name, self._advanced = gate, order, tool_name, False

    def advance(self, callback=None) -> None:
        if self._advanced:
            return
        self._advanced = True
        if not self._gate.begin_in_order(self._order, callback, tool_name=self._tool_name):
            raise _BatchAbandoned(self._tool_name)
