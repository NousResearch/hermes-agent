"""AgentLoop -- extracted iteration control with middleware hooks.

Replaces the raw while-loop inside AIAgent.run_conversation() with a
testable, composable loop class.  Phase 2 of the loop refactor:

  Phase 1  agent/config.py   -- 4 dataclasses for AIAgent params
  Phase 2  agent/loop.py     -- AgentLoop + LoopContext + MiddlewareBase
  (future phases will migrate the real loop body into middleware)

Design goals
------------
- S — AgentLoop owns iteration control, budget, interrupts
- O — extensible via middleware hooks without modifying the loop
- D — depends on abstract callables (api_call_fn, tool_dispatch_fn),
      not on AIAgent directly
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoopContext -- budget + interrupt state
# ---------------------------------------------------------------------------

class LoopContext:
    """Mutable iteration budget and interrupt signal.

    Separated from AgentLoop so middleware and callers can inspect / control
    the loop without reaching into the loop object itself.
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        self._consumed: int = 0
        self._stopped: bool = False       # set by request_interrupt()
        self._grace_remaining: int = 0    # set by enable_grace_call()

    # -- Budget queries -----------------------------------------------------

    @property
    def remaining(self) -> int:
        """Iterations still available (grace calls not included)."""
        base = self.max_iterations - self._consumed
        return max(base, 0)

    def should_continue(self) -> bool:
        """True when the loop may run another iteration.

        Accounts for consumed budget, explicit interrupt, and grace calls.
        """
        if self._stopped:
            return False
        if self.remaining > 0:
            return True
        if self._grace_remaining > 0:
            return True
        return False

    def consume(self) -> bool:
        """Record one iteration.  Returns True if budget remains."""
        self._consumed += 1
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
        return self.should_continue()

    # -- Interrupt ----------------------------------------------------------

    def request_interrupt(self) -> None:
        """Signal the loop to stop before the next iteration."""
        self._stopped = True

    # -- Grace call ---------------------------------------------------------

    def enable_grace_call(self) -> None:
        """Allow one more iteration even though budget is exhausted.

        Mirrors AIAgent._budget_grace_call: the model gets a final chance
        to produce a summary when it just ran out of iterations.
        """
        self._grace_remaining += 1


# ---------------------------------------------------------------------------
# MiddlewareBase -- hook interface
# ---------------------------------------------------------------------------

class MiddlewareBase:
    """Base class for AgentLoop middleware.

    Override any combination of hooks.  All default implementations are
    no-ops so middleware only needs to define the hooks it cares about.
    """

    def before_iteration(self, ctx: LoopContext, iteration: int, **kwargs) -> None:
        """Called before each API call.  kwargs may include 'messages'."""

    def after_iteration(
        self, ctx: LoopContext, iteration: int, result: Dict[str, Any]
    ) -> None:
        """Called after each API call with the raw result dict."""

    def on_interrupt(self, ctx: LoopContext, iteration: int) -> None:
        """Called when an interrupt is detected."""

    def on_budget_exhausted(self, ctx: LoopContext, iteration: int) -> None:
        """Called when the iteration budget runs out."""

    def on_tool_call(
        self, ctx: LoopContext, iteration: int, tool_call: Dict[str, Any]
    ) -> None:
        """Called for each individual tool call in a batch."""

    def on_tool_result(
        self, ctx: LoopContext, iteration: int, tool_result: Any
    ) -> None:
        """Called for each tool result after dispatch."""


# ---------------------------------------------------------------------------
# AgentLoop -- core iteration driver
# ---------------------------------------------------------------------------

class AgentLoop:
    """Iteration loop with middleware hooks.

    Usage::

        ctx    = LoopContext(max_iterations=10)
        loop   = AgentLoop(ctx, middlewares=[my_mw])
        result = loop.run(messages, tools,
                          api_call_fn=my_api,
                          tool_dispatch_fn=my_dispatch)

    ``api_call_fn(messages, tools)`` must return a dict with at least:
      - has_tool_calls: bool
      - content: str  (when has_tool_calls is False)

    When has_tool_calls is True it must also include:
      - tool_calls: list[dict] with keys (id, name, arguments)

    ``tool_dispatch_fn(tool_calls)`` receives the tool_calls list and must
    return a list of result dicts with at least ``tool_call_id``.
    """

    def __init__(
        self,
        context: LoopContext,
        middlewares: Optional[List[MiddlewareBase]] = None,
    ) -> None:
        self.context = context
        self.middlewares: List[MiddlewareBase] = middlewares or []
        self.iteration: int = 0
        self.interrupted: bool = False

    # -- Public API ---------------------------------------------------------

    def run(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        api_call_fn: Callable,
        tool_dispatch_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute the loop until budget exhaustion, interrupt, or final response.

        Returns a dict describing the outcome:
          - Normal completion:  {"content": "<model response>"}
          - Budget exhaustion:  {"budget_exhausted": True, ...}
          - Interrupt:          {"interrupted": True}
        """
        while self.context.should_continue():
            self.iteration += 1
            it = self.iteration

            # --- middleware: before ---
            for mw in self.middlewares:
                mw.before_iteration(self.context, it, messages=messages)

            # --- API call ---
            # NOTE: consume() is the caller's responsibility (mirrors the
            # real loop where AIAgent controls the budget inline).  The test
            # doubles call ctx.consume() inside their fake api_call_fn.
            raw = api_call_fn(messages, tools)

            # --- interrupt check (model or context can request) ---
            if self.context._stopped:
                self.interrupted = True
                for mw in self.middlewares:
                    mw.on_interrupt(self.context, it)
                return {"interrupted": True}

            # --- no tool calls → final response ---
            if not raw.get("has_tool_calls"):
                result = {"content": raw.get("content", "")}
                for mw in self.middlewares:
                    mw.after_iteration(self.context, it, result)
                return result

            # --- tool calls ---
            tool_calls = raw.get("tool_calls", [])
            for tc in tool_calls:
                for mw in self.middlewares:
                    mw.on_tool_call(self.context, it, tc)

            if tool_dispatch_fn is not None:
                tool_results = tool_dispatch_fn(tool_calls)
                for tr in tool_results:
                    for mw in self.middlewares:
                        mw.on_tool_result(self.context, it, tr)

            # --- middleware: after (tool iteration) ---
            after_result = {
                "has_tool_calls": True,
                "tool_calls": tool_calls,
            }
            if tool_dispatch_fn is not None:
                after_result["tool_results"] = tool_results

            for mw in self.middlewares:
                mw.after_iteration(self.context, it, after_result)

            # --- budget exhausted after consuming? ---
            if not self.context.should_continue():
                for mw in self.middlewares:
                    mw.on_budget_exhausted(self.context, it)
                return {"budget_exhausted": True, **after_result}

        # Should not normally reach here, but defensive:
        return {"budget_exhausted": True}
