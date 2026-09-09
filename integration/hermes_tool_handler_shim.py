"""Tool-handler shim — example wrapping ``model_tools.handle_function_call``.

The kernel calls into ``ToolHandlerProtocol``:

    def handle(calls: list[ToolCall]) -> list[ToolOutput]: ...

In hermes today, tool dispatch goes through ``model_tools.handle_function_call``
(``model_tools.py:688``). It runs one call at a time; parallelization and
conflict detection live as private helpers inside ``run_agent.py``:

    _should_parallelize_tool_batch(tool_calls)
    _extract_parallel_scope_path(tool_name, function_args)
    _paths_overlap(left, right)

The shim is the right home for those helpers. The kernel hands the
shim the **full batch** of calls that governance has already allowed;
the shim decides whether to sequence or parallelize.
"""

from __future__ import annotations

from typing import Any

from agent.runtime import ToolCall, ToolOutput


# ----- placeholder for hermes-side dispatch ----------------------------------
#
# In hermes-agent, replace these stubs with the real imports:
#
#     from model_tools import handle_function_call
#     from run_agent import (
#         _should_parallelize_tool_batch,
#         _extract_parallel_scope_path,
#         _paths_overlap,
#     )


def _placeholder_handle_function_call(name: str, arguments: dict, **kwargs) -> Any:
    raise NotImplementedError("replace with hermes ``handle_function_call``")


def _placeholder_should_parallelize(calls: list[ToolCall]) -> bool:
    return False


# ----- the actual shim -------------------------------------------------------


class HermesToolHandler:
    """Wraps hermes tool dispatch into ``ToolHandlerProtocol``.

    Pattern:
      * receive the batch of allowed ``ToolCall`` objects (governance has
        already filtered),
      * decide parallel vs sequential via the existing hermes helpers,
      * dispatch each call through ``handle_function_call``,
      * translate any raise into a ``ToolOutput(is_error=True)`` so the
        loop sees a typed observation rather than an exception.

    Behavior preserved:
      * Approval gating (``tools/approval.py``) and guardrails
        (``agent/tool_guardrails.py``) continue to wrap individual tools
        as they already do — they are orthogonal to this shim.
      * Final-answer tool detection: the shim sets ``is_final_answer=True``
        when the tool name matches the kernel's final-answer convention.
    """

    def __init__(
        self,
        *,
        dispatch=_placeholder_handle_function_call,
        should_parallelize=_placeholder_should_parallelize,
        final_answer_tool_name: str = "final_answer",
    ) -> None:
        self._dispatch = dispatch
        self._should_parallelize = should_parallelize
        self._final = final_answer_tool_name

    def handle(self, calls: list[ToolCall]) -> list[ToolOutput]:
        if not calls:
            return []
        if self._should_parallelize(calls):
            return self._run_parallel(calls)
        return self._run_sequential(calls)

    # ---- execution paths ----------------------------------------------------

    def _run_sequential(self, calls: list[ToolCall]) -> list[ToolOutput]:
        return [self._run_one(call) for call in calls]

    def _run_parallel(self, calls: list[ToolCall]) -> list[ToolOutput]:
        # TODO(hermes): use the existing thread/async pool. The hermes
        # codebase has ``_get_tool_loop()`` / ``_get_worker_loop()`` in
        # ``model_tools.py`` that handle this; mirror those here.
        # For the kernel contract, this method MUST return outputs in the
        # same order as ``calls`` (the loop pairs by id, not by position,
        # but ordering preservation aids debugging).
        return [self._run_one(call) for call in calls]

    def _run_one(self, call: ToolCall) -> ToolOutput:
        try:
            result = self._dispatch(call.name, call.arguments)
        except Exception as exc:  # narrow boundary — convert to typed observation
            return ToolOutput(
                id=call.id,
                name=call.name,
                output={"error_type": type(exc).__name__, "message": str(exc)},
                is_error=True,
                is_final_answer=False,
            )
        return ToolOutput(
            id=call.id,
            name=call.name,
            output=result,
            is_error=False,
            is_final_answer=(call.name == self._final),
        )
