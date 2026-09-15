"""Outer-loop exception handler for the conversation turn loop: phase-aware classification
(interpreter shutdown, deterministic local post-processing bug vs API-path failure),
unanswered tool_call error results, and the per-turn error cap that stops the loop from
spinning until the budget is gone. Nothing here imports ``agent.conversation_loop`` at
module level (cycle); loop-internal constants resolve lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import sys
from typing import Any

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class OuterErrorVerdict:
    """``action`` is ``"break"`` (turn ends with ``final_response``/``turn_exit_reason`` set)
    or ``"fallthrough"`` (retry the iteration). The other fields are the loop locals the
    handler rebinds."""

    action: str
    _outer_error_count: Any
    _turn_exit_reason: Any
    failed: Any
    final_response: Any


def handle_outer_loop_error(
    agent: Any, *, e: Any, _outer_error_count: Any, api_call_count: Any, messages: Any,
    conversation_history: Any, _turn_exit_reason: Any, failed: Any, final_response: Any,
) -> OuterErrorVerdict:
    """Handle an exception that escaped the response-processing block. Shutdown and
    local-processing errors are deterministic and end the turn; API-path errors retry until
    ``min(_MAX_OUTER_LOOP_ERRORS, max_iterations)`` escaped exceptions. The assistant
    message is never appended here: a prefill/interim assistant may already be the tail
    (assistant→assistant); ``finalize_turn`` appends only when safe."""
    from agent.conversation_loop import (
        _API_CALL_MODULES, _LOCAL_PROCESSING_MODULES, _MAX_OUTER_LOOP_ERRORS,
        _is_interpreter_shutdown_error, _ra,
    )

    def _verdict(action: str) -> OuterErrorVerdict:
        return OuterErrorVerdict(
            action=action, _outer_error_count=_outer_error_count,
            _turn_exit_reason=_turn_exit_reason, failed=failed, final_response=final_response,
        )

    # Count every escaped exception before classification so permanent failures
    # terminate even with an unlimited turn budget.
    _outer_error_count += 1

    # Interpreter shutdown makes every executor op raise: break.
    # Phase-aware error classification. The huge outer try/except spans both the actual API request and all
    # local post-processing of the returned assistant message. Deterministic local bugs (e.g. passing a
    # multimodal content list into a regex helper after a vision turn or context compaction) should not be
    # retried: they will fail identically on every iteration and only burn the iteration budget. We classify
    # an error as local by inspecting the traceback: if the exception propagated through any of the known
    # local post-processing helpers and never entered the interruptible API-call helpers, it is almost
    # certainly a local processing bug. (#66267) Interpreter shutdown: if the process is tearing down, every
    # executor-backed operation (API call, tool dispatch, memory sync) raises ``RuntimeError: cannot
    # schedule new futures after interpreter shutdown``. Retrying is pointless — the executor is gone for
    # good — and each retry just spams another traceback. Break immediately so the turn exits cleanly.
    # (#93217)
    if sys.is_finalizing() or _is_interpreter_shutdown_error(e):
        error_msg = f"Interpreter is shutting down — cannot continue (API call #{api_call_count}): {e}"
        try:
            agent._safe_print(f"❌ {error_msg}")
        except (OSError, ValueError):
            pass
        logger.warning(error_msg)
        # Best-effort persist — the dying executor may raise the same error; don't let
        # it mask the shutdown exit. finalize_turn retries.
        try:
            agent._persist_session(messages, conversation_history)
        except Exception:
            pass
        _turn_exit_reason = "interpreter_shutdown"
        final_response = "Session is shutting down. Your conversation can be resumed with: hermes --resume <session-id>"
        return _verdict("break")

    # Deterministic local post-processing bugs (traceback via local helpers, never API
    # helpers) aren't retried.
    tb_module_names: set[str] = set()
    _tb = e.__traceback__
    while _tb is not None:
        tb_module_names.add(os.path.splitext(os.path.basename(_tb.tb_frame.f_code.co_filename))[0])
        _tb = _tb.tb_next
    _is_local_processing_error = bool(tb_module_names & _LOCAL_PROCESSING_MODULES) and not (
        tb_module_names & _API_CALL_MODULES
    )

    if _is_local_processing_error:
        error_msg = f"Error during local message processing after OpenAI-compatible API call #{api_call_count}: {str(e)}"
    else:
        error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
    # Honor the _vprint contract: suppress_status_output silences hard failures;
    # quiet_mode -q still shows them. Traceback is logged below.
    if getattr(agent, "suppress_status_output", False):
        logger.error(error_msg)
    else:
        try:
            print(f"❌ {error_msg}")
        except (OSError, ValueError):
            logger.error(error_msg)

    # ERROR level with traceback so outer-loop failures land in agent.log AND errors.log.
    logger.exception("Outer loop error in API call #%d", api_call_count)

    # An appended assistant tool_calls message needs a role="tool" result per
    # tool_call_id; fill in error results for unanswered ones.
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            break
        if msg.get("role") == "tool":
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            answered_ids = {
                m["tool_call_id"]
                for m in messages[idx + 1:]
                if isinstance(m, dict) and m.get("role") == "tool"
            }
            for tc in msg["tool_calls"]:
                if tc and isinstance(tc, dict) and tc["id"] not in answered_ids:
                    append_message(messages, {
                        "role": "tool",
                        "name": _ra().AIAgent._get_tool_call_name_static(tc),
                        "tool_call_id": tc["id"],
                        "content": f"Error executing tool: {error_msg}",
                    })
        break

    # Non-tool errors are already printed; a synthetic message would pollute history
    # and risk breaking role alternation.

    # Local errors are deterministic: stop early instead of retrying until the budget is
    # gone; a small per-turn cap prevents infinite spinning.
    _outer_error_cap = min(_MAX_OUTER_LOOP_ERRORS, max(1, agent.max_iterations))
    if (
        _is_local_processing_error
        or api_call_count >= agent.max_iterations - 1
        or _outer_error_count >= _outer_error_cap
    ):
        if _is_local_processing_error:
            _turn_exit_reason = f"local_processing_error({error_msg[:80]})"
            final_response = f"I apologize, but I encountered an error while processing the model response: {error_msg}"
        elif _outer_error_count >= _outer_error_cap:
            failed = True
            _turn_exit_reason = f"repeated_outer_errors({error_msg[:80]})"
            final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
        else:
            _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
            final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
        return _verdict("break")
    return _verdict("fallthrough")
