"""Bounded, iteration-neutral recovery of tool requests that stopped before effects."""

from dataclasses import dataclass
from typing import Any

from agent.message_metadata import append_message
from agent.message_sanitization import coalesce_tool_call_id
from agent.tool_snapshot import ToolSnapshotRefreshError, refresh_tool_snapshot_after_stale, stale_tool_result


@dataclass
class SnapshotRecovery:
    action: str
    api_call_count: int
    stale_tool_snapshot_retries: int
    final_response: Any
    failed: bool
    _turn_exit_reason: str


def recover_stale_tool_snapshot(
    agent, *, error, assistant_message, messages, conversation_history, api_call_count,
    stale_tool_snapshot_retries, final_response, failed, _turn_exit_reason,
):
    # A call row may already be durable when execution rejects its snapshot.
    # Pair only outstanding IDs; completed results and already-paired stale calls stay intact.
    emitted = {coalesce_tool_call_id(tc): tc for tc in assistant_message.tool_calls}
    pending = set()
    for message in messages:
        if message.get("role") == "assistant":
            pending.update(coalesce_tool_call_id(tc) for tc in message.get("tool_calls", []))
        elif message.get("role") == "tool":
            pending.discard(coalesce_tool_call_id({"id": message.get("tool_call_id")}))
    for call_id in emitted:
        if call_id not in pending:
            continue
        from agent.tool_dispatch_helpers import make_tool_result_message
        tc = emitted[call_id]
        append_message(messages, make_tool_result_message(
            tc.function.name, stale_tool_result(tc.function.name), call_id, effect_disposition="none"
        ))
    if pending & emitted.keys():
        if agent._flush_messages_to_session_db(messages, conversation_history) is False:
            return SnapshotRecovery("break", api_call_count, stale_tool_snapshot_retries,
                                    "", True, "session_persistence_failed")
    if getattr(agent, "_incremental_persistence_failed", False):
        return SnapshotRecovery("break", api_call_count, stale_tool_snapshot_retries,
                                "", True, "session_persistence_failed")
    stale_tool_snapshot_retries += 1
    if not isinstance(error, ToolSnapshotRefreshError):
        try:
            refresh_tool_snapshot_after_stale(agent, assistant_message)
        except ToolSnapshotRefreshError as exc:
            error = exc
    if isinstance(error, ToolSnapshotRefreshError):
        final_response = "Tool configuration changed and could not be refreshed safely; no further tool call was executed."
        _turn_exit_reason = "tool_snapshot_refresh_failed"
    elif stale_tool_snapshot_retries >= 3:
        final_response = "Tool configuration kept changing before execution; the turn stopped without retrying further."
        _turn_exit_reason = "stale_tool_snapshot_exhausted"
    else:
        api_call_count -= 1
        agent._api_call_count = api_call_count
        agent.iteration_budget.refund()
        agent._buffer_status("Tool configuration changed; retrying with the current tools.")
        return SnapshotRecovery("continue", api_call_count, stale_tool_snapshot_retries,
                                final_response, failed, _turn_exit_reason)
    agent._emit_status(final_response)
    append_message(messages, {"role": "assistant", "content": final_response})
    return SnapshotRecovery("break", api_call_count, stale_tool_snapshot_retries,
                            final_response, True, _turn_exit_reason)
