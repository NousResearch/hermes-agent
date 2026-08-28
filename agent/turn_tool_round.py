"""One tool-calling round of the conversation turn loop: validate/cap/dedupe the model's
tool calls, persist the tool-call turn BEFORE any side effect, execute the tools, honour
guardrail halts / persistence failures, then compress after tool results. Nothing here
imports ``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve
lazily.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

from agent.message_metadata import append_message
from agent.message_sanitization import coalesce_tool_call_id
from agent.turn_preflight import compress_after_tool_results
from agent.turn_tool_validation import validate_tool_calls

logger = logging.getLogger("agent.conversation_loop")

# Post-response housekeeping tools: a round made only of these mutes tool progress.
_HOUSEKEEPING_TOOLS = frozenset({"memory", "todo_list", "skill_manage", "session_search"})


@dataclass
class ToolRoundVerdict:
    """``action``: ``"continue"`` (tools ran, next API call), ``"break"`` (turn ends:
    persistence failure, guardrail halt, post-tool compression end) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals the round
    rebinds."""

    action: str
    messages: Any
    conversation_history: Any
    active_system_prompt: Any
    compression_attempts: Any
    final_response: Any
    failed: Any
    _turn_exit_reason: Any
    truncated_tool_call_retries: Any
    result: Optional[Dict[str, Any]] = None


def run_tool_round(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any,
    conversation_history: Any, api_call_count: Any, effective_task_id: Any, user_message: Any,
    system_message: Any, active_system_prompt: Any, compression_attempts: Any,
    max_compression_attempts: Any, final_response: Any, failed: Any, _turn_exit_reason: Any,
    truncated_tool_call_retries: Any,
) -> ToolRoundVerdict:
    """Execute one tool round in the exact original order. Persist-before-execute is a
    durability invariant: resume must see the executed block if a destructive tool restarts
    Hermes; a failed canonical append ends the turn rather than running tools from
    process-only state."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ToolRoundVerdict:
        return ToolRoundVerdict(
            action=action, messages=messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, compression_attempts=compression_attempts,
            final_response=final_response, failed=failed, _turn_exit_reason=_turn_exit_reason,
            truncated_tool_call_retries=truncated_tool_call_retries, result=result,
        )

    if not agent.quiet_mode:
        agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

    if agent.verbose_logging:
        for tc in assistant_message.tool_calls:
            raw_args = tc.function.arguments
            args_preview = raw_args[:200] if isinstance(raw_args, str) else repr(raw_args)[:200]
            logging.debug("Tool call: %s with args: %s...", tc.function.name, args_preview)

    _tvv = validate_tool_calls(
        agent, assistant_message, finish_reason, messages=messages,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    if _tvv.action == "return":
        return _verdict("return", _tvv.result)
    if _tvv.action == "continue":
        return _verdict("continue")

    # Post-call guardrails.
    assistant_message.tool_calls = agent._deduplicate_tool_calls(
        agent._cap_delegate_task_calls(assistant_message.tool_calls)
    )

    # Mixed batch: the assistant message keeps EVERY emitted call (each tool_call needs a
    # matching result) while only valid ones dispatch.
    _invalid_batch_calls = [
        tc for tc in assistant_message.tool_calls if tc.function.name not in agent.valid_tool_names
    ] if _tvv.mixed_invalid_batch else []

    assistant_msg, duplicate_previous_interim = stage_tool_call_message(
        agent, assistant_message=assistant_message, finish_reason=finish_reason, messages=messages
    )
    append_message(messages, assistant_msg)

    # Mixed batch: error-result invalid calls and drop them from execution.
    if _invalid_batch_calls:
        for tc in _invalid_batch_calls:
            append_message(messages, {
                "role": "tool",
                "name": tc.function.name,
                "tool_call_id": coalesce_tool_call_id(tc),
                "content": _invalid_tool_name_error_content(
                    tc.function.name, agent.valid_tool_names
                ),
            })
        assistant_message.tool_calls = [
            tc for tc in assistant_message.tool_calls if tc.function.name in agent.valid_tool_names
        ]

    # Persist the tool-call turn before any tool side effects so resume sees the executed
    # block if a destructive tool restarts Hermes.
    try:
        _tool_turn_persisted = agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception as exc:
        _tool_turn_persisted = False
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning(
            "Incremental tool-call persistence failed before execution "
            "(session=%s): %s",
            agent.session_id or "none",
            exc,
        )

    if _tool_turn_persisted is False:
        # Canonical append failed: never project the row or run tools from process-only
        # state; break rather than retry. No recorded cause means genuinely unknown.
        if getattr(agent, "_last_persistence_error_cause", None) is None:
            agent._last_persistence_error_cause = "unknown"
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    # A UI must never observe an assistant/tool-call row that is only an in-memory
    # projection: emit interim commentary after the DB append.
    if not duplicate_previous_interim:
        agent._emit_interim_assistant_message(assistant_msg)

    # Flush open streaming boxes before tools so early content doesn't wrap tool feed
    # lines. Display callback only — TTS (_stream_callback) must NOT receive None (EOS).
    agent._close_stream_display_segment()

    agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

    if getattr(agent, "_incremental_persistence_failed", False):
        # Tool result could not be made canonical: never send the in-memory result to
        # the model or project later events from this turn.
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    if agent._tool_guardrail_halt_decision is not None:
        decision = agent._tool_guardrail_halt_decision
        _turn_exit_reason = "guardrail_halt"
        final_response = agent._toolguard_controlled_halt_response(decision)
        agent._emit_status(f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}")
        append_message(messages, {"role": "assistant", "content": final_response})
        # Emit the halt so it isn't mistaken for a crash; the stream callback is still
        # alive, so SSE/TUI clients see the explanation.
        if final_response:
            agent._safe_print(f"\n{final_response}\n")
            if agent.stream_delta_callback:
                with suppress(Exception):
                    agent.stream_delta_callback(final_response)
                    agent.stream_delta_callback(None)
        return _verdict("break")

    # Reset per-turn retry counters so one truncation can't poison the turn.
    truncated_tool_call_retries = 0
    # Defer the paragraph break: _fire_stream_delta() prepends one "\n\n" when real
    # text arrives, so tool iterations don't stack blank lines.
    agent._stream_needs_break = True
    # Refund the iteration when the ONLY tool was execute_code (programmatic tool
    # calling) — cheap RPC-style calls shouldn't eat the budget.
    if {tc.function.name for tc in assistant_message.tool_calls} == {"execute_code"}:
        agent.iteration_budget.refund()

    _ptc = compress_after_tool_results(
        agent, messages=messages, system_message=system_message, user_message=user_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, effective_task_id=effective_task_id,
        final_response=final_response, turn_exit_reason=_turn_exit_reason,
    )
    messages = _ptc.messages
    active_system_prompt = _ptc.active_system_prompt
    conversation_history = _ptc.conversation_history
    compression_attempts = _ptc.compression_attempts
    final_response = _ptc.final_response
    _turn_exit_reason = _ptc.turn_exit_reason
    if _ptc.end_turn:
        return _verdict("break")

    # Save session log incrementally (so progress is visible even if interrupted)
    agent._session_messages = messages
    # Touch activity so slow post-tool work plus a slow follow-up API call can't exceed
    # the gateway inactivity timeout (HERMES_AGENT_TIMEOUT).
    # Touch activity before continuing so the gateway's inactivity monitor never sees a stale timestamp
    # between tool completion and the start of the next API call. Without this, a tool-call result (which
    # takes ~0s to process) followed by slow post-tool processing (compression, persist) and a slow
    # follow-up API call can exceed the gateway inactivity timeout (HERMES_AGENT_TIMEOUT, default 1800s) and
    # the gateway kills the session before the next activity touch fires (#69559, #69131).
    agent._touch_activity(f"tool results posted, continuing iteration #{api_call_count}")
    return _verdict("continue")


def stage_tool_call_message(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any
) -> Tuple[Dict[str, Any], bool]:
    """Build the assistant tool-call row and update the per-turn fallback/mute state.

    Drops a bare bracketed marker beside a call (#78148), classifies housekeeping-only
    rounds, keeps visible content as the empty-follow-up fallback, pops thinking-only
    prefills (resetting their counters), re-arms the post-tool nudge and the
    dropped-tool-call stall budget. Returns ``(assistant_msg, duplicate_previous_interim)``;
    the flag suppresses re-emitting interim commentary the previous ``incomplete`` row
    already showed."""
    from agent.conversation_loop import _STALE_MARKER_RE

    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)

    turn_content = assistant_message.content or ""

    # A bare bracketed token (e.g. ``[memory]``) beside a function call is protocol
    # scaffolding; persisting it lets the post-tool fallback replay it forever (#78148).
    if assistant_message.tool_calls and _STALE_MARKER_RE.fullmatch(turn_content.strip()):
        logger.warning(
            "Discarding bare tool-call marker from assistant content: %s", turn_content
        )
        turn_content = ""
        assistant_msg["content"] = ""

    # Classify tools regardless of visible content: a substantive tool-only turn must
    # invalidate any older housekeeping fallback (so a two-turn-old housekeeping
    # narration isn't attributed to the preceding tool turn), and clear the mute flag a
    # prior housekeeping turn set, else _vprint suppresses this turn's tool progress.
    _all_housekeeping = all(
        tc.function.name in _HOUSEKEEPING_TOOLS for tc in assistant_message.tool_calls
    )
    if assistant_message.tool_calls and not _all_housekeeping:
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        agent._mute_post_response = False

    # Content + tool_calls in one turn: keep the content as a fallback final response in
    # case the follow-up turn after tools is empty. Mute only when EVERY tool call is
    # post-response housekeeping; substantive tools keep output on.
    if turn_content and agent._has_content_after_think_block(turn_content):
        agent._last_content_with_tools = turn_content
        agent._last_content_tools_all_housekeeping = _all_housekeeping
        if _all_housekeeping and agent._has_stream_consumers():
            agent._mute_post_response = True
        elif agent._should_emit_quiet_tool_messages():
            clean = agent._strip_think_blocks(turn_content).strip()
            if clean:
                agent._vprint(f"  ┊ 💬 {clean}")

    # Pop thinking-only prefill message(s) before appending (same rationale as the
    # final-response path). Tool calls after a prefill recovery reset the prefill
    # counter, so each tool-call success is a fresh start, not a cumulative burn.
    _had_prefill = False
    while messages and isinstance(messages[-1], dict) and messages[-1].get("_thinking_prefill"):
        messages.pop()
        _had_prefill = True
    if _had_prefill:
        agent._thinking_prefill_retries = 0
        agent._empty_content_retries = 0
    # Re-arm the post-tool nudge so it can fire on a LATER tool round; a landed tool call
    # recovers any dropped-tool-call stall, so refresh that budget per stall.
    agent._post_tool_empty_retried = False
    agent._dropped_toolcall_retries = 0

    previous_msg = messages[-1] if messages else None
    current_interim_visible = agent._interim_assistant_visible_text(assistant_msg)
    duplicate_previous_interim = (
        bool(current_interim_visible)
        and isinstance(previous_msg, dict)
        and previous_msg.get("role") == "assistant"
        and previous_msg.get("finish_reason") == "incomplete"
        and agent._interim_assistant_visible_text(previous_msg) == current_interim_visible
    )
    return assistant_msg, duplicate_previous_interim
