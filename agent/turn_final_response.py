"""No-tool-call (final text) branch of the conversation turn loop: empty/think-only recovery,
intent-ack / stall-guard continuation, length-continuation joining, dropped-tool-call
re-prompt, scaffolding pop, stop gates, then the durable final flush. Extracted from
``run_conversation``; nothing here imports ``agent.conversation_loop`` at module level
(cycle) — loop-internal nudge constants resolve lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from agent.message_metadata import append_message
from agent.turn_empty_response import recover_empty_response
from agent.turn_stop_gates import apply_stop_gates

logger = logging.getLogger("agent.conversation_loop")

# Ephemeral retry scaffolding rows popped before the final answer becomes durable.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_thinking_prefill", "_empty_recovery_synthetic", "_empty_terminal_sentinel",
    "_dropped_toolcall_nudge",
)


@dataclass
class FinalResponseVerdict:
    """``action``: ``"break"`` (turn ends with ``final_response``), ``"continue"`` (a
    continuation/re-prompt/stop-gate asked for another API call) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals rebound."""

    action: str
    active_system_prompt: Any
    final_response: Any
    _turn_exit_reason: Any
    _preflight_compression_blocked: Any
    codex_ack_continuations: Any
    truncated_response_parts: Any
    length_continue_retries: Any
    _pending_verification_response: Any
    _pending_verification_response_previewed: Any
    result: Optional[Dict[str, Any]] = None


def finish_text_response(
    agent: Any, *, assistant_message: Any, response: Any, finish_reason: Any, messages: Any,
    api_messages: Any, conversation_history: Any, api_call_count: Any, user_message: Any,
    active_system_prompt: Any, final_response: Any, _turn_exit_reason: Any,
    _preflight_compression_blocked: Any, codex_ack_continuations: Any,
    truncated_response_parts: Any, length_continue_retries: Any,
    _pending_verification_response: Any, _pending_verification_response_previewed: Any,
) -> FinalResponseVerdict:
    """Finish (or defer) a text-only assistant response in the original guard order. Every
    continuation path sets ``final_response = None`` so an acknowledgment never suppresses
    iteration-limit summarization; the final message is appended and flushed only after the
    stop gates accept it."""
    from agent.conversation_loop import (
        _CODEX_ACK_CONTINUATION_NUDGE, _DROPPED_TOOLCALL_NUDGE_CONTENT, _join_truncated_parts
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> FinalResponseVerdict:
        return FinalResponseVerdict(
            action=action, active_system_prompt=active_system_prompt, final_response=final_response,
            _turn_exit_reason=_turn_exit_reason,
            _preflight_compression_blocked=_preflight_compression_blocked,
            codex_ack_continuations=codex_ack_continuations,
            truncated_response_parts=truncated_response_parts,
            length_continue_retries=length_continue_retries,
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=_pending_verification_response_previewed,
            result=result,
        )

    final_response = assistant_message.content or ""
    # Unmute: _mute_post_response from a housekeeping tool turn must not silence
    # empty-response warnings on the final response path.
    agent._mute_post_response = False

    # Think-block-only / empty content: recovery path.
    if not agent._has_content_after_think_block(final_response):
        _ev = recover_empty_response(
            agent, assistant_message, response, finish_reason, final_response=final_response,
            messages=messages, api_messages=api_messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, api_call_count=api_call_count,
            turn_exit_reason=_turn_exit_reason,
            preflight_compression_blocked=_preflight_compression_blocked,
        )
        final_response = _ev.final_response
        _turn_exit_reason = _ev.turn_exit_reason
        active_system_prompt = _ev.active_system_prompt
        _preflight_compression_blocked = _ev.preflight_compression_blocked
        if _ev.action == "return":
            return _verdict("return", _ev.result)
        if _ev.action == "break":
            return _verdict("break")
        return _verdict("continue")

    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    # Surface the one-shot fallback switch notice before dropping the retry buffer so a
    # provider/model switch stays visible on success.
    agent._emit_pending_fallback_notice()
    agent._clear_status_buffer()

    # Defensive: repair malformed role-alternation before API call. Catches cases where the history got
    # wedged into a ``tool → user`` or ``user → user`` tail (e.g. after empty- response scaffolding was
    # stripped and a new user message landed after an orphan tool result). Most providers return empty
    # content on malformed sequences, which would otherwise retrigger the empty-retry loop indefinitely.
    # repair_message_sequence_with_cursor also recomputes the SessionDB flush cursor (_last_flushed_db_idx)
    # when repair compacts the list, so the turn-end flush doesn't skip the assistant/tool chain (#44837).
    # One-time repeated-heal escalation notice (#96870): if the sanitizer above just crossed the per-session
    # heal threshold, deliver the queued notice through the status/warning callback — the normal out-of-band
    # delivery channel (gateway status message / CLI print). NEVER appended to messages/api_messages:
    # conversation context and the cached prompt prefix stay byte-identical.
    from agent.agent_runtime_helpers import (
        intent_ack_continuation_mode, trailing_continue_intent
    )

    _ack_mode = intent_ack_continuation_mode(agent)
    # Said-continue-but-stopped guard: no tool calls but the short reply TAILS with an
    # announced next action. Reuses the SAME bounded continuation counter (max 2 per turn).
    _stall_continue_intent = (
        bool(getattr(agent, "_stall_guards", True))
        and agent.valid_tool_names
        and codex_ack_continuations < 2
        and trailing_continue_intent(agent._strip_think_blocks(final_response or ""))
    )
    if _stall_continue_intent or (
        _ack_mode != "off"
        and agent.valid_tool_names
        and codex_ack_continuations < 2
        and agent._looks_like_codex_intermediate_ack(
            user_message=user_message, assistant_content=final_response, messages=messages,
            require_workspace=(_ack_mode == "codex_only"),
        )
    ):
        if _stall_continue_intent:
            logger.info(
                "Stall guard: turn ending on trailing continue-"
                "intent with no tool calls — re-prompting to act "
                "(%d/2)", codex_ack_continuations + 1,
            )
        codex_ack_continuations += 1
        interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
        append_message(messages, interim_msg)
        agent._emit_interim_assistant_message(interim_msg)
        append_message(messages, {"role": "user", "content": _CODEX_ACK_CONTINUATION_NUDGE})
        agent._session_messages = messages
        # An acknowledgment is non-final: its text must not suppress iteration-limit
        # summarization if the continuation exhausts budget.
        final_response = None
        return _verdict("continue")

    codex_ack_continuations = 0

    if truncated_response_parts:
        final_response = _join_truncated_parts([*truncated_response_parts, final_response])
        truncated_response_parts = []
        length_continue_retries = 0
        # The continuation recovered, so the fragments stay in the transcript.
        for _frag in messages:
            if isinstance(_frag, dict):
                _frag.pop("_length_continuation_fragment", None)
                _frag.pop("_length_continuation_nudge", None)

    final_response = agent._strip_think_blocks(final_response).strip()

    final_msg = agent._build_assistant_message(assistant_message, finish_reason)

    # Dropped tool-call recovery (copilot/Claude): finish_reason="tool_calls" with empty
    # tool_calls would end the turn unstarted; re-prompt (max 3 CONSECUTIVE stalls).
    if (
        finish_reason == "tool_calls"
        and not assistant_message.tool_calls
        and getattr(agent, "_dropped_toolcall_retries", 0) < 3
    ):
        agent._dropped_toolcall_retries = getattr(agent, "_dropped_toolcall_retries", 0) + 1
        logger.warning(
            "finish_reason=tool_calls with empty tool_calls array "
            "(narration only) — re-prompting to emit the call "
            "(retry %d/3, model=%s provider=%s)",
            agent._dropped_toolcall_retries, agent.model, agent.provider,
        )
        agent._emit_status(
            "↻ Model signaled a tool call but sent none — "
            f"re-prompting ({agent._dropped_toolcall_retries}/3)"
        )
        # Both halves of the re-prompt pair are ephemeral scaffolding: never persisted,
        # and the finalization pop strips an unanswered tail pair.
        final_msg["_dropped_toolcall_nudge"] = True
        append_message(messages, final_msg)
        append_message(messages, {
            "role": "user",
            "content": _DROPPED_TOOLCALL_NUDGE_CONTENT,
            "_dropped_toolcall_nudge": True,
        })
        agent._session_messages = messages
        final_response = None
        return _verdict("continue")

    # Genuine turn end (no dropped-tool-call mismatch): clear stall budget.
    agent._dropped_toolcall_retries = 0

    # Pop prefill / empty-retry scaffolding before the final response or
    # verification follow-up; it must not become durable transcript.
    while (
        messages
        and isinstance(messages[-1], dict)
        and any(messages[-1].get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS)
    ):
        messages.pop()

    _sg = apply_stop_gates(
        agent, final_msg, final_response=final_response, messages=messages,
        conversation_history=conversation_history,
        pending_verification_response=_pending_verification_response,
        pending_verification_response_previewed=_pending_verification_response_previewed,
    )
    _pending_verification_response = _sg.pending_verification_response
    _pending_verification_response_previewed = _sg.pending_verification_response_previewed
    if _sg.continue_turn:
        final_response = None
        return _verdict("continue")

    append_message(messages, final_msg)
    # Make the answer durable before leaving the loop (_DB_PERSISTED_MARKER keeps
    # _persist_session idempotent). Failure must NOT abort the turn: finalize retries.
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.warning(
            "final text-turn flush failed (session=%s) — reply is "
            "not yet durable; relying on finalize_turn retry",
            getattr(agent, "session_id", None) or "none",
            exc_info=True,
        )

    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
    if not agent.quiet_mode:
        agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
    return _verdict("break")
