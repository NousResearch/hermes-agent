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
    "_dropped_toolcall_nudge", "_degenerate_final_nudge",
)

# Degenerate-final guard. A text stop whose ENTIRE answer is a fragment — a stray token, a
# wrong-script word, or a truncated non-sentence — after the turn already ran real tool work is
# a provider-side collapse, not an answer: the loop accepts it, the turn ends ``completed``, and
# an unattended job silently abandons the task. Reported class on the Responses wire with
# muse-spark (aggregated upstream); the tool work always executes correctly and only the final
# text collapses. Re-prompt ONCE, keeping the fragment as the fallback so a second collapse
# still ends the turn with today's behaviour.
_DEGENERATE_FINAL_NUDGE_CONTENT = (
    "Your previous message ended the turn with a fragment that is not a usable answer while the "
    "task was still in progress. Resume the task and finish it, then give a complete user-facing "
    "answer. If you believe the task IS finished, say so explicitly and summarize what was done "
    "and verified."
)

#: Above this length a reply is treated as a real (if terse) answer, never a collapse.
_DEGENERATE_FINAL_MAX_CHARS = 24
#: Whitespace-token ceiling for the "short, but not a sentence" arm.
_DEGENERATE_FINAL_MAX_TOKENS = 3
#: Legitimate terse answers that must never trip the guard.
_DEGENERATE_FINAL_ALLOWLIST = frozenset({
    "done", "ok", "okay", "yes", "no", "yep", "nope", "fixed", "finished", "complete",
    "completed", "ready", "correct", "agreed", "understood", "ack", "none", "got it",
    # Plausible one-word legitimate answers after tool work; the allowlist is the tuning
    # surface when a false positive shows up in the warning log.
    "approved", "confirmed", "verified", "noted", "working", "checking", "pass", "passed",
    "failed", "error", "cancelled", "canceled",
})
_DEGENERATE_FINAL_ON = {"true", "always", "yes", "on"}
_DEGENERATE_FINAL_OFF = {"false", "never", "no", "off"}
#: Model substrings carrying the reported collapse class, used by the ``auto`` scope.
_DEGENERATE_FINAL_AUTO_MODELS = ("muse-spark",)


def _looks_like_degenerate_final(text: Any) -> bool:
    """Whether a text stop reads as a collapsed answer rather than a real one.

    True only for a short fragment carrying a concrete degeneration signal: a single token, any
    non-ASCII character (the reported collapse class is a stray wrong-script word), or a
    sub-sentence run with no terminal punctuation. Allowlisted terse answers never match, and a
    legitimate short answer above the char ceiling never matches.
    """
    t = str(text or "").strip()
    if not t or len(t) > _DEGENERATE_FINAL_MAX_CHARS:
        return False
    if t.lower().strip(".!") in _DEGENERATE_FINAL_ALLOWLIST:
        return False
    if any(ord(ch) > 0x7F for ch in t):
        return True
    tokens = t.split()
    if len(tokens) <= 1:
        return True
    return len(tokens) <= _DEGENERATE_FINAL_MAX_TOKENS and not t.endswith((".", "!", "?", ":"))


def _tool_results_since_last_user(messages: Any) -> int:
    """Tool-result rows after the most recent user row — the turn's mid-task evidence."""
    count = 0
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            break
        if role == "tool":
            count += 1
    return count


def _degenerate_final_guard_mode(agent: Any) -> str:
    """``"off"``, ``"all"`` or ``"auto"`` for the degenerate-final re-prompt.

    Mirrors ``agent_runtime_helpers.intent_ack_continuation_mode``: ``agent._degenerate_final_guard``
    overrides (``True``/true-ish -> all, ``False``/false-ish -> off, a list -> all when a substring
    matches the model), and the default ``auto`` fires only on the wire/family the collapse is
    reported on. Deliberately config-free for now so the guard ships with no new config surface.
    """
    mode = getattr(agent, "_degenerate_final_guard", "auto")
    if mode is False or (isinstance(mode, str) and mode.lower() in _DEGENERATE_FINAL_OFF):
        return "off"
    if mode is True or (isinstance(mode, str) and mode.lower() in _DEGENERATE_FINAL_ON):
        return "all"
    if isinstance(mode, list):
        model_lower = (getattr(agent, "model", "") or "").lower()
        return "all" if any(str(p).lower() in model_lower for p in mode if p) else "off"
    if getattr(agent, "api_mode", "") == "codex_responses":
        return "all"
    model_lower = (getattr(agent, "model", "") or "").lower()
    return "all" if any(p in model_lower for p in _DEGENERATE_FINAL_AUTO_MODELS) else "off"


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

    # Reasoning-only clean stop: some reasoning parsers (vLLM nemotron_v3 past ~500K
    # prompt tokens) file the whole answer as reasoning when the model omits the closing
    # delimiter. ``finish_reason == "stop"`` means the provider considers generation
    # complete, so the empty-response ladder would only re-bill the same input to arrive
    # at a truncated preview of this text; promote the reasoning to the visible answer
    # BEFORE the ladder. ``length`` (cut off mid-thought) stays on the continuation path,
    # and the promoted text is persisted as ordinary content so the next turn replays it.
    _content = assistant_message.content
    if (
        finish_reason == "stop"
        and not assistant_message.tool_calls
        and (_content is None or (isinstance(_content, str) and not _content.strip()))
    ):
        _promoted = agent._extract_reasoning(assistant_message)
        if _promoted:
            logger.info(
                "Reasoning-only clean stop (%d chars) — using reasoning as the final response",
                len(_promoted),
            )
            assistant_message.content = _promoted
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

    # Degenerate-final recovery (see _DEGENERATE_FINAL_NUDGE_CONTENT above): a text stop whose
    # whole answer is a fragment after real tool work is a provider-side collapse. Bounded to one
    # re-prompt per turn (the counter clears on the next genuine turn end), scoped by
    # _degenerate_final_guard_mode, and gated on the turn having actually run tool work — a terse
    # answer from a chat-only turn is a legitimate answer, not a collapse.
    if (
        bool(getattr(agent, "_stall_guards", True))
        and _degenerate_final_guard_mode(agent) != "off"
        and not assistant_message.tool_calls
        and getattr(agent, "_degenerate_final_nudges", 0) < 1
        and _tool_results_since_last_user(messages) >= 2
        and _looks_like_degenerate_final(final_response)
    ):
        agent._degenerate_final_nudges = getattr(agent, "_degenerate_final_nudges", 0) + 1
        logger.warning(
            "Degenerate final: text stop ended the turn with a %d-char fragment after %d "
            "tool result(s) — re-prompting once (model=%s provider=%s api_mode=%s): %r",
            len(final_response or ""), _tool_results_since_last_user(messages),
            agent.model, agent.provider, getattr(agent, "api_mode", ""),
            (final_response or "")[:40],
        )
        agent._emit_status(
            "↻ Model ended the turn on a fragment — re-prompting once to finish"
        )
        # Both halves of the re-prompt pair are ephemeral scaffolding: never persisted, and the
        # finalization pop strips an unanswered tail pair.
        final_msg["_degenerate_final_nudge"] = True
        append_message(messages, final_msg)
        append_message(messages, {
            "role": "user",
            "content": _DEGENERATE_FINAL_NUDGE_CONTENT,
            "_degenerate_final_nudge": True,
        })
        agent._session_messages = messages
        final_response = None
        return _verdict("continue")

    # Genuine turn end (no dropped-tool-call mismatch): clear stall budget.
    agent._dropped_toolcall_retries = 0
    agent._degenerate_final_nudges = 0

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
