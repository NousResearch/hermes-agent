"""No-tool-call (final text) branch of the conversation turn loop: empty/think-only recovery,
intent-ack / stall-guard continuation, length-continuation joining, dropped-tool-call
re-prompt, scaffolding pop, stop gates, then the durable final flush. Extracted from
``run_conversation``; nothing here imports ``agent.conversation_loop`` at module level
(cycle) — loop-internal nudge constants resolve lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Optional

from agent.leaked_tool_call import find_leaked_tool_call, leaked_tool_call_nudge
from agent.message_metadata import append_message
from agent.reasoning_promotion import answer_in_reasoning_capability
from agent.repetition_guard import STOP_PATH_MIN_CHARS, is_runaway_repetition
from agent.stream_delivery import publish_held_final_row
from agent.turn_failure_copy import stamp_failure
from agent.turn_empty_response import (
    housekeeping_reply_line, recover_empty_response, take_housekeeping_reply_line,
)
from agent.turn_stop_gates import apply_stop_gates
from agent.turn_truncation import partial_result, repetition_copy

_REPETITION_STOPPED = repetition_copy(
    "before delivery",
    "so the repeated output was discarded.",
    "; refusing to return a",
)
logger = logging.getLogger("agent.conversation_loop")

# Ephemeral retry scaffolding rows popped before the final answer becomes durable.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_thinking_prefill", "_empty_recovery_synthetic", "_empty_terminal_sentinel",
    "_dropped_toolcall_nudge",
)
# Re-prompts for a tool call left as text (see ``_reprompt_leaked_tool_call``); the pair reuses the
# ``_dropped_toolcall_nudge`` flag, so persistence and compression already treat it as scaffolding.
_LEAKED_TOOLCALL_MAX_RETRIES = 2


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
    api_call_count: int
    result: Optional[dict[str, Any]] = None


def _bare_marker_after_reply(agent: Any, final_response: Any, *, truncated_parts: Any = None,
                             promoted: Any = None) -> bool:
    """End this turn on the reply written beside a housekeeping round instead of its bare marker:
    only when the host says a marker may not stay silent (``_silence_marker_needs_reply``), for a
    complete uninterrupted response, and only when a real (non-marker) reply line exists."""
    from gateway.response_filters import is_intentional_silence_response

    if (getattr(agent, "_silence_marker_needs_reply", False) is not True or truncated_parts or promoted
            or getattr(agent, "_interrupt_requested", False)):
        return False
    reply = housekeeping_reply_line(agent)
    return (bool(reply) and not is_intentional_silence_response(reply)
            and is_intentional_silence_response(agent._strip_think_blocks(final_response or "").strip()))


def _recover_marker_reply(
    agent: Any, final_response: Any, verdict: Callable[[str], FinalResponseVerdict], *,
    truncated_parts: Any, promoted: Any,
) -> Optional[FinalResponseVerdict]:
    """Select a completed housekeeping reply before output transforms and persistence.

    Keep the caller's loop state, changing only the chosen final and its exit reason.
    Other hosts and permitted silence continue through normal text handling.
    """
    if not _bare_marker_after_reply(agent, final_response, truncated_parts=truncated_parts, promoted=promoted):
        return None
    result = verdict("break")
    result.final_response = take_housekeeping_reply_line(agent)
    result._turn_exit_reason = "fallback_prior_turn_content"
    logger.info("Silence marker after a written reply: using that reply as the final response")
    return result


def _absorb_continuation_fragments(final_response: Any, parts: Any, messages: Any) -> tuple[Any, bool]:
    """Stitch length-continuation fragments in front of the closing piece; True when there were
    any. The continuation recovered, so the fragment rows stay in the transcript, unflagged."""
    if not parts:
        return final_response, False
    from agent.conversation_loop import _join_truncated_parts

    for frag in messages:
        if isinstance(frag, dict):
            frag.pop("_length_continuation_fragment", None)
            frag.pop("_length_continuation_nudge", None)
    return _join_truncated_parts([*parts, (final_response, False)]), True


def _append_nudge_pair(agent: Any, messages: Any, assistant_row: dict, nudge: str) -> None:
    # Both halves of a re-prompt pair are ephemeral scaffolding: never persisted, and the
    # finalization pop strips an unanswered tail pair.
    assistant_row["_dropped_toolcall_nudge"] = True
    append_message(messages, assistant_row)
    append_message(messages, {"role": "user", "content": nudge, "_dropped_toolcall_nudge": True})
    agent._session_messages = messages


def _reprompt_dropped_tool_call(agent: Any, final_msg: dict, assistant_message: Any, finish_reason: Any,
                                messages: Any, nudge: str) -> bool:
    """Dropped tool-call recovery (seen with Claude via gateways): finish_reason="tool_calls" with
    empty tool_calls would end the turn unstarted; re-prompt (max 3 CONSECUTIVE stalls). False
    is a genuine turn end, which clears both tool-call stall budgets."""
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
        agent._emit_diagnostic_status(
            "↻ Model signaled a tool call but sent none — "
            f"re-prompting ({agent._dropped_toolcall_retries}/3)"
        )
        _append_nudge_pair(agent, messages, final_msg, nudge)
        return True
    agent._dropped_toolcall_retries = 0
    agent._leaked_toolcall_retries = 0
    return False


def _reprompt_leaked_tool_call(agent: Any, assistant_message: Any, finish_reason: Any, messages: Any) -> bool:
    """Chat-completions reply with no ``tool_calls`` but a call to an offered tool written as text
    in its reasoning or content (local llama.cpp/vLLM tool parsers missing the markup).
    Re-prompt, never execute the recovered text: bounded to ``_LEAKED_TOOLCALL_MAX_RETRIES``,
    reset by a real tool round or a clean final answer. Past the budget the turn falls through to
    the normal path, loudly. ``length`` stops keep the continuation path (markup cut mid-call)."""
    if (
        str(getattr(agent, "api_mode", "") or "chat_completions").strip().lower() != "chat_completions"
        or assistant_message.tool_calls
        or finish_reason == "length"
        or not getattr(agent, "valid_tool_names", None)
    ):
        return False
    for where, text in (("reasoning", agent._extract_reasoning(assistant_message)),
                        ("reply", assistant_message.content)):
        if leak := find_leaked_tool_call(text, agent.valid_tool_names):
            break
    else:
        return False
    retries = getattr(agent, "_leaked_toolcall_retries", 0)
    if retries >= _LEAKED_TOOLCALL_MAX_RETRIES:
        logger.warning(
            "Model left a %s as text in the %s again after %d re-prompts — NOT executed; ending "
            "the turn on the normal path (format=%s complete=%s model=%s provider=%s): %r",
            leak.label, where, retries, leak.fmt, leak.complete, agent.model, agent.provider,
            leak.snippet,
        )
        agent._emit_diagnostic_status(
            f"⚠ Model wrote a {leak.label} as text instead of calling it — not executed"
        )
        return False
    agent._leaked_toolcall_retries = retries + 1
    logger.warning(
        "Model wrote a %s as text in the %s with no structured tool_calls (format=%s "
        "complete=%s finish_reason=%s) — re-prompting (%d/%d, model=%s provider=%s): %r",
        leak.label, where, leak.fmt, leak.complete, finish_reason, agent._leaked_toolcall_retries,
        _LEAKED_TOOLCALL_MAX_RETRIES, agent.model, agent.provider, leak.snippet,
    )
    agent._emit_diagnostic_status(
        f"↻ Model wrote a {leak.label} as text — re-prompting "
        f"({agent._leaked_toolcall_retries}/{_LEAKED_TOOLCALL_MAX_RETRIES})"
    )
    _append_nudge_pair(agent, messages, agent._build_assistant_message(assistant_message, finish_reason),
                       leaked_tool_call_nudge(leak, where))
    return True


def _promote_reasoning_only_stop(agent: Any, assistant_message: Any, finish_reason: Any, messages: Any,
                                 api_call_count: Any) -> Optional[str]:
    """The reasoning to return as the answer on a reasoning-only clean stop, else None.

    Trusted routes only (elsewhere reasoning is private and takes the ladder): some parsers (vLLM
    nemotron_v3 past ~500K prompt tokens) file the whole answer as reasoning when the closing
    delimiter is missing; ``stop`` means generation is complete, so promote it BEFORE the ladder.
    ``length`` (cut off mid-thought) stays on the continuation path. The promoted text is RETURNED
    as the answer but never written into the assistant row's ``content``: chain-of-thought stored
    as ordinary content is indistinguishable from a real reply on every history surface (#111761).
    The row keeps ``content`` empty with the text in its reasoning fields and carries the promoted
    text as the ``api_content`` sidecar, so the next turn still replays it byte-identically.
    Anthropic thinking (signed ``thinking`` block, or a plugin's ``*.native_assistant`` carrier of
    native Claude turns) is a summary written by a separate model, never the answer: it takes the
    empty-response continuation instead."""
    _content = assistant_message.content
    if not (
        finish_reason == "stop"
        and not assistant_message.tool_calls
        and (_content is None or (isinstance(_content, str) and not _content.strip()))
        and not any(
            isinstance(d, dict) and (
                (d.get("type") in ("thinking", "redacted_thinking") and (d.get("signature") or d.get("data")))
                or str(d.get("type") or "").endswith(".native_assistant")
            )
            for d in getattr(assistant_message, "reasoning_details", None) or ()
        )
        and answer_in_reasoning_capability(agent)
    ):
        return None
    promoted = agent._extract_reasoning(assistant_message) or None
    if promoted:
        # WARNING, not INFO: a model that keeps ending turns this way is stalled
        # (planning monologue, zero tool calls) while the turn reports "complete".
        logger.warning(
            "Reasoning-only clean stop (%d chars) — returning the reasoning as the final "
            "response (model=%s provider=%s api_calls=%d tool_turns=%d)",
            len(promoted), agent.model, agent.provider, api_call_count,
            sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")),
        )
    return promoted


def finish_text_response(
    agent: Any, *, assistant_message: Any, response: Any, finish_reason: Any, messages: Any,
    api_messages: Any, conversation_history: Any, api_call_count: Any, user_message: Any,
    active_system_prompt: Any, final_response: Any, _turn_exit_reason: Any,
    _preflight_compression_blocked: Any, codex_ack_continuations: Any,
    truncated_response_parts: Any, length_continue_retries: Any,
    _pending_verification_response: Any, _pending_verification_response_previewed: Any,
    effective_task_id: Any,
) -> FinalResponseVerdict:
    """Finish (or defer) a text-only assistant response in the original guard order. Every
    continuation path sets ``final_response = None`` so an acknowledgment never suppresses
    iteration-limit summarization; the final message is appended and flushed only after the
    stop gates accept it."""
    from agent.conversation_loop import (
        _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE, _DROPPED_TOOLCALL_NUDGE_CONTENT,
    )

    def _verdict(action: str, result: Optional[dict[str, Any]] = None) -> FinalResponseVerdict:
        return FinalResponseVerdict(
            action=action, active_system_prompt=active_system_prompt, final_response=final_response,
            _turn_exit_reason=_turn_exit_reason,
            _preflight_compression_blocked=_preflight_compression_blocked,
            codex_ack_continuations=codex_ack_continuations,
            truncated_response_parts=truncated_response_parts,
            length_continue_retries=length_continue_retries,
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=_pending_verification_response_previewed,
            api_call_count=api_call_count,
            result=result,
        )

    # Checked before reasoning promotion and before prose is accepted: a call left as text
    # would otherwise end the turn "complete" with the action never run.
    if _reprompt_leaked_tool_call(agent, assistant_message, finish_reason, messages):
        final_response = None
        return _verdict("continue")

    _promoted = _promote_reasoning_only_stop(agent, assistant_message, finish_reason, messages, api_call_count)
    final_response = _promoted or assistant_message.content or ""
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
        api_call_count = _ev.api_call_count
        return _verdict(_ev.action, _ev.result)

    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    # Surface the one-shot fallback switch notice before dropping the retry buffer so a
    # provider/model switch stays visible on success.
    agent._emit_pending_fallback_notice()
    agent._clear_status_buffer()

    if marker_verdict := _recover_marker_reply(
            agent, final_response, _verdict, truncated_parts=truncated_response_parts, promoted=_promoted):
        return marker_verdict

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
        intent_ack_continuation_mode, looks_like_degenerate_final, promoted_reasoning_announces_action,
        tool_results_this_turn, trailing_continue_intent,
    )

    _ack_mode = intent_ack_continuation_mode(agent)
    # Said-continue-but-stopped guard: no tool calls but the short reply TAILS with an
    # announced next action. Reuses the SAME bounded continuation counter (max 2 per turn).
    # Promoted reasoning gets the broader first-person-plan tail detector: with tools offered
    # and zero tool calls, chain-of-thought ending on "Let me batch the terminal calls..." is a
    # stalled model, and returning it as the answer aborts the tool loop while reporting
    # "complete" (#111761). Same cap, so a model that never acts still ends after 2 nudges.
    _stall_text = agent._strip_think_blocks(final_response or "")
    _stall_continue_intent = (
        bool(getattr(agent, "_stall_guards", True))
        and agent.valid_tool_names
        and codex_ack_continuations < 2
        and (
            trailing_continue_intent(_stall_text)
            or (bool(_promoted) and promoted_reasoning_announces_action(_stall_text))
        )
    )
    # Degenerate-final guard (#103483): the turn did real tool work and then stopped on a
    # fragment. Same scope knob and the SAME bounded counter as the ack continuation; the nudge
    # row itself closes the tool-work window, so a second fragment ends the turn as the answer.
    _tool_rows = tool_results_this_turn(messages)
    _degenerate_final = (
        bool(getattr(agent, "_stall_guards", True))
        and _ack_mode != "off"
        and codex_ack_continuations < 2
        and _tool_rows > 0
        and looks_like_degenerate_final(_stall_text, user_message=user_message)
    )
    # Precedence: an announced next action outranks the fragment shape; the codex ack is last.
    if _stall_continue_intent:
        _continuation_kind = "stall"
    elif _degenerate_final:
        _continuation_kind = "degenerate"
    elif (
        _ack_mode != "off"
        and agent.valid_tool_names
        and codex_ack_continuations < 2
        and agent._looks_like_codex_intermediate_ack(
            user_message=user_message, assistant_content=final_response, messages=messages,
            require_workspace=(_ack_mode == "codex_only"),
        )
    ):
        _continuation_kind = "ack"
    else:
        _continuation_kind = None
    if _continuation_kind:
        if _continuation_kind == "stall":
            logger.info(
                "Stall guard: turn ending on trailing continue-"
                "intent with no tool calls — re-prompting to act "
                "(%d/2)", codex_ack_continuations + 1,
            )
        elif _continuation_kind == "degenerate":
            logger.warning(
                "Degenerate final: %d-char fragment %r ended the turn after %d tool result(s) — "
                "re-prompting (%d/2)", len(_stall_text), _stall_text[:40], _tool_rows,
                codex_ack_continuations + 1,
            )
        codex_ack_continuations += 1
        interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
        if _promoted:
            # Same sidecar as the final row: the wire copy must carry the promoted text, not only
            # ``reasoning_content``, or the continuation replays an empty assistant turn.
            interim_msg["api_content"] = final_response
        append_message(messages, interim_msg)
        agent._emit_interim_assistant_message(interim_msg)
        append_message(messages, {
            "role": "user",
            "content": (
                _DEGENERATE_FINAL_NUDGE if _continuation_kind == "degenerate"
                else _CODEX_ACK_CONTINUATION_NUDGE
            ),
        })
        agent._session_messages = messages
        # An acknowledgment is non-final: its text must not suppress iteration-limit
        # summarization if the continuation exhausts budget.
        final_response = None
        return _verdict("continue")

    codex_ack_continuations = 0

    final_response, _joined = _absorb_continuation_fragments(final_response, truncated_response_parts, messages)
    if _joined:
        truncated_response_parts, length_continue_retries = [], 0

    final_response = agent._strip_think_blocks(final_response).strip()

    # A provider may end a degenerate loop normally with finish_reason="stop" instead of
    # exhausting its output cap (#100716). Check every completed visible text response before
    # any verify/kanban interim emission or durable transcript write.
    # Runaway scale and shape only: a completed answer the user asked to be repetitive is
    # delivered, unlike a length-truncated fragment that burned the whole budget.
    if (
        final_response
        and len(final_response) >= STOP_PATH_MIN_CHARS
        and is_runaway_repetition(final_response)
    ):
        line, user_response, error = _REPETITION_STOPPED
        agent._vprint(f"{agent.log_prefix}{line}", force=True, diagnostic=True)
        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", stamp_failure(
            partial_result(messages, api_call_count, user_response, error), "truncated", True,
        ))

    final_msg = agent._build_assistant_message(assistant_message, finish_reason)
    if _promoted:
        # Replay sidecar only: ``content`` stays empty so the row is never mistaken for a
        # real reply; ``build_api_messages`` substitutes ``api_content`` on the wire.
        final_msg["api_content"] = final_response

    if _reprompt_dropped_tool_call(agent, final_msg, assistant_message, finish_reason, messages,
                                   _DROPPED_TOOLCALL_NUDGE_CONTENT):
        final_response = None
        return _verdict("continue")

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

    # Plugins rewrite the reply BEFORE it is appended and flushed: SQLite treats a non-blank
    # assistant row as settled, so a transform after this write would reach the user but never
    # the stored/replayed transcript (#44239). finalize_turn reads the recorded outcome; like
    # there, an interrupted turn keeps the raw text.
    from agent.turn_finalizer import apply_llm_output_transform
    _transformed = False
    if not getattr(agent, "_interrupt_requested", False):
        final_response, _transformed, _ = apply_llm_output_transform(
            agent, final_response, turn_id=getattr(agent, "_current_turn_id", "") or "", logger=logger,
        )
    if _transformed:
        if _promoted:
            final_msg["api_content"] = final_response
        else:
            final_msg["content"] = final_response
    publish_held_final_row(agent, final_msg, final_response, _joined)

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
