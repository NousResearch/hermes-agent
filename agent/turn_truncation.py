"""Truncation recovery (``finish_reason == "length"``) for the conversation turn loop.

Handles thinking-budget exhaustion, repetition-dominated truncation, content-filter stream
stalls escalated to the fallback chain, text continuation nudges (up to 4, with the ceiling
exit that drops the fragment trail), truncated tool-call retries with max_tokens boosts, and
the final roll-back. Nothing here imports ``agent.conversation_loop`` at module level
(cycle); loop-internal helpers are imported lazily so tests patching them keep working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.error_classifier import FailoverReason
from agent._truncation_boost_cap import resolve_truncation_boost_cap
from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence
from agent.repetition_guard import is_repetition_dominated
from agent.turn_api_call import stop_thinking_spinner
from agent.turn_retry_state import TurnRetryState
from hermes_constants import PARTIAL_STREAM_STUB_ID

logger = logging.getLogger("agent.conversation_loop")

_CONTINUABLE_MODES = {"chat_completions", "bedrock_converse", "anthropic_messages"}
_THINK_TAG_RE = re.compile(r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>', re.IGNORECASE)
_TRUNCATED_FINAL = "Response truncated due to output length limit"
_FIRST_TRUNCATED_FINAL = "First response truncated due to output length limit"

_THINKING_EXHAUSTED = (
    "💭 Reasoning exhausted the output token budget — no visible response was produced.",
    "⚠️ **Thinking Budget Exhausted**\n\nThe model used all its output tokens on reasoning "
    "and had none left for the actual response.\n\nTo fix this:\n"
    "→ Lower reasoning effort: `/reasoning low` or `/reasoning minimal`\n"
    "→ Or switch to a larger/non-reasoning model with `/model`",
    "Model used all output tokens on reasoning with none left "
    "for the response. Try lowering reasoning effort or increasing max_tokens.",
)
_REPETITION_DOMINATED = (
    "🔁 Response dominated by repeated text — stopping instead of continuing a degenerate response.",
    "⚠️ **Response Stopped — Repetition Detected**\n\nThe model fell into a repetition loop while "
    "writing this response, so continuing would only produce more repeated text. The partial response "
    "was discarded.\n\n→ Switch to a different model with `/model`\n"
    "→ Or resend your message (your conversation history is preserved)",
    "Model output entered a repetition loop and was truncated mid-loop; refusing to continue a "
    "degenerate response.",
)
_CEILING_NO_TEXT = (
    "⚠️ **No visible answer was produced.** The model hit its output-token limit on every "
    "continuation attempt — its reasoning consumed the entire budget each time.\n\nTo fix this:\n"
    "→ Lower reasoning effort: `/reasoning low` or `/reasoning none`\n→ Or raise max_tokens for this model"
)


def normalize_response_for_agent(agent: Any, response: Any) -> Any:
    """One OpenAI-style message from any transport; Anthropic strips the OAuth tool prefix."""
    if agent.api_mode == "anthropic_messages":
        return agent._get_transport().normalize_response(
            response, strip_tool_prefix=agent._is_anthropic_oauth
        )
    return agent._get_transport().normalize_response(response)


def partial_result(
    messages: List[Dict[str, Any]], api_call_count: int, final_response: str,
    error: Optional[str] = None, *, failed: bool = False,
) -> Dict[str, Any]:
    """Typed incomplete-turn result (``partial`` unless ``failed``); ``error`` defaults to
    ``final_response``."""
    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        ("failed" if failed else "partial"): True,
        "error": final_response if error is None else error,
    }


@dataclass
class TruncationVerdict:
    """Outcome of ``recover_from_truncation``.

    ``action``: ``"return"`` (end the turn with ``result``), ``"break"`` (a
    ``_retry.restart_with_*`` flag is set — restart the API call), ``"continue"``
    (re-issue the same call immediately) or ``"fallthrough"`` (unreachable in practice:
    every path exits, kept for the contract). The remaining fields are the loop locals
    the handler may have rebound."""

    action: str
    result: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    length_continue_retries: int
    truncated_response_parts: List[str]
    truncated_tool_call_retries: int
    retry_count: int
    compression_attempts: int


@dataclass(kw_only=True)
class _Trunc(TruncationVerdict):
    """Working state for the truncation phases — the verdict itself, plus the read-only
    call context; phases mutate the loop-local fields and ``done()`` stamps the action."""

    agent: Any
    response: Any
    finish_reason: str
    conversation_history: Any
    api_call_count: int
    effective_task_id: Any
    current_turn_user_idx: Any
    action: str = "fallthrough"
    result: Optional[Dict[str, Any]] = None

    def done(self, action: str, result: Optional[Dict[str, Any]] = None) -> TruncationVerdict:
        self.action, self.result = action, result
        return self

    def end_turn(
        self, final_response: str, error: Optional[str] = None, *,
        result_messages: Optional[List[Dict[str, Any]]] = None, cleanup: bool = True,
        failed: bool = False,
    ) -> TruncationVerdict:
        """Persist and end the turn as partial (or ``failed``)."""
        agent = self.agent
        if cleanup:
            agent._cleanup_task_resources(self.effective_task_id)
        agent._persist_session(self.messages, self.conversation_history)
        return self.done("return", partial_result(
            self.messages if result_messages is None else result_messages, self.api_call_count,
            final_response, error, failed=failed,
        ))

    @property
    def is_stub(self) -> bool:
        return getattr(self.response, "id", "") == PARTIAL_STREAM_STUB_ID


def _abort_reason(agent: Any, content: Any, has_tool_calls: bool) -> Optional[tuple]:
    """``(vprint, user response, error)`` when continuation must NOT be attempted:
    thinking exhausted the budget (reasoning blocks with no visible text after them —
    ``content=None`` from non-<think> models is normal truncation), or a repetition loop
    burned the budget on one fragment (reasoning stripped first)."""
    if has_tool_calls:
        return None
    if content and _THINK_TAG_RE.search(content) and not agent._has_content_after_think_block(content):
        return _THINKING_EXHAUSTED
    visible = agent._strip_think_blocks(content) if isinstance(content, str) else content
    if visible and is_repetition_dominated(visible):
        return _REPETITION_DOMINATED
    return None


def _content_filter_fallback(st: _Trunc, _retry: TurnRetryState) -> Optional[TruncationVerdict]:
    """Content-filter stream stall → fallback. ``_content_filter_terminated`` is
    content-deterministic, so escalate before retrying the primary; without a fallback
    fall through to normal continuation (best-effort, may loop)."""
    agent = st.agent
    if not (
        getattr(st.response, "_content_filter_terminated", False)
        and agent._fallback_index < len(agent._fallback_chain)
    ):
        return None
    agent._vprint(
        f"{agent.log_prefix}🛡️  Content filter terminated stream — activating fallback provider...",
        force=True,
    )
    agent._emit_status("Content filter terminated stream; switching to fallback...")
    if agent._try_activate_fallback():
        # Roll partial content back to the last clean turn so the fallback gets a
        # coherent continuation point; unmark survivors (their text left the partial).
        if st.truncated_response_parts:
            st.messages = agent._get_messages_up_to_last_assistant(st.messages)
        for _frag in st.messages:
            if isinstance(_frag, dict):
                _frag.pop("_length_continuation_fragment", None)
                _frag.pop("_length_continuation_nudge", None)
        agent._session_messages = st.messages
        st.length_continue_retries = 0
        st.truncated_response_parts = []
        st.retry_count = 0
        st.compression_attempts = 0
        _retry.primary_recovery_attempted = False
        _retry.restart_with_rebuilt_messages = True
        return st.done("break")
    agent._vprint(
        f"{agent.log_prefix}⚠️  No fallback provider configured — retrying with same provider "
        f"(may re-hit filter)...",
        force=True,
    )
    return None


def _continue_text(st: _Trunc, _retry: TurnRetryState, assistant_message: Any) -> TruncationVerdict:
    """Text truncation (no tool calls): append the fragment + a continuation nudge (up to
    4), then the ceiling exit that drops the fragment trail and keeps the stitched partial.
    Never appends an interim assistant row with NO visible content — strict providers
    reject it with 400 — only the nudge."""
    from agent.conversation_loop import _get_continuation_prompt, _join_truncated_parts

    agent = st.agent
    messages = st.messages
    st.length_continue_retries += 1
    n = st.length_continue_retries
    _interim_content = getattr(assistant_message, "content", None)
    if not _interim_content and not st.is_stub:
        # Thinking-only truncation: continuing with thinking ON re-burns the budget.
        agent._ephemeral_reasoning_off = True
    if _interim_content:
        interim_msg = agent._build_assistant_message(assistant_message, st.finish_reason)
        interim_msg["_length_continuation_fragment"] = True  # ceiling exit drops these
        append_message(messages, interim_msg)
        st.truncated_response_parts.append(_interim_content)

    if n < 4:
        _dropped_tools = getattr(st.response, "_dropped_tool_names", None)
        if st.is_stub and _dropped_tools:
            agent._vprint(
                f"{agent.log_prefix}↻ Stream interrupted mid "
                f"tool-call ({', '.join(_dropped_tools[:3])}) — requesting chunked retry ({n}/4)..."
            )
        elif st.is_stub:
            agent._vprint(f"{agent.log_prefix}↻ Stream interrupted — requesting continuation ({n}/4)...")
        else:
            agent._vprint(f"{agent.log_prefix}↻ Requesting continuation ({n}/4)...")
        append_message(messages, {
            "role": "user", "content": _get_continuation_prompt(st.is_stub, _dropped_tools),
            "_length_continuation_nudge": True,
        })
        agent._session_messages = messages
        _retry.restart_with_length_continuation = True
        return st.done("break")

    partial_response = agent._strip_think_blocks(_join_truncated_parts(st.truncated_response_parts)).strip()
    # The one-shot reasoning-off override must not leak into the next turn.
    agent._ephemeral_reasoning_off = False
    agent._vprint(
        f"{agent.log_prefix}⚠️  Response still truncated after {n} continuation attempts — "
        + ("keeping the partial response received so far." if partial_response
           else "no visible text was produced."),
        force=True,
    )
    # Unanswered continue nudges made every later turn re-truncate: drop the trail.
    idx = st.current_turn_user_idx
    _turn_start = idx + 1 if isinstance(idx, int) and idx >= 0 else 0
    messages[_turn_start:] = [
        m for m in messages[_turn_start:]
        if not (isinstance(m, dict) and (
            m.get("_length_continuation_fragment") or m.get("_length_continuation_nudge")
        ))
    ]
    if partial_response:
        append_message(messages, {
            "role": "assistant", "content": partial_response, "finish_reason": "length"
        })
    agent._session_messages = messages
    return st.end_turn(
        partial_response or _CEILING_NO_TEXT,
        "Response remained truncated after 4 continuation attempts",
    )


def _retry_truncated_tool_call(st: _Trunc, api_kwargs: Any) -> TruncationVerdict:
    """Truncated tool call: re-run the same call (up to 4×) with a boosted max_tokens —
    a real output-cap truncation needs it, harmless for a network stall — else refuse to
    execute incomplete arguments."""
    agent = st.agent
    if st.truncated_tool_call_retries < 4:
        st.truncated_tool_call_retries += 1
        n = st.truncated_tool_call_retries
        if st.is_stub:
            agent._buffer_vprint(f"⚠️  Stream interrupted mid tool-call — retrying ({n}/4)...")
        else:
            agent._buffer_vprint(f"⚠️  Truncated tool call detected — retrying API call ({n}/4)...")
        _tc_boost = (agent.max_tokens if agent.max_tokens else 4096) * (2 ** n)
        _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
        if _tc_requested_cap is not None:
            _tc_boost = max(_tc_boost, _tc_requested_cap)
        agent._ephemeral_max_output_tokens = min(
            _tc_boost,
            resolve_truncation_boost_cap(
                requested_cap=_tc_requested_cap, provider=agent.provider, model=agent.model,
            ),
        )
        return st.done("continue")  # don't append the broken response
    agent._flush_status_buffer()
    if st.is_stub:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Stream kept dropping mid tool-call after 4 retries — the action was not executed.",
            force=True,
        )
        _final_response = "Stream repeatedly dropped mid tool-call (network); the tool was not executed"
    else:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
            force=True,
        )
        _final_response = _TRUNCATED_FINAL
    agent._cleanup_task_resources(st.effective_task_id)
    # Prior tool batches can leave a tool-result tail; this path never reaches finalize_turn.
    close_interrupted_tool_sequence(st.messages, _final_response)
    return st.end_turn(_final_response, cleanup=False)


def recover_from_truncation(
    agent: Any, response: Any, finish_reason: str, _retry: TurnRetryState, *,
    messages: List[Dict[str, Any]], conversation_history: Any, api_kwargs: Any, api_call_count: int,
    effective_task_id: Any, current_turn_user_idx: Any, length_continue_retries: int,
    truncated_response_parts: List[str], truncated_tool_call_retries: int, retry_count: int,
    compression_attempts: int,
) -> TruncationVerdict:
    """Recover from a truncated response. Order is load-bearing: thinking exhaustion and
    repetition abort BEFORE any continuation; a content-filter stall escalates to the
    fallback chain BEFORE the primary is retried; text continuation (no tool calls) then
    truncated tool-call retry; finally roll back to the last complete assistant turn."""
    st = _Trunc(
        agent=agent, response=response, finish_reason=finish_reason,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id, current_turn_user_idx=current_turn_user_idx,
        messages=messages, length_continue_retries=length_continue_retries,
        truncated_response_parts=truncated_response_parts,
        truncated_tool_call_retries=truncated_tool_call_retries, retry_count=retry_count,
        compression_attempts=compression_attempts,
    )
    agent._vprint(
        f"{agent.log_prefix}⚠️  Response truncated — stream ended before completion"
        if st.is_stub else
        f"{agent.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens",
        force=True,
    )

    _trunc_msg = normalize_response_for_agent(agent, response)
    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

    abort = _abort_reason(agent, _trunc_content, _trunc_has_tool_calls)
    if abort is not None:
        line, user_response, error = abort
        agent._vprint(f"{agent.log_prefix}{line}", force=True)
        return st.end_turn(user_response, error)

    if agent.api_mode in _CONTINUABLE_MODES:
        cf = _content_filter_fallback(st, _retry)
        if cf is not None:
            return cf
        if _trunc_msg is not None:
            if not _trunc_has_tool_calls:
                return _continue_text(st, _retry, _trunc_msg)
            return _retry_truncated_tool_call(st, api_kwargs)

    if len(messages) > 1:
        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
        return st.end_turn(
            _TRUNCATED_FINAL, result_messages=agent._get_messages_up_to_last_assistant(messages)
        )
    # First message was truncated - mark as failed
    agent._flush_status_buffer()
    agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
    return st.end_turn(_FIRST_TRUNCATED_FINAL, cleanup=False, failed=True)


_CODEX_REPLAY_KEYS = (
    "content", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items",
)


def continue_codex_incomplete(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int,
) -> Optional[Dict[str, Any]]:
    """Codex Responses ``status=incomplete`` continuation (max 3 per turn).

    Appends the interim assistant message (deduped on visible content only — opaque
    provider state drifts per continuation; ``codex_reasoning_items`` are merged, not
    overwritten, because the earlier response holds the only native-compaction
    checkpoint) and, when a bare retry would be byte-identical, a user-role nudge — only
    after an assistant row, to preserve role alternation. Returns ``None`` to continue
    the turn loop, or the terminal ``partial`` result once retries are exhausted."""
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE

    agent._codex_incomplete_retries += 1
    n = agent._codex_incomplete_retries

    interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
    interim_has_content = bool((interim_msg.get("content") or "").strip())
    _reasoning = interim_msg.get("reasoning")
    interim_has_reasoning = isinstance(_reasoning, str) and bool(_reasoning.strip())
    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
    interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning or interim_has_codex_message_items:
        last_msg = messages[-1] if messages else None
        last_is_dict = isinstance(last_msg, dict)
        last_interim_visible = agent._interim_assistant_visible_text(last_msg) if last_is_dict else ""
        current_interim_visible = agent._interim_assistant_visible_text(interim_msg)
        if last_interim_visible or current_interim_visible:
            same_visible_output = last_interim_visible == current_interim_visible
        else:
            # Neither has text eligible for interim delivery: compare raw content+reasoning.
            same_visible_output = last_is_dict and (
                (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
            )
        if (
            last_is_dict
            and last_msg.get("role") == "assistant"
            and last_msg.get("finish_reason") == "incomplete"
            and same_visible_output
        ):
            # Duplicate: refresh replay state in place, no re-emitted commentary.
            for _key in _CODEX_REPLAY_KEYS:
                if _key not in interim_msg:
                    continue
                if _key == "codex_reasoning_items":
                    from agent.native_compaction import merge_interim_reasoning_items
                    last_msg[_key] = merge_interim_reasoning_items(last_msg.get(_key), interim_msg[_key])
                else:
                    last_msg[_key] = interim_msg[_key]
        else:
            append_message(messages, interim_msg)
            agent._emit_interim_assistant_message(interim_msg)

    if n < 3:
        # If the interim has nothing the Responses converter will replay, a bare retry is
        # byte-identical; a replayable interim holding only a ``compaction`` checkpoint
        # ALSO re-sends identically. One bare retry, then always nudge.
        interim_replayable = interim_has_content or interim_has_codex_reasoning or interim_has_codex_message_items
        if not interim_replayable or n >= 2:
            _last_msg = messages[-1] if messages else None
            if isinstance(_last_msg, dict):
                _already_nudged = (
                    _last_msg.get("role") == "user" and _last_msg.get("content") == _CODEX_INCOMPLETE_NUDGE
                )
                # Alternation guard: the nudge may only follow an assistant row.
                if not _already_nudged and _last_msg.get("role") == "assistant":
                    append_message(messages, {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE})
        if not agent.quiet_mode:
            agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({n}/3)")
        # Spinner/heartbeat notice: these retries can take minutes and otherwise look
        # like infinite thinking.
        # #70773: same FD-recycle corruption vector as #67142. The shared OpenAI client's connection pool
        # must NOT be closed from this watchdog/poll thread — worker threads from previous stale-killed
        # attempts may still be unwinding their SSL BIOs. The request-local client is already closed above
        # via _close_request_client_once. The shared client will be replaced lazily by
        # _ensure_primary_openai_client on the next request.
        # Surface the continuation on the live spinner/status line (CLI/TUI/Desktop) and gateway heartbeat:
        # each of these retries can spend minutes waiting on the provider, and without a distinct notice the
        # user only sees a generic thinking spinner ("infinite thinking", #64434).
        agent._emit_wait_notice(
            f"↻ model returned reasoning with no final answer — asking it to continue ({n}/3)"
        )
        agent._session_messages = messages
        return None

    agent._codex_incomplete_retries = 0
    agent._persist_session(messages, conversation_history)
    return partial_result(
        messages, api_call_count, "Codex response remained incomplete after 3 continuation attempts"
    )


@dataclass
class RefusalVerdict:
    """Outcome of ``handle_content_policy_refusal``: ``"break"`` (fallback activated —
    restart armed on ``_retry``; caller resets retry/compression counters) or
    ``"return"`` (the typed content-policy result in ``result``). ``active_system_prompt``
    is the possibly re-synced system prompt."""

    action: str
    result: Optional[Dict[str, Any]]
    active_system_prompt: Any


def handle_content_policy_refusal(
    agent: Any, response: Any, _retry: TurnRetryState, *, thinking_spinner: Any,
    messages: List[Dict[str, Any]], api_messages: Any, api_kwargs: Any, active_system_prompt: Any,
    conversation_history: Any, api_call_count: int, effective_task_id: Any, turn_id: Any,
    api_request_id: Any, api_start_time: float, retry_count: int, max_retries: int,
) -> RefusalVerdict:
    """HTTP-200 refusal (``finish_reason`` ``content_filter`` / ``guardrail_intervened``).
    Deterministic for the unchanged prompt — never retried: one configured-fallback try,
    else surface the refusal (explanation may live only in the reasoning channel)."""
    from agent.conversation_loop import (
        _CONTENT_POLICY_RECOVERY_HINT, _arm_fallback_restart, _content_policy_blocked_result
    )

    _refusal_result = normalize_response_for_agent(agent, response)
    _refusal_text = (getattr(_refusal_result, "content", None) or "").strip()
    if not _refusal_text:
        _refusal_text = (agent._extract_reasoning(_refusal_result) or "").strip()

    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type="ContentPolicyBlocked",
        error_message=_refusal_text or "model declined to respond (content_filter)",
        status_code=None, retry_count=retry_count, max_retries=max_retries, retryable=False,
        reason=FailoverReason.content_policy_blocked.value,
    )
    stop_thinking_spinner(agent, thinking_spinner)

    if agent._has_pending_fallback():
        agent._buffer_status("⚠️ Model declined to respond (safety refusal) — trying fallback...")
    if agent._try_activate_fallback():
        active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
        return RefusalVerdict("break", None, active_system_prompt)

    agent._flush_status_buffer()
    _refusal_log = _refusal_text[:500] + "..." if len(_refusal_text) > 500 else _refusal_text
    logger.warning(
        "%sModel declined to respond (finish_reason=content_filter). model=%s provider=%s refusal=%s",
        agent.log_prefix, agent.model, agent.provider,
        _refusal_log or "(no text)",
    )
    agent._emit_status("⚠️ The model declined to respond to this request (safety refusal).")
    _refusal_detail = (
        f"Model's explanation: {_refusal_text}" if _refusal_text else "The model returned no explanation."
    )
    _refusal_response = (
        "⚠️  The model declined to respond to this request (safety refusal — not a Hermes/gateway failure).\n\n"
        f"{_refusal_detail}\n\n"
        f"{_CONTENT_POLICY_RECOVERY_HINT}"
    )
    agent._cleanup_task_resources(effective_task_id)
    agent._persist_session(messages, conversation_history)
    return RefusalVerdict("return", _content_policy_blocked_result(
        messages, api_call_count, final_response=_refusal_response,
        error_detail=_refusal_text or "model declined (content_filter)",
    ), active_system_prompt)
