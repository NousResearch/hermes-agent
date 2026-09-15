"""Truncation recovery (``finish_reason == "length"``) for the conversation turn loop.

Handles thinking-budget exhaustion, repetition-dominated truncation, content-filter stream
stalls escalated to the fallback chain, text continuation nudges (up to 4, with the ceiling
exit that drops the fragment trail), bounded chunking recovery for output-limit-truncated
tool calls (both the provider-flagged ``length`` path and the router-hidden
``tool_calls``/``stop`` path), and the final roll-back. Nothing here imports
``agent.conversation_loop`` at module level (cycle); loop-internal helpers are imported
lazily so tests patching them keep working.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.error_classifier import FailoverReason
from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence
from agent.repetition_guard import is_repetition_dominated
from agent.turn_api_call import stop_thinking_spinner
from agent.turn_retry_state import TurnRetryState
from agent.usage_pricing import normalize_usage
from hermes_constants import PARTIAL_STREAM_STUB_ID

logger = logging.getLogger("agent.conversation_loop")

_CONTINUABLE_MODES = {"chat_completions", "bedrock_converse", "anthropic_messages"}
_THINK_TAG_RE = re.compile(r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>', re.IGNORECASE)
_TRUNCATED_FINAL = "Response truncated due to output length limit"
_FIRST_TRUNCATED_FINAL = "First response truncated due to output length limit"

# Chunking-recovery budget for an output-limit-truncated tool call. Deliberately separate from
# ``truncated_tool_call_retries`` (the max_tokens-boost path): that counter is only reset by a
# successful tool round, so a run of truncations with nothing executed in between was capped at
# 4 no matter how clearly the model was converging. The base allowance still refuses a model that
# just repeats its oversized payload; the allowance only grows while the truncated payload keeps
# shrinking, and it stops growing at ``_CHUNKING_MAX_EXTENSIONS`` — so the bound stays hard
# (base 4 + at most 4 extensions x step 4 = 20 attempts).
_CHUNKING_BASE_ATTEMPTS = 4
_CHUNKING_MAX_EXTENSIONS = 4
# Attempts between re-arms: extension N is granted on attempt N*STEP + 1 (5, 9, 13, 17).
_CHUNKING_EXTENSION_STEP = 4
# "Measurable progress" = the truncated payload shrank by at least this fraction.
_CHUNKING_PROGRESS_RATIO = 0.05
# #106260: a stream that died on a context-overflow error after partial delivery must not seed a
# continuation — the transcript already cannot fit, and appending the partial stub grows every
# later request into the same overflow. End the turn via the recovery contract instead.
_CONTEXT_OVERFLOW_PARTIAL_FINAL = (
    "The request no longer fits the model's context window, so the partial "
    "response was not continued. Continue in a fresh session (/new; gateway "
    "chats are reset automatically)."
)

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
# Below this many free tokens the prompt itself filled the window: a continuation nudge +
# fragment costs ~100 tokens per attempt, so retrying only shrinks the room (#106120).
_MIN_CONTINUATION_HEADROOM = 512
_WINDOW_FILLED = (
    "⚠️ **Context window full.** The prompt used {prompt:,} of this model's {ctx:,}-token "
    "context window, leaving no room to answer in. This is a context-window limit, not an "
    "output-length limit.\n\nTo fix this:\n→ Compress the conversation with `/compress` or start "
    "a new session\n→ Or raise the model's context window (e.g. Ollama `num_ctx`)"
)


def _prompt_filled_window(agent: Any, response: Any) -> Optional[tuple[int, int]]:
    """``(prompt_tokens, context_length)`` when this response's usage shows the prompt left
    less than ``_MIN_CONTINUATION_HEADROOM`` in the window compression resolves for the
    model; ``None`` (keep continuing) when either number is unknown."""
    ctx = int(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0)
    usage = getattr(response, "usage", None)
    if not (ctx and usage):
        return None
    prompt = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode).prompt_tokens
    return (prompt, ctx) if prompt and ctx - prompt < _MIN_CONTINUATION_HEADROOM else None


def normalize_response_for_agent(agent: Any, response: Any) -> Any:
    """One OpenAI-style message from any transport; Anthropic strips the OAuth tool prefix."""
    if agent.api_mode == "anthropic_messages":
        return agent._get_transport().normalize_response(
            response, strip_tool_prefix=agent._is_anthropic_oauth
        )
    return agent._get_transport().normalize_response(response)


def partial_result(
    messages: List[Dict[str, Any]], api_call_count: int, final_response: str,
    error: Optional[str] = None, *, failed: bool = False, compression_exhausted: bool = False,
) -> Dict[str, Any]:
    """Typed incomplete-turn result (``partial`` unless ``failed``); ``error`` defaults to
    ``final_response``. ``compression_exhausted`` carries the #98722 typed bit the gateway
    consumes to reset/move future input to a clean session (see run_turn.py)."""
    result = {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        ("failed" if failed else "partial"): True,
        "error": final_response if error is None else error,
    }
    if compression_exhausted:
        result["compression_exhausted"] = True
    return result


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
    # Turn-scoped chunking-recovery budget (owned by the loop, passed through). Not a plain
    # loop local: a fresh ``TurnRetryState`` is built every attempt, so the budget has to
    # survive that rebuild to bound a run of truncations with no tool round in between.
    chunking_progress: Any = None


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
    # Normalized assistant message for the truncated response — computed once by the caller so
    # the chunking path can judge payload progress without re-normalizing (which would re-run
    # the sanitizer over already-sanitized arguments).
    trunc_msg: Any = None
    action: str = "fallthrough"
    result: Optional[Dict[str, Any]] = None
    window_filled: Optional[tuple[int, int]] = None  # (prompt_tokens, context_length)

    def done(self, action: str, result: Optional[Dict[str, Any]] = None) -> TruncationVerdict:
        self.action, self.result = action, result
        return self

    def end_turn(
        self, final_response: str, error: Optional[str] = None, *,
        result_messages: Optional[List[Dict[str, Any]]] = None, cleanup: bool = True,
        failed: bool = False, compression_exhausted: bool = False,
    ) -> TruncationVerdict:
        """Persist and end the turn as partial (or ``failed``).

        ``compression_exhausted`` forwards the #98722 typed bit so the gateway can
        move future input off a bloated session (run_turn.py consumes it).
        """
        agent = self.agent
        if cleanup:
            agent._cleanup_task_resources(self.effective_task_id)
        agent._persist_session(self.messages, self.conversation_history)
        return self.done("return", partial_result(
            self.messages if result_messages is None else result_messages, self.api_call_count,
            final_response, error, failed=failed, compression_exhausted=compression_exhausted,
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

    filled = st.window_filled
    if n < 4 and filled is None:
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
        f"{agent.log_prefix}⚠️  Not continuing — each attempt would only grow the prompt."
        if filled is not None else
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
    if filled is not None:
        notice = _WINDOW_FILLED.format(prompt=filled[0], ctx=filled[1])
        return st.end_turn(
            f"{partial_response}\n\n{notice}" if partial_response else notice,
            f"Prompt used {filled[0]} of {filled[1]} context tokens; no room to answer",
        )
    return st.end_turn(
        partial_response or _CEILING_NO_TEXT,
        "Response remained truncated after 4 continuation attempts",
    )


def _incomplete_tool_call_names(assistant_message: Any) -> List[str]:
    """Names of tool calls whose argument JSON is incomplete/unrepairable.

    Mirrors the streaming assembler's verdict (``json.loads`` fails *and* the repair
    fallback gives up), because a non-streaming provider response reaches this module
    without that flag. A payload that repairs cleanly is a complete call, not a
    truncation — nothing here may weaken that distinction.
    """
    from agent.message_sanitization import _repair_tool_call_arguments

    names: List[str] = []
    for tc in getattr(assistant_message, "tool_calls", None) or []:
        function = getattr(tc, "function", None)
        args = getattr(function, "arguments", None)
        if not isinstance(args, str) or not args.strip():
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError:
            name = getattr(function, "name", None) or "?"
            if _repair_tool_call_arguments(args, name) == "{}":
                names.append(name)
    return names


@dataclass
class _ChunkingProgress:
    """Mutable chunking-recovery budget, owned by the turn loop (``_LoopState``) so it survives
    the per-iteration ``TurnRetryState`` rebuild. See ``_chunking_recovery_state``.

    ``window_baseline`` is the payload size at the start of the current extension window: the
    budget qualifies for the next extension on *cumulative* shrinkage since that baseline, so a
    short plateau on the boundary attempt cannot erase convergence earned just before it.
    """

    attempts: int = 0
    last_size: int = 0
    extensions: int = 0
    window_baseline: int = 0


def _truncated_payload_size(assistant_message: Any) -> int:
    """UTF-8 byte length of the truncated tool-call arguments that survived into the message.

    Progress is judged on this: a model that is converging on a deliverable payload emits
    measurably shorter argument JSON each attempt, whatever the tool. Reads the raw argument
    strings (before any repair) so the measurement reflects what the provider actually sent.
    Encoded as UTF-8 so the number matches the "bytes" the log and budget constants talk about
    (``len(str)`` would count code points, which differ for non-ASCII payloads).
    """
    total = 0
    for tc in getattr(assistant_message, "tool_calls", None) or []:
        args = getattr(getattr(tc, "function", None), "arguments", None)
        if isinstance(args, str):
            total += len(args.encode("utf-8", "surrogatepass"))
    return total


def _chunking_recovery_state(progress: "_ChunkingProgress", progress_size: int) -> tuple[int, int]:
    """Advance the chunking-recovery budget and report ``(attempt, limit)``.

    A dedicated counter — deliberately NOT ``truncated_tool_call_retries``, which the
    max_tokens-boost path shares and only a successful tool round resets. That coupling is
    what starved this recovery: a run of truncations with no tool round between them always
    died at 4, however clearly the model was converging.

    An extension is granted on the boundary attempts (``BASE + N*STEP + 1``) when the payload
    has shrunk by at least ``_CHUNKING_PROGRESS_RATIO`` **cumulatively since the last
    extension** (``window_baseline``), not merely on the boundary attempt itself. Judging the
    window instead of the single step is what stops a short plateau on the boundary from
    discarding genuine convergence earned on the attempts just before it.

    Both directions stay hard: noise or an identical payload never reaches the cumulative
    threshold, so the budget never re-arms and the run is refused at the base limit; a
    converging run stops growing after ``_CHUNKING_MAX_EXTENSIONS``.
    """
    attempts = progress.attempts + 1
    if progress.window_baseline <= 0 and progress_size > 0:
        # First measurement of the base window: the baseline everything else is judged against.
        progress.window_baseline = progress_size
    on_boundary = (
        attempts > _CHUNKING_BASE_ATTEMPTS
        and attempts % _CHUNKING_EXTENSION_STEP == 1
    )
    window_shrank = (
        progress.window_baseline > 0
        and progress_size > 0
        and progress_size <= progress.window_baseline * (1.0 - _CHUNKING_PROGRESS_RATIO)
    )
    if on_boundary and window_shrank and progress.extensions < _CHUNKING_MAX_EXTENSIONS:
        progress.extensions += 1
        # New window starts here: the next extension is earned by shrinkage from this payload.
        progress.window_baseline = progress_size
    progress.attempts = attempts
    if progress_size > 0:
        progress.last_size = progress_size
    limit = _CHUNKING_BASE_ATTEMPTS + progress.extensions * _CHUNKING_EXTENSION_STEP
    return attempts, limit


def _request_chunked_tool_retry(
    st: _Trunc, _retry: TurnRetryState, progress: Optional["_ChunkingProgress"],
    broken_tools: List[str],
) -> Optional[TruncationVerdict]:
    """An output-limit truncation cut a tool call off mid-arguments: discard the broken
    response and ask for smaller calls instead of a bigger output budget.

    Boosting ``max_tokens`` here is the pathological path — the model regenerates the same
    oversized payload and truncates again, so the turn ends as an error after N identical
    requests. Returns ``None`` once the chunking budget is spent, so the caller's terminal
    refusal still reports that the incomplete arguments were never executed.
    """
    from agent.conversation_loop import _get_continuation_prompt

    agent = st.agent
    if progress is None:
        # No loop-scoped budget threaded in (direct unit use): behave like one attempt.
        progress = _ChunkingProgress()
    _progress = _truncated_payload_size(st.trunc_msg)
    attempt, limit = _chunking_recovery_state(progress, _progress)
    if attempt > limit:
        return None
    agent._vprint(
        f"{agent.log_prefix} Tool call ({', '.join(broken_tools[:3])}) was cut off by the output "
        f"limit — requesting chunked retry ({attempt}/{limit}, args {_progress} bytes)..."
    )
    # Broken response discarded (never appended, never executed); the nudge is a user row like
    # the network-drop continuation, and the pre-call sequence repair folds it onto a user tail.
    append_message(st.messages, {
        "role": "user", "content": _get_continuation_prompt(False, broken_tools),
        "_length_continuation_nudge": True,
    })
    agent._session_messages = st.messages
    _retry.restart_with_chunking_nudge = True
    return st.done("break")


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
        agent._ephemeral_max_output_tokens = min(_tc_boost, max(32768, _tc_requested_cap or 0))
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


def _recover_hidden_truncation_phase(
    agent: Any, request: Any, _retry: TurnRetryState, *, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int, effective_task_id: Any,
    current_turn_user_idx: Any, truncated_tool_call_retries: int, retry_count: int,
    compression_attempts: int, chunking_progress: Optional["_ChunkingProgress"] = None,
) -> TruncationVerdict:
    """Bounded chunking recovery for a truncation the router hid behind a non-``length``
    ``finish_reason`` (see ``turn_tool_validation.HiddenTruncationRequest``).

    The ``finish_reason`` phase never saw a truncation, so this is the ONLY place the same
    recovery can run for such an attempt: it shares ``chunking_progress`` with the
    ``finish_reason="length"`` path, so both write into one hard-bounded budget. Reuses
    ``_request_chunked_tool_retry`` and its terminal refusal — no duplicated recovery logic
    and no second, weaker bound.

    ``_retry`` is the loop's LIVE ``TurnRetryState`` (not a fresh throwaway instance): the
    restart flag ``_request_chunked_tool_retry`` arms has to be visible to
    ``apply_retry_restarts`` after this phase returns, exactly as on the ``length`` path. The
    broken response is discarded here and never staged, persisted or executed.
    """
    st = _Trunc(
        agent=agent, response=None, finish_reason=request.finish_reason,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id, current_turn_user_idx=current_turn_user_idx,
        messages=messages, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=truncated_tool_call_retries, retry_count=retry_count,
        compression_attempts=compression_attempts, chunking_progress=chunking_progress,
        trunc_msg=request.assistant_message,
    )
    _broken_tools = list(request.broken_tools)
    _chunked = _request_chunked_tool_retry(st, _retry, chunking_progress, _broken_tools)
    if _chunked is not None:
        return _chunked
    # Chunking budget spent without the payload ever shrinking: refuse deterministically,
    # identical to the output-limit path (no max_tokens boost, no execution).
    agent._flush_status_buffer()
    agent._vprint(
        f"{agent.log_prefix}⚠️  Tool call ({', '.join(_broken_tools[:3])}) stayed over "
        "the output limit after chunked retries — refusing to execute incomplete "
        "tool arguments.",
        force=True,
    )
    close_interrupted_tool_sequence(st.messages, _TRUNCATED_FINAL)
    return st.end_turn(_TRUNCATED_FINAL, cleanup=False)


def recover_from_truncation(
    agent: Any, response: Any, finish_reason: str, _retry: TurnRetryState, *,
    messages: List[Dict[str, Any]], conversation_history: Any, api_kwargs: Any, api_call_count: int,
    effective_task_id: Any, current_turn_user_idx: Any, length_continue_retries: int,
    truncated_response_parts: List[str], truncated_tool_call_retries: int, retry_count: int,
    compression_attempts: int, chunking_progress: Optional["_ChunkingProgress"] = None,
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
        compression_attempts=compression_attempts, chunking_progress=chunking_progress,
    )
    st.window_filled = _prompt_filled_window(agent, response)
    agent._vprint(
        f"{agent.log_prefix}⚠️  Response truncated — stream ended before completion"
        if st.is_stub else
        f"{agent.log_prefix}⚠️  Response truncated (finish_reason='length') - the prompt filled the "
        f"context window ({st.window_filled[0]:,}/{st.window_filled[1]:,} tokens)"
        if st.window_filled else
        f"{agent.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens",
        force=True,
    )

    # #106260: a context-overflow error after partial delivery must not seed a
    # continuation. _partial_stream_stub marks such stubs _overflow_terminal and
    # leaves content empty; continuing would only re-send a larger request into
    # the same overflow. The stub path never raises, so this class never reached
    # recover_from_overflow's compress-and-retry on main either — ending the turn
    # replaces a growth loop, not a compression attempt.
    if getattr(st.response, "_overflow_terminal", False):
        agent._flush_status_buffer()
        agent._vprint(
            f"{agent.log_prefix}⚠️ Stream ended on a context-overflow error after "
            "partial delivery — not continuing (the request no longer fits the model's "
            "context window).",
            force=True,
        )
        # Prior tool batches can leave a tool-result tail; this path never reaches
        # finalize_turn (same as the truncated-tool-call terminal above).
        close_interrupted_tool_sequence(st.messages, _CONTEXT_OVERFLOW_PARTIAL_FINAL)
        # Carry the #98722 typed exhaustion bit so the gateway resets/moves future
        # input to a clean session instead of leaving this bloated one authoritative
        # for the next turn.
        return st.end_turn(
            _CONTEXT_OVERFLOW_PARTIAL_FINAL,
            error=_CONTEXT_OVERFLOW_PARTIAL_FINAL,
            failed=True,
            compression_exhausted=True,
        )

    _trunc_msg = normalize_response_for_agent(agent, response)
    st.trunc_msg = _trunc_msg
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
            if not st.is_stub:
                # A real output-limit truncation (not a network drop) whose argument JSON never
                # completed: only chunking guidance can break the regenerate-and-truncate cycle
                # that the max_tokens boost feeds.
                _broken_tools = _incomplete_tool_call_names(_trunc_msg)
                if _broken_tools:
                    _chunked = _request_chunked_tool_retry(
                        st, _retry, st.chunking_progress, _broken_tools
                    )
                    if _chunked is not None:
                        return _chunked
                    # Chunking budget spent without the payload ever shrinking: the model keeps
                    # regenerating the same oversized call. Boosting max_tokens here only
                    # re-sends it, so refuse deterministically instead of burning more calls.
                    agent._flush_status_buffer()
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Tool call ({', '.join(_broken_tools[:3])}) stayed over "
                        "the output limit after chunked retries — refusing to execute incomplete "
                        "tool arguments.",
                        force=True,
                    )
                    close_interrupted_tool_sequence(st.messages, _TRUNCATED_FINAL)
                    return st.end_turn(_TRUNCATED_FINAL, cleanup=False)
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
