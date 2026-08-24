"""Overflow recovery for the conversation turn loop: 413 payload-too-large and
context-length errors after ``classify_api_error``.

Each path either compresses and signals a restart, defers softly (compression lock /
transient block), or ends the turn with a typed result. Nothing here imports
``agent.conversation_loop`` at module level (cycle); loop-internal helpers and the token
estimators that tests patch on the loop module are imported lazily inside the handlers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.conversation_compression import (
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE, COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, compression_blocked_transiently,
    compression_skipped_due_to_lock, context_compression_timed_out,
)
from agent.error_classifier import FailoverReason
from agent.message_sanitization import serialized_messages_bytes
from agent.model_metadata import (
    get_context_length_from_provider_error, is_output_cap_error,
    parse_available_output_tokens_from_error,
)
from agent.turn_retry_state import TurnRetryState
from utils import base_url_host_matches

logger = logging.getLogger("agent.conversation_loop")

_RETRY_HINT = "   💡 Try /new to start a fresh conversation, or /compress to retry compression."

_GITHUB_MODELS_HINT = (
    "   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
    "      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
    "      exceeds that floor, so this endpoint cannot run an agentic loop.",
    "      Use the `copilot` provider with a Copilot subscription token (`hermes",
    "      setup` → GitHub Copilot), or pick any other provider.",
)

_MINIMAX_ANTHROPIC_URLS = ("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic")


@dataclass
class OverflowVerdict:
    """Outcome of ``recover_from_overflow``.

    ``action`` is one of ``"return"`` (end the turn with ``result``), ``"break"``
    (restart the API call — a ``_retry.restart_with_*`` flag is set), ``"continue"``
    (retry the call immediately) or ``"fallthrough"`` (not an overflow error, or
    overflow recovery declined — continue generic error handling). The remaining
    fields are the loop locals the handler may have rebound."""

    action: str
    result: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    active_system_prompt: Any
    conversation_history: Any
    approx_tokens: int
    compression_attempts: int
    provider_overflow_recovery_pending: bool
    is_context_length_error: bool


@dataclass(kw_only=True)
class _Recovery(OverflowVerdict):
    """Working state shared by the overflow sub-handlers — the verdict itself, plus the
    read-only call context; handlers mutate the loop-local fields and ``done()`` stamps
    the action."""

    agent: Any
    api_messages: Any
    system_message: Any
    effective_task_id: Any
    api_call_count: int
    max_compression_attempts: int
    action: str = "fallthrough"
    result: Optional[Dict[str, Any]] = None
    provider_overflow_recovery_pending: bool = False
    is_context_length_error: bool = False

    def done(self, action: str, result: Optional[Dict[str, Any]] = None) -> OverflowVerdict:
        self.action, self.result = action, result
        return self

    def fail_turn(
        self, final_response: str, *, notices: tuple = (), log: Optional[tuple] = None,
        compression_exhausted: bool = True, **extra: Any,
    ) -> OverflowVerdict:
        """End the turn as failed/partial. ``notices`` flush the buffered retry trace
        first so the user sees what compression attempts were made."""
        agent = self.agent
        if notices:
            agent._flush_status_buffer()
            for line in notices:
                agent._vprint(f"{agent.log_prefix}{line}", force=True)
        if log:
            logger.error(*log)
        agent._persist_session(self.messages, self.conversation_history)
        result = {
            "final_response": final_response,
            "messages": self.messages,
            "completed": False,
            "api_calls": self.api_call_count,
            "error": final_response,
            "partial": True,
            "failed": True,
        }
        if compression_exhausted:
            # Reuse the gateway's existing context-recovery contract (#98722, salvaged from #98741). The
            # bloated transcript remains intact while future input can move to a clean session instead of
            # replaying the summarize-timeout loop.
            result["compression_exhausted"] = True
        result.update(extra)
        return self.done("return", result)

    def count_attempt(self, *, payload_too_large: bool = False) -> Optional[OverflowVerdict]:
        """Bump ``compression_attempts``; the terminal verdict once the cap is exceeded."""
        self.compression_attempts += 1
        cap = self.max_compression_attempts
        if self.compression_attempts <= cap:
            return None
        if payload_too_large:
            return self.fail_turn(
                f"Request payload too large: max compression attempts ({cap}) reached.",
                notices=(
                    f"❌ Max compression attempts ({cap}) reached for payload-too-large error.",
                    _RETRY_HINT,
                ),
                log=("%s413 compression failed after %d attempts.", self.agent.log_prefix, cap),
            )
        return self.fail_turn(
            f"Context length exceeded: max compression attempts ({cap}) reached.",
            notices=(f"❌ Max compression attempts ({cap}) reached.", _RETRY_HINT),
            log=("%sContext compression failed after %d attempts.", self.agent.log_prefix, cap),
        )

    def compress(self, request_tokens: int, *, fail_on_timeout: bool = False) -> Optional[OverflowVerdict]:
        """One compression pass with the summary-failure cooldown bypassed (the
        provider proved the request doesn't fit). Returns ``None`` when history was
        compressed, or a soft-defer verdict when another path holds the compression
        lock or a timed guard no-oped the pass: the attempt is refunded and the turn
        ends WITHOUT ``compression_exhausted`` so the gateway does not auto-reset. With
        ``fail_on_timeout`` a host timeout (recovery spent its wait budget with no
        committed summary) ends the turn via the typed contract, since re-sending would
        hit the same overflow."""
        from agent.conversation_compression import conversation_history_after_compression
        from agent.conversation_loop import _COMPRESSION_TIMEOUT_FINAL_RESPONSE, _compression_deferred_result

        agent = self.agent
        before = self.messages
        self.messages, self.active_system_prompt = agent._compress_context(
            before, self.system_message, approx_tokens=request_tokens,
            task_id=self.effective_task_id, bypass_cooldown=True,
        )
        if self.messages is before:
            deferred = None
            if compression_skipped_due_to_lock(agent):
                deferred = _compression_deferred_result(agent, self.messages, self.api_call_count)
            elif compression_blocked_transiently(agent):
                deferred = _compression_deferred_result(
                    agent, self.messages, self.api_call_count, reason="transient_block",
                )
            if deferred is not None:
                self.compression_attempts -= 1
                agent._persist_session(self.messages, self.conversation_history)
                return self.done("return", deferred)
        if fail_on_timeout and context_compression_timed_out(agent):
            return self.fail_turn(
                _COMPRESSION_TIMEOUT_FINAL_RESPONSE, turn_exit_reason="context_compression_timeout"
            )
        self.conversation_history = conversation_history_after_compression(
            agent, self.messages, self.conversation_history
        )
        return None

    def compress_scored_by_tokens(
        self, request_tokens: int, *, fail_on_timeout: bool = False,
    ) -> Tuple[Optional[OverflowVerdict], bool, int]:
        """``compress`` scored in message count / tokens (context-overflow errors ARE
        token-budget errors). Same-message-count compression (tool-result pruning,
        in-place summarization) can shrink the request, so re-estimate rather than trust
        the array length. Returns ``(deferred_verdict, shrank, new_tokens)``."""
        from agent.model_metadata import estimate_messages_tokens_rough

        original_len = len(self.messages)
        original_tokens = estimate_messages_tokens_rough(self.messages)
        deferred = self.compress(request_tokens, fail_on_timeout=fail_on_timeout)
        if deferred is not None:
            return deferred, False, original_tokens
        messages = self.messages
        # Re-measure after compression. Same-message-count compression (tool-result pruning, in-place
        # summarization) can materially reduce request size without reducing the message array (#39550), and
        # — the image-dominated case — compaction's historical-media aging (#97160) can free megabytes of
        # base64 that the token estimate never counted. Bytes are the yardstick for a 413; tokens are kept
        # only for status display.
        # Re-estimate tokens after compression. Same-message-count compression (tool-result pruning,
        # in-place summarization) can materially reduce request size without reducing the message array.
        # (#39550)
        new_tokens = estimate_messages_tokens_rough(messages)
        shrank_tokens = new_tokens > 0 and new_tokens < original_tokens * 0.95
        if len(messages) < original_len:
            self.agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
        elif shrank_tokens:
            self.agent._buffer_status(COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=original_tokens, after=new_tokens))
        return None, len(messages) < original_len or shrank_tokens, new_tokens

    def request_tokens(self) -> int:
        """Overhead-aware request size (msgs + tools + system) so LCM forced-overflow
        recovery arms on the TRUE request, not the tool-blind message count."""
        from agent.model_metadata import estimate_request_tokens_rough

        return estimate_request_tokens_rough(self.api_messages, tools=self.agent.tools or None)


def _recover_payload_too_large(st: _Recovery, _retry: TurnRetryState) -> OverflowVerdict:
    """413: compress and retry. A 413 is a BYTE-size error, so progress is scored in
    payload bytes — never the token estimate, which is deliberately byte-blind to images
    and wedged sessions on "no progress"."""
    from agent.model_metadata import estimate_messages_tokens_rough

    agent = st.agent
    exhausted = st.count_attempt(payload_too_large=True)
    if exhausted is not None:
        return exhausted
    agent._buffer_status(
        f"⚠️  Request payload too large (413) — compression attempt "
        f"{st.compression_attempts}/{st.max_compression_attempts}..."
    )

    messages = st.messages
    original_len = len(messages)
    # A 413 is a BYTE-size error, so this branch scores progress in BYTES of the serialized messages payload
    # — exact and free — never the token estimate. The estimator prices every image at a flat per-image
    # token cost (see estimate_messages_tokens_rough) so screenshots don't trigger premature compaction;
    # that deliberate byte-blindness means compaction can free megabytes of base64 (real case: two vision
    # results = 96.6% of the request body but ~3.7% of the estimate) while the token delta stays under any
    # threshold. Token-scored progress here burned all attempts on "no progress" and wedged the session
    # permanently. (#88960 / #47339)
    original_bytes = serialized_messages_bytes(messages)
    deferred = st.compress(st.request_tokens())
    if deferred is not None:
        return deferred

    # Re-measure: same-count compression and media aging can shrink the request
    # without shrinking the array. Tokens only for status display.
    messages = st.messages
    st.approx_tokens = estimate_messages_tokens_rough(messages)
    new_bytes = serialized_messages_bytes(messages)
    if len(messages) < original_len or (new_bytes > 0 and new_bytes < original_bytes * 0.95):
        if len(messages) < original_len:
            agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
        else:
            agent._buffer_status(
                f"🗜️ Compressed {original_bytes:,} → {new_bytes:,} " f"payload bytes, retrying..."
            )
        time.sleep(2)  # Brief pause between compression retries
        _retry.restart_with_compressed_messages = True
        return st.done("break")

    if agent._try_strip_image_parts_from_tool_messages(st.api_messages, remember_model=False):
        agent._buffer_status(
            "📐 Compression could not reduce the request further — "
            "removed retained vision payloads and retrying..."
        )
        return st.done("continue")

    return st.fail_turn(
        "Request payload too large (413). Cannot compress further.",
        notices=("❌ Payload too large and cannot compress further.", _RETRY_HINT),
        log=("%s413 payload too large. Cannot compress further.", agent.log_prefix),
    )


def _clamp_output_cap(st: _Recovery, _retry: TurnRetryState, available_out: int, old_ctx: int) -> OverflowVerdict:
    """Output-cap error ("max_tokens too large": input fits but input + max_tokens >
    window). The provider's available_tokens is the authoritative bound; also estimate
    the real request shape (API-only content) and use the smaller minus a margin."""
    agent = st.agent
    request_input_estimate = st.request_tokens()
    local_available_out = old_ctx - request_input_estimate
    if local_available_out > 0:
        safe_out = max(1, min(available_out, local_available_out) - 64)
    else:
        # Local estimate can overshoot; fall back to the provider-reported budget.
        safe_out = max(1, available_out - 64)
    agent._ephemeral_max_output_tokens = safe_out
    agent._buffer_vprint(
        f"⚠️  Output cap too large for current prompt — retrying with max_tokens={safe_out:,} "
        f"(provider_available={available_out:,}, estimated_request_tokens={request_input_estimate:,}; "
        f"context_length unchanged at {old_ctx:,})"
    )
    # Still count against compression_attempts so a recurring error can't loop forever.
    exhausted = st.count_attempt()
    if exhausted is not None:
        return exhausted
    # Also compress history so the retry doesn't spin on max_tokens alone; dropping the
    # middle window makes the total fit. Compression must never turn an output-cap error
    # fatal — on error, fall through and retry on max_tokens alone.
    try:
        deferred, _shrank, _new_tokens = st.compress_scored_by_tokens(request_input_estimate)
        if deferred is not None:
            return deferred
    except Exception:
        logger.warning(
            "%sOutput-cap compression hit an error; retrying on max_tokens only.", agent.log_prefix
        )
    _retry.restart_with_compressed_messages = True
    return st.done("break")


def _adopt_provider_context_limit(st: _Recovery, error_msg: str, old_ctx: int) -> Optional[int]:
    """Shrink context_length only when the provider reports the real limit; else keep
    the window and compress. Guessed probe tiers can turn a configured 1M window into
    256K/128K/64K. Returns the provider-reported limit, or ``None``."""
    from agent.model_metadata import save_context_length

    agent = st.agent
    compressor = agent.context_compressor
    new_ctx = get_context_length_from_provider_error(error_msg, old_ctx)
    if new_ctx is not None:
        agent._buffer_vprint(f"Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})")
        compressor.update_model(
            model=agent.model, context_length=new_ctx, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )
        # Persist the provider-reported limit BEFORE compression/retry: rate limit,
        # missing usage, or restart must not lose confirmed metadata. Probe flags
        # remain a fallback if this write fails.
        save_context_length(agent.model, agent.base_url, new_ctx)
        # Probe flags only on the built-in compressor (plugin engines manage their
        # own); provider-sourced value, so safe to cache.
        if hasattr(compressor, "_context_probed"):
            compressor._context_probed = True
            compressor._context_probe_persistable = True
        agent._buffer_vprint(f"⚠️  Context length exceeded — using provider limit: {old_ctx:,} → {new_ctx:,} tokens")
        return new_ctx

    is_minimax_provider = (
        (getattr(agent, "provider", "") or "").lower() in {"minimax", "minimax-cn"}
        or (getattr(agent, "base_url", "") or "").rstrip("/").lower().startswith(_MINIMAX_ANTHROPIC_URLS)
    )
    if is_minimax_provider and "context window exceeds limit (" in error_msg:
        agent._buffer_vprint(
            f"Provider reported overflow amount only; "
            f"keeping context_length at {old_ctx:,} tokens and compressing."
        )
    else:
        agent._buffer_vprint(
            f"⚠️  Context length exceeded, but provider did not report a max context length; "
            f"keeping context_length at {old_ctx:,} tokens and compressing."
        )
    return None


def _recover_context_length(st: _Recovery, _retry: TurnRetryState, error_msg: str) -> OverflowVerdict:
    """Context-length error. Two shapes: "prompt too long" = INPUT overflows the window
    (shrink context_length + compress); "max_tokens too large" = input fits but
    input + max_tokens > window (shrink the OUTPUT cap only)."""
    agent = st.agent
    old_ctx = agent.context_compressor.context_length

    available_out = parse_available_output_tokens_from_error(error_msg)
    if available_out is not None:
        return _clamp_output_cap(st, _retry, available_out, old_ctx)

    # Output-cap error with unparseable budget: compression can't help (input already
    # fits) and would death-loop on the same 400. Fail fast.
    if is_output_cap_error(error_msg):
        return st.fail_turn(
            "max_tokens exceeds the provider's output cap for this model. "
            "Lower model.max_tokens in config.yaml.",
            notices=(
                "❌ The provider rejected the request because max_tokens exceeds its output cap for this model.",
                "   💡 Lower model.max_tokens in your config.yaml to at or below the model's max-output limit. "
                "(This is an output-cap error, not a context overflow — compression cannot fix it.)",
            ),
            log=(
                f"{agent.log_prefix}Output-cap error not routed into compression "
                f"(max_tokens over provider cap): {error_msg[:200]}",
            ),
            compression_exhausted=False,
        )

    new_ctx = _adopt_provider_context_limit(st, error_msg, old_ctx)

    exhausted = st.count_attempt()
    if exhausted is not None:
        return exhausted
    agent._buffer_status(COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(
        tokens=st.approx_tokens, attempt=st.compression_attempts, cap=st.max_compression_attempts,
    ))

    deferred, shrank, new_tokens = st.compress_scored_by_tokens(st.request_tokens(), fail_on_timeout=True)
    if deferred is not None:
        return deferred
    st.approx_tokens = new_tokens
    if shrank or (new_ctx and new_ctx < old_ctx):
        time.sleep(2)  # Brief pause between compression retries
        # Rebuild the full request and force normal preflight to honor it; message
        # count alone doesn't prove system/tool-inclusive pressure fell.
        st.provider_overflow_recovery_pending = True
        _retry.restart_with_compressed_messages = True
        return st.done("break")

    # Can't compress further and already at minimum tier.
    return st.fail_turn(
        f"Context length exceeded ({new_tokens:,} tokens). Cannot compress further.",
        notices=(
            "❌ Context length exceeded and cannot compress further.",
            "   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.",
        ),
        log=("%sContext length exceeded: %s tokens. Cannot compress further.", agent.log_prefix, f"{new_tokens:,}"),
    )


def recover_from_overflow(
    agent: Any, api_error: Exception, classified: Any, _retry: TurnRetryState, *,
    status_code: Optional[int], error_msg: str, wrapped_output_cap_budget: Optional[int],
    messages: List[Dict[str, Any]], api_messages: Any, system_message: Any,
    active_system_prompt: Any, conversation_history: Any, approx_tokens: int,
    compression_attempts: int, max_compression_attempts: int, api_call_count: int,
    effective_task_id: Any,
) -> OverflowVerdict:
    """413 payload-too-large and context-length recovery (compress + retry, output-cap
    clamp, provider-reported context limit, GitHub Models free-tier hint). Order is
    load-bearing: 413 is checked BEFORE the generic 4xx handler, and context-length
    errors (incl. relay-wrapped output-cap 429s) BEFORE non-retryable client errors.
    Compression progress is scored in payload BYTES for 413 (never the byte-blind token
    estimate) and in tokens/message count for context overflow."""
    st = _Recovery(
        agent=agent, api_messages=api_messages, system_message=system_message,
        effective_task_id=effective_task_id, api_call_count=api_call_count,
        max_compression_attempts=max_compression_attempts, messages=messages,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        approx_tokens=approx_tokens, compression_attempts=compression_attempts,
    )

    # GitHub Models free tier caps requests at 8K tokens, under the system prompt +
    # tool schema floor; compression can't help, so say so.
    if (
        status_code == 413
        and isinstance(agent.base_url, str)
        and base_url_host_matches(agent.base_url, "models.inference.ai.azure.com")
    ):
        for line in _GITHUB_MODELS_HINT:
            agent._vprint(f"{agent.log_prefix}{line}", force=True)

    if classified.reason == FailoverReason.payload_too_large:
        return _recover_payload_too_large(st, _retry)

    # Relay-wrapped output-cap 429s (parsed by the caller) go to the clamp, not
    # failover or generic retries. The classifier also covers 400/disconnect +
    # large-session heuristics.
    st.is_context_length_error = (
        # Check for context-length errors BEFORE generic 4xx handler. The classifier detects context
        # overflow from: explicit error messages, generic 400 + large session heuristic (#1630), and server
        # disconnect + large session pattern (#2153).
        classified.reason == FailoverReason.context_overflow
        or wrapped_output_cap_budget is not None
    )
    if st.is_context_length_error:
        return _recover_context_length(st, _retry, error_msg)
    return st.done("fallthrough")
