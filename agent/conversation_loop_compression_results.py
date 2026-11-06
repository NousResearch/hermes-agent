"""Typed compression recovery results shared by the conversation loop and its phase siblings."""

from __future__ import annotations

import logging
from typing import Any

from agent.turn_failure_copy import site_copy

logger = logging.getLogger("agent.conversation_loop")


def _compression_deferred_result(
    agent, messages: list[dict], api_call_count: int, reason: str = "lock", *,
    provider_rejected: bool = False, request_tokens: int | None = None,
) -> dict[str, Any]:
    """Soft turn result for a transiently-deferred compression. Both reasons must end as
    ``compression_deferred``, never ``compression_exhausted`` — the gateway wipes the
    session on exhaustion (#9893/#35809). ``failed`` stays False; the turn persists."""
    from agent.conversation_loop import _partial_turn_result

    if provider_rejected and reason == "transient_block":
        structural = _uncompressible_request_result(agent, messages, api_call_count, request_tokens)
        if structural is not None:
            return structural
    session = agent.session_id or "none"
    if reason == "transient_block":
        block = getattr(agent, "_compression_blocked_transient", None)
        logger.info(
            "turn deferred: compression transiently blocked (%s) (session=%s) — not counting as "
            "compression exhaustion", block if isinstance(block, str) else "unknown guard", session,
        )
        _final = (
            "Context compression is paused. Retry after the pause ends, or run /compress to request "
            "a fresh attempt. Your conversation has been preserved."
        )
    else:
        holder = getattr(agent, "_compression_skipped_due_to_lock", None)
        logger.info(
            "turn deferred: compression lock held by another path (session=%s holder=%s) — not "
            "counting as compression exhaustion", session, holder if isinstance(holder, str) else "unconfirmed",
        )
        _final = (
            "Context compression is already running for this session. Please retry in a moment — "
            "your next message will be processed once the concurrent compression finishes."
        )
    try:
        agent._flush_status_buffer()
    except Exception:
        pass
    return _partial_turn_result(
        _final, messages, api_call_count,
        failed=False, compression_deferred=True, session_id=agent.session_id,
    )


def _uncompressible_request_result(
    agent, messages: list[dict], api_call_count: int, request_tokens: int | None, *,
    provider_rejected: bool = True,
) -> dict[str, Any] | None:
    """For a proven input rejection or typed definite-budget refusal, recheck structural eligibility."""
    from agent.conversation_loop import _partial_turn_result

    if not _no_current_compaction_window(agent, messages, request_tokens):
        return None
    # Exhaustion rotates gateway sessions. An impossible current request must leave its history intact.
    limit = "provider's context or payload limit" if provider_rejected else "configured model context window"
    return _partial_turn_result(
        f"This request exceeds the {limit}, and there is no eligible earlier "
        "history to summarize. Shorten the current message or attachments, or use a larger context. "
        "Your conversation has been preserved.",
        messages, api_call_count, failed=False, compression_deferred=True, session_id=agent.session_id,
        failure_reason="request_uncompressible", failure_retryable=False,
        turn_exit_reason="request_no_compressible_history",
    )


def _no_current_compaction_window(agent, messages: list[dict], request_tokens: int | None) -> bool:
    """Recheck the actual default engine's current bounds, never a stale backoff or display string."""
    from agent.context_compressor import ContextCompressor

    compressor = getattr(agent, "context_compressor", None)
    if not isinstance(compressor, ContextCompressor):
        return False
    bounds = compressor._window_bounds(compressor._drop_blank_echoes(messages), request_tokens)
    return bounds.start >= bounds.end


def _provider_overflow_exhausted_result(
    agent, messages: list[dict], conversation_history, api_call_count: int,
    request_pressure_tokens: int, max_compression_attempts: int,
) -> dict[str, Any]:
    """Fail closed when a rebuilt request is still too large after recovery."""
    from agent.conversation_loop import _partial_turn_result

    structural = _uncompressible_request_result(agent, messages, api_call_count, request_pressure_tokens)
    if structural is not None:
        agent._persist_session(messages, conversation_history)
        return structural
    agent._flush_status_buffer()
    logger.error(
        "%sContext compression failed after %d attempts; rebuilt request "
        "remains over threshold at ~%s tokens.",
        agent.log_prefix, max_compression_attempts, f"{request_pressure_tokens:,}",
    )
    # Host progress-aware timeout (#98722, salvaged from #98741): the provider proved the request does not
    # fit, but this recovery pass spent the full wait budget without a committed summary. Re-sending the
    # unchanged request would bounce off the same overflow error and re-enter compression in the same turn.
    # End the turn with the typed recovery contract instead — transcript intact, no further doomed provider
    # sends.
    # Prior <3 retries (or an earlier successful tool batch) leave a tool-result tail. Closing it here
    # matches interrupt aborts (#48879 / #52592) so the next user turn is not tool→user for strict
    # providers.
    agent._persist_session(messages, conversation_history)
    return _partial_turn_result(
        site_copy("context_overflow", model=agent.model),
        messages, api_call_count, failed=True, compression_exhausted=True,
        turn_exit_reason="context_compression_exhausted",
        failure_reason="context_overflow", failure_retryable=False,
    )


def _preflight_refusal_result(agent, exc, conversation_history) -> dict[str, Any]:
    """Resolve typed turn-start refusal provenance without mistaking no progress for elapsed time."""
    from agent.conversation_loop import _partial_turn_result
    from agent.turn_context import PreflightCompressionInsufficientProgress

    if isinstance(exc, PreflightCompressionInsufficientProgress) and exc.messages is not None:
        structural = _uncompressible_request_result(
            agent, exc.messages, 0, exc.request_tokens, provider_rejected=False,
        )
        if structural is not None:
            agent._persist_session(exc.messages, exc.conversation_history)
            return structural
    logger.warning(
        "Turn-start preflight compression could not complete — ending turn with typed recovery result: %s", exc,
    )
    # Clear the tripwire slot note_turn_start registered (the early return skips the persist
    # funnel). The user row is deliberately NOT persisted (#7100).
    from agent.agent_runtime_helpers import note_turn_persisted
    note_turn_persisted(agent)
    # Not _COMPRESSION_TIMEOUT_FINAL_RESPONSE — that describes a different state
    # (compression ran, could not reduce); the exception text carries the guidance.
    return _partial_turn_result(
        str(exc), list(conversation_history or []), 0,
        failed=True, compression_exhausted=True,
        turn_exit_reason=("context_compression_no_progress" if isinstance(exc, PreflightCompressionInsufficientProgress)
                          else "context_compression_timeout"),
        failure_reason="context_overflow", failure_retryable=False,
    )

