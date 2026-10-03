"""Per-response usage accounting for the conversation turn loop.

After every successful model API call, ``record_response_usage`` folds ``response.usage``
into: the context compressor (``update_from_response`` + the compression-budget rearm
latch), the usage anchor for display/compression math, per-session token/cost counters,
the state.db token-delta queue, and the observability log line. MoA sessions additionally
fold advisor fan-out usage into the reported counts and price the aggregator at its REAL
model/provider. Logger name stays ``agent.conversation_loop`` for caplog parity.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.image_token_cost import calibrate_from_usage
from agent.usage_anchor import capture_usage_anchor, set_usage_anchor
from agent.usage_pricing import estimate_usage_cost, normalize_usage, with_served_service_tier

logger = logging.getLogger("agent.conversation_loop")


def _agent_session_source(agent: Any) -> str:
    """The surface the agent's own row create would stamp (``_ensure_db_session``), so an
    accounting guard that wins the row-creation race never mints an anonymous session."""
    from run_agent import _session_source_for_agent  # late: run_agent imports this module
    return _session_source_for_agent(getattr(agent, "platform", None))


@dataclass
class ResponseUsageOutcome:
    """``compression_attempts`` is the (possibly rearmed-to-zero) budget counter;
    ``rearmed`` tells the loop to also clear its preflight-block latch."""

    compression_attempts: int
    rearmed: bool = False


def _loop_mod():
    """Lazy ``agent.conversation_loop`` import (avoids an import cycle)."""
    import agent.conversation_loop as _cl

    return _cl


def decode_duration_for_tps(api_duration: float, api_start_time=None, first_chunk_at=None) -> float:
    """Decode-phase duration for throughput math: wall clock minus time-to-first-token.

    A cold local model spends most of a call loading (queue + prefill) before the
    first token streams; dividing output tokens by the full wall clock then reports
    the load wait as generation slowness. The first stream chunk timestamp
    (``agent._last_api_first_chunk_at``) marks where generation actually started,
    so throughput divides by ``api_duration - ttft`` instead. Falls back to the
    full ``api_duration`` whenever the chunk timing is missing or implausible
    (non-streamed call, stale timestamp from another attempt, clock skew).
    """
    try:
        duration = float(api_duration)
    except (TypeError, ValueError):
        return api_duration
    try:
        if api_start_time is None or first_chunk_at is None:
            return duration
        ttft = float(first_chunk_at) - float(api_start_time)
    except (TypeError, ValueError):
        return duration
    if not (ttft == ttft and 0 <= ttft < duration):  # NaN / negative / beyond the call
        return duration
    decode = duration - ttft
    return decode if decode > 0 else duration


def reset_throughput_history(agent: Any) -> None:
    """Clear the rolling histories behind ``avg_tps``/``avg_latency_s`` (latency, decode
    phase, output — appended together per call).

    The next completed call re-seeds them, so the status-bar readout restarts from
    current conditions instead of dragging a stale (e.g. cold-start) average.
    """
    with suppress(Exception):
        hist = getattr(agent, "_api_latency_history", None)
        if hist is not None:
            hist.clear()
    with suppress(Exception):
        dhist = getattr(agent, "_api_decode_duration_history", None)
        if dhist is not None:
            dhist.clear()
    with suppress(Exception):
        ohist = getattr(agent, "_api_output_history", None)
        if ohist is not None:
            ohist.clear()


def _fold_moa_usage(agent, canonical_usage):
    """MoA: fold advisor fan-out usage into REPORTED token counts (only aggregator usage is
    returned, so advisor spend would be invisible) and flush the full-turn trace when
    ``moa.save_traces`` is on. Returns ``(client, canonical_usage, advisor_cost)``."""
    _moa_ref_cost = None
    _moa_client = getattr(agent, "client", None)
    if _moa_client is not None and hasattr(_moa_client, "consume_reference_usage"):
        try:
            _ref_usage, _moa_ref_cost = _moa_client.consume_reference_usage()
            if _ref_usage is not None:
                canonical_usage = canonical_usage + _ref_usage
        except Exception as _moa_acct_exc:  # pragma: no cover - defensive
            logger.debug("MoA reference usage accounting failed: %s", _moa_acct_exc)
    if _moa_client is not None and hasattr(_moa_client, "consume_and_save_trace"):
        try:
            # Streaming path: pass the streamed acting text so the trace is self-contained.
            _agg_streamed_text = getattr(agent, "_current_streamed_assistant_text", "") or ""
            _moa_client.consume_and_save_trace(
                agent.session_id, aggregator_output_fallback=_agg_streamed_text or None
            )
        except Exception as _moa_trace_exc:  # pragma: no cover - defensive
            logger.debug("MoA trace flush failed: %s", _moa_trace_exc)
    return _moa_client, canonical_usage, _moa_ref_cost


def record_response_usage(
    agent: Any, response: Any, *, messages: List[Dict[str, Any]], api_call_count: int,
    api_duration: float, compression_attempts: int, max_compression_attempts: int,
    api_start_time: float | None = None, first_chunk_at: float | None = None,
) -> ResponseUsageOutcome:
    """Fold ``response.usage`` into compressor, anchors, session counters, state.db
    and the API-call log line (see module docstring). No-usage responses only
    consume a pending compaction verdict. Returns the loop-visible outcome.

    ``api_start_time`` + ``first_chunk_at`` (default: the attempt's
    ``agent._last_api_first_chunk_at``) feed the rolling throughput history with
    the decode phase only, so model-load / prefill wait doesn't read as
    generation slowness.
    """
    rearmed = False
    compressor = agent.context_compressor
    # Count every completed provider attempt, including providers that omit usage.
    # Token/cost accounting below stays gated on real usage, but the request itself
    # must remain observable.
    agent.session_api_calls += 1
    if not (hasattr(response, 'usage') and response.usage):
        if getattr(compressor, "awaiting_real_usage_after_compression", False):
            # No usage -> cannot adjudicate the prior compaction; consume the
            # pending verdict so later readings aren't charged to it and
            # preflight deferral isn't latched indefinitely.
            compressor.update_from_response({})
        _note_usage_less = getattr(compressor, "note_usage_less_response", None)
        if callable(_note_usage_less):
            _note_usage_less()
        logger.info(
            "API call #%d: model=%s provider=%s in=? out=? total=? latency=%.1fs usage=unavailable",
            agent.session_api_calls, agent.model, agent.provider or "unknown", api_duration,
        )
        return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)

    canonical_usage = with_served_service_tier(
        normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode), response)
    # Aggregator-only usage kept for pricing: advisor tokens are priced at each advisor's
    # OWN model rate and added as dollars below.
    aggregator_usage = canonical_usage
    _moa_client, canonical_usage, _moa_ref_cost = _fold_moa_usage(agent, canonical_usage)
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    # Canonical token + cache buckets for context engines; legacy keys stay for back-compat.
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }
    # Capture the boundary latch before update_from_response() consumes it: only the real
    # prompt count right after a compaction rearms the budget.
    _completed_compaction_pending = bool(
        getattr(compressor, "_verify_compaction_cleared_threshold", False)
    )
    compressor.update_from_response(usage_dict)
    # Usage-anchored accounting: snapshot exact provider usage against the durable
    # transcript (main-loop ONLY; MoA uses pre-fold aggregator usage). The display meter
    # anchors on the turn's FIRST response: later same-turn responses inflate
    # prompt_tokens with replayed thinking. Display-only; compression math uses real usage.
    # The provider just priced this request exactly: if the delta since the previous anchor
    # introduced images, the residual is their real per-image cost (learned before re-anchoring).
    calibrate_from_usage(agent, messages, aggregator_usage.prompt_tokens)
    _new_anchor = capture_usage_anchor(
        aggregator_usage.prompt_tokens, aggregator_usage.output_tokens, messages
    )
    if _new_anchor is not None:
        set_usage_anchor(agent, _new_anchor, turn_base=api_call_count == 1)
    _compression_threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    if _loop_mod()._should_rearm_compression_budget(
        compression_attempts, completed_compaction_pending=_completed_compaction_pending,
        prompt_tokens=prompt_tokens, threshold_tokens=_compression_threshold,
    ):
        logger.info(
            "Compression budget rearmed after provider-confirmed "
            "recovery: prompt=%s < threshold=%s (attempts were %s/%s)",
            f"{prompt_tokens:,}",
            f"{_compression_threshold:,}",
            compression_attempts,
            max_compression_attempts,
        )
        compression_attempts = 0
        # Confirmed recovery also clears the loop's stale insufficient-progress verdict
        # (``_preflight_compression_blocked``), else a later pressure spike grows unchecked.
        rearmed = True

    # Stash canonical usage for on_turn_complete(); keep the latest call's.
    agent._last_turn_usage = dict(usage_dict)
    # The parent's CURRENT prompt size for headroom math (delegate summary budgets): the
    # aggregator's own prompt, never the MoA-folded total (advisor prompts are not in this context).
    agent._last_prompt_size_tokens = int(aggregator_usage.prompt_tokens or 0)

    # Persist only provider-confirmed context lengths, not probe tiers.
    if getattr(compressor, "_context_probed", False):
        ctx = compressor.context_length
        if getattr(compressor, "_context_probe_persistable", False):
            from agent.model_metadata import save_provider_context_length

            save_provider_context_length(agent.model, agent.base_url, ctx, agent.provider)
            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
        compressor._context_probed = False
        compressor._context_probe_persistable = False

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens
    # Rolling history for status-bar averages (last 10). Latency keeps the full wall
    # clock; throughput divides by the decode phase (first token → completion), so a
    # cold local model's load/prefill wait doesn't read as generation slowness.
    with suppress(Exception):
        if first_chunk_at is None:
            first_chunk_at = getattr(agent, "_last_api_first_chunk_at", None)
        _tps_duration = decode_duration_for_tps(api_duration, api_start_time, first_chunk_at)
        hist = getattr(agent, "_api_latency_history", None)
        if hist is not None:
            hist.append(float(api_duration))
        dhist = getattr(agent, "_api_decode_duration_history", None)
        if dhist is not None:
            dhist.append(float(_tps_duration))
        ohist = getattr(agent, "_api_output_history", None)
        if ohist is not None:
            ohist.append(int(canonical_usage.output_tokens or 0))

    _cache_pct = ""
    if canonical_usage.cache_read_tokens and prompt_tokens:
        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
    # write= is the money (cache writes cost 50x a read); id= is what a provider needs to look the
    # request up; upstream= is who actually served it when the route reports that (OpenRouter's
    # `provider`). Diagnosing the 1,393-agent run's cache misses took a DB join and a live probe
    # because none of the three were on this line.
    if canonical_usage.cache_write_tokens:
        _cache_pct += f" write={canonical_usage.cache_write_tokens}"
    _rid = getattr(response, "id", None)
    _ident = f" id={_rid}" if isinstance(_rid, str) and _rid else ""
    _upstream = getattr(response, "provider", None)
    if isinstance(_upstream, str) and _upstream:
        _ident += f" upstream={_upstream}"
    logger.info(
        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s%s",
        agent.session_api_calls, agent.model, agent.provider or "unknown",
        prompt_tokens, completion_tokens, total_tokens,
        api_duration, _cache_pct, _ident,
    )
    # nous.anthropic_wire=auto: the session's wire is decided once, from this first response.
    if agent.session_api_calls == 1 and (agent.provider or "") == "nous":
        with suppress(Exception):
            from agent.nous_wire import maybe_switch_wire_after_first_response
            maybe_switch_wire_after_first_response(agent, response, agent.session_api_calls)

    # MoA: agent.model/provider are the virtual preset/"moa" with no pricing entry, silently
    # dropping aggregator spend. Price at the REAL model/provider from the aggregator slot.
    _agg_cost_model, _agg_cost_provider, _agg_cost_base_url = agent.model, agent.provider, agent.base_url
    _agg_slot = getattr(_moa_client, "last_aggregator_slot", None) if _moa_client is not None else None
    if _agg_slot and _agg_slot.get("model"):
        _agg_cost_model = _agg_slot["model"]
        _agg_cost_provider = _agg_slot.get("provider") or agent.provider
        _agg_cost_base_url = _agg_slot.get("base_url") or agent.base_url
    cost_result = estimate_usage_cost(
        _agg_cost_model, aggregator_usage, provider=_agg_cost_provider,
        base_url=_agg_cost_base_url, api_key=getattr(agent, "api_key", ""),
    )
    # Cost delta = aggregator + MoA advisor cost (already priced per-advisor at each
    # advisor's own model rate), so state.db's estimated_cost_usd matches the folded
    # token counts.
    _cost_delta = None
    if cost_result.amount_usd is not None:
        _cost_delta = float(cost_result.amount_usd)
        agent.session_estimated_cost_usd += _cost_delta
    if _moa_ref_cost is not None:
        try:
            _moa_cost = float(_moa_ref_cost)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            _moa_cost = None
        if _moa_cost is not None:
            agent.session_estimated_cost_usd += _moa_cost
            _cost_delta = (_cost_delta or 0.0) + _moa_cost
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    # Persist per-call token deltas for any session_id so non-CLI runs can't lose
    # accounting; gateway/session-store writes use absolute totals and safely overwrite
    # these deltas. Enqueued, not written (a cold state.db UPDATE here stalled the tool
    # loop); drained at finalize via _persist_session.
    if agent._session_db and agent.session_id:
        try:
            # Ensure the row exists: under concurrent SQLite load the initial
            # _ensure_db_session() may fail, and UPDATE on a missing row affects 0 rows.
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.queue_token_counts(
                agent.session_id,
                source=_agent_session_source(agent),
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=_cost_delta,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as e:  # silent loss here undercounts analytics
            logger.debug(
                "Token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, e,
            )

    if agent.verbose_logging:
        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

    # Report cache stats for any provider that returns ``prompt_tokens_details.cached_tokens``,
    # not only when we inject cache_control markers.
    cached = canonical_usage.cache_read_tokens
    written = canonical_usage.cache_write_tokens
    prompt = usage_dict["prompt_tokens"]
    if (cached or written) and not agent.quiet_mode:
        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
        agent._vprint(
            f"{agent.log_prefix}   💾 Cache: "
            f"{cached:,}/{prompt:,} tokens "
            f"({hit_pct:.0f}% hit, {written:,} written)"
        )
    return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)
