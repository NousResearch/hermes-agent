"""Usage and billing attribution for a finished claude-agent-sdk turn: canonical
token accounting, the subscription/metered cost status and the per-session
counters. Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional
from agent.claude_sdk_runtime_context import (
    _coerce_usage_int,
    _sdk_context_prompt_tokens,
    _sync_context_length_from_cli,
)

from agent.claude_sdk_runtime_state import _SdkTurnState

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


def _claude_sdk_billing_accounting(turn) -> tuple[str, str, str, Optional[float]]:
    """Return (mode, status, source, actual_cost) from child evidence.

    ``ResultMessage.total_cost_usd`` is a usage value, not proof of which lane
    paid it. It becomes an actual cost only when the CLI's init/rate-limit
    messages identified a metered lane. The default fail-closed session guard
    labels safe turns ``subscription_included``; an explicit metered opt-in
    with incomplete evidence stays unknown rather than being called included.
    """
    mode = str(getattr(turn, "billing_mode", None) or "unknown")
    actual_cost: Optional[float] = None
    if mode == "sdk_reported_metered":
        raw_cost = getattr(turn, "total_cost_usd", None)
        if not isinstance(raw_cost, bool):
            try:
                candidate = float(raw_cost)
            except (TypeError, ValueError):
                candidate = -1.0
            if candidate >= 0 and math.isfinite(candidate):
                actual_cost = candidate
        return (
            mode,
            "reported" if actual_cost is not None else "unknown",
            "claude-agent-sdk",
            actual_cost,
        )
    if mode == "subscription_included":
        return mode, "included", "claude-subscription", None
    return "unknown", "unknown", "claude-agent-sdk-unverified", None


def _record_claude_sdk_usage(agent, turn) -> dict[str, Any]:
    """Translate SDK ResultMessage usage into Hermes accounting.

    The SDK reports Anthropic-shaped usage: input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens. Billing labels come
    from the child evidence captured by ``ClaudeAgentSdkSession``; never call
    an explicitly allowed API-key/Extra-Usage turn subscription-included."""
    agent.session_api_calls += 1

    billing_mode, cost_status, cost_source, actual_cost = (
        _claude_sdk_billing_accounting(turn)
    )
    agent.session_cost_status = cost_status
    agent.session_cost_source = cost_source

    # Attribution: the configured model wins; when it is unset (documented
    # default — the CLI picks), back-fill from the model id the SDK itself
    # reported so usage rows stop reading model='unknown'.
    resolved_model = agent.model or getattr(turn, "model_last", None) or ""

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=resolved_model,
                    actual_cost_usd=actual_cost,
                    cost_status=cost_status,
                    cost_source=cost_source,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode=billing_mode,
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude-sdk api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        if billing_mode == "subscription_included":
            return {}
        return {
            "estimated_cost_usd": None,
            "actual_cost_usd": actual_cost,
            "cost_status": cost_status,
            "cost_source": cost_source,
        }

    from agent.usage_pricing import CanonicalUsage

    input_tokens = _coerce_usage_int(usage.get("input_tokens"))
    output_tokens = _coerce_usage_int(usage.get("output_tokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cache_read_input_tokens"))
    cache_write_tokens = _coerce_usage_int(
        usage.get("cache_creation_input_tokens")
    )

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    # Context pressure != tokens billed. See _sdk_context_prompt_tokens.
    context_prompt_tokens = _sdk_context_prompt_tokens(usage, prompt_tokens)
    if context_prompt_tokens != prompt_tokens:
        logger.debug(
            "claude-sdk context tokens: using last-iteration %d (turn aggregate "
            "was %d) for context pressure; aggregate retained for billing.",
            context_prompt_tokens, prompt_tokens,
        )

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        # BEFORE update_from_response, which compares last_prompt_tokens against
        # threshold_tokens: the setter invalidates the derived budgets, so
        # syncing first means even the first turn's compression decision is
        # taken against the CLI's real window rather than the metadata one.
        _sync_context_length_from_cli(agent)
        try:
            compressor_usage = dict(usage_dict)
            compressor_usage["prompt_tokens"] = context_prompt_tokens
            compressor_usage["total_tokens"] = (
                context_prompt_tokens + completion_tokens
            )
            compressor.update_from_response(compressor_usage)
        except Exception:
            logger.debug("claude-sdk usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=0,
                actual_cost_usd=actual_cost,
                cost_status=cost_status,
                cost_source=cost_source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode=billing_mode,
                model=resolved_model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude-sdk token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        # Context-pressure value: drives the runtime footer's context_pct and
        # gateway session hygiene. usage_dict still carries the BILLING
        # aggregate under "prompt_tokens".
        "last_prompt_tokens": context_prompt_tokens,
        "estimated_cost_usd": None,
        "actual_cost_usd": actual_cost,
        "cost_status": cost_status,
        "cost_source": cost_source,
    }


def _account_turn(agent, state: _SdkTurnState) -> None:
    """Post-turn accounting: the skill-nudge counter, usage attribution, the
    skill-review cadence and the external memory sync."""
    turn = state.turn
    # Counter ticks — _turns_since_memory/_user_turn_count are incremented by
    # run_conversation()'s pre-loop block; only _iters_since_skill is ours.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    state.usage_result = (
        _record_claude_sdk_usage(agent, turn)
        if getattr(turn, "api_call_made", True)
        else {}
    )

    state.should_review_skills = False
    # Skill-review cadence belongs to review policy, not foreground tool
    # availability. If routed, a distinct normal runtime owns optional writes.
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
    ):
        state.should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=state.original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=state.messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)
