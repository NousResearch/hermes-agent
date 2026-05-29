"""Runtime hooks for the opt-in provider gateway."""

from __future__ import annotations

import logging
from typing import Any

from agent.usage_pricing import estimate_usage_cost, normalize_usage
from provider_gateway.config import GatewayConfig, load_gateway_config
from provider_gateway.policy import (
    ProviderRouteCandidate,
    build_gateway_policy,
    should_consider_gateway_fallback,
)
from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker

logger = logging.getLogger(__name__)


def record_provider_response_usage(
    agent: Any,
    response: Any,
    *,
    latency_seconds: float,
) -> bool:
    """Record a successful OpenAI-compatible provider response if enabled."""
    config = _get_gateway_config(agent)
    if not _should_track_usage(agent, config):
        return False

    raw_usage = getattr(response, "usage", None)
    usage = normalize_usage(
        raw_usage,
        provider=getattr(agent, "provider", None),
        api_mode=getattr(agent, "api_mode", None),
    )
    estimated_cost = _estimate_cost_usd(agent, usage, config)

    return _record_usage(
        agent,
        ProviderUsageRecord(
            provider=_agent_str(agent, "provider", "unknown"),
            model=_agent_str(agent, "model", "unknown"),
            api_mode=_agent_str(agent, "api_mode", "chat_completions"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            estimated_cost_usd=estimated_cost,
            latency_ms=round(max(0.0, latency_seconds) * 1000.0, 2),
            status="success",
            session_id=getattr(agent, "session_id", None),
        ),
    )


def record_provider_error_usage(
    agent: Any,
    error: BaseException,
    *,
    latency_seconds: float,
) -> bool:
    """Record an OpenAI-compatible provider request error if enabled."""
    config = _get_gateway_config(agent)
    if not _should_track_usage(agent, config):
        return False

    return _record_usage(
        agent,
        ProviderUsageRecord(
            provider=_agent_str(agent, "provider", "unknown"),
            model=_agent_str(agent, "model", "unknown"),
            api_mode=_agent_str(agent, "api_mode", "chat_completions"),
            latency_ms=round(max(0.0, latency_seconds) * 1000.0, 2),
            status="error",
            session_id=getattr(agent, "session_id", None),
            error_type=type(error).__name__,
        ),
    )


def observe_gateway_route_selection(
    agent: Any,
    reason: Any,
) -> ProviderRouteCandidate | None:
    """Record the next gateway route candidate without mutating runtime routing."""
    config = _get_gateway_config(agent)
    if not config.enabled or not should_consider_gateway_fallback(reason):
        return None

    policy = build_gateway_policy(agent, config)
    candidate = policy.next_after(
        _agent_str(agent, "provider", ""),
        _agent_str(agent, "model", ""),
        base_url=getattr(agent, "base_url", None),
    )
    if candidate is None:
        return None

    try:
        setattr(agent, "_provider_gateway_last_route_candidate", candidate)
    except Exception:
        pass
    logger.debug(
        "Provider gateway observed next route candidate: provider=%s model=%s source=%s",
        candidate.provider,
        candidate.model,
        candidate.source,
    )
    return candidate


def _get_gateway_config(agent: Any) -> GatewayConfig:
    config = getattr(agent, "_provider_gateway_config", None)
    if isinstance(config, GatewayConfig):
        return config

    try:
        config = load_gateway_config()
    except Exception as exc:  # pragma: no cover - defensive config fallback
        logger.debug("Provider gateway config load failed: %s", exc)
        config = GatewayConfig()

    try:
        setattr(agent, "_provider_gateway_config", config)
    except Exception:
        pass
    return config


def _should_track_usage(agent: Any, config: GatewayConfig) -> bool:
    return (
        config.enabled
        and config.track_usage
        and _agent_str(agent, "api_mode", "") == "chat_completions"
    )


def _get_usage_tracker(agent: Any) -> Any:
    tracker = getattr(agent, "_provider_usage_tracker", None)
    if tracker is not None:
        return tracker

    tracker = ProviderUsageTracker()
    try:
        setattr(agent, "_provider_usage_tracker", tracker)
    except Exception:
        pass
    return tracker


def _record_usage(agent: Any, record: ProviderUsageRecord) -> bool:
    try:
        _get_usage_tracker(agent).record_usage(record)
        return True
    except Exception as exc:  # pragma: no cover - tracking must not break chat
        logger.debug("Provider gateway usage tracking failed: %s", exc)
        return False


def _estimate_cost_usd(agent: Any, usage: Any, config: GatewayConfig) -> float:
    if not config.track_cost:
        return 0.0
    try:
        result = estimate_usage_cost(
            _agent_str(agent, "model", ""),
            usage,
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
            api_key=getattr(agent, "api_key", ""),
        )
    except Exception as exc:  # pragma: no cover - cost failure should be silent
        logger.debug("Provider gateway cost estimate failed: %s", exc)
        return 0.0

    if result.amount_usd is None:
        return 0.0
    return float(result.amount_usd)


def _agent_str(agent: Any, name: str, default: str) -> str:
    value = getattr(agent, name, default)
    if value is None:
        return default
    text = str(value).strip()
    return text or default
