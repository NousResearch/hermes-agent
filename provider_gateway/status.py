"""Status helpers for the opt-in provider gateway."""

from __future__ import annotations

from typing import Any

from provider_gateway.config import GatewayConfig
from provider_gateway.policy import ProviderRouteCandidate, build_gateway_policy


def build_gateway_status(agent: Any) -> dict[str, Any]:
    """Build a compact provider gateway status snapshot for CLI/status surfaces."""
    config = getattr(agent, "_provider_gateway_config", None)
    if not isinstance(config, GatewayConfig):
        config = GatewayConfig()

    policy = build_gateway_policy(agent, config)
    tracker = getattr(agent, "_provider_usage_tracker", None)
    usage_summary = []
    if tracker is not None and hasattr(tracker, "summarize_by_provider"):
        try:
            usage_summary = tracker.summarize_by_provider()
        except Exception:
            usage_summary = []

    last_candidate = getattr(agent, "_provider_gateway_last_route_candidate", None)

    return {
        "enabled": config.enabled,
        "backend": config.backend,
        "routing_strategy": config.routing_strategy,
        "track_usage": config.track_usage,
        "track_cost": config.track_cost,
        "candidate_count": len(policy.candidates),
        "last_observed_candidate": _candidate_to_dict(last_candidate),
        "usage_summary": usage_summary,
    }


def format_gateway_status_lines(status: dict[str, Any]) -> list[str]:
    """Format a status snapshot as compact terminal lines."""
    usage_summary = status.get("usage_summary") or []
    if not status.get("enabled") and not usage_summary:
        return []

    enabled_label = "enabled" if status.get("enabled") else "disabled"
    lines = [
        (
            "Provider Gateway: "
            f"{enabled_label} "
            f"(backend={status.get('backend', 'native')}, "
            f"strategy={status.get('routing_strategy', 'round-robin')}, "
            f"candidates={int(status.get('candidate_count') or 0)})"
        ),
        (
            "Tracking: "
            f"usage={_on_off(status.get('track_usage'))}, "
            f"cost={_on_off(status.get('track_cost'))}"
        ),
    ]

    candidate = status.get("last_observed_candidate")
    if isinstance(candidate, dict) and candidate.get("provider") and candidate.get("model"):
        lines.append(
            "Next observed: "
            f"{candidate['provider']}/{candidate['model']} "
            f"via {candidate.get('source') or 'unknown'}"
        )

    for row in usage_summary:
        if not isinstance(row, dict):
            continue
        provider = row.get("provider") or "unknown"
        request_count = int(row.get("request_count") or 0)
        success_count = int(row.get("success_count") or 0)
        error_count = int(row.get("error_count") or 0)
        total_tokens = int(row.get("total_tokens") or 0)
        cost = float(row.get("estimated_cost_usd") or 0.0)
        avg_latency = float(row.get("avg_latency_ms") or 0.0)
        lines.append(
            f"Usage {provider}: "
            f"{request_count:,} req, "
            f"{success_count:,} ok, "
            f"{error_count:,} error, "
            f"{total_tokens:,} tokens, "
            f"${cost:.4f}, "
            f"avg {avg_latency:.1f}ms"
        )

    return lines


def _candidate_to_dict(candidate: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, ProviderRouteCandidate):
        return None
    return {
        "provider": candidate.provider,
        "model": candidate.model,
        "source": candidate.source,
        "base_url": candidate.base_url,
    }


def _on_off(value: Any) -> str:
    return "on" if bool(value) else "off"
