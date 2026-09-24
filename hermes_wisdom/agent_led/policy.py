"""Policy values for agent-led sharing.

Organization limits are authoritative. Missing or invalid server policy defers
proactive work; local defaults never bypass an unavailable policy service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_AGGREGATE_COUNT = 3
DEFAULT_MAX_CANDIDATES = 3
DEFAULT_DISMISS_SUPPRESSION_DAYS = 30
DEFAULT_RESUGGEST_COOLDOWN_DAYS = 14
DEFAULT_POPULAR_INSTALL_THRESHOLD = 10
DEFAULT_REVIEW_INTERVAL_HOURS = 24 * 7
DEFAULT_DELIVERY_RETRIES = 3
DEFAULT_DELIVERY_BACKOFF_SECONDS = 1.0
DEFAULT_RECOMMENDATION_TTL_HOURS = 24 * 7

_INT_FLOORS = {
    "window_days": 1,
    "min_aggregate_count": 1,
    "max_candidates": 0,
    "dismiss_suppression_days": 1,
    "resuggest_cooldown_days": 0,
    "popular_install_threshold": 1,
    "review_interval_hours": 1,
    "delivery_retries": 0,
    "recommendation_ttl_hours": 1,
}


@dataclass(frozen=True)
class AgentLedPolicy:
    enabled: bool = True
    window_days: int = DEFAULT_WINDOW_DAYS
    min_aggregate_count: int = DEFAULT_MIN_AGGREGATE_COUNT
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    dismiss_suppression_days: int = DEFAULT_DISMISS_SUPPRESSION_DAYS
    resuggest_cooldown_days: int = DEFAULT_RESUGGEST_COOLDOWN_DAYS
    popular_install_threshold: int = DEFAULT_POPULAR_INSTALL_THRESHOLD
    review_interval_hours: int = DEFAULT_REVIEW_INTERVAL_HOURS
    delivery_retries: int = DEFAULT_DELIVERY_RETRIES
    delivery_backoff_seconds: float = DEFAULT_DELIVERY_BACKOFF_SECONDS
    recommendation_ttl_hours: int = DEFAULT_RECOMMENDATION_TTL_HOURS
    consecutive_day_usage_counts: bool = True
    repeated_edits_count: bool = True
    notification_defaults: dict[str, bool] = field(default_factory=lambda: {
        "skill_ready_to_share": True, "teammate_published": True, "update_available": True,
    })
    source: str = "defaults"
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "window_days": self.window_days,
            "min_aggregate_count": self.min_aggregate_count,
            "max_candidates": self.max_candidates,
            "dismiss_suppression_days": self.dismiss_suppression_days,
            "resuggest_cooldown_days": self.resuggest_cooldown_days,
            "popular_install_threshold": self.popular_install_threshold,
            "review_interval_hours": self.review_interval_hours,
            "delivery_retries": self.delivery_retries,
            "delivery_backoff_seconds": self.delivery_backoff_seconds,
            "recommendation_ttl_hours": self.recommendation_ttl_hours,
            "consecutive_day_usage_counts": self.consecutive_day_usage_counts,
            "repeated_edits_count": self.repeated_edits_count,
            "notification_defaults": dict(self.notification_defaults),
            "source": self.source,
        }


def _coerce_int(value: Any, fallback: int, floor: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(floor, number)


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, number)


def _apply(policy: AgentLedPolicy, values: Mapping[str, Any], *, source: str) -> AgentLedPolicy:
    updates: dict[str, Any] = {}
    if isinstance(values.get("enabled"), bool):
        updates["enabled"] = values["enabled"]
    for key, floor in _INT_FLOORS.items():
        if key in values:
            updates[key] = _coerce_int(values[key], getattr(policy, key), floor)
    if "delivery_backoff_seconds" in values:
        updates["delivery_backoff_seconds"] = _coerce_float(
            values["delivery_backoff_seconds"], policy.delivery_backoff_seconds
        )
    known = set(_INT_FLOORS) | {"enabled", "delivery_backoff_seconds"}
    extras = {k: v for k, v in values.items() if k not in known}
    if extras:
        updates["extras"] = {**policy.extras, **extras}
    if updates:
        updates["source"] = source
    return replace(policy, **updates) if updates else policy


def _local_config_block() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config

        wisdom = (load_config() or {}).get("wisdom") or {}
    except Exception:
        return {}
    if not isinstance(wisdom, dict):
        return {}
    block = wisdom.get("agent_led")
    if isinstance(block, bool):
        return {"enabled": block}
    return block if isinstance(block, dict) else {}


def load_policy(*, client: Any = None, local: Mapping[str, Any] | None = None) -> AgentLedPolicy:
    """Read the member-scoped policy when agent delivery has not been disabled."""
    policy = AgentLedPolicy()
    local_block = local if local is not None else _local_config_block()
    policy = _apply(policy, local_block, source="local_config")
    from ..mediation import delivery_mode

    if not policy.enabled or delivery_mode() != "agent":
        return replace(policy, enabled=False)
    try:
        from ..client import AgentLedPolicyResponse

        server = client.agent_led_policy()
        server = AgentLedPolicyResponse.model_validate(server)
        if not client.display_org_id or server.org_id != client.display_org_id:
            raise ValueError("Recommendation policy organization mismatch")
    except Exception as exc:
        logger.debug("Agent-led policy unavailable (%s); deferring", type(exc).__name__)
        return replace(policy, enabled=False, source="server_unavailable")
    return replace(
        policy,
        window_days=server.usage_evidence_window_days,
        min_aggregate_count=server.min_aggregate_invocations,
        max_candidates=server.max_recommendations_per_user_per_week,
        dismiss_suppression_days=server.not_now_suppression_days,
        popular_install_threshold=server.install_popularity_threshold,
        consecutive_day_usage_counts=server.consecutive_day_usage_counts,
        repeated_edits_count=server.repeated_edits_count,
        notification_defaults=server.notification_defaults.model_dump(),
        source="server_policy",
    )
