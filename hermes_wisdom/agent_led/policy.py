"""Policy values for agent-led sharing.

Server-provided policy wins when the sync client exposes it; otherwise the
local ``wisdom.agent_led`` config block applies, falling back to documented
defaults. Every value is clamped so a malformed config cannot disable the
safety-relevant floors.
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


def _server_policy_block(client: Any) -> Mapping[str, Any]:
    """Return the ``agentLed`` block of the org policy when the client exposes it.

    The Gateway policy endpoint is admin-scoped today, so a 401/403 or a
    missing method is expected for most members and is treated as "no server
    policy". Only a dict-shaped ``agentLed``/``agent_led`` block is honored.
    """
    if client is None:
        return {}
    getter = getattr(client, "org_policy", None) or getattr(client, "policy", None)
    if not callable(getter):
        return {}
    try:
        body = getter()
    except Exception as exc:
        logger.debug("Agent-led policy: server policy unavailable (%s)", type(exc).__name__)
        return {}
    if not isinstance(body, dict):
        return {}
    block = body.get("agentLed", body.get("agent_led"))
    return block if isinstance(block, dict) else {}


def load_policy(*, client: Any = None, local: Mapping[str, Any] | None = None) -> AgentLedPolicy:
    """Resolve the effective agent-led policy (defaults < local config < server)."""
    policy = AgentLedPolicy()
    local_block = local if local is not None else _local_config_block()
    policy = _apply(policy, local_block, source="local_config")
    server_block = _server_policy_block(client)
    policy = _apply(policy, server_block, source="server_policy")
    from ..mediation import delivery_mode

    # Organization policy and legacy local flags cannot opt a profile into
    # proactive model work. There is one user-visible rollout switch.
    return replace(policy, enabled=policy.enabled and delivery_mode() == "agent")
