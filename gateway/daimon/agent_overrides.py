"""Compute AIAgent construction overrides based on Daimon tier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gateway.daimon.config import load_daimon_config
from gateway.daimon.tier import Tier, resolve_tier


@dataclass
class AgentOverrides:
    """Overrides to apply to AIAgent construction for a Daimon session."""

    model: Optional[str] = None  # Override the model
    max_iterations: Optional[int] = None  # Override iteration cap
    disabled_toolsets: Optional[list[str]] = None  # ADDITIONAL disabled toolsets (merge with existing)
    gateway_timeout: Optional[int] = None  # Override gateway timeout
    ephemeral_system_prompt: Optional[str] = None  # Daimon persona prompt
    tier: Optional[Tier] = Tier.USER  # None = user should be silently ignored


def compute_overrides(
    raw_config: dict,
    user_id: str,
    platform: str,
    role_ids: Optional[list[str]] = None,
) -> Optional[AgentOverrides]:
    """Compute tier-based overrides for agent construction.

    Returns None if Daimon is not configured (no admin_users and no admin_roles set)
    or if the platform is not Discord.
    Returns AgentOverrides with tier=None if the user should be silently ignored.
    Returns AgentOverrides with the appropriate values for the user's tier.
    """
    if platform != "discord":
        return None

    cfg = load_daimon_config(raw_config)

    # Daimon is only active if at least one access control list is configured
    if not cfg.admin_users and not cfg.admin_roles:
        return None

    tier = resolve_tier(user_id, cfg, role_ids=role_ids)

    if tier is None:
        # User should be silently ignored — return sentinel with tier=None
        return AgentOverrides(tier=None)

    if tier.is_admin:
        return AgentOverrides(
            model=cfg.admin_model,
            tier=tier,
        )

    # User tier: apply limits
    # Disable toolsets where limit=0
    disabled = [tool for tool, limit in cfg.tool_limits.items() if limit == 0]

    return AgentOverrides(
        model=cfg.user_model,
        max_iterations=cfg.max_iterations,
        disabled_toolsets=disabled,
        gateway_timeout=cfg.gateway_timeout,
        tier=tier,
    )
