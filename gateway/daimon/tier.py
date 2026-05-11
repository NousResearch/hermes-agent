from __future__ import annotations

from enum import Enum
from typing import Optional

from gateway.daimon.config import DaimonConfig


class Tier(Enum):
    """User access tier."""

    ADMIN = "admin"
    USER = "user"

    def model(self, cfg: DaimonConfig) -> str:
        """Return the model string for this tier."""
        if self is Tier.ADMIN:
            return cfg.admin_model
        return cfg.user_model

    @property
    def is_admin(self) -> bool:
        """Return True if this tier has admin privileges."""
        return self is Tier.ADMIN


def resolve_tier(
    user_id: str,
    cfg: DaimonConfig,
    role_ids: Optional[list[str]] = None,
) -> Optional[Tier]:
    """Determine the tier for a given user ID and roles based on config.

    Resolution order (highest privilege wins):
      1. debug_force_tier override → forced tier for all users
      2. user_id in admin_users → ADMIN
      3. any role in admin_roles → ADMIN
      4. user_roles empty (not configured) → USER (open access)
      5. user_id in user_users → USER
      6. any role in user_roles → USER
      7. Otherwise → None (silent ignore)

    Returns None when the user should be silently ignored (user_roles is
    configured but the user matches neither admin nor user criteria).
    """
    # Debug override — force all users to a specific tier
    if cfg.debug_force_tier:
        try:
            return Tier(cfg.debug_force_tier)
        except ValueError:
            pass  # Invalid tier name in config — fall through to normal resolution

    # Admin checks (highest privilege wins)
    if user_id in cfg.admin_users:
        return Tier.ADMIN
    if role_ids and cfg.admin_roles:
        if set(role_ids) & set(cfg.admin_roles):
            return Tier.ADMIN

    # User checks
    if not cfg.user_roles:
        # No user_roles configured = open access (everyone is user tier)
        return Tier.USER
    if user_id in cfg.user_users:
        return Tier.USER
    if role_ids and set(role_ids) & set(cfg.user_roles):
        return Tier.USER

    # No match + user_roles configured = silent ignore
    return None
