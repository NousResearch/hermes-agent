"""Operator-facing diagnostics for Feishu admission configuration."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Optional


def blocked_group_access_warning(
    *,
    default_group_policy: str,
    allowed_group_users: Collection[str],
    admins: Collection[str],
    group_rules: Mapping[str, Any],
) -> Optional[str]:
    """Explain when the effective default policy cannot admit any human sender."""
    if (
        default_group_policy == "allowlist"
        and not allowed_group_users
        and not admins
        and not group_rules
    ):
        return (
            "[Feishu] Group policy is 'allowlist', but no FEISHU_ALLOWED_USERS, "
            "admins, or group_rules are configured; all human group messages will "
            "be rejected. Configure group access in this profile's .env or config.yaml."
        )
    return None
