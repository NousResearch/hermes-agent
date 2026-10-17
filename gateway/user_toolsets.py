"""Resolve per-user gateway toolsets from top-level ``user_toolsets`` config."""

from __future__ import annotations

from typing import List, Optional

from gateway.whatsapp_identity import canonical_whatsapp_identifier
from hermes_cli.toolset_validation import parse_platform_toolsets_value


def _source_identifiers(source: object, platform: str) -> list[str]:
    """Ordered sender identifiers: primary, alternate, then platform-derived aliases."""
    identifiers: list[str] = []
    for field in ("user_id", "user_id_alt"):
        value = getattr(source, field, None)
        identifier = str(value).strip() if value is not None else ""
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)

    if platform == "whatsapp":
        for identifier in tuple(identifiers):
            canonical = canonical_whatsapp_identifier(identifier)
            if canonical and canonical not in identifiers:
                identifiers.append(canonical)
    return identifiers


def resolve_user_toolsets(
    config: dict, source: object, platform: str,
) -> Optional[List[str]]:
    """Return a matching user's replacement toolset list, or ``None``.

    Platform keys match source identifiers after string coercion, so quoted and
    numeric YAML keys behave identically. Platform-specific alias expansion belongs
    exclusively to :func:`_source_identifiers`; configured keys remain literal.
    Invalid values are ignored and safely fall back to the platform toolsets. An
    explicit empty list remains authoritative.
    """
    user_toolsets = config.get("user_toolsets")
    if not isinstance(user_toolsets, dict):
        return None
    platform_users = user_toolsets.get(platform)
    if not isinstance(platform_users, dict):
        return None

    source_ids = _source_identifiers(source, platform)
    if not source_ids:
        return None

    configured_users: dict[str, object] = {}
    for configured_id, raw_toolsets in platform_users.items():
        candidate = str(configured_id).strip()
        if candidate:
            configured_users.setdefault(candidate, raw_toolsets)

    for identifier in source_ids:
        if identifier in configured_users:
            return parse_platform_toolsets_value(configured_users[identifier])
    return None
