"""Conservative opt-in policy for isolated night-business worker routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai-codex"})
_DEFAULT_TIMEZONE = "Asia/Seoul"
_DEFAULT_START = (21, 0)
_DEFAULT_END = (6, 0)


@dataclass(frozen=True)
class NightBusinessRouteDecision:
    """An audit-safe route selection; profile is set only for an allowed route."""

    profile: str | None
    provider: str
    reason: str
    timezone: str
    local_time: str


def _parse_time(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, str):
        return default
    try:
        hour, minute = (int(part) for part in value.strip().split(":", 1))
    except (TypeError, ValueError):
        return default
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return default
    return hour, minute


def _is_in_window(now: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
    current = (now.hour, now.minute)
    if start == end:
        return False
    if start < end:
        return start <= current < end
    return current >= start or current < end


def decide_night_business_route(
    settings: Mapping[str, Any] | None,
    *,
    provider: str,
    personal_has_credentials: bool,
    personal_has_available: bool,
    business_has_available: bool,
    now: datetime | None = None,
) -> NightBusinessRouteDecision:
    """Select a business profile only after the matching personal pool is exhausted.

    This pure policy intentionally accepts only pool availability evidence. It
    never receives tokens, pool entries, or account labels, keeping its result
    safe for durable Kanban events and Discord status reporting.
    """
    config = settings if isinstance(settings, Mapping) else {}
    normalized_provider = str(provider or "").strip().lower()
    timezone_name = str(config.get("timezone") or _DEFAULT_TIMEZONE)
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        zone = ZoneInfo(_DEFAULT_TIMEZONE)
        timezone_name = _DEFAULT_TIMEZONE
    local_now = (now or datetime.now(zone)).astimezone(zone)
    decision = lambda profile, reason: NightBusinessRouteDecision(
        profile=profile,
        provider=normalized_provider,
        reason=reason,
        timezone=timezone_name,
        local_time=local_now.isoformat(timespec="minutes"),
    )

    if config.get("enabled") is not True:
        return decision(None, "disabled")
    if normalized_provider not in _SUPPORTED_PROVIDERS:
        return decision(None, "unsupported_provider")
    start = _parse_time(config.get("start_time"), _DEFAULT_START)
    end = _parse_time(config.get("end_time"), _DEFAULT_END)
    if not _is_in_window(local_now, start, end):
        return decision(None, "outside_window")
    if not personal_has_credentials:
        return decision(None, "personal_unknown")
    if personal_has_available:
        return decision(None, "personal_available")
    profiles = config.get("profiles")
    profile = profiles.get(normalized_provider) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, str) or not profile.strip():
        return decision(None, "business_profile_missing")
    if not business_has_available:
        return decision(None, "business_unavailable")
    return decision(profile.strip(), "personal_exhausted")
