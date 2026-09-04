"""Notification policy and plugin-local config resolution for Discord."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

NOTIFICATIONS_ALL = "all"
NOTIFICATIONS_IMPORTANT = "important"
VALID_NOTIFICATION_MODES = frozenset({NOTIFICATIONS_ALL, NOTIFICATIONS_IMPORTANT})
SUPPRESS_NOTIFICATIONS_FLAG = 1 << 12


def normalize_notification_mode(value: Any, *, default: str = NOTIFICATIONS_ALL) -> str:
    """Return a supported mode, falling back to the backward-compatible default."""
    normalized = str(value or default).strip().lower()
    return normalized if normalized in VALID_NOTIFICATION_MODES else default


def resolve_notification_mode(config: Any) -> str:
    """Resolve Discord notifications from profile display config or adapter config.

    The display setting is plugin-owned and does not belong in the gateway's
    generic YAML bridge. ``PlatformConfig.extra`` remains a construction-time
    fallback for direct adapter users and tests.
    """
    extra = getattr(config, "extra", None)
    raw = None
    try:
        from hermes_cli.config import load_config_readonly

        loaded = load_config_readonly() or {}
        display = loaded.get("display") if isinstance(loaded, Mapping) else None
        platforms = display.get("platforms") if isinstance(display, Mapping) else None
        discord = platforms.get("discord") if isinstance(platforms, Mapping) else None
        configured = discord.get("notifications") if isinstance(discord, Mapping) else None
        if configured not in {None, ""}:
            raw = configured
    except Exception:
        logger.debug("Could not load Discord notification mode", exc_info=True)
    if raw is None and isinstance(extra, Mapping):
        raw = extra.get("notifications")

    normalized = normalize_notification_mode(raw)
    candidate = str(raw or NOTIFICATIONS_ALL).strip().lower()
    if candidate not in VALID_NOTIFICATION_MODES:
        logger.warning(
            "Unknown Discord notifications mode %r; defaulting to 'all' (valid: all, important)",
            raw,
        )
    return normalized


def notification_kwargs(
    *,
    mode: str,
    metadata: Mapping[str, Any] | None,
    final_chunk: bool,
) -> dict[str, bool]:
    """Return discord.py kwargs for one physical message.

    ``notify`` is the authoritative notification-worthy marker, matching the
    existing Telegram contract and preserving actionable sends. Ordinary
    ``_interim_send`` deliveries do not carry it. For a split notification-worthy
    delivery, only its last physical Discord message omits ``silent`` and can
    generate a push.
    """
    if mode != NOTIFICATIONS_IMPORTANT:
        return {}
    metadata = metadata or {}
    notification_worthy = bool(metadata.get("notify"))
    if notification_worthy and final_chunk:
        return {}
    return {"silent": True}


def with_suppress_notifications_flag(flags: int, *, silent: bool) -> int:
    """Set Discord's SUPPRESS_NOTIFICATIONS bit for raw REST payloads."""
    return flags | SUPPRESS_NOTIFICATIONS_FLAG if silent else flags
