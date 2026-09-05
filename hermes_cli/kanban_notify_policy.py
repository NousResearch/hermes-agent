"""Kanban notification auto-subscribe routing policies.

Auto-subscribe call sites start from a source-session subscription target. Most
platforms keep the historical single ``notify+wake`` row. iMessage-like sources
(Photon and BlueBubbles) are the exception: raw terminal status pings are noisy
there, so when a private Telegram home channel is configured we split the route
into source wake-only plus Telegram passive-notify.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

IMESSAGE_KANBAN_SOURCE_PLATFORMS = frozenset({"photon", "bluebubbles"})
_RAW_NOTIFY_HOME_PLATFORM = "telegram"


def _clean_str(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _routing_key(target: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(target.get("platform") or "").lower(),
        str(target.get("chat_id") or ""),
        str(target.get("thread_id") or ""),
    )


def _dedupe_targets(targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        key = _routing_key(target)
        if key in seen or not key[0] or not key[1]:
            continue
        seen.add(key)
        out.append(target)
    return out


def _telegram_dm_metadata(thread_id: Optional[str]) -> Optional[dict[str, Any]]:
    metadata: dict[str, Any] = {"chat_type": "dm"}
    if thread_id:
        metadata["thread_id"] = thread_id
        # Telegram DM topic mode needs the same placement metadata used by
        # gateway-origin replies so passive pings do not fall into the root lobby.
        metadata["telegram_dm_topic_reply_fallback"] = True
        if thread_id != "1":
            metadata["direct_messages_topic_id"] = thread_id
    return metadata


def _telegram_home_notify_target(notifier_profile: Optional[str]) -> Optional[dict[str, Any]]:
    """Passive raw-notify target for the configured Telegram home channel.

    Returns None when Telegram has no configured home channel. The caller then
    keeps the historical source ``notify+wake`` route so terminal notifications
    are not silently dropped on installs without Telegram.
    """
    try:
        from gateway.config import Platform, load_gateway_config

        home = load_gateway_config().get_home_channel(Platform(_RAW_NOTIFY_HOME_PLATFORM))
    except Exception:
        logger.debug("kanban notify policy: Telegram home lookup failed", exc_info=True)
        return None
    if home is None or not _clean_str(getattr(home, "chat_id", "")):
        return None
    thread_id = _clean_str(getattr(home, "thread_id", None))
    target: dict[str, Any] = {
        "platform": _RAW_NOTIFY_HOME_PLATFORM,
        "chat_id": str(home.chat_id),
        "chat_type": "dm",
        "thread_id": thread_id,
        "user_id": _clean_str(getattr(home, "user_id", None)),
        "notifier_profile": notifier_profile,
        "delivery_mode": "notify",
        "delivery_metadata": _telegram_dm_metadata(thread_id),
    }
    return {key: value for key, value in target.items() if value is not None}


def kanban_auto_subscribe_targets(source_target: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return ``add_notify_sub`` kwargs for a new task auto-subscription.

    ``source_target`` is the historical single route back to the creating
    session. For Photon/BlueBubbles/iMessage-like sources, split supported
    installs into two rows:

    * source platform: ``delivery_mode='wake'`` (Molly can narrate there)
    * Telegram home: ``delivery_mode='notify'`` (raw terminal events only)

    Explicit ``hermes kanban notify-subscribe`` rows do not call this helper, so
    operator-chosen subscriptions remain last-write-wins and untouched.
    """
    if not source_target:
        return []
    source = dict(source_target)
    platform = str(source.get("platform") or "").lower()
    if platform not in IMESSAGE_KANBAN_SOURCE_PLATFORMS:
        return _dedupe_targets([source])

    telegram_target = _telegram_home_notify_target(source.get("notifier_profile"))
    if telegram_target is None:
        return _dedupe_targets([source])

    wake_source = dict(source)
    wake_source["delivery_mode"] = "wake"
    return _dedupe_targets([wake_source, telegram_target])
