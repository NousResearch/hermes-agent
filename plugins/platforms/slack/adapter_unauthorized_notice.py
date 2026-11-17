"""Opt-in ephemeral "not enabled for you" notice for a channel @mention from an unauthorized sender.

Without it (the default) such a mention is dropped with only a log line, so the person sees the bot
ignore them. With ``unauthorized_mention_notice: true`` they get one ephemeral reply (visible only
to them) per ``DECLINE_DEDUPE_SECONDS``, sharing the stamps of the DM ``decline`` behaviour so one
sender hears at most one refusal per window. DMs keep their ``unauthorized_dm_behavior`` flow.
"""
import asyncio
import logging
import re
from typing import Any, Optional

from agent.i18n import t
from gateway.pairing import PairingStore

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def unauthorized_mention_notice_enabled(extra: dict) -> bool:
    raw = (extra or {}).get("unauthorized_mention_notice", False)
    return raw.strip().lower() in _TRUE if isinstance(raw, str) else bool(raw)


def _mentions_bot(text: str, bot_uid: Optional[str]) -> bool:
    return bool(bot_uid and text and re.search(rf"<@{re.escape(bot_uid)}(?:\|[^>]*)?>", text))


async def notify_unauthorized_mention(
    adapter: Any, event: dict, user_id: str, channel_id: str, team_id: str, is_dm: bool,
    sender_is_bot: bool) -> None:
    """Send the notice when enabled and the rejected message is a person's @mention in a channel."""
    bot_uid = adapter._team_bot_user_ids.get(team_id, adapter._bot_user_id)
    if (
        is_dm or sender_is_bot or not user_id or not _mentions_bot(event.get("text") or "", bot_uid)
        or not unauthorized_mention_notice_enabled(adapter.config.extra)):
        return
    store = PairingStore(profile=getattr(adapter, "_owner_profile", None))
    # Stamp BEFORE sending (the decline rule): a failed send must not become a notice storm.
    try:
        if store.has_recent_decline("slack", user_id):
            return
        await asyncio.to_thread(store.record_decline, "slack", user_id)
    except (OSError, TimeoutError):
        logger.warning("[Slack] Could not record an unauthorized-mention notice", exc_info=True)
        return
    try:
        await adapter._get_client(channel_id, team_id=team_id or None).chat_postEphemeral(
            channel=channel_id, user=user_id, text=t("platform.slack.unauthorized_mention_notice"),
            thread_ts=event.get("thread_ts"))
    except Exception:
        logger.warning("[Slack] Failed to send unauthorized-mention notice in %s", channel_id, exc_info=True)
