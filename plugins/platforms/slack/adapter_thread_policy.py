"""Slack thread follow-up admission; legacy auto-engagement remains the default."""
from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger(__name__)


def thread_followup_mode(extra: dict) -> str:
    mode = extra.get("thread_followup_mode", "auto")
    if mode not in {"auto", "root_mention"}:
        raise ValueError("Slack thread_followup_mode must be auto or root_mention")
    return mode


async def root_mention_allows(adapter, channel_id: str, thread_ts: str, team_id: str, mention_text) -> bool:
    """Read only the head of an unmentioned thread; failures never enroll a conversation."""
    bot_uid = adapter._team_bot_user_ids.get(team_id, adapter._bot_user_id)
    if not bot_uid:
        return False
    key = (team_id, channel_id, thread_ts)
    cached = adapter._root_mention_cache.get(key)
    if cached and time.monotonic() - cached[0] < 60:
        return cached[1]
    try:
        result = await adapter._get_client(channel_id, team_id=team_id).conversations_replies(
            channel=channel_id, ts=thread_ts, limit=1, inclusive=True)
        messages = result.get("messages") or []
        allowed = bool(messages and messages[0].get("ts") == thread_ts
                       and re.search(rf"<@{re.escape(bot_uid)}(?:\|[^>]*)?>", mention_text(messages[0])))
    except Exception:
        logger.debug("Slack thread head unavailable; ignoring unmentioned reply", exc_info=True)
        return False
    adapter._root_mention_cache[key] = (time.monotonic(), allowed)
    if len(adapter._root_mention_cache) > 2500:
        adapter._root_mention_cache.pop(next(iter(adapter._root_mention_cache)))
    return allowed


async def should_wake_on_unmentioned_message(
    adapter, event_thread_ts, channel_id: str, user_id: str, is_thread_reply: bool,
    team_id: str, chat_type: str, mention_text,
) -> bool:
    """Opt-in root gating excludes legacy session/mention wake signals without changing auto mode."""
    if (thread_followup_mode(adapter.config.extra) == "root_mention"
            and chat_type != "dm"):
        return bool(is_thread_reply and event_thread_ts) and await root_mention_allows(
            adapter, channel_id, event_thread_ts, team_id, mention_text)
    if not event_thread_ts:
        return False
    thread_marker = adapter._workspace_message_marker(team_id, event_thread_ts)
    # Check scoped marker AND bare ts: entries recorded before team_id was
    # known are bare, and a scoped-vs-bare mismatch must not silence the bot.
    if is_thread_reply and (
        thread_marker in adapter._bot_message_ts or event_thread_ts in adapter._bot_message_ts):
        return True
    if thread_marker in adapter._mentioned_threads or event_thread_ts in adapter._mentioned_threads:
        return True
    if is_thread_reply and adapter._has_active_session_for_thread(
        channel_id=channel_id, thread_ts=event_thread_ts, user_id=user_id, team_id=team_id,
        chat_type=chat_type):
        return True
    if is_thread_reply and await adapter._bot_authored_thread_root(
        channel_id=channel_id, thread_ts=event_thread_ts, team_id=team_id, chat_type=chat_type):
        return True
    # Thread PARENT @-mentioned the bot before this process (restart): a bare "run" is for us.
    # 5th check (#24848): the thread PARENT @-mentioned the bot, but the mention event predates this
    # process (restart) or the parent asked the bot to wait for a follow-up (e.g. A plain reply like
    # "run" in that thread is addressed to the bot even though the reply itself carries no mention.
    if is_thread_reply:
        bot_uid = adapter._team_bot_user_ids.get(team_id, adapter._bot_user_id)
        if bot_uid:
            parent_text = await adapter._fetch_thread_parent_text(
                channel_id=channel_id, thread_ts=event_thread_ts, team_id=team_id,
                strip_bot_mention=False)
            if parent_text and f"<@{bot_uid}>" in parent_text:
                # Remember so later replies skip the fetch.
                if not adapter._slack_strict_mention():
                    adapter._register_mentioned_thread(event_thread_ts)
                return True
    return False
