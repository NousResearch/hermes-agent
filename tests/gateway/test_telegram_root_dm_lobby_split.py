"""``ignore_root_dm`` keeps the TEXT remainder of a client-split root-DM command.

Telegram splits a >4096-char message: a long ``/queue <prompt>`` arrives as a near-limit COMMAND chunk
plus plain-TEXT continuations. The lobby gate admits the command; it must also admit those
continuations, or the command runs truncated, while root chatter stays dropped.
"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import PlatformConfig
from tests.gateway.conftest import _ensure_telegram_mock

_ensure_telegram_mock()
from plugins.platforms.telegram.adapter import TelegramAdapter

CHAT, USER = 111, 42


def _update(message_id, text):
    m = SimpleNamespace(
        chat=SimpleNamespace(id=CHAT, type="private", title=None, full_name="T"),
        from_user=SimpleNamespace(id=USER, full_name="T", is_bot=False), text=text,
        message_thread_id=None, is_topic_message=False, message_id=message_id,
        reply_to_message=None, date=datetime.now(UTC), forum_topic_created=None)
    return SimpleNamespace(message=m, effective_message=m, update_id=message_id)


@pytest.mark.asyncio
async def test_split_root_dm_command_keeps_its_continuation_and_drops_later_chatter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "allow_from": [str(USER)], "ignore_root_dm": True, "dm_topics": [{"chat_id": CHAT, "topics": []}]}))

    async def _no_commands(*_a):
        return None

    a._ensure_forum_commands = _no_commands
    a._reload_dm_topics_from_config = lambda: None  # config.yaml would re-supply the same dm_topics
    first = "/queue " + "x" * (4096 - len("/queue "))
    tail = "IMPORTANT: do not post to the public channel"
    try:
        await a._handle_command(_update(1, first), None)
        await a._handle_text_message(_update(2, tail), None)
        await a._handle_text_message(_update(3, "unrelated root chatter"), None)  # last chunk was short
        texts = [e.text for e in a._pending_text_batches.values()]
        assert len(texts) == 1 and texts[0].startswith(first) and texts[0].endswith(tail)
    finally:
        tasks = list(a._pending_text_batch_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
