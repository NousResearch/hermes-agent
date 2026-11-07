"""Text debounce batching must never merge messages from different authors.

Batches key on the SESSION. With a shared group session (``group_sessions_per_user: false``)
one key covers the whole group, so rapid messages from different people used to be
concatenated into one event carrying only the first sender's ``source`` — the gateway then
attributed the second person's words to the first (seen live in a WhatsApp group). Every
adapter batches through ``BasePlatformAdapter._enqueue_text_event``, whose author-change flush
lives in one helper (``flush_pending_if_author_changed``); these tests drive it through the
real WhatsApp and Telegram adapters. Feishu keeps its own batcher and gates merges on the same
``batch_authors_match`` predicate.
"""

import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import batch_authors_match
from gateway.session import SessionSource
from tests.waiting import async_wait_until

GROUP_CHAT = "120363000000000000@g.us"


def _event(text, user_id, user_name, *, platform=Platform.WHATSAPP, user_id_alt=None):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=platform, chat_id=GROUP_CHAT, chat_type="group",
            user_id=user_id, user_id_alt=user_id_alt, user_name=user_name,
        ),
    )


def _whatsapp_adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    return WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "session_name": "test", "group_sessions_per_user": False}))


def _telegram_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={
        "group_sessions_per_user": False}))


ADAPTERS = {
    Platform.WHATSAPP: _whatsapp_adapter,
    Platform.TELEGRAM: _telegram_adapter,
}


def _recording(adapter, *, fail_for=None):
    """Replace the adapter's pipeline entry with a recorder; ``fail_for`` raises for one author."""
    dispatched = []

    async def handle_message(event):
        dispatched.append((event.source.user_name, event.text))
        if event.source.user_name == fail_for:
            raise RuntimeError("handler exploded")

    adapter.handle_message = handle_message
    adapter._text_batch_delay_seconds = 0.05
    adapter._text_batch_split_delay_seconds = 0.05
    return dispatched


def test_batch_authors_match():
    alice = _event("a", "111@lid", "Alice")
    assert batch_authors_match(alice, _event("b", "111@lid", "Alice"))
    assert not batch_authors_match(alice, _event("c", "222@lid", "Bruna"))
    # The alt id identifies the same person when the primary id differs in form.
    assert batch_authors_match(_event("a", "primary-form", "Alice", user_id_alt="111@lid"),
                               _event("b", "111@lid", "Alice"))
    # One side identified, the other not: never the same author.
    assert not batch_authors_match(alice, _event("d", None, "Alice"))
    # No ids anywhere: the display name decides; no identity at all keeps legacy batching.
    assert batch_authors_match(_event("a", None, "Name"), _event("b", None, "Name"))
    assert not batch_authors_match(_event("a", None, "Name"), _event("b", None, "Other"))
    assert batch_authors_match(_event("a", None, None), _event("b", None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", list(ADAPTERS))
async def test_author_change_flushes_prior_batch_as_its_own_turn(platform):
    adapter = ADAPTERS[platform]()
    dispatched = _recording(adapter)

    adapter._enqueue_text_event(_event("first line", "111@lid", "Alice", platform=platform))
    adapter._enqueue_text_event(_event("second line", "111@lid", "Alice", platform=platform))
    adapter._enqueue_text_event(_event("quick interjection", "222@lid", "Bruna", platform=platform))

    await async_wait_until(lambda: len(dispatched) >= 2, what="both authors dispatched")
    assert dispatched == [("Alice", "first line\nsecond line"), ("Bruna", "quick interjection")]
    assert not adapter._pending_text_batches


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", list(ADAPTERS))
async def test_same_author_still_batches(platform):
    adapter = ADAPTERS[platform]()
    dispatched = _recording(adapter)

    adapter._enqueue_text_event(_event("line 1", "111@lid", "Alice", platform=platform))
    adapter._enqueue_text_event(_event("line 2", "111@lid", "Alice", platform=platform))

    await async_wait_until(lambda: dispatched, what="batch dispatched")
    assert dispatched == [("Alice", "line 1\nline 2")]


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", list(ADAPTERS))
async def test_failing_author_flush_dispatch_is_logged_not_lost(platform, caplog):
    adapter = ADAPTERS[platform]()
    dispatched = _recording(adapter, fail_for="Alice")

    with caplog.at_level(logging.ERROR, logger="gateway.platforms.helpers"):
        adapter._enqueue_text_event(_event("from alice", "111@lid", "Alice", platform=platform))
        adapter._enqueue_text_event(_event("from bruna", "222@lid", "Bruna", platform=platform))
        await async_wait_until(lambda: len(dispatched) >= 2, what="both authors dispatched")
        await async_wait_until(lambda: not adapter._background_tasks, what="author flush settled")

    assert dispatched == [("Alice", "from alice"), ("Bruna", "from bruna")]
    failures = [r for r in caplog.records if "Text batch dispatch failed" in r.getMessage()]
    assert len(failures) == 1
    assert isinstance(failures[0].exc_info[1], RuntimeError)


def test_feishu_batches_merge_only_within_one_author():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    alice = _event("a", "ou_alice", "Alice", platform=Platform.FEISHU)
    assert FeishuAdapter._text_batch_is_compatible(alice, _event("b", "ou_alice", "Alice", platform=Platform.FEISHU))
    assert not FeishuAdapter._text_batch_is_compatible(alice, _event("c", "ou_bruna", "Bruna", platform=Platform.FEISHU))


@pytest.mark.asyncio
async def test_wecom_author_flush_does_not_promote_the_flushed_attachment():
    """WeCom promotes a buffered attachment to TEXT when text joins it; another author's text
    flushes the attachment instead, so it must leave the flushed event untouched."""
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"group_sessions_per_user": False}))
    dispatched = _recording(adapter)
    adapter._attachment_text_merge_delay_seconds = 0.05
    attachment = _event("", "alice", "Alice", platform=Platform.WECOM)
    attachment.message_type = MessageType.PHOTO
    attachment.media_urls, attachment.media_types = ["cache/a.png"], ["image/png"]
    reply = _event("about that", "bruna", "Bruna", platform=Platform.WECOM)
    reply.reply_to_message_id, reply.reply_to_text = "quote:1", "quoted"

    adapter._enqueue_text_event(attachment)
    adapter._enqueue_text_event(reply)

    await async_wait_until(lambda: len(dispatched) >= 2, what="both authors dispatched")
    assert dispatched == [("Alice", ""), ("Bruna", "about that")]
    assert attachment.message_type == MessageType.PHOTO
    assert attachment.reply_to_message_id is None
