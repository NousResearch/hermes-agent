"""Text-debounce batching for the WhatsApp adapter (issue #35301).

WhatsApp delivers rapid multi-message bursts (forwarded batches, paste-splits)
individually.  Without debounce each fragment triggers a separate agent
invocation, wasting tokens and flooding the user with reply fragments.  This
mirrors the Telegram/WeCom/Feishu pattern.

Batch delays are read from ``config.extra`` (config.yaml), not env vars.
"""

import asyncio

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from gateway.session import SessionSource


def _make_adapter(**extra):
    base = {"session_name": "test"}
    base.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base))


def _event(text):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="chat123",
        chat_type="dm",
        user_id="user1",
        user_name="tester",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)


def test_batch_delays_default_from_config():
    adapter = _make_adapter()
    assert adapter._text_batch_delay_seconds == 5.0
    assert adapter._text_batch_split_delay_seconds == 10.0


def test_batch_delays_overridden_via_config_extra():
    adapter = _make_adapter(
        text_batch_delay_seconds="2.5",
        text_batch_split_delay_seconds=7,
    )
    assert adapter._text_batch_delay_seconds == 2.5
    assert adapter._text_batch_split_delay_seconds == 7.0


def test_invalid_config_value_falls_back_to_default():
    adapter = _make_adapter(
        text_batch_delay_seconds="garbage",
        text_batch_split_delay_seconds=-3,
    )
    assert adapter._text_batch_delay_seconds == 5.0
    assert adapter._text_batch_split_delay_seconds == 10.0


def test_env_var_is_ignored(monkeypatch):
    # Config-only path: the legacy HERMES_* env var must NOT influence delays.
    monkeypatch.setenv("HERMES_WHATSAPP_TEXT_BATCH_DELAY_SECONDS", "99")
    adapter = _make_adapter()
    assert adapter._text_batch_delay_seconds == 5.0


def test_rapid_texts_collapse_into_single_dispatch():
    adapter = _make_adapter(
        text_batch_delay_seconds=0.05,
        text_batch_split_delay_seconds=0.05,
    )
    dispatched = []

    async def _capture(event):
        dispatched.append(event.text)

    adapter.handle_message = _capture

    async def _drive():
        adapter._enqueue_text_event(_event("one"))
        adapter._enqueue_text_event(_event("two"))
        adapter._enqueue_text_event(_event("three"))
        assert dispatched == []  # nothing flushed during the burst
        await asyncio.sleep(0.2)

    asyncio.run(_drive())
    assert dispatched == ["one\ntwo\nthree"]


def test_lone_message_dispatched_alone():
    adapter = _make_adapter(
        text_batch_delay_seconds=0.05,
        text_batch_split_delay_seconds=0.05,
    )
    dispatched = []

    async def _capture(event):
        dispatched.append(event.text)

    adapter.handle_message = _capture

    async def _drive():
        adapter._enqueue_text_event(_event("solo"))
        await asyncio.sleep(0.2)

    asyncio.run(_drive())
    assert dispatched == ["solo"]


def test_group_merge_tags_each_appended_author():
    """Shared group batch: each appended message is tagged with its own sender.

    The first message stays unprefixed (run.py prefixes it with its author
    later); subsequent messages from other users must carry their own [Name]
    so authorship isn't collapsed onto the batch owner (issue: author-merge).
    """
    async def _run():
        adapter = _make_adapter(
            group_sessions_per_user=False,
            text_batch_delay_seconds=60,   # don't flush during the test
            text_batch_split_delay_seconds=60,
        )

        def _gev(text, user_id, user_name):
            src = SessionSource(
                platform=Platform.WHATSAPP, chat_id="grp", chat_type="group",
                user_id=user_id, user_name=user_name,
            )
            return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)

        adapter._enqueue_text_event(_gev("hey", "u1", "Maks"))
        adapter._enqueue_text_event(_gev("elo", "u2", "Borrell"))
        key = adapter._text_batch_key(_gev("x", "u1", "Maks"))
        assert adapter._pending_text_batches[key].text == "hey\n[Borrell] elo"
        for t in adapter._pending_text_batch_tasks.values():
            t.cancel()

    asyncio.run(_run())


def test_dm_merge_not_tagged():
    """DMs are single-author: no [Name] tags injected on merge."""
    async def _run():
        adapter = _make_adapter(
            text_batch_delay_seconds=60,
            text_batch_split_delay_seconds=60,
        )
        adapter._enqueue_text_event(_event("a"))
        adapter._enqueue_text_event(_event("b"))
        key = adapter._text_batch_key(_event("x"))
        assert adapter._pending_text_batches[key].text == "a\nb"
        for t in adapter._pending_text_batch_tasks.values():
            t.cancel()

    asyncio.run(_run())
