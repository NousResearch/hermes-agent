import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import ProcessingOutcome


def _make_adapter(*, bot_to_bot=None, media=None, reactions=None, bot_username="hermes_bot"):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    extra = {
        "allowed_chats": [],
        "group_allowed_chats": [],
        "allowed_topics": [],
    }
    if bot_to_bot is not None:
        extra["bot_to_bot"] = bot_to_bot
    if media is not None:
        extra["media"] = media
    if reactions is not None:
        extra["reactions"] = reactions

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(
        id=999,
        username=bot_username,
        set_message_reaction=AsyncMock(),
        send_sticker=AsyncMock(return_value=SimpleNamespace(message_id=456)),
    )
    adapter._reply_to_mode = "first"
    adapter._mention_patterns = []
    adapter._bot_to_bot_rate_windows = {}
    adapter._bot_to_bot_circuit_breakers = {}
    adapter._status_message_ids = {}
    adapter._notification_kwargs = lambda metadata: {}
    adapter._thread_kwargs_for_send = lambda *args, **kwargs: {"message_thread_id": kwargs.get("reply_to_message_id") and None}
    adapter._metadata_thread_id = lambda metadata: (metadata or {}).get("thread_id")
    adapter._should_thread_reply = lambda reply_to, index=0: bool(reply_to)
    return adapter


def _mention_entity(text, mention="@hermes_bot"):
    return SimpleNamespace(type="mention", offset=text.index(mention), length=len(mention))


def _group_message(
    text="hello",
    *,
    chat_id=-100,
    thread_id=903,
    sender_id=111,
    sender_username="peer_bot",
    sender_is_bot=True,
    entities=None,
):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=entities or [],
        caption_entities=[],
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        chat=SimpleNamespace(id=chat_id, type="supergroup", title="Test Group", is_forum=True),
        from_user=SimpleNamespace(
            id=sender_id,
            username=sender_username,
            is_bot=sender_is_bot,
            full_name="Peer Bot" if sender_is_bot else "Alice Example",
            first_name="Peer" if sender_is_bot else "Alice",
        ),
        reply_to_message=None,
        date=None,
    )


def _enabled_config(**overrides):
    cfg = {
        "enabled": True,
        "allowlisted_bot_ids": ["111"],
        "allowlisted_bot_usernames": ["peer_bot"],
        "require_explicit_mention": True,
        "exclusive_bot_mentions": True,
        "enabled_chats": ["-100:903"],
        "max_reply_depth": 3,
        "rate_limit": {"window_seconds": 60, "max_messages": 10},
        "circuit_breaker": {"window_seconds": 300, "max_trips": 3, "cooldown_seconds": 900},
    }
    cfg.update(overrides)
    return cfg


def test_bot_sender_is_ignored_by_default():
    adapter = _make_adapter()
    text = "@hermes_bot please review"
    msg = _group_message(text, entities=[_mention_entity(text)])

    assert adapter._should_process_message(msg) is False


def test_allowlisted_bot_sender_with_explicit_target_is_accepted():
    adapter = _make_adapter(bot_to_bot=_enabled_config())
    text = "@hermes_bot please review"
    msg = _group_message(text, entities=[_mention_entity(text)])

    assert adapter._should_process_message(msg) is True


def test_non_allowlisted_bot_sender_is_ignored_even_when_mentioned():
    adapter = _make_adapter(bot_to_bot=_enabled_config())
    text = "@hermes_bot please review"
    msg = _group_message(
        text,
        sender_id=222,
        sender_username="stranger_bot",
        entities=[_mention_entity(text)],
    )

    assert adapter._should_process_message(msg) is False


def test_bot_message_to_other_bot_is_ignored():
    adapter = _make_adapter(bot_to_bot=_enabled_config())
    text = "@codex_bot please review"
    msg = _group_message(text, entities=[_mention_entity(text, "@codex_bot")])

    assert adapter._should_process_message(msg) is False


def test_bot_to_bot_topic_gate_uses_chat_colon_thread():
    adapter = _make_adapter(bot_to_bot=_enabled_config(enabled_chats=["-100:708"]))
    text = "@hermes_bot please review"
    msg = _group_message(text, thread_id=903, entities=[_mention_entity(text)])

    assert adapter._should_process_message(msg) is False


def test_bot_to_bot_trace_depth_blocks_at_limit_and_strips_before_dispatch():
    adapter = _make_adapter(bot_to_bot=_enabled_config(max_reply_depth=2))
    text = "@hermes_bot continue [trace:hms:abc depth=2]"
    msg = _group_message(text, entities=[_mention_entity(text)])

    assert adapter._should_process_message(msg) is False
    assert adapter._clean_bot_trigger_text(text) == "continue"


def test_bot_to_bot_rate_limit_blocks_after_window_quota():
    adapter = _make_adapter(
        bot_to_bot=_enabled_config(rate_limit={"window_seconds": 60, "max_messages": 1})
    )
    text = "@hermes_bot one"
    msg = _group_message(text, entities=[_mention_entity(text)])

    assert adapter._should_process_message(msg) is True
    assert adapter._should_process_message(msg) is False


def test_reactions_use_nested_media_config_emojis():
    async def _run():
        adapter = _make_adapter(
            media={
                "reactions": {
                    "enabled": True,
                    "on_accept": "⏳",
                    "on_success": "✅",
                    "on_error": "❌",
                }
            }
        )
        event = SimpleNamespace(source=SimpleNamespace(chat_id="-100"), message_id="42")
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        calls = adapter._bot.set_message_reaction.await_args_list
        assert calls[0].kwargs["reaction"] == "⏳"
        assert calls[1].kwargs["reaction"] == "✅"

    asyncio.run(_run())


def test_sticker_alias_send_preserves_thread_metadata():
    async def _run():
        adapter = _make_adapter(
            media={"stickers": {"enabled": True, "aliases": {"shipit": "FILE_ID"}}}
        )
        result = await adapter.send_sticker_alias("-100", "shipit", metadata={"thread_id": "903"})

        assert result.success is True
        assert result.message_id == "456"
        adapter._bot.send_sticker.assert_awaited_once()
        assert adapter._bot.send_sticker.await_args.kwargs["sticker"] == "FILE_ID"

    asyncio.run(_run())
