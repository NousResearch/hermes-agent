import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _utf16_span(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    prefix = text[:start]
    return (
        len(prefix.encode("utf-16-le")) // 2,
        len(needle.encode("utf-16-le")) // 2,
    )


def _entity(text: str, typ: str, needle: str, **extra):
    offset, length = _utf16_span(text, needle)
    return SimpleNamespace(type=typ, offset=offset, length=length, **extra)


def _make_adapter(*, require_mention=True, bot_username="hermes_bot"):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True,
        token="[REDACTED]",
        extra={
            "require_mention": require_mention,
            "allowed_chats": ["-100"],
            "allowed_topics": [],
            "group_allowed_chats": ["-100"],
        },
    )
    adapter._bot = SimpleNamespace(id=999, username=bot_username)
    adapter._message_handler = AsyncMock(return_value="response produced")
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = []
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._dm_topic_chat_ids = set()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    adapter._is_user_authorized_from_message = lambda _msg: True
    adapter._apply_topic_recovery = lambda _event: None
    adapter._should_drop_delayed_delivery = lambda: False
    async def _handle_message(event):
        await adapter._message_handler(event)
    adapter.handle_message = _handle_message
    return adapter


def _group_message(text: str, *, entities, message_id=42):
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        caption=None,
        entities=entities,
        caption_entities=[],
        message_thread_id=7,
        is_topic_message=True,
        chat=SimpleNamespace(id=-100, type="supergroup", title="Jay Command Center", is_forum=True),
        from_user=SimpleNamespace(id=111, full_name="Andres", first_name="Andres", is_bot=False),
        reply_to_message=None,
        date=None,
    )


def test_canonicalizes_supported_ios_rich_text_entities_with_utf16_offsets():
    text = "@hermes_bot bold italic under strike spoil code pre link quote expand 😎 custom"
    entities = [
        _entity(text, "bold", "bold"),
        _entity(text, "italic", "italic"),
        _entity(text, "underline", "under"),
        _entity(text, "strikethrough", "strike"),
        _entity(text, "spoiler", "spoil"),
        _entity(text, "code", "code"),
        _entity(text, "pre", "pre", language="python"),
        _entity(text, "text_link", "link", url="https://example.com"),
        _entity(text, "blockquote", "quote"),
        _entity(text, "expandable_blockquote", "expand"),
        _entity(text, "custom_emoji", "😎", custom_emoji_id="emoji-1"),
    ]

    msg = _group_message(text, entities=entities)
    event = _make_adapter()._build_message_event(msg, msg_type=__import__("gateway.platforms.base", fromlist=["MessageType"]).MessageType.TEXT, update_id=7001)

    assert event.metadata["telegram_plain_text"] == text
    assert {e["type"] for e in event.metadata["telegram_entities"]} >= {
        "bold", "italic", "underline", "strikethrough", "spoiler", "code", "pre",
        "text_link", "blockquote", "expandable_blockquote", "custom_emoji",
    }
    assert "**bold**" in event.text
    assert "*italic*" in event.text
    assert "<u>under</u>" in event.text
    assert "~~strike~~" in event.text
    assert "||spoil||" in event.text
    assert "`code`" in event.text
    assert "```python\npre\n```" in event.text
    assert "[link](https://example.com)" in event.text
    assert "> quote" in event.text
    assert "> expand" in event.text
    assert "😎" in event.text


def test_unrelated_ios_formatting_entities_do_not_suppress_raw_mention_fallback():
    text = "**client-rendered** @hermes_bot please handle this iOS rich instruction"
    # iOS-style formatted paste may include formatting entities but no mention entity.
    entities = [_entity(text, "bold", "client-rendered")]
    adapter = _make_adapter(require_mention=True)

    assert adapter._message_mentions_bot(_group_message(text, entities=entities)) is True
    assert adapter._should_process_message(_group_message(text, entities=entities)) is True


def test_ios_rich_update_reaches_dispatch_with_canonical_markdown_once():
    async def _run():
        text = "@hermes_bot run bold instruction part 1 " + "x" * 3900
        text2 = "continue with italic part 2"
        msg1 = _group_message(text, entities=[_entity(text, "bold", "bold")], message_id=100)
        msg2 = _group_message(text2, entities=[_entity(text2, "italic", "italic")], message_id=101)
        adapter = _make_adapter(require_mention=True)

        await adapter._handle_text_message(SimpleNamespace(update_id=8001, message=msg1, effective_message=msg1), SimpleNamespace())
        await adapter._handle_text_message(SimpleNamespace(update_id=8002, message=msg2, effective_message=msg2), SimpleNamespace())
        await asyncio.sleep(0.03)

        adapter._message_handler.assert_awaited_once()
        event = adapter._message_handler.await_args.args[0]
        assert "**bold**" in event.text
        assert "*italic*" in event.text
        assert event.text.count("continue with") == 1
        assert event.metadata["telegram_plain_text"].startswith("@hermes_bot run bold")
        assert event.platform_update_id == 8001

    asyncio.run(_run())


def test_sanitized_ios_fixture_reproduces_routing_and_normalization_contract():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "telegram_ios_rich_text_updates.json").read_text())
    ios = fixture["ios_formatted"]
    text = ios["text"]
    entities = [SimpleNamespace(**entity) for entity in ios["entities"]]
    msg = _group_message(text, entities=entities)
    adapter = _make_adapter(require_mention=True)

    assert adapter._should_process_message(msg) is True
    event = adapter._build_message_event(msg, msg_type=__import__("gateway.platforms.base", fromlist=["MessageType"]).MessageType.TEXT, update_id=9001)
    assert event.metadata["telegram_plain_text"] == text
    assert event.metadata["telegram_canonical_text"] == event.text
    assert "**bold**" in event.text
    assert "[link](https://example.com)" in event.text
