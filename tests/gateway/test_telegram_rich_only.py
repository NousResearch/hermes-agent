"""Rich-only Telegram messages (Bot API 10.1 rich_message blocks) must not be
silently dropped — they are flattened to plaintext and dispatched as TEXT."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._running = True
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._held_inbound_events = []
    adapter.HELD_INBOUND_MAX = 64
    adapter.handle_message = AsyncMock()
    return adapter


def _rich_update(text: str = "富文本内容", *, with_text: bool = False):
    blocks = [{"type": "paragraph", "text": [{"type": "plain", "text": text}]}]
    api_kwargs = {"rich_message": {"blocks": blocks}}
    msg = SimpleNamespace(
        text=text if with_text else None,
        caption=None,
        api_kwargs=api_kwargs,
        chat=SimpleNamespace(id=1, type="private", title=None),
        from_user=SimpleNamespace(id=1, full_name="T", is_bot=False),
        message_id=10,
        date=None,
    )
    return SimpleNamespace(update_id=1, message=msg)


@pytest.mark.asyncio
async def test_rich_only_message_recovered_as_text(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda m: True)
    monkeypatch.setattr(adapter, "_should_process_message", lambda m, **k: True)
    monkeypatch.setattr(adapter, "_ensure_forum_commands", AsyncMock())
    monkeypatch.setattr(adapter, "_cache_replied_media", AsyncMock())
    monkeypatch.setattr(adapter, "_apply_telegram_group_observe_attribution", lambda e: e)
    monkeypatch.setattr(adapter, "_clean_bot_trigger_text", lambda t: t)
    monkeypatch.setattr(adapter, "_build_message_event", lambda m, t, update_id=None: SimpleNamespace(text=""))
    enqueued = []
    monkeypatch.setattr(adapter, "_enqueue_text_event", enqueued.append)

    await adapter._handle_rich_only_message(_rich_update(), context=None)

    assert len(enqueued) == 1
    assert "富文本内容" in enqueued[0].text


@pytest.mark.asyncio
async def test_plain_message_not_touched(monkeypatch):
    """A message with text is owned by the group-0 TEXT handler — never enqueue here."""
    adapter = _make_adapter()
    enqueued = []
    monkeypatch.setattr(adapter, "_enqueue_text_event", enqueued.append)

    await adapter._handle_rich_only_message(_rich_update(with_text=True), context=None)

    assert enqueued == []


@pytest.mark.asyncio
async def test_unauthorized_user_content_never_parsed(monkeypatch):
    """Authorize before dispatch: an unauthorized user's rich payload is never
    parsed, let alone enqueued."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda m: False)
    calls = []
    orig = TelegramAdapter._extract_rich_reply_text

    def counting(cls, msg):
        calls.append(msg)
        return orig(msg)

    monkeypatch.setattr(TelegramAdapter, "_extract_rich_reply_text", classmethod(counting))
    enqueued = []
    monkeypatch.setattr(adapter, "_enqueue_text_event", enqueued.append)

    await adapter._handle_rich_only_message(_rich_update(), context=None)

    assert calls == []
    assert enqueued == []


def _nested_list_blocks(depth: int):
    blocks = [{"type": "paragraph", "text": "deep"}]
    for _ in range(depth):
        blocks = [{"type": "list", "items": [{"blocks": blocks}]}]
    return blocks


def test_deeply_nested_blocks_are_bounded():
    """Bounded nesting: hostile nesting depth cannot blow the stack, while
    ordinary nesting still recovers content."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    assert isinstance(TelegramAdapter._flatten_rich_blocks(_nested_list_blocks(60)), str)
    assert "deep" in TelegramAdapter._flatten_rich_blocks(_nested_list_blocks(3))


@pytest.mark.asyncio
async def test_typed_rich_message_attribute_supported(monkeypatch):
    """A typed rich_message attribute (future PTB modeling) is honored the
    same way as the api_kwargs payload."""
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda m: True)
    monkeypatch.setattr(adapter, "_should_process_message", lambda m, **k: True)
    monkeypatch.setattr(adapter, "_ensure_forum_commands", AsyncMock())
    monkeypatch.setattr(adapter, "_cache_replied_media", AsyncMock())
    monkeypatch.setattr(adapter, "_apply_telegram_group_observe_attribution", lambda e: e)
    monkeypatch.setattr(adapter, "_clean_bot_trigger_text", lambda t: t)
    monkeypatch.setattr(adapter, "_build_message_event", lambda m, t, update_id=None: SimpleNamespace(text=""))
    enqueued = []
    monkeypatch.setattr(adapter, "_enqueue_text_event", enqueued.append)

    msg = SimpleNamespace(
        text=None,
        caption=None,
        rich_message=SimpleNamespace(blocks=[{"type": "paragraph", "text": "typed富文本"}]),
        chat=SimpleNamespace(id=1, type="private", title=None),
        from_user=SimpleNamespace(id=1, full_name="T", is_bot=False),
        message_id=10,
        date=None,
    )
    await adapter._handle_rich_only_message(SimpleNamespace(update_id=2, message=msg), context=None)

    assert len(enqueued) == 1
    assert "typed富文本" in enqueued[0].text


@pytest.mark.asyncio
async def test_no_match_logs_field_names_only(monkeypatch, caplog):
    """No recoverable content: nothing is enqueued and the debug log names
    the inspected fields, never message content."""
    import logging

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda m: True)
    enqueued = []
    monkeypatch.setattr(adapter, "_enqueue_text_event", enqueued.append)

    msg = SimpleNamespace(
        text=None,
        caption=None,
        api_kwargs={"some_other_field": 1},
        chat=SimpleNamespace(id=1, type="private", title=None),
        from_user=SimpleNamespace(id=1, full_name="T", is_bot=False),
        message_id=10,
        date=None,
    )
    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.telegram.adapter"):
        await adapter._handle_rich_only_message(SimpleNamespace(update_id=3, message=msg), context=None)

    assert enqueued == []
    debug_records = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "inspected fields" in r.message
    ]
    assert debug_records, "expected a field-name-only debug log"
    logged = debug_records[0].message
    assert "rich_message" in logged
    assert "some_other_field" in logged
