"""Native previews remain prefix-stable and share Telegram's typing-action limits."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from plugins.platforms.telegram import adapter as tg_mod
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def draft_adapter(monkeypatch):
    now = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(
        tg_mod.asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: now.value),
    )

    async def await_request(awaitable, **kwargs):
        return await awaitable

    monkeypatch.setattr(tg_mod, "_await_with_thread_deadline", await_request)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.send_chat_action = AsyncMock()
    return adapter, now


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["```python\nx = ", "😀" * 2100], ids=["open-code-fence", "astral-limit"])
async def test_plain_draft_keeps_original_prefix_within_utf16_limit(draft_adapter, content):
    adapter, _ = draft_adapter
    adapter.format_message = lambda text: f"changed::{text}"

    result = await adapter.send_draft("123", 7, content)

    kwargs = adapter._bot.send_message_draft.call_args.kwargs
    assert result.success
    assert "parse_mode" not in kwargs
    assert content.startswith(kwargs["text"])
    assert utf16_len(kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH
    assert kwargs["text"] == ("😀" * 2048 if content.startswith("😀") else content)


@pytest.mark.asyncio
async def test_drafts_and_typing_share_both_windows_for_normalized_peer(draft_adapter):
    adapter, now = draft_adapter
    for _ in range(10):
        await adapter.send_typing("00123")
    for index in range(10):
        await adapter.send_draft("123", 7, f"first {index}")

    skipped = await adapter.send_draft("00123", 7, "latest first batch")
    assert skipped.success and skipped.raw_response["skipped"]
    assert adapter._bot.send_message_draft.await_count == 10
    assert adapter._bot.send_chat_action.await_count == 10

    now.value += 5.01
    for index in range(20):
        await adapter.send_draft("123", 7, f"second {index}")
    skipped = await adapter.send_draft("123", 7, "latest second batch")
    assert skipped.success and skipped.raw_response["skipped"]
    assert adapter._bot.send_message_draft.await_count == 30

    other_peer = await adapter.send_draft("124", 8, "independent chat")
    assert other_peer.success and not other_peer.raw_response
    now.value = 1030.01
    latest = await adapter.send_draft("00123", 7, "latest, not a queued older frame")
    assert latest.success and not latest.raw_response
    assert adapter._bot.send_message_draft.call_args.kwargs["text"] == "latest, not a queued older frame"


@pytest.mark.asyncio
@pytest.mark.parametrize("rich", [False, True])
async def test_retry_after_skips_preview_and_typing_then_resumes_native_draft(draft_adapter, rich):
    adapter, now = draft_adapter
    error = RuntimeError("Flood control exceeded")
    error.retry_after = timedelta(seconds=3)
    draft_api = adapter._bot.send_message_draft
    if rich:
        adapter._rich_messages_enabled = adapter._rich_drafts_enabled = True
        adapter._allow_cjk_rich_messages = True
        adapter._rich_content_ok = lambda content: True
        draft_api = adapter._bot.do_api_request = AsyncMock()
    draft_api.side_effect = [error, True]

    rejected = await adapter.send_draft("123", 7, "first preview")
    assert rejected.success and rejected.raw_response["skipped"]
    assert rejected.retry_after == pytest.approx(3)
    assert draft_api.await_count == 1
    if rich:
        adapter._bot.send_message_draft.assert_not_awaited()

    await adapter.send_typing("00123")
    during_cooldown = await adapter.send_draft("00123", 7, "newer preview")
    assert during_cooldown.success and during_cooldown.raw_response["skipped"]
    adapter._bot.send_chat_action.assert_not_awaited()
    assert draft_api.await_count == 1

    now.value += 3.01
    resumed = await adapter.send_draft("123", 7, "latest preview")
    assert resumed.success and not resumed.raw_response
    assert draft_api.await_count == 2
    assert not adapter._rich_draft_disabled


@pytest.mark.asyncio
async def test_recent_native_draft_replaces_redundant_typing_refresh(draft_adapter):
    adapter, now = draft_adapter
    await adapter.send_draft("123", 7, "working preview")
    await adapter.send_typing("00123")
    adapter._bot.send_chat_action.assert_not_awaited()

    now.value += 4.01
    await adapter.send_typing("123")
    adapter._bot.send_chat_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_unsupported_draft_still_reports_failure_for_edit_fallback(draft_adapter):
    adapter, _ = draft_adapter
    adapter._bot.send_message_draft.side_effect = RuntimeError("method not found")

    result = await adapter.send_draft("123", 7, "preview")

    assert not result.success
    assert "method not found" in result.error
