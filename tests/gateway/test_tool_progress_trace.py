from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.tool_progress_trace import (
    ToolProgressTrace,
    format_tool_progress_call,
    format_tool_progress_detail,
    format_tool_progress_summary,
)


class FakeButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _snapshot_with_three_calls():
    trace = ToolProgressTrace("trace1", max_args_chars=200, max_result_chars=200)
    trace.started("read_file", "adapter.py", {"path": "adapter.py"})
    trace.completed("read_file", duration=0.05, result={"content": "hello"})
    trace.started("terminal", "pytest", {"command": "pytest tests/foo.py"})
    trace.completed("terminal", duration=2.2, result="passed")
    trace.started("web_search", "Hermes docs", {"query": "Hermes Agent docs"})
    return trace.snapshot()


def test_tool_progress_summary_is_inline_for_one_call():
    trace = ToolProgressTrace("trace1")
    trace.started("read_file", "adapter.py", {"path": "adapter.py"})
    snapshot = trace.snapshot()

    summary = format_tool_progress_summary(snapshot, inline_limit=2)

    assert "Tools: 1 call" in summary
    assert "#1" in summary
    assert "read_file" in summary
    assert "adapter.py" in summary


def test_tool_progress_summary_collapses_after_inline_limit():
    snapshot = _snapshot_with_three_calls()

    summary = format_tool_progress_summary(snapshot, inline_limit=2)

    assert "Tools: 3 calls" in summary
    assert "Counts:" in summary
    assert "Sequence: read_file → terminal → web_search" in summary
    assert "Tap Details" in summary
    assert "pytest tests/foo.py" not in summary


def test_tool_progress_detail_and_call_views_include_drilldown_data():
    snapshot = _snapshot_with_three_calls()

    detail = format_tool_progress_detail(snapshot)
    terminal_output = format_tool_progress_call(snapshot, 2, section="output")

    assert "Tool details: 3 calls" in detail
    assert "pytest tests/foo.py" in detail
    assert "Output preview:" in terminal_output
    assert "passed" in terminal_output


def test_tool_progress_zero_detail_budget_suppresses_args_and_output():
    trace = ToolProgressTrace("trace1", max_args_chars=0, max_result_chars=0)
    trace.started("terminal", "run secret command", {"command": "echo SECRET"})
    trace.completed("terminal", duration=0.1, result="SECRET output", call_id=None)
    snapshot = trace.snapshot()
    call = snapshot["calls"][0]

    assert call["args_preview"] == ""
    assert call["result_preview"] == ""
    assert "SECRET" not in format_tool_progress_call(snapshot, 1)


def test_tool_progress_completion_uses_call_id_for_same_named_tools():
    trace = ToolProgressTrace("trace1")
    trace.started("terminal", args={"command": "first"}, call_id="call-1")
    trace.started("terminal", args={"command": "second"}, call_id="call-2")
    trace.completed("terminal", result="second result", call_id="call-2")
    trace.completed("terminal", result="first result", call_id="call-1")
    snapshot = trace.snapshot()

    assert snapshot["calls"][0]["args_preview"].find("first") >= 0
    assert "first result" in snapshot["calls"][0]["result_preview"]
    assert snapshot["calls"][1]["args_preview"].find("second") >= 0
    assert "second result" in snapshot["calls"][1]["result_preview"]


@pytest.mark.asyncio
async def test_telegram_tool_progress_send_registers_keyboard(monkeypatch):
    from gateway.config import PlatformConfig
    import plugins.platforms.telegram.adapter as telegram_adapter_mod

    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardMarkup", FakeMarkup)
    monkeypatch.setattr(telegram_adapter_mod, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    monkeypatch.setattr(telegram_adapter_mod, "TELEGRAM_AVAILABLE", True)

    adapter = telegram_adapter_mod.TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._send_message_with_thread_fallback = AsyncMock(
        return_value=SimpleNamespace(message_id=123)
    )
    snapshot = _snapshot_with_three_calls()
    content = format_tool_progress_summary(snapshot, inline_limit=2)

    result = await adapter.send_tool_progress_message("42", content, tool_trace=snapshot)

    assert result.success is True
    assert result.message_id == "123"
    kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
    markup = kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].text == "Подробнее"
    assert markup.inline_keyboard[0][0].callback_data == "tp:trace1:d"
    assert kwargs["disable_notification"] is True
    assert [button.callback_data for button in markup.inline_keyboard[1]] == [
        "tp:trace1:c:1",
        "tp:trace1:c:2",
        "tp:trace1:c:3",
    ]
    assert adapter._tool_progress_state["trace1"]["total"] == snapshot["total"]
    assert adapter._tool_progress_state["trace1"]["telegram_chat_id"] == "42"
    assert adapter._tool_progress_state["trace1"]["telegram_message_id"] == "123"


@pytest.mark.asyncio
async def test_telegram_tool_progress_callback_expands_details(monkeypatch):
    from gateway.config import PlatformConfig
    import plugins.platforms.telegram.adapter as telegram_adapter_mod

    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardMarkup", FakeMarkup)
    monkeypatch.setattr(telegram_adapter_mod, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    monkeypatch.setattr(telegram_adapter_mod, "TELEGRAM_AVAILABLE", True)

    adapter = telegram_adapter_mod.TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._tool_progress_state["trace1"] = _snapshot_with_three_calls()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)

    query = SimpleNamespace(
        from_user=SimpleNamespace(id=7, first_name="Mikhail"),
        message=SimpleNamespace(
            chat_id=42,
            message_thread_id=99,
            chat=SimpleNamespace(type="supergroup"),
        ),
        edit_message_text=AsyncMock(),
        answer=AsyncMock(),
    )

    await adapter._handle_tool_progress_callback(query, "tp:trace1:d")

    query.edit_message_text.assert_awaited_once()
    kwargs = query.edit_message_text.await_args.kwargs
    assert "Tool details" in kwargs["text"]
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "tp:trace1:s"
    query.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_tool_progress_callback_rejects_wrong_message(monkeypatch):
    from gateway.config import PlatformConfig
    import plugins.platforms.telegram.adapter as telegram_adapter_mod

    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardMarkup", FakeMarkup)
    monkeypatch.setattr(telegram_adapter_mod, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    monkeypatch.setattr(telegram_adapter_mod, "TELEGRAM_AVAILABLE", True)

    adapter = telegram_adapter_mod.TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    snapshot = _snapshot_with_three_calls()
    snapshot["telegram_chat_id"] = "42"
    snapshot["telegram_message_id"] = "123"
    adapter._tool_progress_state["trace1"] = snapshot
    adapter._is_callback_user_authorized = MagicMock(return_value=True)

    query = SimpleNamespace(
        from_user=SimpleNamespace(id=7, first_name="Mikhail"),
        message=SimpleNamespace(
            chat_id=42,
            message_id=999,
            message_thread_id=99,
            chat=SimpleNamespace(type="supergroup"),
        ),
        edit_message_text=AsyncMock(),
        answer=AsyncMock(),
    )

    await adapter._handle_tool_progress_callback(query, "tp:trace1:d")

    query.edit_message_text.assert_not_awaited()
    query.answer.assert_awaited_once_with(text="Tool details expired.")


@pytest.mark.asyncio
async def test_telegram_tool_progress_send_does_not_plain_retry_timeout(monkeypatch):
    from gateway.config import PlatformConfig
    import plugins.platforms.telegram.adapter as telegram_adapter_mod

    class TimedOut(Exception):
        pass

    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter_mod, "InlineKeyboardMarkup", FakeMarkup)
    monkeypatch.setattr(telegram_adapter_mod, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    monkeypatch.setattr(telegram_adapter_mod, "TELEGRAM_AVAILABLE", True)

    adapter = telegram_adapter_mod.TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._send_message_with_thread_fallback = AsyncMock(side_effect=TimedOut("timed out"))
    snapshot = _snapshot_with_three_calls()
    content = format_tool_progress_summary(snapshot, inline_limit=2)

    result = await adapter.send_tool_progress_message("42", content, tool_trace=snapshot)

    assert result.success is False
    assert result.retryable is False
    assert adapter._send_message_with_thread_fallback.await_count == 1
