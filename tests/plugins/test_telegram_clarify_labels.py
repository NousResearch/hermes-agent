"""Telegram clarify captions are presentation only; bodies and answers stay lossless."""

import html
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import hermes_yaml as yaml
from wcwidth import wcswidth

from gateway.config import Platform, PlatformConfig, load_gateway_config
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config
from plugins.platforms.telegram.telegram_clarify_binding import ClarifyPrompt
from tools import clarify_gateway


@pytest.fixture(autouse=True)
def clear_pending():
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        clarify_gateway._notify_cbs.clear()
    yield
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        clarify_gateway._notify_cbs.clear()


def make_adapter(extra=None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter._bot.send_message.return_value = SimpleNamespace(
        chat_id=12345, message_id=100, message_thread_id=42, text="Pick", chat=SimpleNamespace(type="private"))
    return adapter


async def render(adapter, choices, *, multi=False):
    if choices:
        clarify_gateway.register("caption", "session", "Pick <&>", choices, multi_select=multi)
    result = await adapter.send_clarify(
        "12345", "Pick <&>", choices, "caption", "session", metadata={"thread_id": "42"})
    assert result.success
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["parse_mode"] == "HTML"
    assert kwargs["message_thread_id"] == 42
    expected = "❓ Pick &lt;&amp;&gt;"
    if choices:
        expected += "\n\n" + "\n".join(f"{i + 1}. {html.escape(str(c))}" for i, c in enumerate(choices))
    if multi:
        assert kwargs["text"].startswith(expected + "\n\n<i>")
    else:
        assert kwargs["text"] == expected
    assert adapter._clarify_state["caption"] == ClarifyPrompt("session", "12345", "100", "42")
    return kwargs.get("reply_markup")


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{}, {"clarify_button_labels": "numbers"},
    {"clarify_button_labels": "bogus"}, {"clarify_button_labels": None},
    {"clarify_button_labels": []}, {"clarify_button_labels": True}])
async def test_default_and_invalid_modes_keep_numeric_buttons(extra):
    markup = await render(make_adapter(extra), ["Да", "界" * 35])
    assert [r[0].text for r in markup.inline_keyboard[:-1]] == ["1", "2"]
    assert [r[0].callback_data for r in markup.inline_keyboard] == [
        "cl:caption:0", "cl:caption:1", "cl:caption:other"]


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,expected", [
    ("a" * 28, "a" * 28), ("a" * 29, "1"),
    ("я" * 28, "я" * 28), ("я" * 29, "1"),
    ("界" * 14, "界" * 14), ("界" * 15, "1"),
    ("🙂" * 14, "🙂" * 14), ("🙂" * 15, "1"),
    ("e\u0301" * 28, "e\u0301" * 28),
    ("👩‍💻" * 14, "👩‍💻" * 14),
    ("🏴\U000e0067\U000e0062\U000e007f", "🏴\U000e0067\U000e0062\U000e007f"),
    ("🇷🇺" * 14, "🇷🇺" * 14),
    ("", "1"), (" \n\t", "1"), ("\u200b\u202e", "1"),
    ("yes\n\tplease\x00", "yes please"),
    ("<b>& yes", "<b>& yes"),
    ("Yes (Recommended)", "Yes (Recommended)"),
])
async def test_auto_uses_display_columns_per_choice(choice, expected):
    markup = await render(make_adapter({"clarify_button_labels": "auto"}), [choice, "Short", "x" * 29])
    assert [r[0].text for r in markup.inline_keyboard[:-1]] == [expected, "Short", "3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["x" * 120, "界" * 40, "🙂" * 40, "👩‍💻" * 40, "e\u0301" * 100])
async def test_text_caption_is_bounded_without_losing_body(choice):
    markup = await render(make_adapter({"clarify_button_labels": "text"}), [choice])
    caption = markup.inline_keyboard[0][0].text
    assert caption.endswith("…")
    assert choice.startswith(caption[:-1])
    assert 0 < wcswidth(caption) <= 60
    assert len(caption) <= 64
    assert not caption[:-1].endswith("\u200d")
    assert markup.inline_keyboard[0][0].callback_data == "cl:caption:0"


@pytest.mark.asyncio
async def test_text_sanitizes_controls_and_falls_back_for_empty_caption():
    markup = await render(make_adapter({"clarify_button_labels": "text"}),
                          ["<b>& yes\n\tplease\x00\u202e", "", "\ud800"])
    assert [r[0].text for r in markup.inline_keyboard[:-1]] == ["<b>& yes please", "2", "3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("choices", [None, []])
@pytest.mark.parametrize("mode", ["numbers", "text", "auto"])
async def test_open_ended_prompt_has_no_keyboard(mode, choices):
    assert await render(make_adapter({"clarify_button_labels": mode}), choices) is None


async def tap(adapter, token, monkeypatch, *, allowed="*"):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", allowed)
    query = AsyncMock()
    query.data = f"cl:caption:{token}"
    query.message = adapter._bot.send_message.return_value
    query.from_user = SimpleNamespace(id=777, first_name="Tester")
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
    return query


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["numbers", "text", "auto"])
async def test_caption_does_not_change_resolved_answer(mode, monkeypatch):
    original = "  <long & original> " + "界" * 70 + " (Recommended)"
    adapter = make_adapter({"clarify_button_labels": mode})
    await render(adapter, [original, "No"])
    entry = clarify_gateway.get_pending("caption")
    assert entry is not None
    await tap(adapter, "0", monkeypatch)
    assert entry.response == original
    assert "caption" not in adapter._clarify_state


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["numbers", "text", "auto"])
async def test_caption_does_not_bypass_authorization_or_other_text_capture(mode, monkeypatch):
    adapter = make_adapter({"clarify_button_labels": mode})
    await render(adapter, ["Yes (Recommended)", "No"])
    entry = clarify_gateway.get_pending("caption")
    assert entry is not None
    await tap(adapter, "0", monkeypatch, allowed="888")
    assert entry.response is None
    assert not entry.event.is_set()
    assert "caption" in adapter._clarify_state
    await tap(adapter, "other", monkeypatch)
    assert entry.awaiting_text
    assert entry.response is None
    assert not entry.event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("numbers", ["☐ 1", "☐ 2"]),
    ("text", ["☐ Yes", "☐ " + "x" * 29]), ("auto", ["☐ Yes", "☐ 2"])])
async def test_multi_select_captions_survive_toggle_and_send(mode, expected, monkeypatch):
    choices = ["Yes", "x" * 29]
    adapter = make_adapter({"clarify_button_labels": mode})
    markup = await render(adapter, choices, multi=True)
    assert [r[0].text for r in markup.inline_keyboard[:2]] == expected
    assert [r[0].callback_data for r in markup.inline_keyboard] == [
        "cl:caption:t0", "cl:caption:t1", "cl:caption:other", "cl:caption:go"]
    entry = clarify_gateway.get_pending("caption")
    assert entry is not None
    query = await tap(adapter, "t0", monkeypatch)
    updated = query.edit_message_reply_markup.call_args.kwargs["reply_markup"]
    assert updated.inline_keyboard[0][0].text == expected[0].replace("☐", "☑")
    assert entry.response is None
    await tap(adapter, "go", monkeypatch)
    assert entry.response is not None
    assert json.loads(entry.response) == ["Yes"]


@pytest.mark.parametrize("mode", ["numbers", "text", "auto", "invalid", None, []])
def test_yaml_hook_bridges_convenience_key_without_environment(mode, monkeypatch):
    monkeypatch.delenv("TELEGRAM_CLARIFY_BUTTON_LABELS", raising=False)
    assert _apply_yaml_config({}, {"clarify_button_labels": mode})["clarify_button_labels"] == mode
    import os
    assert "TELEGRAM_CLARIFY_BUTTON_LABELS" not in os.environ


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["canonical", "convenience", "nested_convenience"])
async def test_yaml_to_real_adapter_end_to_end(form, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    data = {"platforms": {"telegram": {"enabled": True, "token": "test-token"}}}
    if form == "canonical":
        data["platforms"]["telegram"]["extra"] = {"clarify_button_labels": "auto"}
    elif form == "convenience":
        data["telegram"] = {"clarify_button_labels": "auto"}
    else:
        data["telegram"] = {"extra": {"clarify_button_labels": "auto"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_gateway_config(strict=True)
    adapter = make_adapter(config.platforms[Platform.TELEGRAM].extra)
    markup = await render(adapter, ["Да", "界" * 15])
    assert [r[0].text for r in markup.inline_keyboard[:-1]] == ["Да", "2"]
