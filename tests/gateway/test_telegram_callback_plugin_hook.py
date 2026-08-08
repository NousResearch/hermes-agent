"""Tests for forwarding otherwise-unhandled Telegram callbacks to plugins."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _RaisingEq:
    def __eq__(self, other):
        raise RuntimeError("malformed action comparison")


class _PostValidationActionDict(dict):
    def __getitem__(self, key):
        if key == "action":
            raise RuntimeError("post-validation action access")
        return super().__getitem__(key)


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_query(
    data: str,
    *,
    message=True,
    user_id=123,
    inline_message_id=None,
):
    if message:
        chat = SimpleNamespace(type="private")
        message_obj = SimpleNamespace(
            chat_id=456,
            chat=chat,
            message_id=789,
            message_thread_id=None,
        )
    else:
        message_obj = None
    return SimpleNamespace(
        data=data,
        message=message_obj,
        inline_message_id=inline_message_id,
        from_user=SimpleNamespace(id=user_id, first_name="Tester", username="tester"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_known_callback_prefix_takes_precedence_over_plugin_hook(monkeypatch):
    adapter = _make_adapter()
    query = _make_query("cp:reasoning:high")
    choice_handler = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_choice_picker_callback", choice_handler)
    hook = MagicMock(return_value=[{"action": "handled", "answer": "plugin"}])
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    choice_handler.assert_awaited_once_with(query, query.data, "456")
    hook.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_callback_forwards_trusted_metadata_and_applies_valid_result(
    monkeypatch,
):
    adapter = _make_adapter()
    query = _make_query("custom:approve:42")
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    captured = {}

    def hook(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return [{
            "action": "handled",
            "answer": "Approved",
            "edit_text": "Approved by plugin",
            "remove_keyboard": True,
        }]

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert captured == {
        "name": "telegram_callback_query",
        "data": "custom:approve:42",
        "platform": "telegram",
        "chat_id": "456",
        "chat_type": "private",
        "thread_id": None,
        "user_id": "123",
        "user_name": "Tester",
        "message_id": "789",
        "inline_message_id": None,
        "authorized": True,
    }
    query.answer.assert_awaited_once_with(text="Approved")
    query.edit_message_text.assert_awaited_once_with(
        text="Approved by plugin", reply_markup=None
    )
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_only_preserves_existing_keyboard(monkeypatch):
    adapter = _make_adapter()
    query = _make_query("custom:edit-only")
    query.message.reply_markup = "existing-keyboard"
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: [{"action": "handled", "edit_text": "Updated"}],
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    query.edit_message_text.assert_awaited_once_with(
        text="Updated", reply_markup="existing-keyboard"
    )


@pytest.mark.asyncio
async def test_inline_edit_only_fails_closed_when_keyboard_cannot_be_preserved(
    monkeypatch,
):
    adapter = _make_adapter()
    query = _make_query("custom:inline-edit", message=False, inline_message_id="inline-1")
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: [{"action": "handled", "edit_text": "Updated"}],
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    query.answer.assert_awaited_once_with()
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [True, False])
async def test_remove_keyboard_only_uses_reply_markup_edit(monkeypatch, message):
    adapter = _make_adapter()
    query = _make_query(
        "custom:remove-only",
        message=message,
        inline_message_id=None if message else "inline-1",
    )
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: [{"action": "handled", "remove_keyboard": True}],
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    query.answer.assert_awaited_once_with()
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        "handled",
        {"action": _RaisingEq()},
        _PostValidationActionDict(action="handled"),
        {"action": "handled", "answer": 123},
        {"action": "handled", "answer": None},
        {"action": "handled", "answer": "\ud800"},
        {"action": "handled", "remove_keyboard": "yes"},
        {"action": "handled", "remove_keyboard": None},
        {"action": "handled", "edit_text": ""},
        {"action": "handled", "edit_text": None},
        {"action": "handled", "edit_text": "\ud800"},
        {"action": "handled", "unexpected": True},
        {"action": "allow", "answer": "nope"},
    ],
)
async def test_malformed_plugin_result_fails_closed(monkeypatch, result):
    adapter = _make_adapter()
    query = _make_query("custom:malformed")
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [result]
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    query.answer.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_hook_exception_is_isolated(monkeypatch):
    adapter = _make_adapter()
    query = _make_query("custom:boom")

    def boom(*args, **kwargs):
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", boom)

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    query.answer.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_inline_callback_exposes_no_message_metadata(monkeypatch):
    adapter = _make_adapter()
    query = _make_query(
        "custom:inline", message=False, user_id=999, inline_message_id="inline-1"
    )
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: False)
    captured = {}

    def hook(name, **kwargs):
        captured.update(kwargs)
        if not kwargs["authorized"]:
            return [{"action": "handled", "answer": "Not authorized"}]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert captured["chat_id"] is None
    assert captured["chat_type"] is None
    assert captured["thread_id"] is None
    assert captured["message_id"] is None
    assert captured["inline_message_id"] == "inline-1"
    assert captured["user_id"] == "999"
    assert captured["authorized"] is False
    query.answer.assert_awaited_once_with(text="Not authorized")
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()
