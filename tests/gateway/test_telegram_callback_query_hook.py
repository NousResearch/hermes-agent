"""Tests for the pre_callback_query_dispatch plugin hook on TelegramAdapter.

Callback queries (inline keyboard button clicks) are handled by
_handle_callback_query.  The pre_callback_query_dispatch hook fires at the
top of that method so plugins can intercept and claim callbacks before any
built-in prefix handling runs.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._fire_plugin_hook = AsyncMock(return_value=None)
    return adapter


def _make_update(data: str = "nf:yes:42", chat_id: int = -100999, message_id: int = 1,
                 user_id: int = 123):
    msg = SimpleNamespace(
        chat_id=chat_id,
        chat=SimpleNamespace(id=chat_id, type="supergroup"),
        message_id=message_id,
        message_thread_id=None,
    )
    query = SimpleNamespace(
        data=data,
        message=msg,
        from_user=SimpleNamespace(id=user_id, first_name="Tester"),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )
    return SimpleNamespace(update_id=99, callback_query=query)


# ── hook fires with correct kwargs ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_given_callback_query_when_handle_called_then_hook_fires_with_correct_kwargs():
    # Arrange
    adapter = _make_adapter()
    update = _make_update(data="nf:yes:42", chat_id=-100999, message_id=7, user_id=123)

    with patch.object(adapter, "_handle_model_picker_callback", new_callable=AsyncMock):
        # Act
        await adapter._handle_callback_query(update, context=None)

    # Assert
    adapter._fire_plugin_hook.assert_awaited_once_with(
        "pre_callback_query_dispatch",
        data="nf:yes:42",
        chat_id="-100999",
        user_id="123",
        message_id="7",
        raw_query=update.callback_query,
    )


@pytest.mark.asyncio
async def test_given_hook_returns_skip_when_handle_called_then_built_in_handling_suppressed():
    # Arrange
    adapter = _make_adapter()
    adapter._fire_plugin_hook = AsyncMock(return_value={"action": "skip"})
    update = _make_update(data="nf:yes:42")
    handled = []

    with patch.object(adapter, "_handle_model_picker_callback",
                      new_callable=AsyncMock,
                      side_effect=lambda *a, **kw: handled.append("model_picker")):
        # Act
        await adapter._handle_callback_query(update, context=None)

    # Assert — built-in branches never reached
    assert handled == []
    update.callback_query.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_given_hook_returns_allow_when_handle_called_then_built_in_handling_continues():
    # Arrange
    adapter = _make_adapter()
    adapter._fire_plugin_hook = AsyncMock(return_value={"action": "allow"})
    update = _make_update(data="mp:some:model")
    model_picker_called = []

    with patch.object(adapter, "_handle_model_picker_callback",
                      new_callable=AsyncMock,
                      side_effect=lambda *a, **kw: model_picker_called.append(True)):
        # Act
        await adapter._handle_callback_query(update, context=None)

    # Assert — model picker branch was reached
    assert model_picker_called == [True]


@pytest.mark.asyncio
async def test_given_hook_returns_none_when_handle_called_then_built_in_handling_continues():
    # Arrange
    adapter = _make_adapter()
    adapter._fire_plugin_hook = AsyncMock(return_value=None)
    update = _make_update(data="mp:some:model")
    model_picker_called = []

    with patch.object(adapter, "_handle_model_picker_callback",
                      new_callable=AsyncMock,
                      side_effect=lambda *a, **kw: model_picker_called.append(True)):
        # Act
        await adapter._handle_callback_query(update, context=None)

    # Assert — model picker branch was reached
    assert model_picker_called == [True]


@pytest.mark.asyncio
async def test_given_no_query_data_when_handle_called_then_hook_not_fired():
    # Arrange
    adapter = _make_adapter()
    update = SimpleNamespace(
        update_id=99,
        callback_query=SimpleNamespace(data=None, message=None, from_user=None),
    )

    # Act
    await adapter._handle_callback_query(update, context=None)

    # Assert — early-exit before hook
    adapter._fire_plugin_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_given_no_callback_query_when_handle_called_then_hook_not_fired():
    # Arrange
    adapter = _make_adapter()
    update = SimpleNamespace(update_id=99, callback_query=None)

    # Act
    await adapter._handle_callback_query(update, context=None)

    # Assert
    adapter._fire_plugin_hook.assert_not_awaited()


# ── VALID_HOOKS registration ──────────────────────────────────────────────────


def test_pre_callback_query_dispatch_is_in_valid_hooks():
    # Arrange / Act
    from hermes_cli.plugins import VALID_HOOKS

    # Assert
    assert "pre_callback_query_dispatch" in VALID_HOOKS
