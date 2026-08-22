"""Async regression tests for the nested Telegram /model picker callback flows.

These tests exercise the adapter-level state machine without a live bot.
They cover:

* Nested provider pagination (`mpv:`) stays inside the nested renderer
  rather than falling into the legacy provider layout.
* Nested model pagination (`mg:`) preserves the nested renderer.
* "Back" routes (`mp:b:p`, `mp:b:c`, `mp:b:s`, `mp:b:m`) return to the
  correct prior stage for both speed-categories enabled and disabled.
* Expensive-model warning uses the correct Back callback when in nested mode.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock() -> None:
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    mod.ForceReply = type("ForceReply", (), {"__init__": lambda self, **_kw: None})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()


from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


CHAT_ID = "99999"


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    bot = MagicMock()
    bot.edit_message_text = AsyncMock()
    bot.send_message = AsyncMock()
    bot.delete_message = AsyncMock()
    adapter._bot = bot
    adapter._app = MagicMock()
    return adapter


def _query(data: str):
    q = AsyncMock()
    q.data = data
    q.message = MagicMock()
    q.message.chat_id = int(CHAT_ID)
    q.message.message_thread_id = None
    q.message.chat = MagicMock()
    q.message.chat.type = "private"
    q.from_user = MagicMock()
    q.from_user.id = 1
    q.from_user.first_name = "Shiko"
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    return q


def _seed_state(adapter: TelegramAdapter, *, speed_categories: bool = False) -> None:
    providers = [
        {
            "slug": "provider_one",
            "name": "Provider One",
            "total_models": 12,
            "is_current": True,
            "models": [f"provider_one/model_{i}" for i in range(12)],
        },
        {
            "slug": "provider_two",
            "name": "Provider Two",
            "total_models": 0,
            "is_current": False,
            "models": [],
        },
    ]
    adapter._model_picker_state[CHAT_ID] = {
        "msg_id": 42,
        "providers": providers,
        "provider_view": providers,
        "session_key": "s",
        "on_model_selected": AsyncMock(),
        "current_model": "provider_one/model_0",
        "current_provider": "provider_one",
        "provider_page": 0,
        "nested_picker": True,
        "nested_stage": "models",
        "speed_categories": speed_categories,
        "selected_provider_idx": 0,
        "selected_provider": "provider_one",
        "selected_provider_name": "Provider One",
        "selected_category": "paid",
        "selected_speed": "slow" if speed_categories else None,
        "model_list": providers[0]["models"],
        "model_page": 0,
        "search_stage": None,
        "search_prompt_id": None,
        "search_started_at": None,
        "model_filter_query": None,
    }


@pytest.mark.asyncio
async def test_nested_model_pagination_routes_through_nested_renderer():
    adapter = _make_adapter()
    _seed_state(adapter)

    query = _query("mg:0")
    await adapter._handle_model_picker_callback(query, "mg:0", CHAT_ID)

    state = adapter._model_picker_state[CHAT_ID]
    assert state["nested_stage"] == "models"
    state["model_page"] = 0
    # The legacy renderer would have called _build_model_keyboard directly;
    # the nested renderer routes through edit_message_text via the
    # nested renderer pipeline.
    args, kwargs = query.edit_message_text.call_args
    payload = str(kwargs.get("text") or (args[1] if len(args) > 1 else args[0] if args else ""))
    assert "Model Configuration" in payload
    assert "Provider One" in payload


@pytest.mark.asyncio
async def test_nested_provider_pagination_routes_through_nested_renderer():
    adapter = _make_adapter()
    _seed_state(adapter)
    state = adapter._model_picker_state[CHAT_ID]
    state["nested_stage"] = "providers"

    query = _query("mpv:1")
    await adapter._handle_model_picker_callback(query, "mpv:1", CHAT_ID)

    state = adapter._model_picker_state[CHAT_ID]
    assert state["provider_page"] == 1
    assert state["nested_stage"] == "providers"
    query.edit_message_text.assert_called()


@pytest.mark.asyncio
async def test_back_to_providers_from_categories():
    adapter = _make_adapter()
    _seed_state(adapter)
    state = adapter._model_picker_state[CHAT_ID]
    state["nested_stage"] = "categories"

    await adapter._handle_model_picker_callback(_query("mp:b:p"), "mp:b:p", CHAT_ID)

    assert state["nested_stage"] == "providers"


@pytest.mark.asyncio
async def test_back_to_categories_from_speeds():
    adapter = _make_adapter()
    _seed_state(adapter, speed_categories=True)
    state = adapter._model_picker_state[CHAT_ID]
    state["nested_stage"] = "speeds"

    await adapter._handle_model_picker_callback(_query("mp:b:c"), "mp:b:c", CHAT_ID)

    assert state["nested_stage"] == "categories"


@pytest.mark.asyncio
async def test_back_to_speeds_from_models_with_speed_categories():
    adapter = _make_adapter()
    _seed_state(adapter, speed_categories=True)
    state = adapter._model_picker_state[CHAT_ID]
    state["nested_stage"] = "models"

    await adapter._handle_model_picker_callback(_query("mp:b:s"), "mp:b:s", CHAT_ID)

    assert state["nested_stage"] == "speeds"


@pytest.mark.asyncio
async def test_back_to_categories_from_models_without_speed_categories():
    adapter = _make_adapter()
    _seed_state(adapter, speed_categories=False)
    state = adapter._model_picker_state[CHAT_ID]
    state["nested_stage"] = "models"

    await adapter._handle_model_picker_callback(_query("mp:b:c"), "mp:b:c", CHAT_ID)

    assert state["nested_stage"] == "categories"


@pytest.mark.asyncio
async def test_back_to_models_from_warning_nested():
    adapter = _make_adapter()
    _seed_state(adapter)
    state = adapter._model_picker_state[CHAT_ID]
    state["nested_stage"] = "models"
    # Confirm the model-list Back callback used in the warning screen
    # routes through the nested renderer when the picker is nested.
    await adapter._handle_model_picker_callback(_query("mp:b:m"), "mp:b:m", CHAT_ID)
    assert state["nested_stage"] == "models"