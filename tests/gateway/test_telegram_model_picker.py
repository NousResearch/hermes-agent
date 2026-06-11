"""Tests for Telegram model picker thread fallback."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock():
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

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}
            ],
            current_model="model_1",
            current_provider="provider_one",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "provider\\_one" in sent["text"]
        assert "`model_1`" in sent["text"]

    @pytest.mark.asyncio
    async def test_back_button_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}],
            "current_model": "model_1",
            "current_provider": "provider_one",
            "session_key": "s",
            "on_model_selected": AsyncMock(),
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mb"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mb", "12345")

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "provider\\_one" in edit_kwargs["text"]
        assert "`model_1`" in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_model_selected_edits_message_on_success(self):
        """Regression: the mm: (model selected → switch) success path must
        edit the picker message to show the confirmation and remove the
        buttons.  An earlier revision of this PR over-indented the
        edit_message_text block so it lived inside the except branch and
        only fired when the callback raised."""
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Switched to `gpt-5`")
        adapter._model_picker_state["12345"] = {
            "providers": [
                {"slug": "openai", "name": "OpenAI", "total_models": 1, "is_current": True}
            ],
            "current_model": "model_1",
            "current_provider": "openai",
            "session_key": "s",
            "on_model_selected": callback,
            "selected_provider": "openai",
            "model_list": ["gpt-5"],
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mm:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        callback.assert_awaited_once()
        query.edit_message_text.assert_awaited()
        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "`gpt-5`" in edit_kwargs["text"]
        assert "12345" not in adapter._model_picker_state

    @pytest.mark.parametrize(
        ("callback_message_id", "callback_thread_id"),
        [
            (99, 111),
            (42, 222),
        ],
    )
    @pytest.mark.asyncio
    async def test_model_picker_rejects_stale_message_or_thread_callback(
        self,
        callback_message_id,
        callback_thread_id,
        monkeypatch,
    ):
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Switched to `gemini-3.5-flash`")
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: None,
        )

        async def mock_send_message(**kwargs):
            return SimpleNamespace(message_id=42)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {
                    "slug": "gemini",
                    "name": "Google",
                    "models": ["gemini-3.5-flash"],
                    "total_models": 1,
                }
            ],
            current_model="gpt-5.5",
            current_provider="openai-codex",
            session_key="agent:main:telegram:dm:12345:111",
            on_model_selected=callback,
            metadata={"thread_id": "111"},
        )
        assert result.success is True

        state = next(iter(adapter._model_picker_state.values()))
        state["selected_provider"] = "gemini"
        state["model_list"] = ["gemini-3.5-flash"]

        query = AsyncMock()
        query.data = "mm:0"
        query.message = SimpleNamespace(
            chat_id=12345,
            message_id=callback_message_id,
            message_thread_id=callback_thread_id,
        )
        query.from_user = SimpleNamespace(first_name="Sam")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        callback.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_provider_group_folds_and_drills_down(self, monkeypatch):
        """A provider family (e.g. MiniMax) collapses to one mpg: button at
        the top level; tapping it expands to its authenticated members as
        mp: buttons. A group reduced to a single authenticated member shows
        no submenu (direct mp: button).

        Inspects callback_data by recording every InlineKeyboardButton built,
        which is robust to whether `telegram` is the real SDK or the module
        mock (the SDK markup objects don't expose a plain iterable under the
        mock)."""
        import plugins.platforms.telegram.adapter as tg

        built: list = []

        class _RecordingButton:
            def __init__(self, text, callback_data=None, **kw):
                self.text = text
                self.callback_data = callback_data
                built.append(callback_data)

        class _RecordingMarkup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(tg, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(tg, "InlineKeyboardMarkup", _RecordingMarkup)

        adapter = _make_adapter()

        async def mock_send_message(**kwargs):
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        providers = [
            {"slug": "minimax", "name": "MiniMax", "total_models": 2},
            {"slug": "minimax-cn", "name": "MiniMax (China)", "total_models": 3},
            {"slug": "xai", "name": "xAI", "total_models": 1},
        ]

        await adapter.send_model_picker(
            chat_id="12345",
            providers=providers,
            current_model="m",
            current_provider="minimax",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata=None,
        )

        assert "mpg:minimax" in built
        assert "mp:xai" in built
        assert "mp:minimax" not in built
        assert "mp:minimax-cn" not in built

        built.clear()
        query = AsyncMock()
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mpg:minimax", "12345")

        assert "mp:minimax" in built
        assert "mp:minimax-cn" in built
        assert "mb" in built

    @pytest.mark.asyncio
    async def test_expensive_model_requires_confirmation(self, monkeypatch):
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Switched to `openai/gpt-5.5-pro`")
        adapter._model_picker_state["12345"] = {
            "providers": [
                {"slug": "openrouter", "name": "OpenRouter", "total_models": 1, "is_current": True}
            ],
            "current_model": "model_1",
            "current_provider": "openrouter",
            "session_key": "s",
            "on_model_selected": callback,
            "selected_provider": "openrouter",
            "model_list": ["openai/gpt-5.5-pro"],
            "msg_id": 42,
        }
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: SimpleNamespace(
                message="!!! EXPENSIVE MODEL WARNING !!!\ndid you mean to select openai/gpt-5.5?"
            ),
        )

        query = AsyncMock()
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        callback.assert_not_awaited()
        assert "12345" in adapter._model_picker_state
        first_edit = query.edit_message_text.call_args[1]
        assert "EXPENSIVE MODEL WARNING" in first_edit["text"]
        assert first_edit["reply_markup"] is not None

        await adapter._handle_model_picker_callback(query, "mc:0", "12345")

        callback.assert_awaited_once_with("12345", "openai/gpt-5.5-pro", "openrouter")
        assert "12345" not in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_retries_without_thread_when_thread_not_found(self):
        adapter = _make_adapter()
        providers = [{"slug": "openai", "name": "OpenAI", "total_models": 2, "is_current": True}]
        call_log = []

        class FakeBadRequest(Exception):
            pass

        async def mock_send_message(**kwargs):
            call_log.append(dict(kwargs))
            if kwargs.get("message_thread_id") is not None:
                raise FakeBadRequest("Message thread not found")
            return SimpleNamespace(message_id=99)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=providers,
            current_model="gpt-5",
            current_provider="openai",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert len(call_log) == 2
        assert call_log[0]["message_thread_id"] == 99999
        assert "message_thread_id" not in call_log[1] or call_log[1]["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_matching_message_and_thread_callback_switches_model(self, monkeypatch):
        """Positive path: a callback whose msg_id AND thread match the picker
        passes ``_model_picker_callback_matches_state`` and performs the switch.

        The guard's happy path with a real (non-General) thread was previously
        unverified — pre-PR tests exercised only the bare-chat key with no
        thread, so the ``state_thread_id == query_thread_id`` arm never matched
        a truthy thread.  Here the picker is opened in topic ``222`` and the
        callback arrives on the same message in the same topic."""
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Switched to `gemini-3.5-flash`")
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: None,
        )

        async def mock_send_message(**kwargs):
            return SimpleNamespace(message_id=42)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {
                    "slug": "gemini",
                    "name": "Google",
                    "models": ["gemini-3.5-flash"],
                    "total_models": 1,
                }
            ],
            current_model="gpt-5.5",
            current_provider="openai-codex",
            session_key="agent:main:telegram:group:12345:222",
            on_model_selected=callback,
            metadata={"thread_id": "222"},
        )
        assert result.success is True

        # The message-keyed entry carries the real thread; confirm the guard
        # is actually exercised against a truthy thread (not General -> None).
        state = adapter._model_picker_state["12345:42"]
        assert state["thread_id"] == "222"
        state["selected_provider"] = "gemini"
        state["model_list"] = ["gemini-3.5-flash"]

        query = AsyncMock()
        query.data = "mm:0"
        query.message = SimpleNamespace(
            chat_id=12345,
            message_id=42,
            message_thread_id=222,
        )
        query.from_user = SimpleNamespace(first_name="Sam")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        callback.assert_awaited_once_with("12345", "gemini-3.5-flash", "gemini")
        query.edit_message_text.assert_awaited()
        edit_kwargs = query.edit_message_text.call_args[1]
        assert "`gemini-3.5-flash`" in edit_kwargs["text"]
        # Successful switch clears both the message-keyed and bare-chat entries.
        assert "12345:42" not in adapter._model_picker_state
        assert "12345" not in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_two_concurrent_pickers_resolve_by_message_key(self, monkeypatch):
        """Two pickers live in one chat: A (msg 10) then B (msg 20).  Sending B
        OVERWRITES the bare ``chat`` key to B's state, but A still has its own
        ``chat:10`` message-keyed entry.  A callback fired on A must resolve to
        state A via that message key — NOT be shadowed by B's bare-key
        overwrite — and switch A's selected model, leaving B untouched."""
        adapter = _make_adapter()
        callback_a = AsyncMock(return_value="Switched to `model-a`")
        callback_b = AsyncMock(return_value="Switched to `model-b`")
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: None,
        )

        message_ids = iter([10, 20])

        async def mock_send_message(**kwargs):
            return SimpleNamespace(message_id=next(message_ids))

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        # Picker A (msg 10).
        result_a = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "prov_a", "name": "ProvA", "models": ["model-a"], "total_models": 1}
            ],
            current_model="cur",
            current_provider="prov_a",
            session_key="s-a",
            on_model_selected=callback_a,
            metadata=None,
        )
        assert result_a.success is True

        # Picker B (msg 20) — re-issue in the same chat; bare key is overwritten.
        result_b = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "prov_b", "name": "ProvB", "models": ["model-b"], "total_models": 1}
            ],
            current_model="cur",
            current_provider="prov_b",
            session_key="s-b",
            on_model_selected=callback_b,
            metadata=None,
        )
        assert result_b.success is True

        # Both message-keyed entries coexist; bare key now points at B.
        assert "12345:10" in adapter._model_picker_state
        assert "12345:20" in adapter._model_picker_state
        assert adapter._model_picker_state["12345"] is adapter._model_picker_state["12345:20"]

        state_a = adapter._model_picker_state["12345:10"]
        state_a["selected_provider"] = "prov_a"
        state_a["model_list"] = ["model-a"]

        # Callback fired on the OLD picker (msg 10).
        query = AsyncMock()
        query.data = "mm:0"
        query.message = SimpleNamespace(
            chat_id=12345,
            message_id=10,
            message_thread_id=None,
        )
        query.from_user = SimpleNamespace(first_name="Sam")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        # A resolves via its own message key and switches; B is never invoked.
        callback_a.assert_awaited_once_with("12345", "model-a", "prov_a")
        callback_b.assert_not_awaited()
        edit_kwargs = query.edit_message_text.call_args[1]
        assert "`model-a`" in edit_kwargs["text"]

        # A's message-keyed entry is cleared; B's picker remains live and intact.
        assert "12345:10" not in adapter._model_picker_state
        assert "12345:20" in adapter._model_picker_state
        assert adapter._model_picker_state["12345"] is adapter._model_picker_state["12345:20"]
