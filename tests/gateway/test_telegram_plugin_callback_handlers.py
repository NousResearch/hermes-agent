"""Tests for plugin-registered Telegram inline keyboard callback handlers.

Covers:
* ``PluginContext.register_telegram_callback_handler`` validation + queuing
* ``PluginManager.get_telegram_callback_handlers`` accessor
* ``TelegramAdapter._handle_callback_query`` dispatching matching callbacks
  while isolating plugin exceptions from the gateway.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure the repo root is importable when this test runs directly
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported without PTB
# ---------------------------------------------------------------------------
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

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig  # noqa: E402
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    mgr = PluginManager()
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


def _make_adapter() -> TelegramAdapter:
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_callback_update(data: str):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_thread_id = None
    query.message.chat = MagicMock()
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    context = MagicMock()
    return update, context, query


class TestRegisterTelegramCallbackHandlerAPI:
    """Behaviour of ctx.register_telegram_callback_handler()."""

    def test_string_prefix_is_queued(self):
        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover - never called here
            await query.answer()

        ctx.register_telegram_callback_handler("idea:", cb)

        handlers = mgr.get_telegram_callback_handlers()
        assert len(handlers) == 1
        matcher, callback, plugin_name = handlers[0]
        assert matcher == "idea:"
        assert callback is cb
        assert plugin_name == "test_plugin"

    def test_regex_matcher_is_accepted(self):
        import re as _re

        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            await query.answer()

        pat = _re.compile(r"^idea:(love|no):")
        ctx.register_telegram_callback_handler(pat, cb)

        handlers = mgr.get_telegram_callback_handlers()
        assert handlers[0][0] is pat

    def test_non_callable_callback_raises(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_callback_handler("idea:", "not a function")  # type: ignore[arg-type]

    def test_empty_string_matcher_raises(self):
        _mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            await query.answer()

        with pytest.raises(ValueError, match="empty matcher"):
            ctx.register_telegram_callback_handler("   ", cb)

    def test_none_matcher_raises(self):
        _mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            await query.answer()

        with pytest.raises(ValueError, match="empty matcher"):
            ctx.register_telegram_callback_handler(None, cb)

    def test_get_telegram_callback_handlers_returns_copy(self):
        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            await query.answer()

        ctx.register_telegram_callback_handler("idea:", cb)
        handlers = mgr.get_telegram_callback_handlers()
        handlers.clear()
        assert len(mgr.get_telegram_callback_handlers()) == 1


class TestTelegramAdapterPluginCallbackDispatch:
    """_handle_callback_query must dispatch plugin-supplied handlers."""

    @pytest.mark.asyncio
    async def test_plugin_callback_prefix_dispatches_and_stops_builtin_fallback(self):
        adapter = _make_adapter()
        calls: list[tuple[object, str]] = []

        async def my_handler(query, data):
            calls.append((query, data))
            await query.answer(text="handled")

        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("idea:", my_handler, "argus-ideas"),
        ]
        update, context, query = _make_callback_update("idea:love:abc123")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_callback_query(update, context)

        assert calls == [(query, "idea:love:abc123")]
        query.answer.assert_awaited_once_with(text="handled")

    @pytest.mark.asyncio
    async def test_plugin_callback_regex_dispatches(self):
        import re as _re

        adapter = _make_adapter()
        seen: list[str] = []

        async def my_handler(query, data):
            seen.append(data)
            await query.answer()

        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            (_re.compile(r"^gift:(save|reject):"), my_handler, "argus-gifts"),
        ]
        update, context, query = _make_callback_update("gift:save:42")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_callback_query(update, context)

        assert seen == ["gift:save:42"]
        query.answer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plugin_exception_is_answered_and_swallowed(self):
        adapter = _make_adapter()

        async def boom(query, data):
            raise RuntimeError("plugin bug")

        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("idea:", boom, "buggy-plugin"),
        ]
        update, context, query = _make_callback_update("idea:explode")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_callback_query(update, context)

        query.answer.assert_awaited_once()
        assert "failed" in query.answer.call_args.kwargs["text"].lower()
