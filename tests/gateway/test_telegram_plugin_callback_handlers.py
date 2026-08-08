"""Tests for plugin-registered Telegram callback-query handlers.

Covers:
* ``PluginContext.register_telegram_callback_handler`` validation + queuing
* ``PluginManager.get_telegram_callback_handlers`` accessor
* ``TelegramAdapter._handle_callback_query`` offering unclaimed updates to
  plugin handlers (prefix match, truthy-consumes, falsy falls through)
* Built-in callback namespaces keep priority over plugin prefixes
* Defensive wrapping: a plugin handler that raises does NOT take down the
  gateway and the tap still gets answered.

Mirrors ``test_slack_plugin_action_handlers.py`` for the Telegram adapter.
"""

from __future__ import annotations

import os
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
# Minimal Telegram mock so TelegramAdapter can be imported
# ---------------------------------------------------------------------------
def _ensure_telegram_mock():
    """Wire up the minimal mocks required to import TelegramAdapter."""
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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402

from hermes_cli.plugins import (  # noqa: E402
    PluginContext,
    PluginManager,
    PluginManifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    """Build a fresh PluginManager + PluginContext bound to it."""
    mgr = PluginManager()
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        description="test",
    )
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


def _make_adapter():
    """Create a TelegramAdapter with mocked internals."""
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_callback_update(data: str):
    """Build a (update, context, query) triple for _handle_callback_query."""
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.from_user = MagicMock()
    query.from_user.id = "12345"
    query.from_user.first_name = "Norbert"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    context = MagicMock()
    return update, context, query


def _fake_mgr(handlers: list) -> MagicMock:
    mgr = MagicMock()
    mgr.get_telegram_callback_handlers.return_value = handlers
    return mgr


# ---------------------------------------------------------------------------
# PluginContext.register_telegram_callback_handler — validation + queuing
# ---------------------------------------------------------------------------

class TestRegisterTelegramCallbackHandlerAPI:
    """Behaviour of ctx.register_telegram_callback_handler()."""

    def test_prefix_handler_is_queued(self):
        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover - never called here
            return True

        ctx.register_telegram_callback_handler("inbox_sweep:", cb)

        handlers = mgr.get_telegram_callback_handlers()
        assert len(handlers) == 1
        prefix, callback, plugin_name = handlers[0]
        assert prefix == "inbox_sweep:"
        assert callback is cb
        assert plugin_name == "test_plugin"

    def test_non_callable_callback_rejected(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_callback_handler("inbox_sweep:", "not-a-fn")

    def test_empty_prefix_rejected(self):
        _mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            return True

        with pytest.raises(ValueError, match="empty prefix"):
            ctx.register_telegram_callback_handler("", cb)
        with pytest.raises(ValueError, match="empty prefix"):
            ctx.register_telegram_callback_handler("   ", cb)

    def test_non_string_prefix_rejected(self):
        """Telegram callback data is a flat string — no regex/dict matchers."""
        import re as _re
        _mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            return True

        with pytest.raises(ValueError):
            ctx.register_telegram_callback_handler(_re.compile(r"^x:"), cb)

    def test_clear_on_forced_rediscovery(self):
        """discover_and_load(force=True) must drop queued handlers."""
        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            return True

        ctx.register_telegram_callback_handler("inbox_sweep:", cb)
        assert mgr.get_telegram_callback_handlers()

        with patch.object(mgr, "_discover_and_load_inner"):
            mgr.discover_and_load(force=True)

        assert mgr.get_telegram_callback_handlers() == []


# ---------------------------------------------------------------------------
# TelegramAdapter._handle_callback_query — plugin dispatch
# ---------------------------------------------------------------------------

class TestTelegramAdapterPluginCallbackDispatch:
    """Unclaimed callback_query updates are offered to plugin handlers."""

    @pytest.mark.asyncio
    async def test_matching_prefix_handler_consumes_update(self):
        adapter = _make_adapter()
        handler = AsyncMock(return_value=True)
        update, context, query = _make_callback_update("inbox_sweep:approve:7")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=_fake_mgr([("inbox_sweep:", handler, "test_plugin")]),
        ):
            await adapter._handle_callback_query(update, context)

        handler.assert_awaited_once_with(query, "inbox_sweep:approve:7")

    @pytest.mark.asyncio
    async def test_falsy_return_falls_through_to_next_handler(self):
        adapter = _make_adapter()
        first = AsyncMock(return_value=False)
        second = AsyncMock(return_value=True)
        update, context, query = _make_callback_update("inbox_sweep:approve:7")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=_fake_mgr([
                ("inbox_sweep:", first, "plugin_a"),
                ("inbox_", second, "plugin_b"),
            ]),
        ):
            await adapter._handle_callback_query(update, context)

        first.assert_awaited_once()
        second.assert_awaited_once_with(query, "inbox_sweep:approve:7")

    @pytest.mark.asyncio
    async def test_non_matching_prefix_handler_not_called(self):
        adapter = _make_adapter()
        handler = AsyncMock(return_value=True)
        update, context, _query = _make_callback_update("otherns:whatever")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=_fake_mgr([("inbox_sweep:", handler, "test_plugin")]),
        ):
            await adapter._handle_callback_query(update, context)

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_builtin_namespaces_keep_priority(self):
        """A plugin prefix can never shadow the built-in approval buttons."""
        adapter = _make_adapter()
        adapter._approval_state[5] = "agent:main:telegram:group:12345:99"
        handler = AsyncMock(return_value=True)
        update, context, _query = _make_callback_update("ea:once:5")

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), \
             patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve, \
             patch(
                 "hermes_cli.plugins.get_plugin_manager",
                 return_value=_fake_mgr([("ea:", handler, "test_plugin")]),
             ):
            await adapter._handle_callback_query(update, context)

        resolve.assert_called_once()
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raising_handler_is_contained_and_answered(self):
        """A plugin handler that raises must not propagate — and still acks."""
        adapter = _make_adapter()

        async def exploding(query, data):
            raise RuntimeError("plugin bug")

        fallback = AsyncMock(return_value=True)
        update, context, query = _make_callback_update("inbox_sweep:approve:7")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=_fake_mgr([
                ("inbox_sweep:", exploding, "plugin_a"),
                ("inbox_sweep:", fallback, "plugin_b"),
            ]),
        ):
            await adapter._handle_callback_query(update, context)

        # Best-effort answer so the button stops spinning.
        query.answer.assert_awaited_once()
        # The raising handler claimed its prefix — no re-dispatch.
        fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plugin_loader_failure_does_not_break_callbacks(self, tmp_path):
        """If get_plugin_manager() blows up, callbacks must keep working."""
        adapter = _make_adapter()
        update, context, _query = _make_callback_update("otherns:whatever")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugins broken"),
        ):
            await adapter._handle_callback_query(update, context)  # must not raise

        # Built-in callbacks still work while the plugin layer is unhealthy.
        b_update, b_context, b_query = _make_callback_update("update_prompt:y")
        b_query.from_user.id = 123
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugins broken"),
        ), patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
                patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}):
            await adapter._handle_callback_query(b_update, b_context)
        assert (tmp_path / ".update_response").read_text() == "y"

    @pytest.mark.asyncio
    async def test_no_handlers_is_a_noop(self):
        """The common case — no plugin handlers registered — must be silent."""
        adapter = _make_adapter()
        update, context, query = _make_callback_update("otherns:whatever")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=_fake_mgr([]),
        ):
            await adapter._handle_callback_query(update, context)

        query.answer.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_end_to_end_registration_reaches_dispatch(self):
        """register_telegram_callback_handler → manager → adapter dispatch."""
        mgr, ctx = _make_ctx()
        seen = []

        async def cb(query, data):
            seen.append(data)
            return True

        ctx.register_telegram_callback_handler("inbox_sweep:", cb)

        adapter = _make_adapter()
        update, context, _query = _make_callback_update("inbox_sweep:approve:7")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            await adapter._handle_callback_query(update, context)

        assert seen == ["inbox_sweep:approve:7"]
