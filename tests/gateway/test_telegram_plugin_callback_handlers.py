"""Tests for plugin-registered Telegram inline-keyboard callbacks.

Covers:
* ``PluginContext.register_telegram_callback_handler`` validation + queuing
* ``PluginManager.get_telegram_callback_handlers`` accessor
* Dispatch inside ``_handle_callback_query`` after built-in prefixes
* Longest-prefix match, unknown prefix no-op, auth fail-closed
* ``send()`` metadata: ``reply_markup`` and per-send ``disable_link_preview``
* Rich path skipped when markup is present so buttons are not dropped
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    mgr = PluginManager()
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


def _make_adapter(extra=None) -> TelegramAdapter:
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class _AuthRunner:
    def __init__(self, authorized: bool):
        self.authorized = authorized

    async def _handle_message(self, event):
        return None

    def _is_user_authorized(self, source):
        return self.authorized


def _callback_update(data: str, user_id: int = 99, chat_id: int = 12345):
    query = MagicMock()
    query.data = data
    query.from_user = SimpleNamespace(id=user_id, first_name="Ada")
    query.message = SimpleNamespace(
        chat_id=chat_id,
        chat=SimpleNamespace(type="private"),
        message_thread_id=None,
        text="card",
    )
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return SimpleNamespace(callback_query=query), query


# ---------------------------------------------------------------------------
# PluginContext.register_telegram_callback_handler
# ---------------------------------------------------------------------------


class TestRegisterTelegramCallbackHandlerAPI:
    def test_string_prefix_is_queued(self):
        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover - never called here
            return None

        ctx.register_telegram_callback_handler("task:", cb)

        handlers = mgr.get_telegram_callback_handlers()
        assert len(handlers) == 1
        prefix, callback, plugin_name = handlers[0]
        assert prefix == "task:"
        assert callback is cb
        assert plugin_name == "test_plugin"

    def test_empty_prefix_rejected(self):
        _mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            return None

        with pytest.raises(ValueError, match="empty or invalid prefix"):
            ctx.register_telegram_callback_handler("  ", cb)

    def test_non_callable_rejected(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_callback_handler("task:", "not-a-fn")

    @pytest.mark.parametrize(
        "prefix",
        ["ea:", "cl:", "gt:", "mp:", "cp:", "sc:", "update_prompt:", "mb", "mx"],
    )
    def test_reserved_prefix_rejected(self, prefix):
        _mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            return None

        with pytest.raises(ValueError, match="reserved"):
            ctx.register_telegram_callback_handler(prefix, cb)

    def test_force_rediscover_clears_queue(self, monkeypatch):
        mgr, ctx = _make_ctx()

        async def cb(query, data):  # pragma: no cover
            return None

        ctx.register_telegram_callback_handler("task:", cb)
        assert mgr.get_telegram_callback_handlers()
        mgr._discovered = True
        monkeypatch.setattr(
            PluginManager, "_discover_and_load_inner", lambda self_inner: None
        )
        mgr.discover_and_load(force=True)
        assert mgr.get_telegram_callback_handlers() == []


# ---------------------------------------------------------------------------
# _handle_callback_query dispatch
# ---------------------------------------------------------------------------


class TestTelegramPluginCallbackDispatch:
    @pytest.mark.asyncio
    async def test_longest_prefix_wins(self):
        adapter = _make_adapter()
        adapter._message_handler = _AuthRunner(True)._handle_message

        short = AsyncMock()
        long = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", short, "short_plugin"),
            ("task:done:", long, "long_plugin"),
        ]

        update, query = _callback_update("task:done:42")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        long.assert_awaited_once()
        assert long.await_args.args[1] == "task:done:42"
        short.assert_not_called()
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_unknown_prefix_is_noop(self):
        adapter = _make_adapter()
        adapter._message_handler = _AuthRunner(True)._handle_message
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", AsyncMock(), "test_plugin"),
        ]

        update, query = _callback_update("other:1")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        query.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_auth_fail_closed_does_not_run_plugin(self):
        adapter = _make_adapter()
        adapter._message_handler = _AuthRunner(False)._handle_message
        plugin_cb = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", plugin_cb, "test_plugin"),
        ]

        update, query = _callback_update("task:1")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        plugin_cb.assert_not_called()
        query.answer.assert_awaited()
        assert "not authorized" in query.answer.await_args.kwargs.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_multiplex_closure_uses_authorization_check(self, monkeypatch):
        """Plugin taps must honor set_authorization_check under multiplex.

        gateway.multiplex_profiles installs a closure as the primary handler,
        so ``handler.__self__`` is absent. The old helper then fail-closed on
        empty TELEGRAM_ALLOWED_USERS and never ran the plugin. The platform
        callback registered via set_authorization_check is the path that
        survives that wrapping (#87132).
        """
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        adapter = _make_adapter()

        def closure_handler(event):
            return None

        adapter._message_handler = closure_handler
        assert getattr(closure_handler, "__self__", None) is None

        def auth_check(user_id, chat_type=None, chat_id=None):
            return str(user_id) == "99" and str(chat_id) == "12345"

        adapter.set_authorization_check(auth_check)

        plugin_cb = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", plugin_cb, "test_plugin"),
        ]

        update, query = _callback_update("task:1", user_id=99, chat_id=12345)
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        plugin_cb.assert_awaited_once()
        query.answer.assert_awaited()
        assert "not authorized" not in (
            query.answer.await_args.kwargs.get("text") or ""
        ).lower()

    @pytest.mark.asyncio
    async def test_multiplex_closure_authorization_check_can_deny(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        adapter = _make_adapter()
        adapter._message_handler = lambda event: None
        adapter.set_authorization_check(lambda user_id, chat_type=None, chat_id=None: False)

        plugin_cb = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", plugin_cb, "test_plugin"),
        ]

        update, query = _callback_update("task:1")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        plugin_cb.assert_not_called()
        assert "not authorized" in query.answer.await_args.kwargs.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_builtin_prefix_still_wins(self):
        adapter = _make_adapter()
        adapter._message_handler = _AuthRunner(True)._handle_message
        plugin_cb = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("ea:", plugin_cb, "rogue_plugin"),
        ]

        update, query = _callback_update("ea:once:1")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        plugin_cb.assert_not_called()
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_plugin_raise_is_swallowed_after_answer(self):
        adapter = _make_adapter()
        adapter._message_handler = _AuthRunner(True)._handle_message

        async def boom(query, data):
            raise RuntimeError("plugin exploded")

        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", boom, "test_plugin"),
        ]
        update, query = _callback_update("task:1")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())

        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_register_handlers_still_dispatches_plugin_taps(self):
        """Reconnect rebuilds the same CallbackQueryHandler; lookup is live."""
        adapter = _make_adapter()
        adapter._message_handler = _AuthRunner(True)._handle_message
        plugin_cb = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_telegram_callback_handlers.return_value = [
            ("task:", plugin_cb, "test_plugin"),
        ]

        captured = []

        def _record_cqh(callback, *args, **kwargs):
            captured.append(callback)
            return SimpleNamespace(callback=callback)

        app = MagicMock()
        with patch(
            "plugins.platforms.telegram.adapter.CallbackQueryHandler",
            side_effect=_record_cqh,
        ):
            adapter._register_handlers(app)
        assert adapter._handle_callback_query in captured

        update, _query = _callback_update("task:9")
        with patch(
            "hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr
        ):
            await adapter._handle_callback_query(update, SimpleNamespace())
        plugin_cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# send() metadata: reply_markup + disable_link_preview
# ---------------------------------------------------------------------------


class TestTelegramSendMarkupAndPreview:
    @pytest.mark.asyncio
    async def test_send_attaches_structured_reply_markup(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        mock_msg = MagicMock()
        mock_msg.message_id = 7
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        rows = [[{"text": "Do it", "callback_data": "task:go"}]]
        result = await adapter.send(
            "12345",
            "hello",
            metadata={"reply_markup": rows},
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_send_passes_through_existing_markup(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        mock_msg = MagicMock()
        mock_msg.message_id = 8
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)
        sentinel = object()

        result = await adapter.send(
            "12345",
            "hello",
            metadata={"reply_markup": sentinel},
        )

        assert result.success is True
        assert adapter._bot.send_message.call_args.kwargs["reply_markup"] is sentinel

    @pytest.mark.asyncio
    async def test_send_keyboard_on_first_chunk_only(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        adapter.MAX_MESSAGE_LENGTH = 80
        adapter._bot.send_message = AsyncMock(
            side_effect=lambda **kwargs: MagicMock(message_id=1)
        )

        content = ("chunk content " * 20).strip()
        result = await adapter.send(
            "123",
            content,
            metadata={
                "reply_markup": [[{"text": "Go", "callback_data": "task:go"}]],
            },
        )

        assert result.success is True
        assert adapter._bot.send_message.await_count > 1
        first = adapter._bot.send_message.await_args_list[0].kwargs
        last = adapter._bot.send_message.await_args_list[-1].kwargs
        assert first.get("reply_markup") is not None
        assert "reply_markup" not in last

    @pytest.mark.asyncio
    async def test_send_honors_per_send_preview_disable(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        adapter._disable_link_previews = False
        mock_msg = MagicMock()
        mock_msg.message_id = 9
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send(
            "12345",
            "see https://example.com",
            metadata={"disable_link_preview": True},
        )

        kwargs = adapter._bot.send_message.call_args.kwargs
        assert "link_preview_options" in kwargs or kwargs.get(
            "disable_web_page_preview"
        ) is True
        assert adapter._link_preview_disabled({"disable_link_preview": True}) is True
        assert adapter._link_preview_kwargs({"disable_link_preview": True})

    @pytest.mark.asyncio
    async def test_rich_path_skipped_when_markup_present(self):
        adapter = _make_adapter(extra={"rich_messages": True})
        adapter._bot.do_api_request = AsyncMock(
            return_value=SimpleNamespace(message_id=123)
        )
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter._bot.send_chat_action = AsyncMock()

        table = (
            "| Case | Status |\n"
            "|---|---|\n"
            "| rich | ok |\n"
        )
        result = await adapter.send(
            "12345",
            table,
            metadata={
                "reply_markup": [[{"text": "Go", "callback_data": "task:go"}]],
            },
        )

        assert result.success is True
        adapter._bot.do_api_request.assert_not_called()
        assert adapter._bot.send_message.call_args.kwargs.get("reply_markup") is not None
