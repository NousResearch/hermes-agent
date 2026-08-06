"""Tests for plugin-registered inline-button callback prefixes.

Covers ``PluginContext.register_callback_prefix`` validation, the
``get_plugin_callback_prefix`` lookup, and the Telegram adapter dispatch:
authorization before the handler, sync and async handlers, bounded answers,
handler failure containment, and built-in prefixes always winning.

Mirrors the fixture pattern of test_telegram_clarify_buttons.py.
"""

import asyncio
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported (mirrors
# test_telegram_clarify_buttons.py)
# ---------------------------------------------------------------------------
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
from hermes_cli.plugins import PluginContext, run_plugin_callback_handler
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_context(plugin_name="p1", registry=None):
    manifest = types.SimpleNamespace(name=plugin_name, key=plugin_name)
    manager = types.SimpleNamespace(_callback_prefixes=registry if registry is not None else {})
    return PluginContext(manifest, manager), manager._callback_prefixes


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_query(data, user_id="777"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.chat.type = "private"
    query.message.message_thread_id = None
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    return update, query


# ===========================================================================
# register_callback_prefix — validation
# ===========================================================================

class TestRegisterCallbackPrefix:
    def test_valid_prefix_stored(self):
        ctx, registry = _make_context()
        handler = lambda data: "ok"  # noqa: E731
        ctx.register_callback_prefix("em:", handler, description="email approvals")
        assert registry["em:"]["handler"] is handler
        assert registry["em:"]["plugin"] == "p1"
        assert registry["em:"]["description"] == "email approvals"

    @pytest.mark.parametrize(
        "bad",
        ["", "em", ":", "EM:", "em :", "a" * 20 + ":", "e m:", "em::", None],
    )
    def test_invalid_shapes_rejected(self, bad):
        ctx, registry = _make_context()
        ctx.register_callback_prefix(bad, lambda data: None)
        assert registry == {}

    @pytest.mark.parametrize(
        "reserved",
        ["ea:", "gt:", "cp:", "cl:", "sc:", "mp:", "mpg:", "mg:", "mb2:", "mxx:"],
    )
    def test_reserved_and_shadowing_rejected(self, reserved):
        # mb2:/mxx: start with the built-in bare prefixes mb/mx and would be
        # consumed by the built-in branch before ever reaching the registry.
        ctx, registry = _make_context()
        ctx.register_callback_prefix(reserved, lambda data: None)
        assert registry == {}

    def test_other_plugins_prefix_not_stolen(self):
        registry = {}
        ctx1, _ = _make_context("p1", registry)
        ctx2, _ = _make_context("p2", registry)
        ctx1.register_callback_prefix("em:", lambda data: "one")
        original = registry["em:"]["handler"]
        ctx2.register_callback_prefix("em:", lambda data: "two")
        assert registry["em:"]["handler"] is original
        assert registry["em:"]["plugin"] == "p1"

    def test_same_plugin_may_rebind(self):
        ctx, registry = _make_context()
        ctx.register_callback_prefix("em:", lambda data: "one")
        replacement = lambda data: "two"  # noqa: E731
        ctx.register_callback_prefix("em:", replacement)
        assert registry["em:"]["handler"] is replacement

    @pytest.mark.parametrize("bad", [None, "not-a-function", 42, object(), ["x"]])
    def test_non_callable_handler_rejected_at_registration(self, bad):
        # A non-callable handler is a bug in the plugin, not a policy outcome:
        # it raises here rather than failing after a user presses the button.
        ctx, registry = _make_context()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_callback_prefix("em:", bad)
        assert registry == {}

    def test_non_callable_does_not_clobber_existing_registration(self):
        ctx, registry = _make_context()
        good = lambda data: "ok"  # noqa: E731
        ctx.register_callback_prefix("em:", good)
        with pytest.raises(ValueError):
            ctx.register_callback_prefix("em:", None)
        assert registry["em:"]["handler"] is good


# ===========================================================================
# run_plugin_callback_handler — off-loop execution and bounded completion
# ===========================================================================

class TestRunPluginCallbackHandler:
    @pytest.mark.asyncio
    async def test_sync_handler_runs_off_the_event_loop(self):
        # A blocking sync handler must not stall the caller's loop: while it
        # blocks, the loop still has to service other coroutines.
        started = threading.Event()
        release = threading.Event()
        thread_names = []

        def handler(data):
            thread_names.append(threading.current_thread().name)
            started.set()
            release.wait(timeout=5)
            return "done"

        task = asyncio.create_task(run_plugin_callback_handler(handler, "em:x"))
        started_at = time.monotonic()
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "handler never started off-loop"

        # The discriminator: we are back on the loop *while* the handler is
        # still blocked. Running it inline could only return control after the
        # handler finished, so the task would already be done here.
        assert not task.done(), "handler ran to completion on the event loop"
        assert time.monotonic() - started_at < 2.0, "event loop was stalled"

        release.set()
        assert await asyncio.wait_for(task, timeout=5) == "done"
        # ...on the dedicated pool, not the process-wide default executor.
        assert thread_names[0].startswith("hermes-plugin-callback")

    @pytest.mark.asyncio
    async def test_async_handler_awaited_on_the_loop(self):
        loop = asyncio.get_running_loop()
        seen = {}

        async def handler(data):
            seen["loop"] = asyncio.get_running_loop()
            return "async ok"

        assert await run_plugin_callback_handler(handler, "em:x") == "async ok"
        assert seen["loop"] is loop

    @pytest.mark.asyncio
    async def test_sync_callable_returning_awaitable_is_resolved(self):
        class AsyncCallable:
            async def __call__(self, data):
                return "resolved"

        assert await run_plugin_callback_handler(AsyncCallable(), "em:x") == "resolved"

    @pytest.mark.asyncio
    async def test_slow_async_handler_is_bounded(self):
        async def handler(data):
            await asyncio.sleep(5)
            return "too late"

        with pytest.raises(asyncio.TimeoutError):
            await run_plugin_callback_handler(handler, "em:x", timeout=0.05)

    @pytest.mark.asyncio
    async def test_slow_sync_handler_is_bounded(self):
        release = threading.Event()

        def handler(data):
            release.wait(timeout=5)
            return "too late"

        try:
            with pytest.raises(asyncio.TimeoutError):
                await run_plugin_callback_handler(handler, "em:x", timeout=0.05)
        finally:
            release.set()  # let the worker thread retire

    @pytest.mark.asyncio
    async def test_handler_exception_propagates(self):
        def handler(data):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await run_plugin_callback_handler(handler, "em:x")


# ===========================================================================
# Telegram adapter dispatch
# ===========================================================================

def _patched_registry(entry_handler, prefix="em:", plugin="p1"):
    entry = {"handler": entry_handler, "plugin": plugin, "description": ""}

    def lookup(data):
        return (prefix, entry) if data.startswith(prefix) else None

    return patch(
        "hermes_cli.plugins.get_plugin_callback_prefix",
        side_effect=lookup,
    )


class TestTelegramPluginCallbackDispatch:
    @pytest.mark.asyncio
    async def test_authorized_press_dispatches_and_answers(self):
        adapter = _make_adapter()
        seen = []

        def handler(data):
            seen.append(data)
            return "✅ archived"

        update, query = _make_query("em:approve:tg-7:archive")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(handler):
                await adapter._handle_callback_query(update, MagicMock())

        assert seen == ["em:approve:tg-7:archive"]
        query.answer.assert_awaited_once()
        assert query.answer.call_args[1]["text"] == "✅ archived"

    @pytest.mark.asyncio
    async def test_unauthorized_press_never_reaches_handler(self):
        adapter = _make_adapter()
        handler = MagicMock()

        update, query = _make_query("em:approve:tg-7:archive", user_id="777")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "999"}, clear=False):
            with _patched_registry(handler):
                await adapter._handle_callback_query(update, MagicMock())

        handler.assert_not_called()
        query.answer.assert_awaited_once()
        assert "not authorized" in query.answer.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_async_handler_awaited(self):
        adapter = _make_adapter()

        async def handler(data):
            return "async ok"

        update, query = _make_query("em:dismiss:tg-9")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(handler):
                await adapter._handle_callback_query(update, MagicMock())

        assert query.answer.call_args[1]["text"] == "async ok"

    @pytest.mark.asyncio
    async def test_answer_is_bounded(self):
        adapter = _make_adapter()

        update, query = _make_query("em:approve:tg-7:archive")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(lambda data: "x" * 5000):
                await adapter._handle_callback_query(update, MagicMock())

        assert len(query.answer.call_args[1]["text"]) <= 180

    @pytest.mark.asyncio
    async def test_none_result_answers_done(self):
        adapter = _make_adapter()

        update, query = _make_query("em:dismiss:tg-9")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(lambda data: None):
                await adapter._handle_callback_query(update, MagicMock())

        assert query.answer.call_args[1]["text"] == "Done."

    @pytest.mark.asyncio
    async def test_handler_exception_contained(self):
        adapter = _make_adapter()

        def handler(data):
            raise RuntimeError("secret internals")

        update, query = _make_query("em:approve:tg-7:archive")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(handler):
                await adapter._handle_callback_query(update, MagicMock())

        text = query.answer.call_args[1]["text"]
        assert text == "❌ Action failed."
        assert "secret internals" not in text

    @pytest.mark.asyncio
    async def test_unmatched_data_falls_through_silently(self):
        adapter = _make_adapter()
        handler = MagicMock()

        update, query = _make_query("zz:whatever")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(handler):
                await adapter._handle_callback_query(update, MagicMock())

        handler.assert_not_called()
        query.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_goes_through_the_bounded_runner(self):
        # The adapter must never call the handler inline — that is what would
        # put plugin code on Telegram's event loop.
        adapter = _make_adapter()
        handler = MagicMock(return_value="unused")

        update, query = _make_query("em:approve:tg-7:archive")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(handler):
                with patch(
                    "hermes_cli.plugins.run_plugin_callback_handler",
                    new=AsyncMock(return_value="✅ via runner"),
                ) as runner:
                    await adapter._handle_callback_query(update, MagicMock())

        handler.assert_not_called()
        runner.assert_awaited_once_with(handler, "em:approve:tg-7:archive")
        assert query.answer.call_args[1]["text"] == "✅ via runner"

    @pytest.mark.asyncio
    async def test_timed_out_handler_answers_and_does_not_hang(self):
        adapter = _make_adapter()

        update, query = _make_query("em:approve:tg-7:archive")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(lambda data: "unused"):
                with patch(
                    "hermes_cli.plugins.run_plugin_callback_handler",
                    new=AsyncMock(side_effect=asyncio.TimeoutError),
                ):
                    await adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_awaited_once()
        assert query.answer.call_args[1]["text"] == "⏳ Action timed out."

    @pytest.mark.asyncio
    async def test_blocking_sync_handler_does_not_stall_the_adapter(self):
        # End-to-end: a blocking handler still gets answered, and the loop
        # stays responsive throughout the press.
        adapter = _make_adapter()
        release = threading.Event()

        def handler(data):
            release.wait(timeout=5)
            return "✅ eventually"

        update, query = _make_query("em:approve:tg-7:archive")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with _patched_registry(handler):
                task = asyncio.create_task(
                    adapter._handle_callback_query(update, MagicMock())
                )
                await asyncio.sleep(0.05)
                assert not task.done()
                release.set()
                await asyncio.wait_for(task, timeout=5)

        assert query.answer.call_args[1]["text"] == "✅ eventually"

    @pytest.mark.asyncio
    async def test_builtin_prefix_wins_over_registry(self):
        # A registry that would greedily match anything must never see a
        # built-in prefix: the mp: branch returns before the plugin lookup.
        adapter = _make_adapter()
        handler = MagicMock()

        update, query = _make_query("mp:whatever")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch(
                "hermes_cli.plugins.get_plugin_callback_prefix",
                side_effect=lambda data: ("mp:", {"handler": handler, "plugin": "p1"}),
            ):
                await adapter._handle_callback_query(update, MagicMock())

        handler.assert_not_called()
