"""Tests for plugin-registered python-telegram-bot handler factories.

Covers the plugin API consolidated onto #59159's factory shape:
* ``PluginContext.register_telegram_handler(factory)`` validation + queuing
* ``PluginManager.get_telegram_handler_factories`` accessor
* ``TelegramAdapter._wire_plugin_handlers`` invoking each factory with
  ``(application, adapter)`` at connect time, before the core handlers
* defensive isolation: a factory that raises does NOT prevent the adapter
  from wiring other factories or continuing to connect.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure the repo root is importable when this test runs directly
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from hermes_cli.plugins import (  # noqa: E402
    PluginContext,
    PluginManager,
    PluginManifest,
)


def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    """Build a fresh PluginManager + PluginContext bound to it."""
    mgr = PluginManager()
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


# ---------------------------------------------------------------------------
# PluginContext.register_telegram_handler — validation + queuing
# ---------------------------------------------------------------------------

class TestRegisterTelegramHandlerAPI:
    """Behaviour of ctx.register_telegram_handler(factory)."""

    def test_factory_is_queued_with_plugin_name(self):
        mgr, ctx = _make_ctx()

        def factory(application, adapter):  # pragma: no cover - never called here
            ...

        ctx.register_telegram_handler(factory)

        factories = mgr.get_telegram_handler_factories()
        assert len(factories) == 1
        fn, plugin_name = factories[0]
        assert fn is factory
        assert plugin_name == "test_plugin"

    def test_non_callable_factory_raises(self):
        """A non-callable factory must be rejected, not silently stored."""
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_handler("not a factory")  # type: ignore[arg-type]

    def test_none_factory_raises(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_handler(None)  # type: ignore[arg-type]

    def test_get_telegram_handler_factories_returns_copy(self):
        """The accessor returns a copy so callers can't mutate plugin state."""
        mgr, ctx = _make_ctx()

        def factory(application, adapter):  # pragma: no cover
            ...

        ctx.register_telegram_handler(factory)
        factories = mgr.get_telegram_handler_factories()
        factories.clear()
        assert len(mgr.get_telegram_handler_factories()) == 1

    def test_multiple_plugins_each_recorded_in_order(self):
        """Registration order is preserved (PTB handler precedence is order-sensitive)."""
        mgr = PluginManager()
        ctx_a = PluginContext(
            manifest=PluginManifest(name="plug_a", version="0", description=""),
            manager=mgr,
        )
        ctx_b = PluginContext(
            manifest=PluginManifest(name="plug_b", version="0", description=""),
            manager=mgr,
        )

        def fa(application, adapter):  # pragma: no cover
            ...

        def fb(application, adapter):  # pragma: no cover
            ...

        ctx_a.register_telegram_handler(fa)
        ctx_b.register_telegram_handler(fb)

        factories = mgr.get_telegram_handler_factories()
        assert [(fn, name) for fn, name in factories] == [(fa, "plug_a"), (fb, "plug_b")]


# ---------------------------------------------------------------------------
# TelegramAdapter connect-path wiring
# ---------------------------------------------------------------------------
# Exercises TelegramAdapter._wire_plugin_handlers() — the connect-time code
# path that consumes get_telegram_handler_factories() and invokes each factory
# with (application, adapter). python-telegram-bot is an optional dep, so mock
# the telegram package the same way tests/gateway/test_telegram_network_reconnect
# .py does.

def _ensure_telegram_mock() -> None:
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _recording_factory(log, key):
    """Build a factory that records the (application, adapter) it receives."""

    def factory(application, adapter):
        log.append((key, application, adapter))

    return factory


class TestTelegramAdapterPluginHandlerWiring:
    """_wire_plugin_handlers() (the connect path) invokes factories with (app, adapter)."""

    def _adapter(self, app) -> TelegramAdapter:
        # object.__new__ skips the heavy __init__. _wire_plugin_handlers only
        # needs self._app and self.name; `name` is a read-only property over
        # self.platform (Platform.TELEGRAM.value.title() -> "Telegram"), so set
        # a stand-in platform rather than the property itself.
        a = object.__new__(TelegramAdapter)
        a.platform = SimpleNamespace(value="telegram")
        a._app = app
        return a

    def _mgr(self, factories):
        mgr = MagicMock()
        mgr.get_telegram_handler_factories.return_value = factories
        return mgr

    def test_factories_invoked_with_app_and_adapter(self):
        """Each factory is called exactly once with (application=app, adapter=self)."""
        app = MagicMock(name="app")
        adapter = self._adapter(app)
        log: list = []
        fa, fb = _recording_factory(log, "a"), _recording_factory(log, "b")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(fa, "plug_a"), (fb, "plug_b")]),
        ):
            adapter._wire_plugin_handlers()

        assert log == [("a", app, adapter), ("b", app, adapter)]

    def test_no_factories_is_a_noop(self):
        """Empty factory list (the common case) wires nothing onto the app."""
        app = MagicMock(name="app")
        adapter = self._adapter(app)

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([]),
        ):
            adapter._wire_plugin_handlers()

        app.add_handler.assert_not_called()

    def test_plugin_manager_load_failure_is_isolated(self):
        """If get_plugin_manager() raises, wiring is skipped — connect stays safe."""
        app = MagicMock(name="app")
        adapter = self._adapter(app)

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugin layer down"),
        ):
            adapter._wire_plugin_handlers()  # must not raise

    def test_one_factory_raising_does_not_block_others(self):
        """A factory that raises must not stop later factories from running."""
        app = MagicMock(name="app")
        adapter = self._adapter(app)
        log: list = []

        def boom(application, adapter):
            raise RuntimeError("plugin bug")

        good = _recording_factory(log, "good")
        other = _recording_factory(log, "other")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(boom, "buggy"), (good, "g"), (other, "o")]),
        ):
            adapter._wire_plugin_handlers()  # must not raise

        assert [key for (key, _app, _adapter) in log] == ["good", "other"]
