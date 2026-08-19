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
# PluginContext.register_telegram_handler: validation + queuing
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

    def test_registration_handle_releases_factory(self):
        """The returned PluginRegistration removes the factory on release,
        so a reloaded plugin's old factory is not re-wired forever."""
        mgr, ctx = _make_ctx()

        def factory(application, adapter):  # pragma: no cover
            ...

        handle = ctx.register_telegram_handler(factory)
        handle.dispose()
        assert mgr.get_telegram_handler_factories() == []

    def test_non_callable_factory_raises(self):
        """A non-callable factory must be rejected, not silently stored."""
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_handler("not a factory")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_handler(None)  # type: ignore[arg-type]

    def test_async_factory_rejected_at_registration(self):
        """async def factories can never be awaited from the synchronous
        wiring path; they must fail at registration, not silently no-op."""
        _mgr, ctx = _make_ctx()

        async def factory(application, adapter):  # pragma: no cover
            ...

        with pytest.raises(ValueError, match="async"):
            ctx.register_telegram_handler(factory)
        assert _mgr.get_telegram_handler_factories() == []

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
        assert factories == [(fa, "plug_a"), (fb, "plug_b")]


# ---------------------------------------------------------------------------
# TelegramAdapter connect-path wiring
# ---------------------------------------------------------------------------
# Exercises TelegramAdapter._wire_plugin_handlers(): the connect-time code
# path that consumes get_telegram_handler_factories() and invokes each factory
# with (application, adapter). The telegram package is mocked by
# tests/gateway/conftest.py at collection time (python-telegram-bot is an
# optional dep).

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _recording_factory(log, key):
    """Build a factory that records the (application, adapter) it receives."""

    def factory(application, adapter):
        log.append((key, application, adapter))

    return factory


class TestTelegramAdapterPluginHandlerWiring:
    """_wire_plugin_handlers() (the connect path) invokes factories with (app, adapter)."""

    def _adapter(self) -> TelegramAdapter:
        # object.__new__ skips the heavy __init__. _wire_plugin_handlers only
        # needs self.name; `name` is a read-only property over self.platform
        # (Platform.TELEGRAM.value.title() -> "Telegram"), so set a stand-in
        # platform rather than the property itself.
        a = object.__new__(TelegramAdapter)
        a.platform = SimpleNamespace(value="telegram")
        return a

    def _mgr(self, factories):
        mgr = MagicMock()
        mgr.get_telegram_handler_factories.return_value = factories
        return mgr

    def test_factories_invoked_with_app_and_adapter(self):
        """Each factory is called exactly once with (application=app, adapter=self)."""
        app = MagicMock(name="app")
        adapter = self._adapter()
        log: list = []
        fa, fb = _recording_factory(log, "a"), _recording_factory(log, "b")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(fa, "plug_a"), (fb, "plug_b")]),
        ):
            adapter._wire_plugin_handlers(app)

        assert log == [("a", app, adapter), ("b", app, adapter)]

    def test_no_factories_is_a_noop(self):
        """Empty factory list (the common case) wires nothing onto the app."""
        app = MagicMock(name="app")
        adapter = self._adapter()

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([]),
        ):
            adapter._wire_plugin_handlers(app)

        app.add_handler.assert_not_called()

    def test_plugin_manager_load_failure_is_isolated(self):
        """If get_plugin_manager() raises, wiring is skipped; connect stays safe."""
        app = MagicMock(name="app")
        adapter = self._adapter()

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugin layer down"),
        ):
            adapter._wire_plugin_handlers(app)  # must not raise

    def test_one_factory_raising_does_not_block_others(self):
        """A factory that raises must not stop later factories from running."""
        app = MagicMock(name="app")
        adapter = self._adapter()
        log: list = []

        def boom(application, adapter):
            raise RuntimeError("plugin bug")

        good = _recording_factory(log, "good")
        other = _recording_factory(log, "other")

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(boom, "buggy"), (good, "g"), (other, "o")]),
        ):
            adapter._wire_plugin_handlers(app)  # must not raise

        assert [key for (key, _app, _adapter) in log] == ["good", "other"]

    @staticmethod
    def _inspectable_app():
        """A minimal app whose handlers map can be inspected (real PTB shape:
        dict[group] -> list of handler objects)."""

        class _App:
            def __init__(self):
                self.handlers = {}

            def add_handler(self, handler, group=0):
                self.handlers.setdefault(group, []).append(handler)

        return _App()

    def test_factories_wire_before_core_handlers_inside_register_handlers(self):
        """_register_handlers must invoke plugin factories BEFORE the core
        add_handler calls: PTB group-0 first-match means a factory wired after
        the core set would be shadowed by it."""
        order: list = []

        app = self._inspectable_app()
        core_add = app.add_handler

        def recording_add(handler, group=0):
            order.append("core")
            core_add(handler, group)

        app.add_handler = recording_add

        def factory(application, adapter):
            order.append("factory")
            application.add_handler(object(), group=1)  # scoped plugin handler

        adapter = self._adapter()
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(factory, "plug_a")]),
        ):
            adapter._register_handlers(app)

        assert order[0] == "factory"
        assert order.count("core") >= 5

    def test_unscoped_group0_addition_is_flagged(self, caplog):
        """A factory adding a group-0 handler gets a warning: it can shadow
        the core handlers under PTB's first-match-per-group rule."""
        import logging

        def careless(application, adapter):
            application.add_handler(object())  # group 0, unscoped

        app = self._inspectable_app()
        adapter = self._adapter()
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(careless, "plug_a")]),
        ):
            with caplog.at_level(
                logging.WARNING, logger="plugins.platforms.telegram.adapter"
            ):
                adapter._wire_plugin_handlers(app)

        assert "group-0" in caplog.text
        assert "plug_a" in caplog.text

    def test_scoped_group_addition_is_not_flagged(self, caplog):
        """A factory registering in a non-zero group stays silent."""
        import logging

        def careful(application, adapter):
            application.add_handler(object(), group=1)

        app = self._inspectable_app()
        adapter = self._adapter()
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(careful, "plug_a")]),
        ):
            with caplog.at_level(
                logging.WARNING, logger="plugins.platforms.telegram.adapter"
            ):
                adapter._wire_plugin_handlers(app)

        assert "group-0" not in caplog.text

    def test_coroutine_returning_factory_is_discarded_with_error(self, caplog):
        """Belt for sync-callable factories that return a coroutine anyway:
        the coroutine is closed and the failure is loud, never a silent
        never-awaited warning at GC time."""
        import logging

        async def _unawaited():
            pass

        def sneaky(application, adapter):
            return _unawaited()

        app = self._inspectable_app()
        adapter = self._adapter()
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._mgr([(sneaky, "plug_a")]),
        ):
            with caplog.at_level(
                logging.ERROR, logger="plugins.platforms.telegram.adapter"
            ):
                adapter._wire_plugin_handlers(app)

        assert "coroutine" in caplog.text
        # No success line for the discarded factory.
        assert "Wired Telegram handlers" not in caplog.text
