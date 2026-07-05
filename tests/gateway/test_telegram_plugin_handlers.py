"""Tests for plugin-registered Telegram PTB handler factories.

Covers:
* ``PluginContext.register_telegram_handler`` validation + queuing
* ``PluginManager.get_telegram_handler_factories`` accessor
* ``TelegramAdapter._wire_plugin_handlers`` invoking factories with
  ``(application, adapter)``
* Defensive isolation: a factory that raises does NOT prevent the
  adapter from wiring other factories or continuing to connect.
* ``discover_and_load(force=True)`` clears queued factories.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable when this test runs directly
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402

from hermes_cli.plugins import (  # noqa: E402
    PluginContext,
    PluginManager,
    PluginManifest,
)


def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    mgr = PluginManager()
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


def _make_adapter() -> TelegramAdapter:
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._app = MagicMock()
    adapter._bot = MagicMock()
    return adapter


# ===========================================================================
# PluginContext.register_telegram_handler — validation + queuing
# ===========================================================================

class TestRegisterTelegramHandlerAPI:
    def test_factory_is_queued_with_plugin_name(self):
        mgr, ctx = _make_ctx()

        def factory(application, adapter):  # pragma: no cover - never called
            pass

        ctx.register_telegram_handler(factory)

        factories = mgr.get_telegram_handler_factories()
        assert len(factories) == 1
        fn, plugin_name = factories[0]
        assert fn is factory
        assert plugin_name == "test_plugin"

    def test_non_callable_factory_raises(self):
        _, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_handler("not-a-callable")  # type: ignore[arg-type]

    def test_accessor_returns_copy(self):
        mgr, ctx = _make_ctx()
        ctx.register_telegram_handler(lambda app, adapter: None)

        got = mgr.get_telegram_handler_factories()
        got.append(("junk", "junk"))
        assert len(mgr.get_telegram_handler_factories()) == 1

    def test_multiple_plugins_each_recorded(self):
        mgr = PluginManager()
        for name in ("plugin_a", "plugin_b"):
            manifest = PluginManifest(name=name, version="0.1.0", description="t")
            ctx = PluginContext(manifest=manifest, manager=mgr)
            ctx.register_telegram_handler(lambda app, adapter: None)

        names = [n for _, n in mgr.get_telegram_handler_factories()]
        assert names == ["plugin_a", "plugin_b"]

    def test_force_rediscovery_clears_factories(self):
        mgr, ctx = _make_ctx()
        ctx.register_telegram_handler(lambda app, adapter: None)
        assert len(mgr.get_telegram_handler_factories()) == 1

        # force=True clears queued registrations before the re-scan; the
        # scan itself finds nothing in an isolated HERMES_HOME.
        mgr.discover_and_load(force=True)
        assert mgr.get_telegram_handler_factories() == []


# ===========================================================================
# TelegramAdapter._wire_plugin_handlers
# ===========================================================================

class TestTelegramAdapterPluginWiring:
    def test_factory_invoked_with_application_and_adapter(self):
        adapter = _make_adapter()
        calls = []

        def factory(application, adp):
            calls.append((application, adp))
            application.add_handler(MagicMock())

        mgr = MagicMock()
        mgr.get_telegram_handler_factories.return_value = [(factory, "biz_plugin")]

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers()

        assert calls == [(adapter._app, adapter)]
        adapter._app.add_handler.assert_called_once()

    def test_no_factories_is_a_noop(self):
        adapter = _make_adapter()
        mgr = MagicMock()
        mgr.get_telegram_handler_factories.return_value = []

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers()

        adapter._app.add_handler.assert_not_called()

    def test_raising_factory_does_not_block_others(self):
        adapter = _make_adapter()
        wired = []

        def bad_factory(application, adp):
            raise RuntimeError("boom")

        def good_factory(application, adp):
            wired.append("good")

        mgr = MagicMock()
        mgr.get_telegram_handler_factories.return_value = [
            (bad_factory, "bad_plugin"),
            (good_factory, "good_plugin"),
        ]

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers()  # must not raise

        assert wired == ["good"]

    def test_manager_load_failure_does_not_raise(self):
        adapter = _make_adapter()
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugin system down"),
        ):
            adapter._wire_plugin_handlers()  # must not raise
