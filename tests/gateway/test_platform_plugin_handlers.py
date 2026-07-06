"""Tests for plugin-registered native platform handler factories.

Covers:
* ``PluginContext.register_platform_handler`` validation + queuing
* ``PluginContext.register_telegram_handler`` back-compat alias
* ``PluginManager.get_platform_handler_factories`` accessor (+ telegram alias)
* ``BasePlatformAdapter._wire_plugin_handlers`` invoking factories with
  ``(native, adapter)`` — exercised through the Telegram adapter
* Defensive isolation: a factory that raises does NOT prevent other
  factories from wiring or the platform from connecting.
* Platform scoping: factories for platform A never fire for platform B.
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
# PluginContext.register_platform_handler — validation + queuing
# ===========================================================================

class TestRegisterPlatformHandlerAPI:
    def test_factory_is_queued_with_plugin_name(self):
        mgr, ctx = _make_ctx()

        def factory(native, adapter):  # pragma: no cover - never called
            pass

        ctx.register_platform_handler("discord", factory)

        factories = mgr.get_platform_handler_factories("discord")
        assert len(factories) == 1
        fn, plugin_name = factories[0]
        assert fn is factory
        assert plugin_name == "test_plugin"

    def test_platform_key_is_normalized(self):
        mgr, ctx = _make_ctx()
        ctx.register_platform_handler("  Slack ", lambda n, a: None)
        assert len(mgr.get_platform_handler_factories("slack")) == 1

    def test_non_callable_factory_raises(self):
        _, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_platform_handler("discord", "nope")  # type: ignore[arg-type]

    def test_empty_platform_raises(self):
        _, ctx = _make_ctx()
        with pytest.raises(ValueError, match="empty platform"):
            ctx.register_platform_handler("  ", lambda n, a: None)

    def test_platform_scoping(self):
        """Factories for platform A never appear in platform B's list."""
        mgr, ctx = _make_ctx()
        ctx.register_platform_handler("discord", lambda n, a: None)
        ctx.register_platform_handler("matrix", lambda n, a: None)
        assert len(mgr.get_platform_handler_factories("discord")) == 1
        assert len(mgr.get_platform_handler_factories("matrix")) == 1
        assert mgr.get_platform_handler_factories("slack") == []

    def test_accessor_returns_copy(self):
        mgr, ctx = _make_ctx()
        ctx.register_platform_handler("telegram", lambda n, a: None)

        got = mgr.get_platform_handler_factories("telegram")
        got.append(("junk", "junk"))
        assert len(mgr.get_platform_handler_factories("telegram")) == 1

    def test_multiple_plugins_each_recorded(self):
        mgr = PluginManager()
        for name in ("plugin_a", "plugin_b"):
            manifest = PluginManifest(name=name, version="0.1.0", description="t")
            ctx = PluginContext(manifest=manifest, manager=mgr)
            ctx.register_platform_handler("telegram", lambda n, a: None)

        names = [n for _, n in mgr.get_platform_handler_factories("telegram")]
        assert names == ["plugin_a", "plugin_b"]

    def test_force_rediscovery_clears_factories(self):
        mgr, ctx = _make_ctx()
        ctx.register_platform_handler("telegram", lambda n, a: None)
        assert len(mgr.get_platform_handler_factories("telegram")) == 1

        mgr.discover_and_load(force=True)
        assert mgr.get_platform_handler_factories("telegram") == []


# ===========================================================================
# Telegram back-compat alias
# ===========================================================================

class TestTelegramAlias:
    def test_register_telegram_handler_routes_to_telegram_bucket(self):
        mgr, ctx = _make_ctx()

        def factory(application, adapter):  # pragma: no cover
            pass

        ctx.register_telegram_handler(factory)

        assert mgr.get_platform_handler_factories("telegram") == [
            (factory, "test_plugin")
        ]
        # Legacy accessor still works.
        assert mgr.get_telegram_handler_factories() == [(factory, "test_plugin")]

    def test_alias_non_callable_raises(self):
        _, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_telegram_handler("not-a-callable")  # type: ignore[arg-type]


# ===========================================================================
# BasePlatformAdapter._wire_plugin_handlers (via TelegramAdapter)
# ===========================================================================

class TestAdapterPluginWiring:
    def test_factory_invoked_with_native_and_adapter(self):
        adapter = _make_adapter()
        calls = []

        def factory(native, adp):
            calls.append((native, adp))
            native.add_handler(MagicMock())

        mgr = MagicMock()
        mgr.get_platform_handler_factories.return_value = [(factory, "biz_plugin")]

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers(adapter._app)

        assert calls == [(adapter._app, adapter)]
        adapter._app.add_handler.assert_called_once()
        # Adapter asked for its own platform's factories.
        mgr.get_platform_handler_factories.assert_called_once_with("telegram")

    def test_no_factories_is_a_noop(self):
        adapter = _make_adapter()
        mgr = MagicMock()
        mgr.get_platform_handler_factories.return_value = []

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers(adapter._app)

        adapter._app.add_handler.assert_not_called()

    def test_raising_factory_does_not_block_others(self):
        adapter = _make_adapter()
        wired = []

        def bad_factory(native, adp):
            raise RuntimeError("boom")

        def good_factory(native, adp):
            wired.append("good")

        mgr = MagicMock()
        mgr.get_platform_handler_factories.return_value = [
            (bad_factory, "bad_plugin"),
            (good_factory, "good_plugin"),
        ]

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers(adapter._app)  # must not raise

        assert wired == ["good"]

    def test_manager_load_failure_does_not_raise(self):
        adapter = _make_adapter()
        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugin system down"),
        ):
            adapter._wire_plugin_handlers(adapter._app)  # must not raise

    def test_native_none_supported(self):
        """Adapters without a separate native client pass None."""
        adapter = _make_adapter()
        seen = []

        mgr = MagicMock()
        mgr.get_platform_handler_factories.return_value = [
            (lambda native, adp: seen.append(native), "p"),
        ]
        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            adapter._wire_plugin_handlers(None)
        assert seen == [None]


# ===========================================================================
# Every adapter calls _wire_plugin_handlers in connect() — source invariant
# ===========================================================================

def test_all_connectable_adapters_wire_plugin_handlers():
    """Invariant: every platform adapter with a connect() implementation
    calls ``_wire_plugin_handlers`` somewhere in its source, so plugins can
    rely on the hook existing on every platform (native may be None)."""
    import glob

    repo = Path(_repo)
    adapter_files = sorted(
        glob.glob(str(repo / "plugins" / "platforms" / "*" / "adapter.py"))
    ) + [
        str(repo / "gateway" / "platforms" / name)
        for name in (
            "api_server.py", "bluebubbles.py", "msgraph_webhook.py",
            "signal.py", "webhook.py", "weixin.py",
            "whatsapp_cloud.py", "yuanbao.py",
        )
    ]
    missing = []
    for f in adapter_files:
        src = Path(f).read_text()
        if "async def connect(" not in src:
            continue
        if "_wire_plugin_handlers" not in src:
            missing.append(f)
    assert not missing, f"adapters missing plugin-handler wiring: {missing}"
