from __future__ import annotations

import logging
import sys
import builtins
from types import ModuleType
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner


@pytest.fixture(autouse=True)
def _restore_feishu_dependency_globals():
    import gateway.platforms.feishu as feishu_pkg
    from gateway.platforms.feishu import adapter as feishu_adapter
    import gateway.platforms.feishu.webhook_guard as webhook_guard

    adapter_names = (
        "FEISHU_AVAILABLE",
        "FEISHU_DOMAIN",
        "FEISHU_WEBHOOK_AVAILABLE",
        "FEISHU_WEBSOCKET_AVAILABLE",
        "LARK_DOMAIN",
        "aiohttp",
        "web",
        "websockets",
        "lark",
    )
    package_names = (
        "FEISHU_AVAILABLE",
        "FEISHU_DOMAIN",
        "FEISHU_WEBHOOK_AVAILABLE",
        "FEISHU_WEBSOCKET_AVAILABLE",
        "LARK_DOMAIN",
    )
    webhook_guard_names = ("WEBHOOK_AVAILABLE", "aiohttp", "web")
    adapter_snapshot = {name: getattr(feishu_adapter, name) for name in adapter_names}
    package_snapshot = {name: getattr(feishu_pkg, name) for name in package_names}
    webhook_guard_snapshot = {
        name: getattr(webhook_guard, name) for name in webhook_guard_names
    }

    yield

    for name, value in adapter_snapshot.items():
        setattr(feishu_adapter, name, value)
    for name, value in package_snapshot.items():
        setattr(feishu_pkg, name, value)
    for name, value in webhook_guard_snapshot.items():
        setattr(webhook_guard, name, value)


def _install_fake_feishu_modules(monkeypatch):
    lark_module = ModuleType("lark_oapi")
    channel_module = ModuleType("lark_oapi.channel")
    core_module = ModuleType("lark_oapi.core")
    const_module = ModuleType("lark_oapi.core.const")

    class FakeFeishuChannel:
        pass

    channel_module.FeishuChannel = FakeFeishuChannel
    const_module.FEISHU_DOMAIN = "https://open.feishu.cn"
    const_module.LARK_DOMAIN = "https://open.larksuite.com"

    aiohttp_module = ModuleType("aiohttp")
    aiohttp_web_module = ModuleType("aiohttp.web")
    aiohttp_module.web = aiohttp_web_module
    websockets_module = ModuleType("websockets")

    monkeypatch.setitem(sys.modules, "lark_oapi", lark_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.channel", channel_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.core", core_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.core.const", const_module)
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_module)
    monkeypatch.setitem(sys.modules, "aiohttp.web", aiohttp_web_module)
    monkeypatch.setitem(sys.modules, "websockets", websockets_module)

    return lark_module


def test_feishu_missing_dependencies_warning_does_not_blame_credentials(monkeypatch, caplog):
    feishu_pkg = ModuleType("gateway.platforms.feishu")

    class FakeFeishuAdapter:
        pass

    feishu_pkg.FeishuAdapter = FakeFeishuAdapter
    feishu_pkg.check_feishu_requirements = lambda: False
    monkeypatch.setitem(sys.modules, "gateway.platforms.feishu", feishu_pkg)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
    )
    platform_config = PlatformConfig(
        enabled=True,
        extra={"app_id": "cli_test_app", "app_secret": "test_secret"},
    )

    caplog.set_level(logging.WARNING, logger="gateway.run")
    adapter = runner._create_adapter(Platform.FEISHU, platform_config)

    assert adapter is None
    assert "required dependencies are unavailable" in caplog.text
    assert "FEISHU_APP_ID/SECRET" not in caplog.text


def test_feishu_requirements_lazy_install_and_rebind_sdk(monkeypatch):
    lark_module = _install_fake_feishu_modules(monkeypatch)

    from gateway.platforms.feishu import adapter as feishu_adapter
    import tools.lazy_deps as lazy_deps

    calls = []

    def fake_ensure(feature, *, prompt=True, force=False):
        calls.append((feature, prompt, force))

    monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBSOCKET_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBHOOK_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "lark", None)
    monkeypatch.setattr(feishu_adapter, "FEISHU_DOMAIN", None)
    monkeypatch.setattr(feishu_adapter, "LARK_DOMAIN", None)
    monkeypatch.setattr(feishu_adapter, "aiohttp", None)
    monkeypatch.setattr(feishu_adapter, "web", None)
    monkeypatch.setattr(feishu_adapter, "websockets", None)

    assert feishu_adapter.check_feishu_requirements() is True

    assert calls == [("platform.feishu", False, True)]
    assert feishu_adapter.FEISHU_AVAILABLE is True
    assert feishu_adapter.FEISHU_WEBHOOK_AVAILABLE is True
    assert feishu_adapter.FEISHU_WEBSOCKET_AVAILABLE is True
    assert feishu_adapter.lark is lark_module
    assert feishu_adapter.FEISHU_DOMAIN == "https://open.feishu.cn"
    assert feishu_adapter.LARK_DOMAIN == "https://open.larksuite.com"


def test_feishu_requirements_syncs_package_availability_after_lazy_install(monkeypatch):
    _install_fake_feishu_modules(monkeypatch)

    import gateway.platforms.feishu as feishu_pkg
    from gateway.platforms.feishu import adapter as feishu_adapter
    import tools.lazy_deps as lazy_deps

    monkeypatch.setattr(lazy_deps, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(feishu_pkg, "FEISHU_AVAILABLE", False, raising=False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_DOMAIN", None)
    monkeypatch.setattr(feishu_adapter, "LARK_DOMAIN", None)
    monkeypatch.setattr(feishu_adapter, "lark", None)

    assert feishu_adapter.check_feishu_requirements() is True

    assert feishu_pkg.FEISHU_AVAILABLE is True
    assert feishu_pkg.FEISHU_DOMAIN == "https://open.feishu.cn"
    assert feishu_pkg.LARK_DOMAIN == "https://open.larksuite.com"


def test_feishu_requirements_refreshes_webhook_guard_after_lazy_install(monkeypatch):
    _install_fake_feishu_modules(monkeypatch)

    from gateway.platforms.feishu import adapter as feishu_adapter
    import gateway.platforms.feishu.webhook_guard as webhook_guard
    import tools.lazy_deps as lazy_deps

    monkeypatch.setattr(lazy_deps, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBHOOK_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "aiohttp", None)
    monkeypatch.setattr(feishu_adapter, "web", None)
    monkeypatch.setattr(webhook_guard, "WEBHOOK_AVAILABLE", False)
    monkeypatch.setattr(webhook_guard, "aiohttp", None)
    monkeypatch.setattr(webhook_guard, "web", None)

    assert feishu_adapter.check_feishu_requirements() is True

    assert webhook_guard.WEBHOOK_AVAILABLE is True
    assert webhook_guard.aiohttp is sys.modules["aiohttp"]
    assert webhook_guard.web is sys.modules["aiohttp.web"]


def test_feishu_requirements_accepts_single_configured_transport(monkeypatch):
    _install_fake_feishu_modules(monkeypatch)

    from gateway.platforms.feishu import adapter as feishu_adapter
    import tools.lazy_deps as lazy_deps

    monkeypatch.setattr(lazy_deps, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBHOOK_AVAILABLE", False)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBSOCKET_AVAILABLE", False)

    real_import = builtins.__import__

    def block_websockets(name, *args, **kwargs):
        if name == "websockets" or name.startswith("websockets."):
            raise ImportError("websockets intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_websockets)

    assert feishu_adapter.check_feishu_requirements() is True
    assert feishu_adapter.FEISHU_WEBHOOK_AVAILABLE is True
    assert feishu_adapter.FEISHU_WEBSOCKET_AVAILABLE is False

    def block_aiohttp(name, *args, **kwargs):
        if name == "aiohttp" or name.startswith("aiohttp."):
            raise ImportError("aiohttp intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_aiohttp)

    assert feishu_adapter.check_feishu_requirements() is True
    assert feishu_adapter.FEISHU_WEBHOOK_AVAILABLE is False
    assert feishu_adapter.FEISHU_WEBSOCKET_AVAILABLE is True


def test_feishu_requirements_rejects_missing_all_transports(monkeypatch):
    _install_fake_feishu_modules(monkeypatch)

    from gateway.platforms.feishu import adapter as feishu_adapter
    import tools.lazy_deps as lazy_deps

    monkeypatch.setattr(lazy_deps, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", False)

    real_import = builtins.__import__

    def block_transports(name, *args, **kwargs):
        if (
            name == "aiohttp"
            or name.startswith("aiohttp.")
            or name == "websockets"
            or name.startswith("websockets.")
        ):
            raise ImportError("transport intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_transports)

    assert feishu_adapter.check_feishu_requirements() is False
    assert feishu_adapter.FEISHU_WEBHOOK_AVAILABLE is False
    assert feishu_adapter.FEISHU_WEBSOCKET_AVAILABLE is False


def test_feishu_requirements_force_upgrades_old_lark_oapi(monkeypatch):
    from importlib import metadata

    _install_fake_feishu_modules(monkeypatch)

    from gateway.platforms.feishu import adapter as feishu_adapter
    import tools.lazy_deps as lazy_deps

    calls = []

    def fake_ensure(feature, *, prompt=True, force=False):
        calls.append((feature, prompt, force))

    monkeypatch.setattr(metadata, "version", lambda name: "1.6.2")
    monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", True)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBHOOK_AVAILABLE", True)

    assert feishu_adapter.check_feishu_requirements() is True

    assert calls == [("platform.feishu", False, True)]


def test_feishu_requirements_force_upgrades_lark_oapi_1_6_4(monkeypatch):
    from importlib import metadata

    _install_fake_feishu_modules(monkeypatch)

    from gateway.platforms.feishu import adapter as feishu_adapter
    import tools.lazy_deps as lazy_deps

    calls = []

    def fake_ensure(feature, *, prompt=True, force=False):
        calls.append((feature, prompt, force))

    monkeypatch.setattr(metadata, "version", lambda name: "1.6.4")
    monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
    monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", True)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(feishu_adapter, "FEISHU_WEBHOOK_AVAILABLE", True)

    assert feishu_adapter.check_feishu_requirements() is True

    assert calls == [("platform.feishu", False, True)]
