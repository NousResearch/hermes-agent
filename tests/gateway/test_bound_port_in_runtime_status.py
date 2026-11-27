"""A port-binding adapter records the port it really bound in runtime status, and the dashboard reports that
port for a profile whose platform plugin the dashboard's own process never loaded (Feishu webhook mode
installed only in another profile)."""

import pytest
from aiohttp import web

from gateway import status
from gateway.platforms.base import BasePlatformAdapter


def _adapter(key: str) -> BasePlatformAdapter:
    adapter = object.__new__(type("_Adapter", (BasePlatformAdapter,), {
        m: (lambda *a, **k: None) for m in ("connect", "disconnect", "get_chat_info", "send")}))
    adapter._runtime_status_platform_key = key
    adapter._shared_listener_profile = None
    adapter._fatal_error_code = adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    return adapter


@pytest.mark.asyncio
async def test_a_plugin_listeners_bound_port_reaches_the_dashboard_without_its_plugin(tmp_path, monkeypatch):
    from gateway.platform_registry import platform_registry
    from gateway.platforms.shared_ingress import bind_listener
    from hermes_cli.web_server_gateway import _profile_platform_ports
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert not platform_registry.is_registered("feishu")  # no table row and no plugin in this process

    adapter = _adapter("feishu")
    app = web.Application()
    app.router.add_post("/feishu/webhook", lambda request: web.Response())
    runner = await bind_listener(adapter, app, "127.0.0.1", 0, "/feishu/webhook")
    try:
        bound = runner.addresses[0][1]
        adapter._mark_connected()
        assert status.flush_runtime_status(timeout=2.0)
        runtime = status.read_runtime_status()
        assert runtime["platforms"]["feishu"]["listener_base"] == f"http://127.0.0.1:{bound}"
        # Another profile's home: no config naming the port, the dashboard has no Feishu plugin.
        assert _profile_platform_ports(tmp_path / "other", runtime) == {"feishu": bound}
    finally:
        await runner.cleanup()


def test_a_dead_listeners_stamp_reports_no_port(tmp_path):
    from hermes_cli.web_server_gateway import _profile_platform_ports
    runtime = {"platforms": {"feishu": {"state": "fatal", "listener_base": "http://127.0.0.1:8765"}}}
    assert _profile_platform_ports(tmp_path, runtime) == {}


@pytest.mark.asyncio
async def test_a_bot_restarted_without_a_listener_clears_its_old_port(tmp_path, monkeypatch):
    """Webhook mode, then a restart in WebSocket mode: the status file must stop naming the old port."""
    from hermes_cli.web_server_gateway import _profile_platform_ports
    from gateway.platforms.shared_ingress import bind_listener
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    webhook_mode = _adapter("feishu")
    app = web.Application()
    app.router.add_post("/feishu/webhook", lambda request: web.Response())
    runner = await bind_listener(webhook_mode, app, "127.0.0.1", 0, "/feishu/webhook")
    webhook_mode._mark_connected()
    assert status.flush_runtime_status(timeout=2.0)
    await runner.cleanup()

    monkeypatch.setattr(status, "_runtime_status_state", None)  # a new gateway process re-reads the file
    _adapter("feishu")._mark_connected()  # WebSocket mode binds nothing
    assert status.flush_runtime_status(timeout=2.0)
    runtime = status.read_runtime_status()
    assert runtime["platforms"]["feishu"]["listener_base"] is None
    assert _profile_platform_ports(tmp_path / "other", runtime) == {}


def test_only_a_record_without_the_field_falls_back_to_config(tmp_path):
    from hermes_cli.web_server_gateway import _profile_platform_ports
    older = {"platforms": {"webhook": {"state": "connected"}}}
    assert _profile_platform_ports(tmp_path, older) == {"webhook": 8644}
    no_listener = {"platforms": {"webhook": {"state": "connected", "listener_base": None}}}
    assert _profile_platform_ports(tmp_path, no_listener) == {}
