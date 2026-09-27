"""Native Feishu plugin handlers use the real SDK dispatcher, without network I/O."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from hermes_cli import plugins
from plugins.platforms.feishu import adapter as fa


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    home = tmp_path / "home"
    directory = home / "plugins" / "feishu-event-fixture"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(
        "name: feishu-event-fixture\nversion: 0.1.0\ndescription: Test native events.\n",
        encoding="utf-8",
    )
    (directory / "__init__.py").write_text('''
events = []
clients = []
def register(ctx):
    def wire(native, adapter):
        clients.append(native)
        builder = getattr(adapter, "event_dispatcher_builder", None)
        if builder is not None:
            builder.register_p2_customized_event(
                "task.task.update_user_access_v2", lambda data: events.append(data.event))
    ctx.register_platform_handler("feishu", wire)
''', encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [feishu-event-fixture]\n", encoding="utf-8"
    )
    empty = tmp_path / "bundled"
    empty.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty))
    from hermes_cli.config import load_config
    assert load_config()["plugins"]["enabled"] == ["feishu-event-fixture"]
    manager = plugins.PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["feishu-event-fixture"]
    assert loaded.error is None
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    pytest.importorskip("lark_oapi")
    assert fa._load_lark_oapi()
    return manager, loaded.module


def dispatch(handler, event_type, event):
    body = json.dumps({"schema": "2.0", "header": {
        "event_type": event_type, "event_id": "fixture-event",
    }, "event": event}).encode()
    # SDK spelling differs across supported releases; both are its WS entry point.
    method = getattr(handler, "do_without_validation", None)
    if method is None:
        method = handler._do_without_validation
    return method(body)


def test_plugin_registers_before_transport_and_survives_client_rebuild(plugin):
    _, module = plugin
    adapter = fa.FeishuAdapter(PlatformConfig(extra={
        "app_id": "fixture-app", "app_secret": "fixture-secret",
    }))
    for revision in (1, 2):
        adapter._prepare_client()
        failure = None
        try:
            dispatch(adapter._event_handler, "task.task.update_user_access_v2", {"revision": revision})
        except Exception as exc:
            failure = str(exc)
        assert module.events == [{"revision": n} for n in range(1, revision + 1)], failure
        assert module.clients[-1] is adapter._client
        adapter.rewire_plugin_handlers()
        assert len(module.clients) == revision
    assert module.clients[0] is not module.clients[1]


def test_late_factories_share_live_dispatcher_and_cannot_replace_core(plugin, monkeypatch, caplog):
    manager, module = plugin
    adapter = fa.FeishuAdapter(PlatformConfig())
    core_events = []
    monkeypatch.setattr(adapter, "_on_message_read_event", core_events.append)
    adapter._prepare_client()
    handler = adapter._event_handler
    calls = []

    def collision(native, adapter):
        calls.append("collision")
        adapter.event_dispatcher_builder.register_p2_im_message_message_read_v1(
            lambda data: pytest.fail("core handler replaced"))

    def broken(native, adapter):
        calls.append("broken")
        raise RuntimeError("fixture factory failure")

    def late(native, adapter):
        calls.append("late")
        assert native is module.clients[-1]
        adapter.event_dispatcher_builder.register_p2_customized_event(
            "fixture.late_v1", lambda data: module.events.append(data.event))

    ctx = plugins.PluginContext(manager._plugins["feishu-event-fixture"].manifest, manager)
    for factory in (collision, broken, late):
        ctx.register_platform_handler("feishu", factory)
    adapter.rewire_plugin_handlers()
    adapter.rewire_plugin_handlers()
    assert calls == ["collision", "broken", "late"]
    assert "already registered" in caplog.text
    assert "fixture factory failure" in caplog.text
    assert adapter._event_handler is handler
    dispatch(handler, "fixture.late_v1", {"text": "task change"})
    assert module.events == [{"text": "task change"}]
    dispatch(handler, "im.message.message_read_v1", {})
    assert len(core_events) == 1


def test_empty_factory_set_keeps_core_dispatch_and_missing_sdk_fallback(plugin, monkeypatch):
    manager, _ = plugin
    manager._platform_handler_factories.clear()
    adapter = fa.FeishuAdapter(PlatformConfig())
    assert adapter.event_dispatcher_builder is None
    events = []
    monkeypatch.setattr(adapter, "_on_message_read_event", events.append)
    adapter._prepare_client()
    dispatch(adapter._event_handler, "im.message.message_read_v1", {})
    assert len(events) == 1
    monkeypatch.setattr(fa, "EventDispatcherHandler", None)
    assert fa.FeishuAdapter(PlatformConfig())._build_event_handler() is None


def test_public_connect_delivers_during_transport_start_and_disconnect_clears_builder(plugin, monkeypatch):
    _, module = plugin
    adapter = fa.FeishuAdapter(PlatformConfig(extra={
        "app_id": "fixture-connect", "app_secret": "fixture-secret",
    }))
    monkeypatch.setattr(adapter, "_hydrate_bot_identity", AsyncMock())
    seen_clients = []

    class LocalWSClient:
        def __init__(self, **kwargs):
            seen_clients.append(kwargs)
            # Delivery at construction must work, not only after connect() returns.
            dispatch(kwargs["event_handler"], "task.task.update_user_access_v2", {"ready": True})

    monkeypatch.setattr(fa, "FeishuWSClient", LocalWSClient)
    monkeypatch.setattr(fa, "_run_official_feishu_ws_client", lambda *args: None)

    async def exercise():
        assert await adapter.connect()
        await adapter._ws_future
        assert module.events == [{"ready": True}]
        assert seen_clients[0]["event_handler"] is adapter._event_handler
        await adapter.disconnect()
        assert adapter.event_dispatcher_builder is None

    asyncio.run(exercise())
