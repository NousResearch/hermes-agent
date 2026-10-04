"""Scoped control-socket MCP reload: no interruption, truthful pending, isolated target."""
import asyncio
import json
import threading
from types import SimpleNamespace

from gateway.control_socket import GatewayControlServer
from tools import mcp_tool_reload as reload_mod


def test_real_control_socket_request_round_trip(tmp_path):
    import asyncio
    from gateway.control_socket import query_gateway_control

    async def run():
        server = GatewayControlServer(tmp_path, verb_handlers={
            "reload-mcp-server": lambda params: {"status": "pending", "name": params["name"]}})
        assert await server.start()
        try:
            answer = await asyncio.to_thread(query_gateway_control, tmp_path, "reload-mcp-server",
                                             params={"name": "smart-web"})
            assert answer == {"status": "pending", "name": "smart-web"}
        finally:
            await server.stop()

    asyncio.run(run())


def test_gate_drains_inflight_and_rejects_new_calls():
    entered = threading.Event()
    release = threading.Event()
    result = {}

    def call():
        with reload_mod.admit_call("smart-web") as admitted:
            assert admitted
            entered.set()
            assert release.wait(3)

    t = threading.Thread(target=call)
    t.start()
    try:
        assert entered.wait(3)
        with reload_mod.drain_calls("smart-web", timeout=0.01) as drained:
            assert drained is False
        with reload_mod.drain_calls("smart-web", timeout=0.01) as drained:
            assert drained is False
            with reload_mod.admit_call("smart-web") as admitted:
                assert admitted is False
            with reload_mod.admit_call("unrelated") as admitted:
                assert admitted is True
        release.set()
        t.join(3)
        assert not t.is_alive()
        with reload_mod.drain_calls("smart-web", timeout=1) as drained:
            assert drained is True
            with reload_mod.admit_call("smart-web") as admitted:
                assert admitted is False
        with reload_mod.admit_call("smart-web") as admitted:
            assert admitted is True
    finally:
        release.set()
        t.join(3)


def test_tool_and_resource_handlers_fail_closed_during_reload(monkeypatch):
    from tools.mcp_tool_handlers import _make_tool_handler, _make_list_resources_handler
    monkeypatch.setattr("tools.mcp_tool_handlers._acquire_call_server",
                        lambda *args: (_ for _ in ()).throw(AssertionError("transport accessed")))
    with reload_mod.drain_calls("smart-web", timeout=0) as drained:
        assert drained
        for handler in (_make_tool_handler("smart-web", "fetch", 1),
                        _make_list_resources_handler("smart-web", 1)):
            assert "reloading" in json.loads(handler({}))["error"]
    with reload_mod.admit_call("smart-web") as admitted:
        assert admitted


def test_control_socket_rejects_invalid_home_and_name(tmp_path, monkeypatch):
    from gateway.run_mcp_reload import reload_scoped_mcp_verb
    home = tmp_path / "gateway"
    home.mkdir()
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    handler = reload_scoped_mcp_verb(runner)
    server = GatewayControlServer(home, verb_handlers={"reload-mcp-server": handler})
    for params in ({"name": ""}, {"name": "smart-web", "home": str(tmp_path / "other")},
                   {"name": ["smart-web"]}):
        request = json.dumps({"verb": "reload-mcp-server", "params": params}).encode()
        answer = json.loads(server.handle_request_line(request))
        assert answer["ok"] is True
        assert answer["result"]["status"] == "rejected"


def test_reload_timeout_returns_pending_without_shutdown(tmp_path, monkeypatch):
    from gateway.run_mcp_reload import reload_scoped_mcp_verb
    from tools import mcp_tool_config, mcp_tool_lifecycle
    home = tmp_path / "gateway"
    home.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {"smart-web": {"command": "dummy"}})
    monkeypatch.setattr(mcp_tool_lifecycle, "shutdown_mcp_servers", lambda **kw: (_ for _ in ()).throw(AssertionError("shutdown")))
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    started = threading.Event()
    release = threading.Event()

    def call():
        with reload_mod.admit_call("smart-web") as admitted:
            assert admitted
            started.set()
            assert release.wait(3)

    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert started.wait(3)
        answer = reload_scoped_mcp_verb(runner)({"name": "smart-web", "drain_timeout": 0.01})
        assert answer["status"] == "pending"
    finally:
        release.set()
        thread.join(3)


def _reload_fixture(tmp_path, monkeypatch):
    from gateway.run_mcp_reload import reload_scoped_mcp_verb
    from tools import mcp_tool as core, mcp_tool_config, mcp_tool_discovery, mcp_tool_lifecycle, mcp_tool_loop
    home = tmp_path / "gateway"
    home.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {"smart-web": {"command": "dummy"}})
    original = SimpleNamespace(session=object())
    other = SimpleNamespace(session=object())
    servers = {"smart-web": original, "unrelated": other}
    monkeypatch.setattr(core, "_servers", servers)
    monkeypatch.setattr(core, "_server_scope_keys", {})
    monkeypatch.setattr(core, "_server_tool_scopes", {})
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(mcp_tool_loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda fn, **kw: asyncio.run(fn()))
    events = []

    async def connect(name, config):
        events.append("preflight")
        async def close():
            events.append("preview-closed")
        return SimpleNamespace(_tools=["schema"], shutdown=close)

    def shutdown(*, scope, names):
        events.append("shutdown")
        assert names == {"smart-web"}
        servers.pop("smart-web")

    def discover(allowed_mcp_names=None):
        events.append("discover")
        assert allowed_mcp_names == ["smart-web"]
        servers["smart-web"] = SimpleNamespace(session=object(), _registered_tool_names=["mcp__smart_web__fetch"])

    monkeypatch.setattr(mcp_tool_discovery, "_connect_server", connect)
    monkeypatch.setattr(mcp_tool_lifecycle, "shutdown_mcp_servers", shutdown)
    monkeypatch.setattr(mcp_tool_discovery, "discover_mcp_tools", discover)
    monkeypatch.setattr("tools.mcp_tool_agent.reprobe_tool_availability", lambda: events.append("reprobe"))
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    return reload_scoped_mcp_verb(runner), servers, original, other, events


def test_lazy_schema_only_registration_is_not_mistaken_for_live_reload(tmp_path, monkeypatch):
    from tools import mcp_tool as core
    handler, servers, original, other, events = _reload_fixture(tmp_path, monkeypatch)
    servers.pop("smart-web")
    monkeypatch.setattr(core, "_lazy_server_configs", {"smart-web": {"command": "dummy", "lazy": True}})
    result = handler({"name": "smart-web"})
    assert result["status"] == "pending"
    assert result["retry"] is False
    assert events == []


def test_existing_connect_attempt_returns_pending_without_second_spawn(tmp_path, monkeypatch):
    from tools import mcp_tool as core
    handler, servers, original, other, events = _reload_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(core, "_server_connecting", {"smart-web"})
    result = handler({"name": "smart-web"})
    assert result["status"] == "pending"
    assert result["retry"] is True
    assert servers["smart-web"] is original
    assert events == []


def test_orphaned_sdk_rpc_prevents_teardown(tmp_path, monkeypatch):
    handler, servers, original, other, events = _reload_fixture(tmp_path, monkeypatch)
    original._inflight_tasks = {object()}
    result = handler({"name": "smart-web"})
    assert result["status"] == "pending"
    assert result["retry"] is True
    assert servers["smart-web"] is original
    assert events == []


def test_shared_connection_refused_without_teardown(tmp_path, monkeypatch):
    from tools import mcp_tool as core
    handler, servers, original, other, events = _reload_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(core, "_server_tool_scopes", {"smart-web": {"other-profile"}})
    result = handler({"name": "smart-web"})
    assert result["status"] == "pending"
    assert result["retry"] is False
    assert servers == {"smart-web": original, "unrelated": other}
    assert events == []


def test_success_replaces_only_target_and_reports_schema(tmp_path, monkeypatch):
    handler, servers, original, other, events = _reload_fixture(tmp_path, monkeypatch)
    result = handler({"name": "smart-web"})
    assert result["status"] == "reloaded"
    assert result["tools"] == ["mcp__smart_web__fetch"]
    assert servers["smart-web"] is not original
    assert servers["unrelated"] is other
    assert events == ["preflight", "preview-closed", "shutdown", "discover", "reprobe"]


def test_between_turns_refresh_is_content_aware(monkeypatch):
    import sys
    from agent.turn_context import _refresh_mcp_tools_between_turns
    from tools import mcp_tool, mcp_tool_agent, mcp_tool_discovery
    assert mcp_tool is sys.modules["tools.mcp_tool"]
    monkeypatch.setattr(mcp_tool_discovery, "has_registered_mcp_tools", lambda: True)
    seen = []
    monkeypatch.setattr(mcp_tool_agent, "refresh_agent_mcp_tools", lambda agent, **kw: seen.append(kw))
    _refresh_mcp_tools_between_turns(SimpleNamespace())
    assert seen == [{"quiet_mode": True, "preserve_prefix": True, "content_aware": True}]


def test_cli_reports_pending_and_success_as_json(monkeypatch, capsys, tmp_path):
    from gateway import mcp_reload_cli
    calls = []

    def request(home, name, *, profile_home, drain_timeout):
        calls.append((home, name, profile_home, drain_timeout))
        return {"status": "pending", "name": name, "retry": True}

    monkeypatch.setattr("gateway.control_socket.reload_gateway_mcp_server", request)
    args = ["smart-web", "--home", str(tmp_path), "--drain-timeout", "2"]
    assert mcp_reload_cli.main(args) == 75
    assert json.loads(capsys.readouterr().out)["status"] == "pending"
    assert calls == [(tmp_path, "smart-web", tmp_path, 2.0)]
    monkeypatch.setattr("gateway.control_socket.reload_gateway_mcp_server",
                        lambda *a, **kw: {"status": "reloaded", "name": "smart-web", "tools": ["fetch"]})
    assert mcp_reload_cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["tools"] == ["fetch"]
    monkeypatch.setattr("gateway.control_socket.reload_gateway_mcp_server",
                        lambda *a, **kw: {"status": "pending", "retry": False})
    assert mcp_reload_cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)["retry"] is False


def test_cli_missing_gateway_fails_closed(monkeypatch, capsys, tmp_path):
    from gateway import mcp_reload_cli
    monkeypatch.setattr("gateway.control_socket.reload_gateway_mcp_server", lambda *a, **kw: None)
    assert mcp_reload_cli.main(["smart-web", "--home", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"


def test_preflight_failure_keeps_old_connection_and_other_servers(tmp_path, monkeypatch):
    from tools import mcp_tool_discovery
    handler, servers, original, other, events = _reload_fixture(tmp_path, monkeypatch)

    async def fail(*args):
        raise RuntimeError("new runtime unavailable")
    monkeypatch.setattr(mcp_tool_discovery, "_connect_server", fail)
    result = handler({"name": "smart-web"})
    assert result["status"] == "failed"
    assert result["old_preserved"] is True
    assert servers == {"smart-web": original, "unrelated": other}
    assert events == []
