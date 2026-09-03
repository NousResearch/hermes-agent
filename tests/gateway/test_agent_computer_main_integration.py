"""Prove BWM-796 wiring on current-main integration surfaces.

These tests exercise live imports against the integrated tree. They do not
read source files. They pin that current-main config, routers, toolsets,
TUI methods, and WebSocket teardown still work, and that the Agent Computer
hooks sit beside them rather than replacing them.
"""

from __future__ import annotations

import asyncio

from hermes_cli.config import DEFAULT_CONFIG, load_config
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS, resolve_toolset


COMPUTER_TOOLS = {
    "computer_ensure",
    "computer_status",
    "computer_wake",
    "computer_observe",
    "computer_act",
}

COMPUTER_RPC = {
    "computer.ensure",
    "computer.status",
    "computer.list",
    "computer.wake",
    "computer.observe",
    "computer.act",
    "computer.takeover",
    "computer.takeover.connect",
    "computer.give_back",
    "computer.identity.create",
    "computer.identity.attach",
    "computer.owner_disconnect",
}

# Current-main methods that must remain registered after the agent-computer
# module is folded into tui_gateway.server.
MAIN_RPC = {
    "gateway.capabilities",
    "session.list",
    "complete.slash",
    "browser.controller.register",
    "browser.controller.detach",
}


def test_config_defaults_keep_current_main_and_resolve_runtime():
    from gateway.agent_computer import resolve_runtime_name

    assert DEFAULT_CONFIG["agent_computer"]["runtime"] == "memory"
    assert "checkpoints" in DEFAULT_CONFIG
    assert "display" in DEFAULT_CONFIG
    loaded = load_config()
    assert loaded["agent_computer"]["runtime"] == "memory"
    assert loaded["display"]["background_process_notifications"]
    assert resolve_runtime_name(env="", config=loaded) == "memory"
    assert resolve_runtime_name(
        env="",
        config={"agent_computer": {"runtime": "chromium"}},
    ) == "chromium"


def test_toolsets_keep_current_main_and_opt_in_agent_computer():
    assert "agent_computer" in TOOLSETS
    assert set(resolve_toolset("agent_computer")) == COMPUTER_TOOLS
    assert COMPUTER_TOOLS.isdisjoint(_HERMES_CORE_TOOLS)
    assert COMPUTER_TOOLS.isdisjoint(resolve_toolset("hermes-cli"))
    assert COMPUTER_TOOLS.isdisjoint(resolve_toolset("hermes-telegram"))
    assert "web_search" in resolve_toolset("hermes-cli")
    assert "terminal" in resolve_toolset("hermes-cli")
    for name, spec in TOOLSETS.items():
        if name == "agent_computer":
            continue
        assert COMPUTER_TOOLS.isdisjoint(set(spec.get("tools") or ())), name


def test_web_server_keeps_existing_routers_and_mounts_agent_computer():
    from hermes_cli.web_server import CONFIG_SCHEMA, app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/agent-computers" in paths
    assert "/api/agent-computers/ensure" in paths
    assert "/api/tools/toolsets" in paths
    assert any(path.startswith("/api/gateway") for path in paths)
    runtime = CONFIG_SCHEMA["agent_computer.runtime"]
    assert runtime["category"] == "agent"
    assert sum(1 for entry in CONFIG_SCHEMA.values() if entry["category"] == "agent_computer") == 0


def test_tui_gateway_keeps_current_main_methods_and_registers_computer():
    import tui_gateway.server as server

    missing_main = MAIN_RPC - set(server._methods)
    missing_computer = COMPUTER_RPC - set(server._methods)
    assert not missing_main, missing_main
    assert not missing_computer, missing_computer


def _run_ws_disconnect(monkeypatch, seed):
    from tui_gateway import server
    from tui_gateway import ws as ws_mod

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_finalize_session", lambda s, end_reason="tui_close": None)

    created = []
    real_transport = ws_mod.WSTransport
    monkeypatch.setattr(
        ws_mod,
        "WSTransport",
        lambda ws, loop, **kw: created.append(real_transport(ws, loop, **kw)) or created[-1],
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            seed(created[0])
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))
    return created


def test_ws_disconnect_runs_agent_computer_release_beside_current_main_cleanup(monkeypatch):
    from tui_gateway import server

    import gateway.agent_computer as ac
    import gateway.browser_control_broker as broker_mod

    ac_released = []
    broker_released = []
    wake_released = []

    class FakeBroker:
        def disconnect_owner(self, transport):
            broker_released.append(transport)
            return 1

    monkeypatch.setattr(broker_mod, "get_browser_control_broker", lambda: FakeBroker())
    monkeypatch.setattr(
        ac,
        "release_owner_for_transport_if_active",
        lambda transport: ac_released.append(transport) or 1,
    )
    monkeypatch.setattr(
        server,
        "_release_wake_for_transport",
        lambda transport: wake_released.append(transport) or True,
    )

    sessions_closed = []
    real_close = server._close_sessions_for_transport

    def _close(transport, end_reason=None):
        sessions_closed.append((transport, end_reason))
        return real_close(transport, end_reason=end_reason)

    monkeypatch.setattr(server, "_close_sessions_for_transport", _close)

    server._sessions.clear()
    server._live_transports.clear()
    try:
        created = _run_ws_disconnect(monkeypatch, lambda _transport: None)

        assert created
        assert ac_released == created
        assert broker_released == created
        assert wake_released == created
        assert sessions_closed == [(created[0], "ws_disconnect")]
        assert not server._live_transports
    finally:
        server._sessions.clear()
        server._live_transports.clear()
