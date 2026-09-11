"""Multiplexed gateways discover and reload MCP servers per profile (#95518)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_constants import get_hermes_home, hermes_home_key


@pytest.mark.asyncio
async def test_gateway_boot_discovers_mcp_under_every_profile_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery

    homes = [("default", tmp_path / "default"), ("worker", tmp_path / "worker")]
    for _name, home in homes:
        home.mkdir()
    seen: list[tuple[Path, str]] = []

    def fake_discover() -> list[str]:
        seen.append((get_hermes_home(), threading.current_thread().name))
        return []

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: homes,
    )
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", fake_discover)

    await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True))

    # Ran once per profile, under that profile's home, off the loop thread.
    assert [home for home, _ in seen] == [home for _, home in homes]
    assert all(thread != threading.current_thread().name for _, thread in seen)


@pytest.mark.asyncio
async def test_reload_mcp_only_touches_requesting_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.run import GatewayRunner
    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_lifecycle as _mcp_lifecycle

    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    worker_scope = hermes_home_key(worker_home)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = MagicMock(return_value=worker_home)
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._async_session_store = SimpleNamespace(
        get_or_create_session=MagicMock(side_effect=RuntimeError("skip transcript")),
    )

    monkeypatch.setattr(mcp_tool, "_servers", {"default-srv": object(), "worker-srv": object()})
    monkeypatch.setattr(
        mcp_tool, "_server_scope_keys",
        {"default-srv": hermes_home_key(tmp_path), "worker-srv": worker_scope},
    )
    seen: list[tuple] = []

    def fake_shutdown(*, scope=None) -> None:
        seen.append(("shutdown", scope, get_hermes_home()))

    def fake_discover() -> list[str]:
        seen.append(("discover", get_hermes_home()))
        return []

    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", fake_shutdown)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", fake_discover)

    event = MessageEvent(
        text="/reload-mcp", message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
            chat_type="dm", profile="worker",
        ),
    )
    result = await runner._execute_mcp_reload(event)

    # Entered worker's scope itself, shut down only worker's servers, and
    # reported only worker's servers (default's untouched connection is not
    # "removed").
    assert seen == [
        ("shutdown", worker_scope, worker_home),
        ("discover", worker_home),
    ]
    assert "default-srv" not in result


@pytest.mark.asyncio
async def test_reload_mcp_reports_a_shared_server_to_a_non_owner_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared connection remains visible when its peer profile reloads MCP."""
    from gateway.run import GatewayRunner
    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_lifecycle as _mcp_lifecycle

    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    worker_scope = hermes_home_key(worker_home)
    launch_scope = hermes_home_key(tmp_path / "default")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = MagicMock(return_value=worker_home)
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._async_session_store = SimpleNamespace(
        get_or_create_session=MagicMock(side_effect=RuntimeError("skip transcript")),
    )

    live_server = SimpleNamespace(session=object(), _config={}, _tools=[], tool_timeout=30,
                                  initialize_result=None, _registered_tool_names=[])
    monkeypatch.setattr(mcp_tool, "_servers", {"shared": live_server})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {"shared": launch_scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {"shared": {launch_scope}}, raising=False)
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())
    monkeypatch.setattr(mcp_tool, "_server_connect_errors", {})
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)

    def fake_discover() -> list[str]:
        from tools import mcp_tool_registration as _mcp_registration
        _mcp_registration.register_connected_into_current_scope({"shared": {}})
        return ["mcp__shared__tool"]

    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda **_kwargs: None)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", fake_discover)

    event = MessageEvent(
        text="/reload-mcp", message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
            chat_type="dm", profile="worker",
        ),
    )
    result = await runner._execute_mcp_reload(event)

    assert "No MCP servers connected." not in result
    assert "shared" in result
    assert mcp_tool._server_scope_keys["shared"] == launch_scope
    assert mcp_tool._server_tool_scopes["shared"] == {launch_scope, worker_scope}


@pytest.mark.parametrize("worker_cfg", [
    {"url": "https://worker.example/mcp"},                                   # different route
    {"url": "https://default.example/mcp", "headers": {"Authorization": "Bearer worker"}},  # same route, other credentials
    {"url": "https://default.example/mcp", "env": {"API_TOKEN": "worker"}},
])
def test_scope_visibility_rejects_a_foreign_or_differently_authenticated_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_cfg: dict
) -> None:
    """A profile may only see a live connection whose route AND credentials match its own config;
    otherwise it would call tools as the owning profile's identity."""
    from tools import mcp_tool
    from tools import mcp_tool_registration as _mcp_registration

    worker_scope = hermes_home_key(tmp_path / "worker")
    launch_scope = hermes_home_key(tmp_path / "default")
    live_server = SimpleNamespace(session=object(), _config={"url": "https://default.example/mcp"})
    monkeypatch.setattr(mcp_tool, "_servers", {"shared": live_server})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {"shared": launch_scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {"shared": {launch_scope}}, raising=False)
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)

    assert _mcp_registration.register_connected_into_current_scope({"shared": worker_cfg}) == 0
    assert mcp_tool._server_tool_scopes["shared"] == {launch_scope}


def test_shared_server_tools_are_callable_and_removed_on_non_owner_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import (
        hermes_home_key,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools import mcp_tool
    from tools import mcp_tool_config as _mcp_config
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools.registry import registry

    worker_home = tmp_path / "profiles" / "worker"
    launch_home = tmp_path / "default"
    worker_home.mkdir(parents=True)
    launch_home.mkdir()
    worker_token = set_hermes_home_override(worker_home)
    previous_multiplex = set_multiplex_active(True)
    worker_scope = hermes_home_key()
    launch_scope = hermes_home_key(launch_home)
    tool = SimpleNamespace(
        name="echo",
        description="Echo a value",
        inputSchema={"type": "object", "properties": {}},
        annotations=None,
    )
    server = SimpleNamespace(
        name="shared",
        session=object(),
        _tools=[tool],
        tool_timeout=30,
        _registered_tool_names=[],
        _config={},
        initialize_result=None,
    )
    owner_tool_name = "mcp__shared__echo"
    registry.register(
        owner_tool_name,
        "mcp-shared",
        {"name": owner_tool_name, "description": "Echo a value", "type": "object"},
        lambda **_kwargs: None,
        scope=launch_scope,
    )
    registry.register_toolset_alias("shared", "mcp-shared")
    server._registered_tool_names = [owner_tool_name]
    with mcp_tool._lock:
        saved = {
            "_servers": dict(mcp_tool._servers),
            "_server_scope_keys": dict(mcp_tool._server_scope_keys),
            "_server_tool_scopes": dict(mcp_tool._server_tool_scopes),
            "_mcp_tool_server_names": dict(mcp_tool._mcp_tool_server_names),
        }
        mcp_tool._servers.clear()
        mcp_tool._server_scope_keys.clear()
        mcp_tool._server_tool_scopes.clear()
        mcp_tool._mcp_tool_server_names.clear()
        mcp_tool._servers["shared"] = server
        mcp_tool._server_scope_keys["shared"] = launch_scope
        mcp_tool._server_tool_scopes["shared"] = {launch_scope}

    try:
        monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", lambda: True)
        monkeypatch.setattr(_mcp_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
        assert _mcp_discovery.register_mcp_servers({"shared": {}})
        tool_names = registry.get_tool_names_for_toolset("mcp-shared")
        assert tool_names
        assert callable(registry.get_entry(tool_names[0]).handler)

        # Changing the worker route removes only the worker overlay; the shared
        # live connection and launch owner remain intact.
        assert _mcp_discovery.register_mcp_servers(
            {"shared": {"url": "https://worker.example/mcp"}}
        ) == []
        assert registry.get_tool_names_for_toolset("mcp-shared") == []
        with mcp_tool._lock:
            assert mcp_tool._server_scope_keys["shared"] == launch_scope
            assert mcp_tool._server_tool_scopes["shared"] == {launch_scope}
            assert mcp_tool._servers["shared"] is server
        assert registry.snapshot_registration(owner_tool_name, scope=launch_scope) is not None
        assert registry.get_toolset_alias_target("shared") == "mcp-shared"

        # Removing the server from the worker config has the same scoped cleanup.
        assert _mcp_discovery.register_mcp_servers({}) == []
        assert registry.get_tool_names_for_toolset("mcp-shared") == []
        with mcp_tool._lock:
            assert mcp_tool._server_scope_keys["shared"] == launch_scope
            assert mcp_tool._server_tool_scopes["shared"] == {launch_scope}
            assert mcp_tool._servers["shared"] is server
    finally:
        for tool_name in list(registry.get_tool_names_for_toolset("mcp-shared")):
            registry.deregister(tool_name, scope=worker_scope)
        registry.deregister(owner_tool_name, scope=launch_scope)
        with mcp_tool._lock:
            for name, value in saved.items():
                target = getattr(mcp_tool, name)
                target.clear()
                target.update(value)
        set_multiplex_active(previous_multiplex)
        reset_hermes_home_override(worker_token)


def test_deregister_scope_kwarg_targets_overlay_and_keeps_plugin_confinement() -> None:
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register("mcp__s__t", "mcp-s", {"name": "mcp__s__t", "description": "d"},
                 lambda **kw: None, scope="/home/p1")
    assert reg.snapshot_registration("mcp__s__t", scope="/home/p1") is not None

    reg.deregister("mcp__s__t")  # unscoped: global slot only, overlay untouched
    assert reg.snapshot_registration("mcp__s__t", scope="/home/p1") is not None

    reg.deregister("mcp__s__t", scope="/home/p1")
    assert reg.snapshot_registration("mcp__s__t", scope="/home/p1") is None

    # A plugin module may not name another profile's overlay.
    reg._plugin_module_scopes["hermes_plugins.p"] = {"/home/p1"}
    reg._caller_module = staticmethod(lambda: "hermes_plugins.p")
    with pytest.raises(PermissionError):
        reg.deregister("anything", scope="/home/p2")
