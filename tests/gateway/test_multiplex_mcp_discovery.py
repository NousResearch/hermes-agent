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
        lambda multiplex: homes,
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
async def test_reload_mcp_formats_scoped_connection_keys_before_refreshing_cached_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connection-ledger tuple keys are internal; reload reports server names and completes refresh."""
    from gateway.run import GatewayRunner
    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_lifecycle as _mcp_lifecycle

    launch_scope = hermes_home_key(tmp_path / "default")
    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    worker_scope = hermes_home_key(worker_home)
    launch_key = (launch_scope, "default-srv")
    worker_key = (worker_scope, "worker-srv")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = MagicMock(return_value=worker_home)
    runner._mcp_reload_refresh_cached_agents = MagicMock()
    runner._async_session_store = SimpleNamespace(
        get_or_create_session=MagicMock(side_effect=RuntimeError("skip transcript")),
    )

    monkeypatch.setattr(mcp_tool, "_servers", {launch_key: object(), worker_key: object()})
    monkeypatch.setattr(
        mcp_tool, "_server_scope_keys",
        {launch_key: launch_scope, worker_key: worker_scope},
    )
    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda **_kwargs: None)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda: [])

    event = MessageEvent(
        text="/reload-mcp", message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
            chat_type="dm", profile="worker",
        ),
    )
    result = await runner._execute_mcp_reload(event)

    assert "MCP reload failed" not in result
    assert "worker-srv" in result
    assert "default-srv" not in result
    runner._mcp_reload_refresh_cached_agents.assert_called_once_with(True, "worker")


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
    private_key = f"shared::profile::{launch_scope}"
    monkeypatch.setattr(mcp_tool, "_servers", {private_key: live_server})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {private_key: "shared"})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {private_key: launch_scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {
        private_key: {launch_scope, worker_scope},
    }, raising=False)
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names_by_scope", {
        worker_scope: {"mcp__shared__tool": private_key},
    }, raising=False)
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
    assert mcp_tool._server_scope_keys[private_key] == launch_scope
    assert mcp_tool._server_tool_scopes[private_key] == {launch_scope, worker_scope}


@pytest.mark.asyncio
async def test_reload_mcp_reports_scoped_lazy_tools_as_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy schema-cache tools count as available without a live transport task."""
    from gateway.run import GatewayRunner
    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_lifecycle as _mcp_lifecycle

    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    worker_scope = hermes_home_key(worker_home)
    private_key = f"shared::profile::{worker_scope}"
    tool_name = "mcp__shared__lazy_reload_invariant"

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = MagicMock(return_value=worker_home)
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._async_session_store = SimpleNamespace(
        get_or_create_session=MagicMock(side_effect=RuntimeError("skip transcript")),
    )

    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {private_key: worker_scope})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {private_key: "shared"})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {private_key: {worker_scope}})
    monkeypatch.setattr(mcp_tool, "_lazy_server_tool_names", {private_key: [tool_name]})
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names_by_scope", {})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)

    def fake_discover() -> list[str]:
        mcp_tool._mcp_tool_server_names_by_scope[worker_scope] = {tool_name: private_key}
        return [tool_name]

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

    assert "No MCP tools available" not in result
    assert "1 tool(s) available" in result
    assert "1 server(s)" in result
    assert "shared" in result


def test_failed_profile_owned_connection_does_not_change_peer_parallel_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed OAuth connection remains an ownership boundary for peer discovery."""
    from tools import mcp_tool
    from tools import mcp_tool_discovery as discovery

    owner_scope = hermes_home_key(tmp_path / "owner")
    worker_scope = hermes_home_key(tmp_path / "worker")
    owner_key = f"shared::profile::{owner_scope}"
    worker_key = f"shared::profile::{worker_scope}"
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {})
    monkeypatch.setattr(mcp_tool, "_server_connect_errors", {owner_key: "expired token"})
    monkeypatch.setattr(mcp_tool, "_server_connect_retry_after", {owner_key: 999999.0})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {owner_key: owner_scope})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {owner_key: "shared"})
    monkeypatch.setattr(mcp_tool, "_parallel_safe_servers", {owner_key})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)

    selected = discovery._select_new_servers({
        "shared": {"auth": "oauth", "supports_parallel_tool_calls": False},
    })

    assert list(selected) == [worker_key]
    assert owner_key in mcp_tool._parallel_safe_servers
    assert worker_key not in mcp_tool._parallel_safe_servers


def test_lazy_profile_overlay_survives_repeated_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached lazy overlay is valid without a live task and is not stale."""
    from tools import mcp_tool
    from tools import mcp_tool_registration as registration
    from tools.registry import registry

    scope = hermes_home_key(tmp_path / "worker")
    key = f"shared::profile::{scope}"
    tool_name = "mcp__shared__echo"
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {key: "shared"})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {key: scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {key: {scope}})
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {key: {"auth": "oauth", "lazy": True}})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: scope)
    registry.register(tool_name, "mcp-shared", {"name": tool_name}, lambda **_kwargs: None, scope=scope)

    assert registration._register_connected_into_current_scope({
        "shared": {"auth": "oauth", "lazy": True},
    }) == 0
    assert registry.snapshot_registration(tool_name, scope=scope) is not None


@pytest.mark.parametrize("current_servers", [
    {},
    {"shared": {"auth": "oauth", "lazy": True, "url": "https://changed.example/mcp"}},
])
def test_stale_lazy_profile_overlay_is_removed_for_changed_or_removed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current_servers: dict
) -> None:
    """A lazy overlay must not hide a changed or removed server from discovery."""
    from tools import mcp_tool
    from tools import mcp_tool_registration as registration
    from tools.mcp_schema_cache import config_fingerprint
    from tools.registry import registry

    scope = hermes_home_key(tmp_path / "worker")
    key = f"shared::profile::{scope}"
    tool_name = "mcp__shared__echo"
    original_config = {"auth": "oauth", "lazy": True}
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {key: "shared"})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {key: scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {key: {scope}})
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {key: original_config})
    monkeypatch.setattr(mcp_tool, "_lazy_server_fingerprints", {
        key: config_fingerprint(original_config),
    })
    monkeypatch.setattr(mcp_tool, "_lazy_server_tool_names", {key: [tool_name]})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: scope)
    registry.register(tool_name, "mcp-shared", {"name": tool_name}, lambda **_kwargs: None, scope=scope)

    assert registration._register_connected_into_current_scope(current_servers) == 0
    assert registry.snapshot_registration(tool_name, scope=scope) is None
    assert key not in mcp_tool._lazy_server_configs
    assert key not in mcp_tool._lazy_server_fingerprints
    assert key not in mcp_tool._lazy_server_tool_names


def test_scoped_teardown_restores_alias_from_surviving_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing one profile's last tool must not remove the surviving profile alias."""
    from tools import mcp_tool
    from tools import mcp_tool_registration as registration
    from tools.registry import registry

    first_scope = hermes_home_key(tmp_path / "first")
    second_scope = hermes_home_key(tmp_path / "second")
    first_key = "shared"
    second_key = "shared::profile::second"
    tool_name = "mcp__shared__alias"
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {first_key: "shared", second_key: "shared"})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {
        first_key: {first_scope}, second_key: {second_scope},
    })
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", {})
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names_by_scope", {})
    for scope, key in ((first_scope, first_key), (second_scope, second_key)):
        registry.register(tool_name, "mcp-shared", {"name": tool_name}, lambda **_kwargs: None, scope=scope)
        registration._track_mcp_tool_server(tool_name, key, scope=scope)
    registry.register_toolset_alias("shared", "mcp-shared")

    registration._deregister_mcp_tool_all_scopes(first_key, tool_name)

    assert registry.snapshot_registration(tool_name, scope=second_scope) is not None
    assert registry.get_toolset_alias_target("shared") == "mcp-shared"


@pytest.mark.asyncio
async def test_reload_mcp_projects_scoped_connection_keys_to_public_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tuple connection keys stay server names in the public reload summary."""
    from gateway.run import GatewayRunner
    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_lifecycle as _mcp_lifecycle

    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    worker_scope = hermes_home_key(worker_home)
    launch_scope = hermes_home_key(tmp_path / "default")
    connection_key = (launch_scope, "shared")

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
    monkeypatch.setattr(mcp_tool, "_servers", {connection_key: live_server})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {connection_key: launch_scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {connection_key: {worker_scope}})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)
    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda **_kwargs: None)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda: [])

    event = MessageEvent(
        text="/reload-mcp", message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
            chat_type="dm", profile="worker",
        ),
    )

    result = await runner._execute_mcp_reload(event)

    assert "shared" in result


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


@pytest.mark.parametrize("auth_config", [
    {"auth": "oauth", "oauth": {"client_id": "shared-client"}},
    {"client_cert": "/profiles/default/client.pem", "client_key": "/profiles/default/client.key"},
])
def test_scope_visibility_rejects_profile_owned_auth_connection_even_when_config_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auth_config: dict
) -> None:
    """Profile-owned OAuth and mTLS sessions never cross a multiplex registry scope.

    Equal config cannot prove equal credentials: OAuth tokens live under each profile home,
    and a live mTLS client can retain profile-owned certificate material. The owning scope
    must continue to reuse its own connection, while a peer scope must connect independently.
    """
    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_registration as _mcp_registration

    worker_scope = hermes_home_key(tmp_path / "worker")
    launch_scope = hermes_home_key(tmp_path / "default")
    live_config = {"url": "https://shared.example/mcp", **auth_config}
    live_server = SimpleNamespace(session=object(), _config=live_config, _tools=[], tool_timeout=30)
    monkeypatch.setattr(mcp_tool, "_servers", {"shared": live_server})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {"shared": "shared"})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {"shared": launch_scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {"shared": {launch_scope}}, raising=False)
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)

    assert _mcp_registration._connection_reusable_in_scope(
        "shared", live_server, live_config, launch_scope
    )
    assert not _mcp_registration._connection_reusable_in_scope(
        "shared", live_server, live_config, worker_scope
    )
    selected = _mcp_discovery._select_new_servers({"shared": live_config})
    assert list(selected.values()) == [live_config]
    assert list(selected) == [f"shared::profile::{worker_scope}"]
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


def test_profile_owned_lazy_and_delimiter_names_keep_public_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy foreign connections get a private key, while configured names remain lossless."""
    from tools import mcp_tool
    from tools import mcp_tool_discovery as discovery

    worker_scope = hermes_home_key(tmp_path / "worker")
    owner_scope = hermes_home_key(tmp_path / "owner")
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {"shared": {"auth": "oauth", "lazy": True}})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {"shared": owner_scope})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {"shared": "shared"})
    monkeypatch.setattr(mcp_tool, "_server_connect_errors", {})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: worker_scope)

    lazy = discovery._select_new_servers({"shared": {"auth": "oauth", "lazy": True}})
    assert list(lazy) == [f"shared::profile::{worker_scope}"]
    assert mcp_tool._server_public_names[list(lazy)[0]] == "shared"

    configured = "acme::profile::prod"
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {})
    selected = discovery._select_new_servers({configured: {"url": "https://example.invalid/mcp"}})
    assert list(selected) == [configured]
    assert mcp_tool._server_public_names[configured] == configured


def test_scoped_mcp_provenance_survives_one_connection_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical public tools retain the surviving profile's parallel-call policy."""
    from tools import mcp_tool
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_registration as registration
    from tools.registry import registry

    first_scope = hermes_home_key(tmp_path / "first")
    second_scope = hermes_home_key(tmp_path / "second")
    tool_name = "mcp__shared__echo"
    first_key = "shared"
    second_key = "shared::profile::second"
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", {})
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names_by_scope", {})
    monkeypatch.setattr(mcp_tool, "_parallel_safe_servers", {first_key, second_key})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {first_key: {first_scope}, second_key: {second_scope}})
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {first_key: "shared", second_key: "shared"})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: second_scope)
    for scope in (first_scope, second_scope):
        registry.register(tool_name, "mcp-shared", {"name": tool_name}, lambda **_kw: None, scope=scope)
    registration._track_mcp_tool_server(tool_name, first_key, scope=first_scope)
    registration._track_mcp_tool_server(tool_name, second_key, scope=second_scope)

    registration._deregister_mcp_tool_all_scopes(first_key, tool_name)

    assert registry.snapshot_registration(tool_name, scope=first_scope) is None
    assert registry.snapshot_registration(tool_name, scope=second_scope) is not None
    assert discovery.is_mcp_tool_parallel_safe(tool_name) is True
