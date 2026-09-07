"""Multiplexed gateways discover and reload MCP servers per profile (#95518)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
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
