"""Behavior contracts for gateway /workspace management."""

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="workspace-user",
        chat_id="workspace-chat",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


@pytest.fixture
def workspace_runner(tmp_path):
    from gateway.run import GatewayRunner

    home = tmp_path / ".hermes"
    home.mkdir()
    token = set_hermes_home_override(home)
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner.session_store = SessionStore(home / "gateway-sessions", config)
    runner._async_session_store = None
    runner._session_db = SimpleNamespace(_db=runner.session_store._db)
    runner._agent_cache = {"unchanged": object()}

    try:
        yield runner, home
    finally:
        from tools.terminal_tool import clear_task_env_overrides

        for entry in runner.session_store._entries.values():
            clear_task_env_overrides(entry.session_id)
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_workspace_switch_updates_tools_without_rebuilding_agent(workspace_runner):
    runner, home = workspace_runner
    target = home / "projects" / "alpha"
    cached_agent = runner._agent_cache["unchanged"]

    created = await runner._handle_workspace_command(
        _event(f'/workspace new alpha "{target}"')
    )
    switched = await runner._handle_workspace_command(_event("/workspace alpha"))

    entry = await runner.async_session_store.get_or_create_session(_source())
    from tools.terminal_tool import resolve_task_overrides

    assert "Registered workspace 'alpha'" in created
    assert "Switched workspace to 'alpha'" in switched
    assert target.is_dir()
    assert entry.workspace_name == "alpha"
    assert entry.workspace_cwd == str(target.resolve())
    overrides = resolve_task_overrides(entry.session_id)
    assert overrides["cwd"] == str(target.resolve())
    assert overrides["env_type"] == "local"
    assert runner._agent_cache["unchanged"] is cached_agent


@pytest.mark.asyncio
async def test_workspace_registry_persists_and_remove_never_deletes_files(workspace_runner):
    runner, home = workspace_runner

    await runner._handle_workspace_command(_event("/workspace new beta"))
    await runner._handle_workspace_command(_event("/workspace switch beta"))
    listed = await runner._handle_workspace_command(_event("/workspace list"))
    target = home / "workspaces" / "beta"
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    removed = await runner._handle_workspace_command(_event("/workspace remove beta"))

    from hermes_cli.config import load_config

    entry = await runner.async_session_store.get_or_create_session(_source())
    assert f"Current: beta ({target.resolve()})" in listed
    assert f"* beta: {target.resolve()}" in listed
    assert "Files were not deleted" in removed
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert "beta" not in load_config().get("gateway", {}).get("workspaces", {})
    assert entry.workspace_name is None
    assert entry.workspace_cwd is None


def test_workspace_selection_survives_session_routing_round_trip():
    from datetime import datetime
    from gateway.session import SessionEntry

    entry = SessionEntry(
        session_key="agent:main:telegram:dm:u:c",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        workspace_name="alpha",
        workspace_cwd="/tmp/alpha",
    )

    restored = SessionEntry.from_dict(entry.to_dict())
    assert (restored.workspace_name, restored.workspace_cwd) == (
        entry.workspace_name,
        entry.workspace_cwd,
    )
