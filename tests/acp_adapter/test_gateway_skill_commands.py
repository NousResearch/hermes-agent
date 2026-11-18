"""Gateway-path editor commands list the session's skills from the owner's ``commands.catalog``."""
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.gateway_session_meta import COMMANDS, SKILL_COMMAND_HINT, available_commands_update

CATALOG = {
    "pairs": [["/help", "Show help"], ["/wire-probe", "Probe   the\nwire"], ["/no-description", ""],
              ["/status", "A skill named like a built-in"]],
    "skills": {"/wire-probe": {"usage": 0, "origin": "local"}, "/no-description": {"usage": 0, "origin": "local"},
               "/status": {"usage": 0, "origin": "local"}},
}


def _names(update):
    return [command.name for command in update.available_commands]


def test_skills_follow_the_built_in_commands():
    update = available_commands_update(CATALOG)
    assert _names(update) == [*COMMANDS, "no-description", "wire-probe"]
    probe = update.available_commands[-1]
    assert probe.description == "Probe the wire"
    assert probe.input.root.hint == SKILL_COMMAND_HINT
    assert update.available_commands[-2].description == "Run the no-description skill"


def test_a_skill_named_like_a_built_in_keeps_the_built_in_entry():
    update = available_commands_update(CATALOG)
    assert _names(update).count("status") == 1
    status = next(command for command in update.available_commands if command.name == "status")
    assert status.description == COMMANDS["status"][0]


@pytest.mark.parametrize("catalog", [None, {}, {"skills": None}, "not a catalog"])
def test_no_catalog_advertises_the_built_in_commands(catalog):
    assert _names(available_commands_update(catalog)) == list(COMMANDS)


async def _published(monkeypatch, rpc):
    agent = GatewayACPAgent()
    owner = AsyncMock()
    owner.rpc.side_effect = rpc
    monkeypatch.setattr(agent, "_client", AsyncMock(return_value=owner))
    monkeypatch.setattr(agent, "_project_usage", AsyncMock())
    agent._conn = AsyncMock()
    await agent._after_attach("session", "editor")
    updates = [call.kwargs["update"] for call in agent._conn.session_update.await_args_list
               if call.kwargs["update"].session_update == "available_commands_update"]
    assert len(updates) == 1
    return owner, updates[0]


@pytest.mark.asyncio
async def test_attach_asks_the_owner_for_this_sessions_catalog(monkeypatch):
    async def rpc(method, **params):
        return CATALOG if method == "commands.catalog" else {}

    owner, update = await _published(monkeypatch, rpc)
    owner.rpc.assert_any_await("commands.catalog", session_id="session")
    assert "wire-probe" in _names(update)


@pytest.mark.asyncio
async def test_a_refused_catalog_still_publishes_the_built_in_commands(monkeypatch):
    async def rpc(method, **params):
        if method == "commands.catalog":
            raise TimeoutError
        return {}

    _owner, update = await _published(monkeypatch, rpc)
    assert _names(update) == list(COMMANDS)
