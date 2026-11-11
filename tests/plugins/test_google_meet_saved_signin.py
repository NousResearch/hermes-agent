"""A Google sign-in saved by ``hermes meet auth`` is used whenever a bot joins a meeting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home

MEET_URL = "https://meet.google.com/abc-defg-hij"


class _BotProcess:
    """Stands in for the detached meet_bot child; records the environment it was spawned with."""

    spawned_env: dict[str, str] | None = None

    def __init__(self, argv, **kwargs):
        type(self).spawned_env = kwargs["env"]
        self.pid = 4242


@pytest.fixture
def saved_signin(monkeypatch) -> Path:
    from plugins.google_meet import process_manager, tools

    # Host prerequisites (Linux/macOS + Playwright) and the child process are the faked boundaries.
    monkeypatch.setattr(tools, "check_meet_requirements", lambda: True)
    monkeypatch.setattr(process_manager.subprocess, "Popen", _BotProcess)
    monkeypatch.setattr(_BotProcess, "spawned_env", None)
    state = get_hermes_home() / "workspace" / "meetings" / "auth.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")
    return state


def test_agent_meet_join_uses_saved_signin(saved_signin):
    from plugins.google_meet import register
    from tools.registry import registry

    class _Ctx:
        def register_tool(self, name, toolset, schema, handler, **kw):
            registry.register(name=name, toolset=toolset, schema=schema, handler=handler, override=True, **kw)

        def register_cli_command(self, **kw):
            pass

        def register_hook(self, *a, **kw):
            pass

    register(_Ctx())
    result = json.loads(registry.dispatch("meet_join", {"url": MEET_URL}))

    assert result["success"] is True
    assert _BotProcess.spawned_env.get("HERMES_MEET_AUTH_STATE") == str(saved_signin)


@pytest.mark.anyio
async def test_remote_node_start_bot_uses_the_node_saved_signin(saved_signin, tmp_path):
    from plugins.google_meet.node import protocol
    from plugins.google_meet.node.server import NodeServer

    server = NodeServer(token_path=tmp_path / "node_token.json")
    request = protocol.make_request("start_bot", server.ensure_token(), {"url": MEET_URL, "guest_name": "Hermes Agent"})
    response = await server._handle_request(request)

    assert response["type"] == "response", response
    assert _BotProcess.spawned_env.get("HERMES_MEET_AUTH_STATE") == str(saved_signin)
