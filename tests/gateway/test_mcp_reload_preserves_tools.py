from collections import OrderedDict
import asyncio
from datetime import datetime
from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def test_reload_mcp_preserves_cached_non_mcp_tools(monkeypatch):
    from gateway.run import GatewayRunner
    import model_tools
    import tools.mcp_tool as mcp_tool

    source = _source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner = object.__new__(GatewayRunner)
    agent = SimpleNamespace(
        tools=[
            _tool("hindsight_recall"),
            _tool("terminal"),
            _tool("mcp_old_search"),
        ],
        valid_tool_names={"hindsight_recall", "terminal", "mcp_old_search"},
    )
    runner._agent_cache = OrderedDict({"telegram:u1:c1": (agent, "sig")})
    runner._agent_cache_lock = Lock()
    runner._running_agents = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry

    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: ["mcp_new_search"])
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda quiet_mode=True: [
            _tool("hindsight_recall"),
            _tool("mcp_new_search"),
        ],
    )

    event = MessageEvent(text="/reload-mcp", source=source, message_id="m1")
    asyncio.run(runner._execute_mcp_reload(event))

    names = [tool["function"]["name"] for tool in agent.tools]
    assert names == ["hindsight_recall", "terminal", "mcp_new_search"]
    assert agent.valid_tool_names == {
        "hindsight_recall",
        "terminal",
        "mcp_new_search",
    }
    runner.session_store.append_to_transcript.assert_called_once()


def test_replace_agent_mcp_tools_removes_stale_mcp_only():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        tools=[
            _tool("plugin_recall"),
            _tool("mcp_old_read"),
            _tool("mcp_old_write"),
        ],
        valid_tool_names={"plugin_recall", "mcp_old_read", "mcp_old_write"},
    )

    changed = GatewayRunner._replace_agent_mcp_tools(agent, [_tool("mcp_new_read")])

    assert changed is True
    assert [tool["function"]["name"] for tool in agent.tools] == [
        "plugin_recall",
        "mcp_new_read",
    ]
    assert agent.valid_tool_names == {"plugin_recall", "mcp_new_read"}
