"""Split-runtime client tools are relayed to the caller on every dispatch path."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import invoke_tool
from agent.tool_executor import execute_tool_calls_sequential
from gateway.platforms.api_server_client_tools import apply_client_tools
from run_agent import AIAgent
from tools import client_tool_gateway as ctg

SESSION_KEY = "run_dispatch_test"
CLIENT_TOOL = {
    "type": "function",
    "function": {"name": "set_timer", "description": "Set a timer on the client device.",
                 "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}}}},
}


@pytest.fixture(autouse=True)
def _clean_client_tool_registry():
    for table in (ctg._entries, ctg._session_index, ctg._notify_cbs):
        table.clear()
    yield
    for table in (ctg._entries, ctg._session_index, ctg._notify_cbs):
        table.clear()


@pytest.fixture
def agent(tmp_path: Path) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True)
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    apply_client_tools(agent, [CLIENT_TOOL], SESSION_KEY)
    return agent


def _answer_relayed_calls(result: str) -> list:
    """Register the run's notify hook: record each relayed call and resolve it from another
    thread, as the caller's POST /tool_result would."""
    relayed: list = []

    def _notify(entry) -> None:
        relayed.append(entry.signature())
        threading.Thread(
            target=ctg.resolve_client_tool, args=(SESSION_KEY, entry.call_id, result), daemon=True).start()

    ctg.register_notify(SESSION_KEY, _notify)
    return relayed


def test_apply_client_tools_exposes_the_tool_to_the_model(agent):
    assert "set_timer" in agent.valid_tool_names
    assert [t["function"]["name"] for t in agent.tools].count("set_timer") == 1


def test_sequential_path_relays_instead_of_dispatching_on_the_host(agent):
    relayed = _answer_relayed_calls("timer set")
    tool_call = SimpleNamespace(
        id="call_seq_1", type="function",
        function=SimpleNamespace(name="set_timer", arguments=json.dumps({"minutes": 5})))
    messages: list = []
    with patch("model_tools.handle_function_call") as host_dispatch:
        execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[tool_call]), messages, "task-1")
    host_dispatch.assert_not_called()
    assert [(c["call_id"], c["name"], c["arguments"]) for c in relayed] == [
        ("call_seq_1", "set_timer", {"minutes": 5})]
    results = [m for m in messages if m.get("role") == "tool"]
    assert len(results) == 1 and results[0]["tool_call_id"] == "call_seq_1"
    assert "timer set" in results[0]["content"]


def test_concurrent_path_relays_instead_of_dispatching_on_the_host(agent):
    relayed = _answer_relayed_calls("timer set")
    with patch("model_tools.handle_function_call") as host_dispatch:
        result = invoke_tool(agent, "set_timer", {"minutes": 5}, "task-1", tool_call_id="call_par_1")
    host_dispatch.assert_not_called()
    assert relayed and relayed[0]["call_id"] == "call_par_1"
    assert result == "timer set"


def test_turn_boundary_rebuild_keeps_client_tools(agent):
    from tools.mcp_tool_agent import _reinject_post_build_tools

    tools_list, name_set = [], set()
    _reinject_post_build_tools(agent, tools_list, name_set)
    assert "set_timer" in name_set
    assert [t["function"]["name"] for t in tools_list] == ["set_timer"]
