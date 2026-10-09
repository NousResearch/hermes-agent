"""Exact tool grants are a session boundary, not merely model schema hints."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

ADS_TOOLS = [
    "web_search", "web_extract", "todo_list", "clarify", "kanban_show",
    "kanban_heartbeat", "kanban_comment", "kanban_complete", "kanban_block",
]


def schema(name):
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}},
    }}


def make_agent(tmp_path, allowed, *, candidates=None, injected=(), receipt_required=False):
    # Exercise the real profile-aware config loader; no real credentials or API calls.
    (tmp_path / "config.yaml").write_text(json.dumps({
        "agent": {"allowed_tools": allowed, "require_execution_receipt": receipt_required}, "compression": {"enabled": False},
    }))
    candidates = candidates or [*ADS_TOOLS, "terminal", "kanban_create", "mcp__demo__delete"]

    def inject(agent):
        agent._context_engine_tool_names = set()
        agent.tools.extend(schema(name) for name in injected)
        agent.valid_tool_names.update(injected)

    with (
        patch("model_tools.get_tool_definitions", return_value=[schema(n) for n in candidates]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.agent_init._inject_context_engine_tools", side_effect=inject),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            model="gpt-4.1", quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def test_exact_grant_applies_after_all_schema_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent(tmp_path, ADS_TOOLS, injected=[
        "plugin_destroy", "desktop_project", "memory_provider_write", "context_engine_write",
    ])
    assert agent.valid_tool_names == set(ADS_TOOLS)
    assert {t["function"]["name"] for t in agent.tools} == set(ADS_TOOLS)


def test_receipt_required_profile_cannot_start_unowned_agent(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    from hermes_cli.kanban_db_policy import PolicyViolation
    with pytest.raises(PolicyViolation, match='approved owned Kanban run'):
        make_agent(tmp_path, ADS_TOOLS, receipt_required=True)


@pytest.mark.parametrize("mode", ["sequential", "concurrent", "invoke"])
@pytest.mark.parametrize("name", ["terminal", "todo_list", "kanban_create", "plugin_destroy", "mcp__demo__delete", "desktop_project"])
def test_execution_denied_even_after_schema_mutation(tmp_path, monkeypatch, mode, name):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent(tmp_path, ["web_search"])
    agent.tools.append(schema(name))
    agent.valid_tool_names.add(name)
    agent._cached_system_prompt = "unchanged cached prefix"
    handler = MagicMock(return_value='{"executed": true}')
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    # Real dispatch, harmless substituted handlers; prove neither inline nor registry runs.
    with (
        patch("model_tools.registry.dispatch", handler),
        patch.dict(INLINE_TOOL_EXECUTORS, {"todo_list": handler}),
    ):
        if mode == "invoke":
            content = agent._invoke_tool(name, {}, "task", pre_tool_block_checked=True)
        else:
            tc = SimpleNamespace(id="denied", type="function", function=SimpleNamespace(name=name, arguments="{}"))
            messages = []
            getattr(agent, f"_execute_tool_calls_{mode}")(SimpleNamespace(tool_calls=[tc]), messages, "task")
            content = messages[0]["content"]
    assert "agent.allowed_tools" in content
    handler.assert_not_called()
    assert agent._cached_system_prompt == "unchanged cached prefix"


@pytest.mark.parametrize("mode", ["sequential", "concurrent", "invoke"])
def test_grant_snapshot_survives_other_profile_and_config_changes(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent(tmp_path, ["web_search"])
    (tmp_path / "config.yaml").write_text('{"agent":{"allowed_tools":[]}}')
    other = tmp_path / "other"; other.mkdir()
    (other / "config.yaml").write_text('{"agent":{"allowed_tools":[]}}')
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(other)
    handler = MagicMock(return_value='{"executed":true}')
    try:
        with patch("model_tools.registry.dispatch", handler):
            if mode == "invoke":
                content = agent._invoke_tool("web_search", {}, "task")
            else:
                tc = SimpleNamespace(id="allowed", type="function", function=SimpleNamespace(name="web_search", arguments="{}"))
                messages = []
                getattr(agent, f"_execute_tool_calls_{mode}")(SimpleNamespace(tool_calls=[tc]), messages, "task")
                content = messages[0]["content"]
        assert json.loads(content)["executed"] is True
        handler.assert_called_once()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_denied_bridge_cannot_be_unwrapped_into_allowed_target(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent(tmp_path, ["mcp_permission_read"], candidates=["mcp_permission_read"])
    tc = SimpleNamespace(id="bridge", type="function", function=SimpleNamespace(name="tool_call", arguments='{"name":"mcp_permission_read","arguments":{}}'))
    handler = MagicMock(return_value='{"executed":true}')
    with (
        patch("tools.tool_search.resolve_underlying_call", return_value=("mcp_permission_read", {}, None)),
        patch("agent.tool_executor._tool_search_scoped_names", return_value=frozenset({"mcp_permission_read"})),
        patch("tools.tool_search.validate_deferred_call_args", return_value=None),
        patch("model_tools.registry.dispatch", handler),
    ):
        messages = []
        getattr(agent, f"_execute_tool_calls_{mode}")(SimpleNamespace(tool_calls=[tc]), messages, "task")
    assert "agent.allowed_tools" in messages[0]["content"]
    handler.assert_not_called()
