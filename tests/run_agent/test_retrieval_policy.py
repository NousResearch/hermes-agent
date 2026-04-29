import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent, _should_parallelize_tool_batch


def _make_agent_with_config(config):
    with (
        patch("run_agent.get_tool_definitions", return_value=[
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            {"type": "function", "function": {"name": "search_files", "parameters": {}}},
            {"type": "function", "function": {"name": "session_search", "parameters": {}}},
        ]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._current_user_message = "Find the packet path"
        agent._reset_retrieval_policy_turn_state()
        return agent


def test_parallelization_is_disabled_for_retrieval_batches():
    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name="read_file", arguments=json.dumps({"path": "a.txt"}))),
        SimpleNamespace(function=SimpleNamespace(name="search_files", arguments=json.dumps({"pattern": "todo", "path": "."}))),
    ]

    assert _should_parallelize_tool_batch(tool_calls) is False


def test_broad_search_is_blocked_when_policy_disallows_it():
    agent = _make_agent_with_config({
        "agent": {},
        "compression": {},
        "auxiliary": {"planner": {"provider": "auto", "timeout": 15, "extra_body": {}}},
        "retrieval_policy": {
            "enabled": True,
            "planner_enabled": False,
            "max_retrieval_calls": 4,
            "max_broad_search_calls": 1,
            "max_subtree_expansions": 2,
            "max_total_retrieval_seconds": 25,
            "allow_unplanned_broad_search": False,
            "debug_log_events": False,
        },
    })

    with patch("run_agent.handle_function_call") as mock_dispatch:
        result = json.loads(agent._invoke_tool(
            "search_files",
            {"pattern": "packet", "path": ".", "target": "files"},
            "task-1",
        ))

    assert "error" in result
    assert "broad search" in result["error"].lower()
    mock_dispatch.assert_not_called()


def test_explicit_read_file_uses_default_plan_without_planner_call():
    agent = _make_agent_with_config({
        "agent": {},
        "compression": {},
        "auxiliary": {"planner": {"provider": "auto", "timeout": 15, "extra_body": {}}},
        "retrieval_policy": {
            "enabled": True,
            "planner_enabled": True,
            "max_retrieval_calls": 4,
            "max_broad_search_calls": 1,
            "max_subtree_expansions": 2,
            "max_total_retrieval_seconds": 25,
            "allow_unplanned_broad_search": False,
            "debug_log_events": False,
        },
    })

    with (
        patch("run_agent.plan_retrieval_tool_use") as mock_plan,
        patch("run_agent.handle_function_call", return_value=json.dumps({"content": "1|ok", "path": "/tmp/demo.txt"})),
    ):
        result = json.loads(agent._invoke_tool(
            "read_file",
            {"path": "/tmp/demo.txt", "offset": 1, "limit": 20},
            "task-1",
        ))

    assert result["path"] == "/tmp/demo.txt"
    mock_plan.assert_not_called()


def test_session_search_first_can_be_allowed_by_planner():
    agent = _make_agent_with_config({
        "agent": {},
        "compression": {},
        "auxiliary": {"planner": {"provider": "auto", "timeout": 15, "extra_body": {}}},
        "retrieval_policy": {
            "enabled": True,
            "planner_enabled": True,
            "max_retrieval_calls": 4,
            "max_broad_search_calls": 1,
            "max_subtree_expansions": 2,
            "max_total_retrieval_seconds": 25,
            "allow_unplanned_broad_search": False,
            "debug_log_events": False,
        },
    })
    agent._session_db = MagicMock()

    planner_plan = SimpleNamespace(
        recommended_sequence=["session_search"],
        max_retrieval_calls=2,
        allow_broad_search=False,
        goal="recall prior discussion",
        source="planner",
        stop_if=[],
    )

    with (
        patch("run_agent.plan_retrieval_tool_use", return_value=planner_plan),
        patch("tools.session_search_tool.session_search", return_value=json.dumps({"success": True, "results": [{"session_id": "abc"}], "count": 1})),
    ):
        result = json.loads(agent._invoke_tool(
            "session_search",
            {"query": "NC OR ITPE", "limit": 3},
            "task-1",
        ))

    assert result["success"] is True
    assert result["count"] == 1


def test_planner_can_block_unplanned_search_files_stage():
    agent = _make_agent_with_config({
        "agent": {},
        "compression": {},
        "auxiliary": {"planner": {"provider": "auto", "timeout": 15, "extra_body": {}}},
        "retrieval_policy": {
            "enabled": True,
            "planner_enabled": True,
            "max_retrieval_calls": 4,
            "max_broad_search_calls": 1,
            "max_subtree_expansions": 2,
            "max_total_retrieval_seconds": 25,
            "allow_unplanned_broad_search": False,
            "debug_log_events": False,
        },
    })

    planner_plan = SimpleNamespace(
        recommended_sequence=["session_search"],
        max_retrieval_calls=2,
        allow_broad_search=False,
        goal="recall prior discussion",
        source="planner",
        stop_if=[],
    )

    with (
        patch("run_agent.plan_retrieval_tool_use", return_value=planner_plan),
        patch("run_agent.handle_function_call") as mock_dispatch,
    ):
        result = json.loads(agent._invoke_tool(
            "search_files",
            {"pattern": "NC", "path": "./notes", "target": "content"},
            "task-1",
        ))

    assert "error" in result
    assert "stage_not_planned" in result["error"]
    mock_dispatch.assert_not_called()
