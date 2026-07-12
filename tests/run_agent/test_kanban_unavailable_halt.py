"""An unroutable Kanban approval parks and halts the worker atomically."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.execution_context import ExecutionRole
from hermes_cli import kanban_db as kb
from run_agent import AIAgent
from tools import approval


def _agent(hermes_home) -> AIAgent:
    hermes_home.mkdir(parents=True, exist_ok=True)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "terminal",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=definitions),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.agent_init.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._execution_role = ExecutionRole.KANBAN_OWNER
    return agent


def _tool_response(arguments: str = "{}", extra_arguments: str | None = None):
    calls = [
        SimpleNamespace(
            id="call-1",
            type="function",
            function=SimpleNamespace(name="terminal", arguments=arguments),
        )
    ]
    if extra_arguments is not None:
        calls.append(
            SimpleNamespace(
                id="call-2",
                type="function",
                function=SimpleNamespace(
                    name="terminal",
                    arguments=extra_arguments,
                ),
            )
        )
    message = SimpleNamespace(content="", tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


def test_unavailable_route_signed_pause_prevents_second_model_call(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("HERMES_KANBAN_SESSION", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE", raising=False)
    agent = _agent(home)

    with kb.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="unroutable", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="test:owner")
        assert claimed is not None
        unavailable = kb.request_task_approval(
            conn,
            task_id=task_id,
            action_kind="terminal",
            action_digest="sha256:unavailable",
            display_target="redacted dangerous action",
            description="operator approval required",
            worker_session_id="worker-session",
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
            profile="worker",
        )
        assert unavailable is not None
        assert unavailable["status"] == "unavailable"
        assert kb.list_task_approvals(conn, task_id=task_id) == []

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    marker = approval._kanban_pending_result(
        unavailable,
        "operator approval required",
    )
    assert marker["outcome"] == "approval_unavailable"
    assert marker["request_id"].startswith("kau_")
    assert marker.get("_hermes_kanban_pause_token")

    agent.client.chat.completions.create.side_effect = [
        _tool_response(),
        AssertionError("unavailable approval pause made an extra model call"),
    ]
    with (
        patch("run_agent.handle_function_call", return_value=json.dumps(marker)),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the gated action")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "kanban_approval_pending"
    assert result["kanban_approval_pending"] is True
    assert result["approval_request"]["request_id"].startswith("kau_")
    assert result["approval_request"]["outcome"] == "approval_unavailable"
    assert "repair the trusted route" in result["final_response"].lower()
    assert "explicitly unblock" in result["final_response"].lower()
    assert "resume automatically" not in result["final_response"].lower()
    assert "_hermes_kanban_pause_token" not in result["messages"][-2]["content"]
    with kb.connect(db_path) as conn:
        parked = kb.get_task(conn, task_id)
        run = kb.get_run(conn, claimed.current_run_id)
        assert parked.status == "blocked"
        assert parked.current_run_id is None
        assert run.outcome == "approval_unavailable"
        assert kb.list_task_approvals(conn, task_id=task_id) == []
        assert kb.list_pending_unnotified_task_approvals(
            conn,
            task_id=task_id,
        ) == []


def test_persistence_failure_halts_through_real_terminal_wrapper(
    tmp_path,
    monkeypatch,
):
    from tools import terminal_tool

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-persistence-failure")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "41")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:claim")
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_SESSION_ID", "worker-session")
    monkeypatch.delenv("HERMES_KANBAN_SESSION", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE", raising=False)
    agent = _agent(home)

    class FakeEnv:
        env = {}
        cwd = "/tmp"

        def execute(self, *_args, **_kwargs):
            raise AssertionError("approval-persistence failure executed command")

    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": "/tmp",
            "timeout": 60,
            "lifetime_seconds": 3600,
            "local_persistent": False,
        },
    )
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_get_kanban_approval_mode", lambda: "ask")
    monkeypatch.setattr(approval, "_request_kanban_approval", lambda **_kwargs: None)
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
    )

    agent.client.chat.completions.create.side_effect = [
        _tool_response(
            json.dumps({"command": "rm -rf /tmp/persistence-fail"}),
            json.dumps({"command": "touch /tmp/must-not-run"}),
        ),
        AssertionError("approval-persistence halt made an extra model call"),
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the gated action")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "kanban_approval_pending"
    assert result["kanban_approval_pending"] is True
    request = result["approval_request"]
    assert request["request_id"].startswith("kaf_")
    assert request["outcome"] == "approval_persistence_failed"
    final = result["final_response"].lower()
    assert "stopped before the action ran" in final
    assert "did not confirm" in final
    assert "card was parked" in final
    assert "worker slot was released" in final
    assert "resume automatically" not in final
    assert "explicitly unblock" not in final
    first_tool_content = result["messages"][-3]["content"]
    skipped_tool_content = result["messages"][-2]["content"]
    assert "_hermes_kanban_pause_token" not in first_tool_content
    assert "_hermes_kanban_pause_token" not in skipped_tool_content
    assert "durably persisted or verified" in skipped_tool_content
    assert "paused pending approval" not in skipped_tool_content
