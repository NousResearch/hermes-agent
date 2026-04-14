"""Runtime autonomy tests for bootstrap preflight and proof-of-done artifacts."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from run_agent import AIAgent
from tests.run_agent.test_run_agent import _make_tool_defs, _mock_response, _mock_tool_call


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("write_file", "terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = patch("run_agent.OpenAI").start()
        yield a
        patch.stopall()


def _setup_agent(agent: AIAgent) -> None:
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False


def test_run_conversation_returns_preflight_failure_artifact(agent, tmp_path, monkeypatch):
    _setup_agent(agent)
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with patch("run_agent.run_bootstrap_preflight", return_value={
        "ok": False,
        "diagnostics": ["Missing Hermes config"],
        "message": "Autonomy bootstrap preflight failed:\n- Missing Hermes config",
    }):
        result = agent.run_conversation("build something")

    assert result["completed"] is False
    assert "preflight failed" in result["final_response"].lower()
    artifact_path = Path(result["proof_of_done_artifact"])
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["preflight"]["ok"] is False


def test_run_conversation_emits_proof_of_done_artifact(agent, tmp_path, monkeypatch):
    _setup_agent(agent)
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    tc = _mock_tool_call(
        name="write_file",
        arguments=json.dumps({"path": "/tmp/example.txt", "content": "hello"}),
        call_id="c1",
    )
    resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
    resp2 = _mock_response(content="Done", finish_reason="stop")
    agent.client.chat.completions.create.side_effect = [resp1, resp2]

    with (
        patch("run_agent.run_bootstrap_preflight", return_value={"ok": True, "diagnostics": [], "message": "ok"}),
        patch("run_agent.handle_function_call", return_value=json.dumps({"status": "ok"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("write the file")

    artifact_path = Path(result["proof_of_done_artifact"])
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["completed"] is True
    assert "/tmp/example.txt" in payload["files_touched"]
    assert payload["preflight"]["ok"] is True
    assert payload["status"] == "passed"
    assert payload["stop_reason"]


def test_proof_of_done_artifact_records_commands_and_gate_outcomes(agent, tmp_path, monkeypatch):
    _setup_agent(agent)
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    tc = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "python scripts/run_readiness.py", "workdir": str(tmp_path)}),
        call_id="c2",
    )
    resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
    resp2 = _mock_response(content="Done", finish_reason="stop")
    agent.client.chat.completions.create.side_effect = [resp1, resp2]

    with (
        patch("run_agent.run_bootstrap_preflight", return_value={"ok": True, "diagnostics": [], "message": "ok"}),
        patch("run_agent.handle_function_call", return_value=json.dumps({"status": "ok", "exit_code": 0})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("verify the repo")

    payload = json.loads(Path(result["proof_of_done_artifact"]).read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert "python scripts/run_readiness.py" in payload["commands_run"]
    assert payload["gates_run"][0]["name"] == "run_readiness"
    assert payload["gates_run"][0]["outcome"] == "passed"
    assert payload["next_required_human_action"] is None
