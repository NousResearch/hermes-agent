"""Open-todo stop gate driven through the real conversation loop (salvage of #80155)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent

OPEN = [{"id": "s4-ci", "content": "commit and verify CI", "status": "in_progress"}]


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")],
        model="test/model", usage=None,
    )


def _make_agent(tmp_path, monkeypatch, *, gate: bool):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(f"agent:\n  todo_stop_gate: {'true' if gate else 'false'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            session_id="todo-stop-gate", api_key="k", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=4,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_a, **_k: None
    agent._save_trajectory = lambda *_a, **_k: None
    agent._todo_store.write(OPEN)
    return agent


def _run(agent, replies):
    calls = []

    def model_call(api_kwargs):
        calls.append(api_kwargs)
        return _response(replies[min(len(calls), len(replies)) - 1])

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        return agent.run_conversation("finish the release"), calls


@pytest.mark.parametrize("gate", [True, False])
def test_open_todos_get_one_nudge_only_when_opted_in(tmp_path, monkeypatch, gate):
    agent = _make_agent(tmp_path, monkeypatch, gate=gate)
    result, calls = _run(agent, ["Next I'll commit.", "Still need to commit.", "third"])

    if not gate:  # legacy default: the narrated stop ends the turn
        assert len(calls) == 1
        assert result["final_response"] == "Next I'll commit."
        return
    # One nudge, then the second stop is accepted even though the item is still open.
    assert len(calls) == 2
    assert result["final_response"] == "Still need to commit."
    nudges = [m for m in result["messages"] if m.get("role") == "user" and "s4-ci" in str(m.get("content"))]
    assert len(nudges) == 1
    # The nudge is a durable row the second request actually carried, after the first answer.
    sent = calls[1]["messages"]
    assert [m.get("role") for m in sent[-2:]] == ["assistant", "user"]
    assert "s4-ci" in sent[-1]["content"]
