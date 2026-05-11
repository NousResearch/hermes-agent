"""Tests for the eval executor's live-agent adapter."""

from __future__ import annotations

import sys
import time
import types

from agent.evals.executor import run_agent_for_eval


def test_run_agent_for_eval_uses_current_aiagent_signature(monkeypatch, tmp_path):
    captured_kwargs = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            if "persist_session" in kwargs:
                raise TypeError("unexpected keyword argument 'persist_session'")
            captured_kwargs.update(kwargs)
            self.iteration_count = 1
            self.model = kwargs.get("model")

        def run_conversation(self, *, user_message, system_message, task_id, **_kwargs):
            assert "current directory" in system_message
            assert task_id.startswith("eval-task-")
            (tmp_path / "output.txt").write_text("hermes-eval-ok", encoding="utf-8")
            return {"final_response": f"done: {user_message}"}

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAIAgent))
    monkeypatch.setattr("agent.evals.executor.resolve_runtime_provider", lambda requested=None: {})

    result = run_agent_for_eval(
        "Create output.txt containing hermes-eval-ok",
        str(tmp_path),
        model="test-model",
    )

    assert result.error is None
    assert result.response_text.startswith("done: Create output.txt")
    assert result.iterations == 1
    assert result.raw["model"] == "test-model"
    assert "persist_session" not in captured_kwargs
    assert captured_kwargs["skip_memory"] is True
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "hermes-eval-ok"


def test_run_agent_for_eval_enforces_timeout(monkeypatch, tmp_path):
    class SlowAIAgent:
        def __init__(self, **kwargs):
            self.iteration_count = 0
            self.model = kwargs.get("model")

        def run_conversation(self, **_kwargs):
            time.sleep(1)
            return {"final_response": "too late"}

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=SlowAIAgent))
    monkeypatch.setattr("agent.evals.executor.resolve_runtime_provider", lambda requested=None: {})

    result = run_agent_for_eval(
        "This should time out",
        str(tmp_path),
        timeout_seconds=0.01,
        model="test-model",
    )

    assert result.error is not None
    assert "timed out" in result.error
    assert result.raw["exception_type"] == "EvalTimeoutError"
