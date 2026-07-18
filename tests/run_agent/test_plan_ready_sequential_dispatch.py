"""Regression: plan_ready must ride the clarify callback in the SEQUENTIAL
executor path (the interactive CLI path), not fall through to the registry
handler with callback=None.

Field bug (session 20260710_150040): in the interactive fire-CLI, plan_ready
returned {"error": "Approval is not available in this execution context."}
because ``execute_tool_calls_sequential`` dispatches tools inline (it does NOT
go through ``agent_runtime_helpers.invoke_tool`` where the plan_ready branch
lived), so plan_ready hit the generic registry ``else`` and lost the
agent.clarify_callback.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names):
    return [
        {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
        for n in names
    ]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


@pytest.fixture
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("plan_ready")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _tool_call(name, arguments="{}"):
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assistant_msg(tool_calls):
    return SimpleNamespace(content="", tool_calls=tool_calls)


def test_plan_ready_uses_clarify_callback_in_sequential_path(agent, hermes_home):
    from hermes_cli.plan_mode import PlanManager

    sid = "seq-plan-sid"
    agent.session_id = sid
    PlanManager(sid).enter()

    seen = {}

    def clarify_cb(question, choices):
        seen["question"] = question
        seen["choices"] = choices
        return "Approve"

    agent.clarify_callback = clarify_cb

    messages: list = []
    tc = _tool_call("plan_ready", json.dumps({"plan_path": ".hermes/plans/p.md"}))
    agent._execute_tool_calls_sequential(_assistant_msg([tc]), messages, "task-1")

    # The clarify callback fired (proving it was NOT None / the registry path).
    assert seen.get("choices") == ["Approve", "Keep planning"]

    # The tool result approved the plan — not the "Approval is not available" error.
    assert messages, "expected a tool result message"
    result = messages[-1]["content"]
    assert "Approval is not available" not in result
    payload = json.loads(result)
    assert payload["status"] == "approved"
    assert not PlanManager(sid).is_active()
