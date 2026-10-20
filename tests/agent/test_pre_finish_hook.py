"""``pre_finish`` keeps a text-only turn going; ``pre_verify`` still waits for file edits."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import plugins
from run_agent import AIAgent

BROKEN = "```mermaid\nflowchart LR\n  A -->\n```"
FIXED = "```mermaid\nflowchart LR\n  A --> B\n```"
HINT = "diagram 1 fails: edge has no target; resend it fixed"


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model", usage=None,
    )


@pytest.fixture
def hooks(monkeypatch):
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    return manager._hooks


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="pre-finish-test", api_key="test-key",
            base_url="https://example.invalid/v1", provider="openai-compat",
            model="test/model", max_iterations=6, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _script(agent, replies):
    seen = []

    def model_call(api_kwargs):
        seen.append([dict(m) for m in api_kwargs["messages"]])
        return _response(replies[min(len(seen), len(replies)) - 1])

    agent._interruptible_api_call = model_call
    return seen


def test_pre_finish_continues_text_turn_without_edits(agent, hooks):
    calls = {"pre_verify": [], "pre_finish": []}
    hooks["pre_verify"] = [lambda **kw: calls["pre_verify"].append(kw)]

    def check(final_response, changed_paths, attempt, **_kw):
        calls["pre_finish"].append((final_response, changed_paths, attempt))
        return {"action": "continue", "message": HINT} if "A -->\n" in final_response else None

    hooks["pre_finish"] = [check]
    seen = _script(agent, [BROKEN, FIXED])

    result = agent.run_conversation("draw the flow")

    assert calls["pre_verify"] == []
    assert calls["pre_finish"] == [(BROKEN, [], 0), (FIXED, [], 1)]
    assert seen[1][-1] == {"role": "user", "content": HINT}
    assert result["final_response"] == FIXED
    assert HINT not in [m.get("content") for m in result["messages"]]


def test_pre_finish_is_bounded_by_max_verify_nudges(agent, hooks):
    hooks["pre_finish"] = [lambda **_kw: {"action": "continue", "message": HINT}]
    seen = _script(agent, ["one", "two", "three", "four"])

    with patch("agent.verify_hooks.max_verify_nudges", return_value=2):
        result = agent.run_conversation("draw the flow")

    assert len(seen) == 3
    assert result["final_response"] == "three"
