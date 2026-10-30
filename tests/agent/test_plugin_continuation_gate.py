"""End-to-end coverage for the plugin stop gate (`plugin_runtime.request_continuation`)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import plugin_runtime
from run_agent import AIAgent

SESSION = "plugin-continuation-test"


def _response(content="answer"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


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
            session_id=SESSION,
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    yield instance
    plugin_runtime.clear_session(SESSION)


def test_a_plugin_continuation_keeps_the_turn_open_with_a_visible_row(agent):
    calls = {"n": 0}

    def model_call(_api_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            queued = plugin_runtime.request_continuation(
                SESSION, "[adviser — advisory only]\nre-read the failing test", source="minir",
            )
            assert queued["ok"] is True
            return _response("first answer")
        return _response("second answer")

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("do the work")

    assert calls["n"] == 2
    assert result["final_response"] == "second answer"
    rows = result["messages"]
    advice = [row for row in rows if row.get("display_kind") == plugin_runtime.DISPLAY_KIND]
    assert len(advice) == 1
    assert advice[0]["role"] == "user"
    assert "re-read the failing test" in advice[0]["content"]
    # The first answer stays in the transcript as an interim assistant row.
    assert any(
        row.get("role") == "assistant" and row.get("content") == "first answer"
        for row in rows
    )


def test_the_continuation_budget_bounds_a_plugin_that_always_asks(agent):
    calls = {"n": 0}
    tokens: list[str] = []

    def model_call(_api_kwargs):
        calls["n"] += 1
        queued = plugin_runtime.request_continuation(SESSION, f"advice {calls['n']}", source="minir")
        if queued.get("token"):
            tokens.append(queued["token"])
        return _response(f"answer {calls['n']}")

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("do the work")

    # Default budget is 2: two continuations, then the third answer ends the turn.
    assert calls["n"] == 3
    assert result["final_response"] == "answer 3"
    assert result["completed"] is True
    # The third request was dropped, not consumed: the token must read expired so a
    # plugin can put its advice back in the pending queue.
    assert len(tokens) == 3
    assert plugin_runtime.continuation_status(SESSION, tokens[-1]) == "expired"


def test_the_injected_row_is_announced_to_the_surface(agent):
    seen: list[dict] = []
    agent.plugin_message_callback = seen.append

    calls = {"n": 0}

    def model_call(_api_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            plugin_runtime.request_continuation(
                SESSION, "[adviser — advisory only]\nkeep going", source="minir")
        return _response("first pass done")

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        agent.run_conversation("do the work")

    assert len(seen) == 1
    assert seen[0]["display_kind"] == "adviser"
    assert len(seen[0]["plugin_message_id"]) == 32


def test_a_plugin_can_trigger_the_next_turn(agent):
    def model_call(_api_kwargs):
        queued = plugin_runtime.queue_post_turn(
            SESSION, "[adviser — advisory only]\nfinish the failing test", source="minir")
        assert queued["ok"] is True
        return _response("first pass done")

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("do the work")

    assert result["final_response"] == "first pass done"
    # The surface (CLI/gateway/TUI) already submits a leftover pending_steer as the next turn.
    assert result["pending_steer"].startswith("[adviser — advisory only]")
    assert result["pending_steer_kind"] == "adviser"


def test_a_continuation_queued_by_a_previous_turn_does_not_leak(agent):
    plugin_runtime.bind_turn(agent)
    queued = plugin_runtime.request_continuation(SESSION, "stale advice", source="minir")
    plugin_runtime.unbind_turn(agent)
    assert plugin_runtime.continuation_status(SESSION, queued["token"]) == "expired"

    agent._interruptible_api_call = lambda _api_kwargs: _response("clean answer")
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("do the work")

    assert result["final_response"] == "clean answer"
    assert not [row for row in result["messages"]
                if row.get("display_kind") == plugin_runtime.DISPLAY_KIND]
