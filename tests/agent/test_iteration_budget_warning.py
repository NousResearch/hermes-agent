"""Behavior contracts for the opt-in iteration-budget warning."""

from __future__ import annotations

from pathlib import Path


def _make_agent(tmp_path: Path, monkeypatch, config_body: str = ""):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(config_body or "{}\n", encoding="utf-8")

    from run_agent import AIAgent

    return AIAgent(
        model="gpt-5.5",
        provider="openai",
        api_key="sk-dummy",
        base_url="https://api.openai.com/v1",
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )


def test_budget_warning_config_is_opt_in(monkeypatch, tmp_path):
    default_agent = _make_agent(tmp_path, monkeypatch)
    assert default_agent.budget_warning_ratio is None

    configured_agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_body="agent:\n  budget_warning_ratio: 0.75\n",
    )
    assert configured_agent.budget_warning_ratio == 0.75


def test_budget_warning_reaches_model_once_via_latest_tool_result(monkeypatch, tmp_path):
    from agent.turn_iteration_prep import (
        ITERATION_BUDGET_WARNING_TEMPLATE,
        _maybe_inject_iteration_budget_warning,
        prepare_iteration,
    )

    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_body="agent:\n  budget_warning_ratio: 0.75\n",
    )
    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result one"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "result two"},
    ]

    assert agent.iteration_budget.consume() is True
    assert agent.iteration_budget.consume() is True
    prepare_iteration(agent, messages=messages, api_call_count=2)
    assert agent._iteration_budget_warning_injected is False
    assert agent.iteration_budget.consume() is True
    prepare_iteration(agent, messages=messages, api_call_count=3)
    assert agent._iteration_budget_warning_injected is True

    expected = ITERATION_BUDGET_WARNING_TEMPLATE.format(used=3, maximum=4)
    assert messages[-1]["content"] == f"result two\n\n{expected}"
    assert messages[2]["content"] == "result one"
    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool", "assistant", "tool",
    ]

    snapshot = [dict(message) for message in messages]
    assert _maybe_inject_iteration_budget_warning(agent, messages) is False
    assert messages == snapshot


def test_budget_warning_resets_for_each_turn(monkeypatch, tmp_path):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning

    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_body="agent:\n  budget_warning_ratio: 0.75\n",
    )

    for turn in ("first", "second"):
        _reset_per_turn_agent_state(agent)
        assert agent._iteration_budget_warning_injected is False
        assert agent.iteration_budget.used == 0

        for _ in range(3):
            assert agent.iteration_budget.consume() is True
        messages = [{"role": "tool", "content": f"{turn} result"}]
        assert _maybe_inject_iteration_budget_warning(agent, messages) is True
        assert agent._iteration_budget_warning_injected is True
        assert "You have used 3 of 4 iterations" in messages[0]["content"]
