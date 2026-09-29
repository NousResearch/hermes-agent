"""Skill iteration limits apply only to the current explicit invocation turn."""

import os
import threading

import pytest

from agent.iteration_budget import normalize_skill_max_turns, skill_max_turns_from_config
from agent.skill_commands import (
    _SKILL_INVOCATION_PREFIX,
    _scaffold_header,
    build_skill_invocation_message,
    skill_invocation_name,
)
from agent.turn_context import _reset_per_turn_agent_state
from hermes_state import SessionDB
from run_agent import AIAgent


def _agent(tmp_path, name, *, baseline=40, limits=None, explicit=False):
    return AIAgent(
        session_db=SessionDB(db_path=tmp_path / f"{name}.db"),
        model="test-model", provider="openai-compat", api_key="test",
        base_url="http://127.0.0.1:1/v1", max_iterations=baseline,
        skill_max_turns=limits, max_iterations_explicit=explicit,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
    )


def _invocation(name="unlazy"):
    return f'{_SKILL_INVOCATION_PREFIX}"{name}" skill, indicating they want you to follow its instructions.]'


def _stacked_invocation(keys):
    """The real stacked/bundle header shape: ``"/a /b" stacked skill bundle``."""
    return _scaffold_header(f'"{" ".join(keys)}" stacked skill bundle', [k.lstrip("/") for k in keys])


def _assert_turn_limit(agent, message, expected):
    _reset_per_turn_agent_state(agent, message)
    assert agent.max_iterations == expected
    assert agent.iteration_budget.max_total == expected


def test_real_single_skill_builder_message_uses_the_skill_limit(tmp_path):
    """The override fires on the message the surfaces actually deliver, not a hand-rolled one."""
    message = build_skill_invocation_message("/unlazy", "fix the leak")
    assert message and message.startswith(_SKILL_INVOCATION_PREFIX)
    assert skill_invocation_name(message) == "unlazy"
    agent = _agent(tmp_path, "built", limits={"unlazy": 80})
    _assert_turn_limit(agent, message, 80)


def test_stacked_invocation_uses_the_first_skill_limit(tmp_path):
    message = _stacked_invocation(["/unlazy", "/other"])
    assert skill_invocation_name(message) == "unlazy"
    agent = _agent(tmp_path, "stacked", limits={"unlazy": 80, "other": 200})
    _assert_turn_limit(agent, message, 80)


def test_invocation_and_plain_followup_reset_to_baseline(tmp_path):
    agent = _agent(tmp_path, "main", limits={"unlazy": 80})
    _assert_turn_limit(agent, _invocation(), 80)
    for message in (
        "continue", "how does /unlazy work?", 'the command "/unlazy"',
        "```\n/unlazy\n```", "docs say " + _invocation().lower(),
    ):
        _assert_turn_limit(agent, message, 40)


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_or_interrupted_turn_cannot_leave_override_armed(tmp_path, interrupted):
    agent = _agent(tmp_path, "main", limits={"unlazy": 80})
    try:
        _assert_turn_limit(agent, _invocation(), 80)
        agent._interrupt_requested = interrupted
        raise RuntimeError("turn failed")
    except RuntimeError:
        pass
    _assert_turn_limit(agent, "continue", 40)


def test_a_directly_assigned_budget_becomes_the_new_baseline(tmp_path):
    """A surface that assigns max_iterations itself must not be reverted at the next turn."""
    agent = _agent(tmp_path, "direct", limits={"unlazy": 80})
    _assert_turn_limit(agent, _invocation(), 80)
    agent.max_iterations = 7
    _assert_turn_limit(agent, "continue", 7)
    _assert_turn_limit(agent, _invocation(), 80)
    _assert_turn_limit(agent, "continue", 7)


def test_concurrent_sessions_resolve_independently(tmp_path):
    """Two agents resolving turns at the same instant must not share the override."""
    first = _agent(tmp_path, "first", limits={"unlazy": 80})
    second = _agent(tmp_path, "second", limits={"unlazy": 12})
    environment = dict(os.environ)
    barrier = threading.Barrier(2, timeout=30)
    errors = []

    def run(agent, message):
        try:
            barrier.wait()  # force real overlap between the two resets
            _reset_per_turn_agent_state(agent, message)
        except Exception as exc:  # surfaced by the assert below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(first, _invocation())),
        threading.Thread(target=run, args=(second, _invocation())),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert (first.max_iterations, first.iteration_budget.max_total) == (80, 80)
    assert (second.max_iterations, second.iteration_budget.max_total) == (12, 12)
    # Neither agent may consult process state, so the environment is untouched.
    assert dict(os.environ) == environment
    _assert_turn_limit(first, "continue", 40)
    _assert_turn_limit(second, "continue", 40)


@pytest.mark.parametrize("baseline,explicit,expected", [
    (200, True, 200), (10, True, 10), (40, False, 80),
])
def test_explicit_limit_precedes_skill_limit(tmp_path, baseline, explicit, expected):
    agent = _agent(tmp_path, "main", baseline=baseline, limits={"unlazy": 80}, explicit=explicit)
    _assert_turn_limit(agent, _invocation(), expected)


def test_cached_gateway_baseline_refresh_stays_in_sync(tmp_path):
    agent = _agent(tmp_path, "main", limits={"unlazy": 80})
    _assert_turn_limit(agent, _invocation(), 80)
    # Mirror the gateway's cached-agent refresh before the next turn
    # (gateway/run_turn_runner.py::_lookup_cached_agent).
    agent.max_iterations = 25
    agent._configured_max_iterations = 25
    _assert_turn_limit(agent, "continue", 25)
    _assert_turn_limit(agent, _invocation(), 80)


def test_only_scaffolded_skill_names_and_valid_config_values_apply(tmp_path):
    assert skill_invocation_name(_invocation("/unlazy /other")) == "unlazy"
    assert skill_invocation_name("quoted " + _invocation()) is None
    assert skill_invocation_name("/unlazy") is None
    limits = {"unlazy": 80, "": 3, "bad": True, "zero": 0, 1: 50, "text": "9"}
    assert normalize_skill_max_turns(limits) == {"unlazy": 80}
    assert skill_max_turns_from_config({"agent": {"skill_max_turns": limits}}) == {"unlazy": 80}
    assert skill_max_turns_from_config({"agent": "invalid"}) == {}
    agent = _agent(tmp_path, "main", limits=limits)
    _assert_turn_limit(agent, _invocation(), 80)
