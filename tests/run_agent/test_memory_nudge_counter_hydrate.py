"""Regression: gateway creates a fresh AIAgent per inbound message but the
persisted conversation_history keeps growing. Without hydration the
``_turns_since_memory`` counter restarts at 0 every turn and
``memory.nudge_interval`` may never trigger (issue #22357).

Rather than instantiating a real AIAgent (heavy: loads providers, plugins,
config), we replicate the exact hydration block from
``AIAgent.run_conversation`` and assert its behavior, plus a source-level
guard so future refactors don't drop the block.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


def _hydrate(agent, conversation_history):
    """Mirror of the hydration block in run_agent.py."""
    if (
        conversation_history
        and agent._user_turn_count == 0
        and agent._memory_nudge_interval > 0
        and agent._turns_since_memory == 0
    ):
        last_memory_idx = -1
        for idx, msg in enumerate(conversation_history):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "tool"
                and msg.get("name") == "memory"
            ):
                last_memory_idx = idx
        if last_memory_idx >= 0:
            relevant = conversation_history[last_memory_idx + 1 :]
        else:
            relevant = conversation_history
        prior_user_turns = sum(
            1
            for msg in relevant
            if isinstance(msg, dict) and msg.get("role") == "user"
        )
        agent._turns_since_memory = (
            prior_user_turns % agent._memory_nudge_interval
        )


def _make_agent(interval=10, turns_since_memory=0, user_turn_count=0):
    return SimpleNamespace(
        _memory_nudge_interval=interval,
        _turns_since_memory=turns_since_memory,
        _user_turn_count=user_turn_count,
    )


def _hist(user_turns, assistant_turns=None):
    if assistant_turns is None:
        assistant_turns = user_turns
    out = []
    for i in range(max(user_turns, assistant_turns)):
        if i < user_turns:
            out.append({"role": "user", "content": f"u{i}"})
        if i < assistant_turns:
            out.append({"role": "assistant", "content": f"a{i}"})
    return out


def test_hydrates_from_prior_history_so_next_turn_can_trigger():
    agent = _make_agent(interval=10)
    _hydrate(agent, _hist(9))
    assert agent._turns_since_memory == 9


def test_hydrate_then_increment_reaches_interval_on_first_new_turn():
    agent = _make_agent(interval=10)
    _hydrate(agent, _hist(9))
    agent._turns_since_memory += 1
    assert agent._turns_since_memory == 10


def test_modulo_keeps_counter_bounded_for_long_histories():
    agent = _make_agent(interval=10)
    _hydrate(agent, _hist(127))
    assert agent._turns_since_memory == 7


def test_no_hydration_when_history_empty():
    agent = _make_agent(interval=10)
    _hydrate(agent, None)
    assert agent._turns_since_memory == 0
    _hydrate(agent, [])
    assert agent._turns_since_memory == 0


def test_no_hydration_when_interval_disabled():
    agent = _make_agent(interval=0)
    _hydrate(agent, _hist(50))
    assert agent._turns_since_memory == 0


def test_no_hydration_after_first_turn_consumed():
    agent = _make_agent(interval=10, user_turn_count=3)
    _hydrate(agent, _hist(50))
    assert agent._turns_since_memory == 0


def test_no_hydration_when_counter_already_advanced():
    """Manual reset by the memory tool sets _turns_since_memory to 0
    mid-session; the _user_turn_count guard ensures hydration is one-shot
    so a non-zero counter is not clobbered on a re-entered first turn."""
    agent = _make_agent(interval=10, turns_since_memory=4, user_turn_count=0)
    _hydrate(agent, _hist(50))
    assert agent._turns_since_memory == 4


def test_messages_without_role_field_are_ignored():
    agent = _make_agent(interval=10)
    history = _hist(3) + [{"content": "no role"}, {"role": "tool", "content": "x"}]
    _hydrate(agent, history)
    assert agent._turns_since_memory == 3


def test_hydration_counts_user_turns_since_last_memory_tool():
    """When the memory tool ran mid-history, runtime resets the counter to 0;
    hydration must mirror that by only counting user turns AFTER that tool call."""
    agent = _make_agent(interval=10)
    history = (
        _hist(8)
        + [{"role": "tool", "name": "memory", "content": "saved"}]
        + [{"role": "user", "content": "u_after_1"}, {"role": "assistant", "content": "a"}]
        + [{"role": "user", "content": "u_after_2"}]
    )
    _hydrate(agent, history)
    assert agent._turns_since_memory == 2


def test_hydration_uses_most_recent_memory_tool_call():
    agent = _make_agent(interval=10)
    history = (
        [{"role": "user", "content": "u0"}]
        + [{"role": "tool", "name": "memory", "content": "first"}]
        + [{"role": "user", "content": "u1"}, {"role": "user", "content": "u2"}]
        + [{"role": "tool", "name": "memory", "content": "second"}]
        + [{"role": "user", "content": "u3"}]
    )
    _hydrate(agent, history)
    assert agent._turns_since_memory == 1


@pytest.mark.parametrize(
    "interval,prior_turns,expected",
    [
        (5, 4, 4),
        (5, 5, 0),
        (5, 6, 1),
        (1, 100, 0),
        (3, 7, 1),
    ],
)
def test_hydration_modulo_table(interval, prior_turns, expected):
    agent = _make_agent(interval=interval)
    _hydrate(agent, _hist(prior_turns))
    assert agent._turns_since_memory == expected


def test_source_level_guard_for_hydration_block():
    """Future refactor must not silently drop the hydration block."""
    import run_agent

    src = inspect.getsource(run_agent.AIAgent.run_conversation)
    assert "_turns_since_memory = prior_user_turns % self._memory_nudge_interval" in src
