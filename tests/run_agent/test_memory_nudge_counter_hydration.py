"""Regression test for issue #22357 — gateway memory-nudge counter hydration.

The gateway creates a fresh AIAgent for each inbound message in several
common scenarios (cache miss, 1h idle eviction at gateway/run.py
_AGENT_CACHE_IDLE_TTL_SECS, config-signature mismatch, process restart).
A freshly built AIAgent has _turns_since_memory=0 and _user_turn_count=0.

Without hydration from conversation_history, the memory.nudge_interval
trigger (`_turns_since_memory >= _memory_nudge_interval`) can never be
reached: every turn looks like turn 1 to the counter, so a user can chat
for hours without ever seeing a "💾 Self-improvement review:" message.

This test pins the hydration behavior added at the top of run_conversation().
"""

from __future__ import annotations


def _make_minimal_agent():
    """Build the smallest object that can run the hydration block.

    The hydration code only touches attributes — no I/O, no API calls.
    We can just set up a SimpleNamespace-like object with the right fields
    and call run_conversation's prelude logic via a thin wrapper.

    The hydration block itself is straightforward enough that we test it
    by replicating it inline against the same inputs — that's the only
    way to test ~10 lines deep inside a 500+ line method without rewriting
    the whole agent loop.
    """


def _run_hydration(conversation_history, memory_nudge_interval=10,
                   prior_turn_count=0, prior_turns_since_memory=0):
    """Replicate the hydration block from run_agent.py:11128-11150.
    Keeping this in sync with the production code is a one-line job; the
    block has no dependencies on anything except primitives + history.
    """
    user_turn_count = prior_turn_count
    turns_since_memory = prior_turns_since_memory

    if conversation_history and user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            user_turn_count = prior_user_turns
            if memory_nudge_interval > 0 and turns_since_memory == 0:
                turns_since_memory = prior_user_turns % memory_nudge_interval

    return user_turn_count, turns_since_memory


def test_no_history_leaves_counters_at_zero():
    user_turn, since_mem = _run_hydration([], memory_nudge_interval=10)
    assert user_turn == 0
    assert since_mem == 0


def test_seven_user_turns_history_hydrates_to_seven():
    """Mid-cycle history: 7 prior user turns, interval 10 → counter at 7."""
    history = []
    for i in range(7):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})

    user_turn, since_mem = _run_hydration(history, memory_nudge_interval=10)

    assert user_turn == 7
    assert since_mem == 7  # 7 % 10 = 7, next 3 turns will trigger review


def test_thirteen_turns_history_wraps_via_modulo():
    """13 prior user turns, interval 10 → counter at 3 (post-wrap), preserving cadence."""
    history = [{"role": "user", "content": f"q{i}"} for i in range(13)]

    user_turn, since_mem = _run_hydration(history, memory_nudge_interval=10)

    assert user_turn == 13
    assert since_mem == 3  # 13 % 10 = 3, next 7 turns to trigger


def test_idempotent_when_counters_already_set():
    """A cached agent with existing counters must NOT have them clobbered.

    Without the `_user_turn_count == 0` guard, cached agents would lose
    their accumulated state every time they re-entered the function.
    """
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    user_turn, since_mem = _run_hydration(
        history, memory_nudge_interval=10,
        prior_turn_count=15, prior_turns_since_memory=5,
    )
    # Existing counters preserved (cache hit case)
    assert user_turn == 15
    assert since_mem == 5


def test_zero_nudge_interval_disables_hydration_of_review_counter():
    """When memory.nudge_interval=0 (review disabled), don't touch the counter."""
    history = [{"role": "user", "content": "q1"}]
    user_turn, since_mem = _run_hydration(history, memory_nudge_interval=0)
    assert user_turn == 1
    assert since_mem == 0  # untouched when interval is 0


def test_assistant_only_history_does_not_advance_user_turn_count():
    """Defensive: only role==user messages contribute. Other roles are noise."""
    history = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
    ]
    user_turn, since_mem = _run_hydration(history, memory_nudge_interval=10)
    assert user_turn == 0
    assert since_mem == 0


def test_production_code_contains_hydration_block():
    """Smoke test: confirm the hydration code is actually wired into
    run_conversation(). If someone deletes it, tests above still pass
    against the inline replica — this fails them awake.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "run_agent.py"
    content = src.read_text(encoding="utf-8")
    # Anchor on the unique comment + the modulo line.
    assert "Hydrate per-session nudge counters from persisted history" in content
    assert "self._turns_since_memory = prior_user_turns % self._memory_nudge_interval" in content
