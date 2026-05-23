"""Regression tests for #30812 — review-cadence counters bound to spawn.

Before this fix, ``conversation_loop.run_conversation`` reset the
``_turns_since_memory`` counter the moment ``_should_review_memory``
crossed its threshold (near the top of the function), and likewise
reset ``_iters_since_skill`` the moment ``_should_review_skills``
flipped True (just before the spawn block). The actual
``_spawn_background_review`` call runs near the end of the function
behind a stricter gate (``final_response and not interrupted and
(...)``). If the iteration loop bailed out before producing a
``final_response`` (user interrupt, exception, API failure), the
counters had already been spent and the user lost an entire nudge
interval before the cadence considered the review again.

The fix moves both resets into the ``else`` branch of the existing
``try`` around ``_spawn_background_review`` so the consumption is
bound to the actual spawn:

* spawn succeeds → only the counter(s) whose flag we passed in get
  reset, mirroring the per-flag decision made earlier;
* spawn raises → both counters stay where they are so the next turn
  re-attempts instead of silently rolling the cadence forward;
* gate skipped (no final_response / interrupted) → counters stay so
  the next turn satisfies the cadence again immediately.

These tests pin every branch of that contract.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ────────────────────────────────────────────────────────────────────────────
# Test helpers (mirror tests/run_agent/test_tool_call_guardrail_runtime.py)
# ────────────────────────────────────────────────────────────────────────────


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name: str = "web_search", arguments: str = "{}", call_id: str | None = None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content: str = "Hello", finish_reason: str = "stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(
    *tool_names: str,
    memory_nudge_interval: int = 3,
    skill_nudge_interval: int = 3,
    has_memory_store: bool = True,
) -> AIAgent:
    """Build a minimal AIAgent wired so the run_conversation cadence
    branches reach the spawn site.

    ``memory_nudge_interval`` / ``skill_nudge_interval`` are kept low
    (default 3) so tests can drive the cadence in a handful of fake
    turns without bloating the LLM-mock side_effect list.

    ``has_memory_store`` toggles the ``agent._memory_store`` guard in
    ``run_conversation`` — when False the memory branch is short-
    circuited even if "memory" is in ``valid_tool_names`` (mirrors a
    config where the user has memory off entirely).
    """
    tool_names = tool_names or ("web_search", "memory", "skill_manage")
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    # Re-seed the cadence config — ``skip_memory=True`` zeros these.
    agent._memory_nudge_interval = memory_nudge_interval
    agent._skill_nudge_interval = skill_nudge_interval
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    agent._user_turn_count = 0
    if has_memory_store:
        # ``run_conversation`` only checks truthiness, never the type.
        agent._memory_store = MagicMock(name="memory_store_stub")
    else:
        agent._memory_store = None
    # Make sure both tool names are in the valid set even if a caller
    # narrowed ``tool_names`` — the cadence guards only read this set.
    agent.valid_tool_names = set(agent.valid_tool_names) | {"memory", "skill_manage"}
    return agent


def _drive_clean_turn(agent: AIAgent, *, spawn_mock=None, response_text: str = "done"):
    """Run a single ``run_conversation`` turn that returns ``response_text``
    cleanly (no tool calls, no interrupt).  Returns the result dict.

    If ``spawn_mock`` is provided it patches
    ``_spawn_background_review`` so tests can assert call args / side-
    effects without actually launching a thread.
    """
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=response_text, finish_reason="stop", tool_calls=None)
    ]
    spawn_ctx = (
        patch.object(agent, "_spawn_background_review", spawn_mock)
        if spawn_mock is not None
        else patch.object(agent, "_spawn_background_review")
    )
    with (
        spawn_ctx as spawn_patch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")
    return result, spawn_patch


# ────────────────────────────────────────────────────────────────────────────
# Memory cadence
# ────────────────────────────────────────────────────────────────────────────


class TestMemoryCadenceBoundToSpawn:

    def test_memory_counter_resets_only_after_successful_spawn(self):
        """Happy path: cadence threshold met, spawn succeeds → reset to 0.

        Pre-fix this was already the observable outcome (the early
        reset happened either way), so this test guards against the
        FIX over-correcting — moving the reset to the spawn site must
        still produce a zero counter after a normal trigger turn.
        """
        agent = _make_agent(memory_nudge_interval=3)
        # Seed the counter just below threshold so this turn crosses it.
        agent._turns_since_memory = 2

        result, spawn_patch = _drive_clean_turn(agent)

        assert result["final_response"] == "done"
        spawn_patch.assert_called_once()
        kwargs = spawn_patch.call_args.kwargs
        assert kwargs["review_memory"] is True
        assert agent._turns_since_memory == 0, (
            "Successful spawn should still reset the memory cadence counter"
        )

    def test_memory_counter_stays_when_spawn_raises(self):
        """If ``_spawn_background_review`` raises, the cadence must
        NOT advance — the user's review is being deferred, not lost.
        """
        agent = _make_agent(memory_nudge_interval=3)
        agent._turns_since_memory = 2

        result, spawn_patch = _drive_clean_turn(
            agent,
            spawn_mock=MagicMock(side_effect=RuntimeError("spawn blew up")),
        )

        assert result["final_response"] == "done"
        spawn_patch.assert_called_once()
        # Counter stays at threshold (3 = 2 + 1) so the next turn
        # immediately re-satisfies the cadence and re-attempts.
        assert agent._turns_since_memory == 3, (
            f"Counter advanced despite spawn failure; got: "
            f"{agent._turns_since_memory}"
        )

    def test_memory_counter_stays_when_loop_interrupted_before_final_response(self):
        """The #30812 reproduction: interrupt fires inside the
        iteration loop before any ``final_response`` is produced.
        ``_spawn_background_review`` is skipped by the gate; the
        counter must NOT be reset so the next turn re-fires.
        """
        agent = _make_agent(memory_nudge_interval=3)
        agent._turns_since_memory = 2
        # Pre-arm the interrupt so the very first iteration aborts.
        agent._interrupt_requested = True
        agent._interrupt_message = "user pressed ctrl-c"

        with (
            patch.object(agent, "_spawn_background_review") as spawn_patch,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        spawn_patch.assert_not_called(), (
            "spawn gate should have skipped the review on interrupt"
        )
        assert result.get("interrupted") is True
        assert agent._turns_since_memory == 3, (
            "Counter wasted on an interrupted turn — the #30812 bug. "
            f"Got: {agent._turns_since_memory}"
        )

    def test_memory_counter_increments_without_resetting_below_threshold(self):
        """Sub-threshold turns must still advance the counter — the
        fix only changes WHEN the reset happens, not the increment.
        """
        agent = _make_agent(memory_nudge_interval=5)
        agent._turns_since_memory = 2  # well below threshold (5)

        result, spawn_patch = _drive_clean_turn(agent)

        spawn_patch.assert_not_called()
        assert result["final_response"] == "done"
        # Incremented from 2 → 3, NOT reset.
        assert agent._turns_since_memory == 3

    def test_memory_counter_accumulates_across_failed_then_successful_turns(self):
        """End-to-end behavior the user reported: a failed trigger turn
        followed by a clean turn fires the review on turn 2, not turn N
        intervals later.
        """
        agent = _make_agent(memory_nudge_interval=3)
        agent._turns_since_memory = 2
        agent._interrupt_requested = True

        # Turn 1 — interrupted before any final_response.
        with (
            patch.object(agent, "_spawn_background_review") as spawn_t1,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("first try")

        spawn_t1.assert_not_called()
        assert agent._turns_since_memory == 3

        # Turn 2 — clean. Counter ticks 3 → 4, still ≥ interval (3),
        # so cadence triggers again; spawn fires and resets to 0.
        # ``clear_interrupt`` is called at the end of every turn by
        # run_conversation, so we don't need to re-arm anything.
        result, spawn_t2 = _drive_clean_turn(agent)
        spawn_t2.assert_called_once()
        assert result["final_response"] == "done"
        assert agent._turns_since_memory == 0


# ────────────────────────────────────────────────────────────────────────────
# Skill cadence — same contract as memory, separate counter
# ────────────────────────────────────────────────────────────────────────────


class TestSkillCadenceBoundToSpawn:

    def test_skill_counter_resets_only_after_successful_spawn(self):
        """Happy path: skill cadence threshold met, spawn succeeds."""
        agent = _make_agent(memory_nudge_interval=1000, skill_nudge_interval=3)
        agent._iters_since_skill = 5  # already past threshold

        result, spawn_patch = _drive_clean_turn(agent)

        assert result["final_response"] == "done"
        spawn_patch.assert_called_once()
        kwargs = spawn_patch.call_args.kwargs
        assert kwargs["review_skills"] is True
        assert agent._iters_since_skill == 0

    def test_skill_counter_stays_when_spawn_raises(self):
        agent = _make_agent(memory_nudge_interval=1000, skill_nudge_interval=3)
        agent._iters_since_skill = 5

        result, spawn_patch = _drive_clean_turn(
            agent,
            spawn_mock=MagicMock(side_effect=RuntimeError("spawn blew up")),
        )

        assert result["final_response"] == "done"
        # Counter NOT reset to 0 by the fix.  Note that the in-iteration
        # increment at the top of each loop pass (``_iters_since_skill
        # += 1``) still fires — the fix only removes the EAGER reset
        # to 0, it doesn't suppress the normal per-iteration counting.
        # 1 iteration ran (one mocked API response), so 5 → 6.
        assert agent._iters_since_skill == 6, (
            f"Skill counter unexpectedly reset on spawn failure; got: "
            f"{agent._iters_since_skill}"
        )

    def test_skill_counter_stays_when_loop_interrupted_before_final_response(self):
        agent = _make_agent(memory_nudge_interval=1000, skill_nudge_interval=3)
        agent._iters_since_skill = 5
        agent._interrupt_requested = True

        with (
            patch.object(agent, "_spawn_background_review") as spawn_patch,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        spawn_patch.assert_not_called()
        assert result.get("interrupted") is True
        assert agent._iters_since_skill == 5


# ────────────────────────────────────────────────────────────────────────────
# Independence — resetting one counter must not touch the other
# ────────────────────────────────────────────────────────────────────────────


class TestCountersResetIndependently:

    def test_only_memory_triggered_does_not_reset_skill_counter(self):
        """Memory cadence met, skill cadence below threshold — only the
        memory counter should reset, the skill counter should keep its
        natural in-iteration increment without being clobbered to 0.
        """
        agent = _make_agent(memory_nudge_interval=3, skill_nudge_interval=100)
        agent._turns_since_memory = 2  # crosses to 3 → triggers
        agent._iters_since_skill = 7   # nowhere near 100

        result, spawn_patch = _drive_clean_turn(agent)

        spawn_patch.assert_called_once()
        kwargs = spawn_patch.call_args.kwargs
        assert kwargs["review_memory"] is True
        assert kwargs["review_skills"] is False
        assert agent._turns_since_memory == 0
        # Skill counter advanced naturally (7 → 8) but was NOT reset.
        # Pre-fix the eager reset adjacent to ``_should_review_skills``
        # would have clobbered this to 0 even though only memory fired.
        assert agent._iters_since_skill == 8, (
            "Skill counter was reset despite its cadence not firing — "
            "would silently push the skill review further into the "
            "future on every memory-only review turn."
        )
        assert result["final_response"] == "done"

    def test_only_skill_triggered_does_not_reset_memory_counter(self):
        """Symmetric test: only the skill cadence fires this turn."""
        agent = _make_agent(memory_nudge_interval=100, skill_nudge_interval=3)
        agent._turns_since_memory = 7
        agent._iters_since_skill = 5

        result, spawn_patch = _drive_clean_turn(agent)

        spawn_patch.assert_called_once()
        kwargs = spawn_patch.call_args.kwargs
        assert kwargs["review_memory"] is False
        assert kwargs["review_skills"] is True
        assert agent._iters_since_skill == 0
        # Memory counter still advances by one because the cadence
        # increment is unconditional — it just doesn't get reset.
        assert agent._turns_since_memory == 8, (
            "Memory counter was reset despite its cadence not firing"
        )
        assert result["final_response"] == "done"

    def test_both_triggered_resets_both(self):
        """When both cadences fire on the same turn, both counters
        reset together after a successful spawn.
        """
        agent = _make_agent(memory_nudge_interval=3, skill_nudge_interval=3)
        agent._turns_since_memory = 2
        agent._iters_since_skill = 5

        result, spawn_patch = _drive_clean_turn(agent)

        spawn_patch.assert_called_once()
        kwargs = spawn_patch.call_args.kwargs
        assert kwargs["review_memory"] is True
        assert kwargs["review_skills"] is True
        assert agent._turns_since_memory == 0
        assert agent._iters_since_skill == 0
        assert result["final_response"] == "done"

    def test_both_triggered_but_spawn_raises_resets_neither(self):
        """The ``else`` branch only runs when the ``try`` block exits
        normally.  A raise must leave BOTH counters intact.
        """
        agent = _make_agent(memory_nudge_interval=3, skill_nudge_interval=3)
        agent._turns_since_memory = 2
        agent._iters_since_skill = 5

        result, spawn_patch = _drive_clean_turn(
            agent,
            spawn_mock=MagicMock(side_effect=RuntimeError("nope")),
        )

        spawn_patch.assert_called_once()
        assert agent._turns_since_memory == 3
        # Skill counter advanced naturally during the iteration loop
        # (5 → 6) but was NOT reset to 0 by the fix.
        assert agent._iters_since_skill == 6
        assert result["final_response"] == "done", (
            "Spawn raising is still a best-effort path — the user's "
            "turn must still complete normally."
        )


# ────────────────────────────────────────────────────────────────────────────
# Contract pin: the conversation_loop file no longer resets eagerly
# ────────────────────────────────────────────────────────────────────────────


class TestConversationLoopWiring:
    """Static checks that pin the structural change — a future
    refactor that re-introduces an eager reset in the cadence-decision
    blocks would silently re-create #30812 without flipping any of the
    behavioral tests above (because the *current* clean-path tests
    would still pass).  These guard against that regression.
    """

    def test_no_eager_reset_in_memory_decision_block(self):
        from pathlib import Path
        source = Path("agent/conversation_loop.py").read_text()
        # Locate the decision block by its decision flag, then verify
        # the eager-reset assignment is no longer next to it.
        idx = source.find("_should_review_memory = True")
        assert idx != -1, "Memory decision block disappeared"
        # Inspect the ~600 characters after the flag set.  The old buggy
        # pattern was a literal ``agent._turns_since_memory = 0``
        # adjacent to the flag set.  The new code puts the reset much
        # later in the file, inside the spawn ``else`` branch.
        window = source[idx:idx + 600]
        assert "agent._turns_since_memory = 0" not in window, (
            "Found eager memory-counter reset adjacent to the decision "
            "block — re-introduces #30812. Move the reset to the "
            "_spawn_background_review else-branch."
        )

    def test_no_eager_reset_in_skill_decision_block(self):
        from pathlib import Path
        source = Path("agent/conversation_loop.py").read_text()
        idx = source.find("_should_review_skills = True")
        assert idx != -1, "Skill decision block disappeared"
        window = source[idx:idx + 400]
        assert "agent._iters_since_skill = 0" not in window, (
            "Found eager skill-counter reset adjacent to the decision "
            "block — re-introduces #30812."
        )

    def test_reset_lives_in_spawn_else_branch(self):
        """The reset must be in the ``else`` of the spawn ``try`` so it
        runs only when ``_spawn_background_review`` returned cleanly.
        """
        from pathlib import Path
        source = Path("agent/conversation_loop.py").read_text()
        # Find the spawn block by its function name; an ``else`` must
        # follow the matching ``except`` and the two resets must live
        # underneath the ``else``.
        spawn_idx = source.find("agent._spawn_background_review")
        assert spawn_idx != -1
        # Look for both resets after the spawn block within a few
        # hundred lines.
        after = source[spawn_idx:spawn_idx + 1500]
        assert "agent._turns_since_memory = 0" in after, (
            "Memory counter reset not found in the spawn try/except/else "
            "region — fix may have been reverted."
        )
        assert "agent._iters_since_skill = 0" in after, (
            "Skill counter reset not found in the spawn try/except/else "
            "region — fix may have been reverted."
        )
        # The reset must be conditional on the per-flag check so that
        # the independence guarantees above hold.
        assert "if _should_review_memory:" in after
        assert "if _should_review_skills:" in after


# ────────────────────────────────────────────────────────────────────────────
# Hydration interaction — fix must not regress #22357
# ────────────────────────────────────────────────────────────────────────────


class TestHydrationStillWorks:

    def test_hydration_block_still_seeds_counter_on_fresh_agent(self):
        """The CLI-mode hydration block runs only when
        ``conversation_history`` is non-empty and ``_user_turn_count==0``
        — both my change AND the original code leave that branch alone,
        so a freshly built agent (gateway cache miss, idle eviction)
        must still pick up the right value from prior history.
        """
        agent = _make_agent(memory_nudge_interval=10)
        # Seed by replaying the hydration math the function performs:
        # 7 prior user turns, interval 10 → counter should land at 7.
        history = []
        for i in range(7):
            history.append({"role": "user", "content": f"q{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="ok", finish_reason="stop", tool_calls=None)
        ]
        with (
            patch.object(agent, "_spawn_background_review") as spawn_patch,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("hello", conversation_history=history)

        # 7 (hydrated) + 1 (this turn's increment) = 8, still below
        # interval 10 → no spawn, no reset.
        spawn_patch.assert_not_called()
        assert agent._turns_since_memory == 8
