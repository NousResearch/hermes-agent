"""Tests for the reconciled autopilot honesty signals folded into /goal.

PR #51565 originally shipped a *parallel* autopilot engine. This reconciled
version drops that engine and instead folds two additive, opt-in, fail-soft
honesty signals into the established ``/goal`` Ralph loop
(``hermes_cli/goals.py``):

    * the deception detector sharpens the continuation directive when the last
      response shows a known reward-seeking cheat pattern, and
    * the off-by-default ADR log records the judge's per-turn decision.

Neither can flip the loop-vs-stop decision or bypass the turn budget; the
``/goal`` loop's own bounded budget remains the single safety cap. This file
proves that additivity and the specific review-nit fixes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber real state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    return home


# --------------------------------------------------------------------------- #
# strict_completion — the "bool('false') is True" verdict-validation nit        #
# (teknium1 @ council_gate.py:281)                                              #
# --------------------------------------------------------------------------- #
class TestStrictCompletion:
    def test_only_real_true_or_truthy_token_completes(self):
        from agent.autopilot import strict_completion

        # Real boolean True is the canonical complete verdict.
        assert strict_completion(True) is True
        # Explicit truthy tokens count.
        for tok in ("true", "TRUE", "yes", "done", "complete", "1"):
            assert strict_completion(tok) is True

    def test_plausible_but_false_never_completes(self):
        from agent.autopilot import strict_completion

        # The exact bug: bool("false") is True, but this must read as NOT done.
        assert strict_completion("false") is False
        assert strict_completion("no") is False
        assert strict_completion("0") is False
        assert strict_completion(0) is False
        assert strict_completion(False) is False
        assert strict_completion(None) is False
        # Any non-boolean/odd type fails safe (keep going), never completes.
        assert strict_completion({"complete": True}) is False
        assert strict_completion(["true"]) is False


# --------------------------------------------------------------------------- #
# Deception addendum folded into the continue branch                            #
# --------------------------------------------------------------------------- #
class TestDeceptionAddendum:
    def test_clean_response_leaves_prompt_unchanged(self):
        from hermes_cli.goals import _apply_deception_addendum

        base = "[Continuing toward your standing goal]\nGoal: ship it"
        out, note = _apply_deception_addendum(base, "I ran the tests and they pass.")
        assert out == base
        assert note == ""

    def test_cheat_pattern_sharpens_the_directive(self):
        from hermes_cli.goals import _apply_deception_addendum

        base = "[Continuing toward your standing goal]\nGoal: ship it"
        # An explicit await-user handoff is a known cheat tell.
        cheat = "This is ready for your review; awaiting your confirmation to proceed."
        out, note = _apply_deception_addendum(base, cheat)
        assert out != base
        assert out.startswith(base)
        assert "CAUGHT" in out
        assert note

    def test_addendum_is_fail_soft(self):
        from hermes_cli.goals import _apply_deception_addendum

        # Empty inputs never raise and never mutate.
        assert _apply_deception_addendum(None, "x") == (None, "")
        assert _apply_deception_addendum("p", "") == ("p", "")


# --------------------------------------------------------------------------- #
# The additive signals do NOT change loop-vs-stop or the turn budget            #
# --------------------------------------------------------------------------- #
class TestAdditivityInLoop:
    def test_continue_branch_appends_addendum_without_changing_decision(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="decep-sid-1", default_max_turns=20)
        mgr.set("do a thing")

        cheat = "The task is complete; I'll let you verify the rest yourself."
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "not done yet", False, None, False),
        ):
            decision = mgr.evaluate_after_turn(cheat)

        # Decision itself is unchanged: still a plain continue under budget.
        assert decision["should_continue"] is True
        assert decision["verdict"] == "continue"
        # But the continuation directive was sharpened with the CAUGHT addendum.
        assert "CAUGHT" in (decision["continuation_prompt"] or "")

    def test_turn_budget_still_the_hard_cap_regardless_of_signals(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="decep-sid-2", default_max_turns=1)
        mgr.set("do a thing")
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "keep going", False, None, False),
        ):
            # Even with a cheat tell present, the budget pause fires at max_turns.
            decision = mgr.evaluate_after_turn("ready for your review, awaiting confirmation")
        assert decision["should_continue"] is False
        assert decision["status"] == "paused"

    def test_adr_records_continue_decision_when_enabled(self, hermes_home, tmp_path, monkeypatch):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        adr_file = tmp_path / "adr.md"
        monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "1")
        monkeypatch.setenv("AUTOPILOT_ADR_PATH", str(adr_file))

        mgr = GoalManager(session_id="decep-sid-3", default_max_turns=20)
        mgr.set("ship the feature")
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "one gap remains", False, None, False),
        ):
            mgr.evaluate_after_turn("made progress on step one")

        assert adr_file.exists()
        body = adr_file.read_text()
        assert "continue" in body
        assert "ship the feature" in body
        assert "one gap remains" in body

    def test_adr_is_noop_when_disabled(self, hermes_home, tmp_path, monkeypatch):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        adr_file = tmp_path / "adr.md"
        monkeypatch.delenv("HERMES_AUTOPILOT_ADR", raising=False)
        monkeypatch.setenv("AUTOPILOT_ADR_PATH", str(adr_file))

        mgr = GoalManager(session_id="decep-sid-4", default_max_turns=20)
        mgr.set("ship the feature")
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "gap", False, None, False),
        ):
            mgr.evaluate_after_turn("progress")

        # Off by default: no file written.
        assert not adr_file.exists()
