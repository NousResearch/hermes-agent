"""Tests for the unbounded goal-turn budget sentinel and the judge kill-switch.

Covers the feature that lets ``goals.max_turns`` be set to an unbounded
sentinel (``0`` / negative int / an unbounded-string) to disable the turn cap,
and ``goals.judge_enabled`` be set false to strip the judge of its power to
end the loop. Both are resolved through ``hermes_cli.goals`` so every consumer
(CLI, gateway, TUI, kanban worker) agrees.
"""

import json

import pytest

from hermes_cli import goals
from hermes_cli.goals import (
    DEFAULT_MAX_TURNS,
    GoalManager,
    GoalState,
    _fmt_turns,
    goal_budget_label,
    goal_judge_enabled,
    resolve_goal_max_turns,
)


# ────────────────────────────────────────────────────────────────────────────
# resolve_goal_max_turns — the sentinel contract
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value,expected",
    [
        # unset / missing -> historical default
        (None, DEFAULT_MAX_TURNS),
        # positive ints unchanged
        (20, 20),
        (5, 5),
        (1, 1),
        # unbounded sentinels -> None
        (0, None),
        (-1, None),
        (-100, None),
        # bools are NOT ints for this purpose (bool subclasses int)
        (True, DEFAULT_MAX_TURNS),
        (False, DEFAULT_MAX_TURNS),
        # unbounded strings (case/whitespace-insensitive)
        ("unbounded", None),
        ("INFINITE", None),
        ("infinity", None),
        ("none", None),
        ("unlimited", None),
        ("  UNBOUNDED ", None),
        # numeric strings
        ("10", 10),
        ("0", None),
        ("-7", None),
        # junk strings fall back to default rather than crash
        ("junk", DEFAULT_MAX_TURNS),
        ("", DEFAULT_MAX_TURNS),
        ("   ", DEFAULT_MAX_TURNS),
        # unknown types -> conservative default
        (3.5, DEFAULT_MAX_TURNS),
        ([], DEFAULT_MAX_TURNS),
        ({}, DEFAULT_MAX_TURNS),
    ],
)
def test_resolve_goal_max_turns(value, expected):
    assert resolve_goal_max_turns(value) == expected


def test_resolve_goal_max_turns_positive_int_unchanged():
    """A valid pre-existing config value must behave exactly as before."""
    for n in (1, 7, 20, 999):
        assert resolve_goal_max_turns(n) == n


# ────────────────────────────────────────────────────────────────────────────
# goal_judge_enabled — the kill-switch flag
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({"goals": {"judge_enabled": True}}, True),
        ({"goals": {"judge_enabled": False}}, False),
        ({"goals": {}}, True),                    # missing -> default True
        ({}, True),                                # no goals block -> True
        ({"goals": {"judge_enabled": "false"}}, False),
        ({"goals": {"judge_enabled": "off"}}, False),
        ({"goals": {"judge_enabled": "no"}}, False),
        ({"goals": {"judge_enabled": "0"}}, False),
        ({"goals": {"judge_enabled": "true"}}, True),
        ({"goals": {"judge_enabled": "junk"}}, True),   # unrecognized -> safe True
        ({"goals": {"judge_enabled": 0}}, False),
        ({"goals": {"judge_enabled": 1}}, True),
    ],
)
def test_goal_judge_enabled(cfg, expected):
    assert goal_judge_enabled(cfg) is expected


def test_goal_judge_enabled_fail_open_on_bad_cfg():
    """A misconfigured value must never silently disable the bounding judge."""
    class _Bad:
        def get(self, *a, **k):
            raise RuntimeError("boom")

    assert goal_judge_enabled(_Bad()) is True


# ────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────────────────

def test_fmt_turns_bounded():
    assert _fmt_turns(3, 20) == "3/20"


def test_fmt_turns_unbounded():
    assert _fmt_turns(3, None) == "3/∞"


def test_goal_budget_label():
    assert goal_budget_label(20) == "20-turn budget"
    assert goal_budget_label(7) == "7-turn budget"
    assert goal_budget_label(None) == "unbounded budget"


# ────────────────────────────────────────────────────────────────────────────
# GoalState JSON round-trip with an unbounded budget
# ────────────────────────────────────────────────────────────────────────────

def test_goal_state_roundtrip_unbounded():
    s = GoalState(goal="g", max_turns=None, turns_used=3)
    s2 = GoalState.from_json(s.to_json())
    assert s2.max_turns is None
    assert s2.turns_used == 3


def test_goal_state_roundtrip_bounded():
    s = GoalState(goal="g", max_turns=7, turns_used=1)
    s2 = GoalState.from_json(s.to_json())
    assert s2.max_turns == 7


def test_goal_state_legacy_row_missing_max_turns_defaults():
    """A row persisted before this feature has no max_turns -> default."""
    s = GoalState.from_json(json.dumps({"goal": "g"}))
    assert s.max_turns == DEFAULT_MAX_TURNS


# ────────────────────────────────────────────────────────────────────────────
# GoalManager budget resolution (DB I/O stubbed)
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def _no_goal_db(monkeypatch):
    monkeypatch.setattr(goals, "load_goal", lambda sid: None)
    monkeypatch.setattr(goals, "save_goal", lambda sid, st: None)


def test_manager_unbounded_default(_no_goal_db):
    mgr = GoalManager(session_id="t", default_max_turns=None)
    assert mgr.default_max_turns is None
    assert mgr.set("goal").max_turns is None


def test_manager_finite_default(_no_goal_db):
    mgr = GoalManager(session_id="t", default_max_turns=20)
    assert mgr.default_max_turns == 20
    assert mgr.set("goal").max_turns == 20


def test_manager_no_arg_default(_no_goal_db):
    assert GoalManager(session_id="t").default_max_turns == DEFAULT_MAX_TURNS


def test_manager_explicit_per_goal_overrides_unbounded_default(_no_goal_db):
    mgr = GoalManager(session_id="t", default_max_turns=None)
    assert mgr.set("goal", max_turns=5).max_turns == 5


def test_manager_per_goal_none_means_use_default(_no_goal_db):
    """set(max_turns=None) means 'use the manager default', not 'unbounded'."""
    mgr = GoalManager(session_id="t", default_max_turns=20)
    assert mgr.set("goal", max_turns=None).max_turns == 20


# ────────────────────────────────────────────────────────────────────────────
# evaluate_after_turn — judge stub that accepts the real call signature
# ────────────────────────────────────────────────────────────────────────────

def _judge_stub(verdict, reason="stub"):
    """Return a judge_goal replacement accepting the production kwargs.

    Must mirror judge_goal's 5-tuple return:
    ``(verdict, reason, parse_failed, wait_directive, transport_failed)``.
    """
    def _judge(goal, last_response, *, subgoals=None, background_processes=None,
               contract=None):
        return (verdict, reason, False, None, False)
    return _judge


def _mgr_with_state(monkeypatch, state, judge_verdict, judge_on):
    """Build a GoalManager whose _state is preset and judge/flag are stubbed."""
    monkeypatch.setattr(goals, "judge_goal", _judge_stub(judge_verdict))
    monkeypatch.setattr(goals, "goal_judge_enabled", lambda cfg=None: judge_on)
    monkeypatch.setattr(goals, "save_goal", lambda sid, st: None)
    mgr = GoalManager(session_id="t", default_max_turns=state.max_turns)
    mgr._state = state
    return mgr


# ────────────────────────────────────────────────────────────────────────────
# evaluate_after_turn — budget skip when unbounded
# ────────────────────────────────────────────────────────────────────────────

def test_evaluate_skips_budget_when_unbounded(monkeypatch):
    """With max_turns=None the budget check must not fire even after many turns."""
    state = GoalState(goal="g", status="active", max_turns=None, turns_used=999)
    mgr = _mgr_with_state(monkeypatch, state, "continue", judge_on=True)

    result = mgr.evaluate_after_turn("latest response")
    assert result["status"] != "paused"
    assert result["should_continue"] is True


def test_evaluate_budget_fires_when_bounded(monkeypatch):
    """Sanity: a finite budget still pauses the loop when exhausted."""
    state = GoalState(goal="g", status="active", max_turns=2, turns_used=0)
    mgr = _mgr_with_state(monkeypatch, state, "continue", judge_on=True)

    mgr.evaluate_after_turn("r1")          # turns_used -> 1
    result = mgr.evaluate_after_turn("r2")  # turns_used -> 2 == budget
    assert result["status"] == "paused"
    assert result["should_continue"] is False


# ────────────────────────────────────────────────────────────────────────────
# evaluate_after_turn — judge kill-switch coerces done -> continue
# ────────────────────────────────────────────────────────────────────────────

def test_judge_disabled_coerces_done_to_continue(monkeypatch):
    """goals.judge_enabled=false: a 'done' verdict must not end the loop."""
    state = GoalState(goal="g", status="active", max_turns=None, turns_used=0)
    mgr = _mgr_with_state(monkeypatch, state, "done", judge_on=False)

    result = mgr.evaluate_after_turn("response")
    assert result["status"] != "done"
    assert result["should_continue"] is True
    # The judge's true verdict is still recorded for diagnostics.
    assert state.last_verdict == "done"


def test_judge_enabled_done_ends_loop(monkeypatch):
    """Default (enabled): a 'done' verdict completes the goal."""
    state = GoalState(goal="g", status="active", max_turns=None, turns_used=0)
    mgr = _mgr_with_state(monkeypatch, state, "done", judge_on=True)

    result = mgr.evaluate_after_turn("response")
    assert result["should_continue"] is False
    assert state.last_verdict == "done"


# ────────────────────────────────────────────────────────────────────────────
# Hardening: a literal ``0`` must never survive as a finite-0 budget
# (regression for the two dormant paths the adversarial review flagged:
# GoalState.from_dict and GoalManager.set). A finite-0 would make the budget
# guard fire ``turns_used(0) >= 0`` immediately; it must collapse to unbounded.
# ────────────────────────────────────────────────────────────────────────────

def test_from_json_literal_zero_collapses_to_unbounded():
    """A hand-edited state row with max_turns: 0 loads as unbounded, not finite-0."""
    state = GoalState.from_json(json.dumps({"goal": "g", "status": "active", "max_turns": 0}))
    assert state.max_turns is None


def test_from_json_negative_collapses_to_unbounded():
    state = GoalState.from_json(json.dumps({"goal": "g", "status": "active", "max_turns": -3}))
    assert state.max_turns is None


def test_from_json_positive_int_preserved():
    state = GoalState.from_json(json.dumps({"goal": "g", "status": "active", "max_turns": 7}))
    assert state.max_turns == 7


def test_from_json_none_preserved():
    """A persisted unbounded row (max_turns: null) round-trips as None."""
    state = GoalState.from_json(json.dumps({"goal": "g", "status": "active", "max_turns": None}))
    assert state.max_turns is None


def test_from_json_missing_uses_default():
    """A legacy row with no max_turns key loads with the historical default."""
    state = GoalState.from_json(json.dumps({"goal": "g", "status": "active"}))
    assert state.max_turns == DEFAULT_MAX_TURNS


def test_manager_set_literal_zero_is_unbounded():
    """GoalManager.set(max_turns=0) resolves to unbounded, never a finite-0."""
    mgr = GoalManager(session_id="t", default_max_turns=DEFAULT_MAX_TURNS)
    state = mgr.set("g", max_turns=0)
    assert state.max_turns is None


def test_manager_set_positive_int_preserved():
    mgr = GoalManager(session_id="t", default_max_turns=DEFAULT_MAX_TURNS)
    state = mgr.set("g", max_turns=5)
    assert state.max_turns == 5


def test_manager_set_none_uses_manager_default():
    mgr = GoalManager(session_id="t", default_max_turns=9)
    state = mgr.set("g", max_turns=None)
    assert state.max_turns == 9
