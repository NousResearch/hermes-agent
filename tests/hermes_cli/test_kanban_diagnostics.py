"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


def _comment(body, ts=None, author="worker", comment_id=1):
    return {
        "id": comment_id,
        "task_id": "t_demo00",
        "author": author,
        "body": body,
        "created_at": int(ts if ts is not None else time.time()),
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48






def test_repeated_crashes_truncates_huge_tracebacks():
    """Full Python tracebacks can be tens of KB. The title stays one
    line (≤160 chars); the detail caps at 500 chars + ellipsis so the
    card doesn't explode visually."""
    huge = "Traceback (most recent call last):\n" + ("  File\n" * 500)
    task = _task(status="ready")
    runs = [
        _run(outcome="crashed", run_id=1, error=huge),
        _run(outcome="crashed", run_id=2, error=huge),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    d = diags[0]
    # Title only the first line, capped.
    assert "\n" not in d.title
    assert len(d.title) < 250
    # Detail contains the snippet with ellipsis.
    assert d.detail.endswith("…") or len(d.detail) < 700


# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True


# ---------------------------------------------------------------------------
# optimization_missing_cost_baseline rule
#
# A card that reads as optimization/cost-reduction/routing/caching work but
# never quantifies (a) what the status quo costs today or (b) how much it
# must save to be worth doing. See t_2f9851d5: three full instrumentation
# rounds ran on the model-routing card before anyone asked if it was worth
# starting, because the card model had no field to ask for that in.
# ---------------------------------------------------------------------------


def test_optimization_card_missing_cost_baseline_fires():
    task = _task(
        id="t_opt0001",
        title="Route OpenCode's non-flagship agent roles to cheaper models",
        body="Static config change, no dynamic routing exists yet.",
        status="todo",
    )
    diags = kd.compute_task_diagnostics(task, [], [])
    hits = [d for d in diags if d.kind == "optimization_missing_cost_baseline"]
    assert len(hits) == 1
    assert hits[0].severity == "warning"
    assert hits[0].data["has_baseline"] is False
    assert hits[0].data["has_threshold"] is False


def test_optimization_card_with_quantified_baseline_and_threshold_clears():
    task = _task(
        id="t_opt0002",
        title="Route OpenCode's non-flagship agent roles to cheaper models",
        body=(
            "Measured baseline cost: $42/day across non-flagship roles today. "
            "Savings threshold: must save at least $10/day to be worth building."
        ),
        status="todo",
    )
    diags = kd.compute_task_diagnostics(task, [], [])
    hits = [d for d in diags if d.kind == "optimization_missing_cost_baseline"]
    assert hits == []


def test_optimization_rule_ignores_non_optimization_cards():
    task = _task(
        id="t_plain001",
        title="Fix typo in onboarding doc",
        body="The word 'recieve' is misspelled on line 12.",
        status="todo",
    )
    diags = kd.compute_task_diagnostics(task, [], [])
    hits = [d for d in diags if d.kind == "optimization_missing_cost_baseline"]
    assert hits == []


def test_optimization_rule_number_does_not_cross_satisfy_both_fields():
    """coderabbit finding: a single number sitting between the two keyword
    phrases must not satisfy BOTH fields at once — only the field it's
    actually closer to."""
    task = _task(
        id="t_opt0003",
        title="Route OpenCode's non-flagship agent roles to cheaper models",
        # A lone "$10/day" between the two keyword phrases, but only
        # actually describing the savings threshold, not the baseline.
        body=(
            "Static config change, no dynamic routing exists yet. "
            "Measured cost today: unclear. Savings threshold: must save $10/day to be worth it."
        ),
    )
    diags = kd.compute_task_diagnostics(task, [], [])
    hits = [d for d in diags if d.kind == "optimization_missing_cost_baseline"]
    assert len(hits) == 1
    assert hits[0].data["has_baseline"] is False
    assert hits[0].data["has_threshold"] is True
    # coderabbit finding: title must name the field that's ACTUALLY missing
    # (threshold is present here, only baseline is missing).
    assert "savings threshold" not in hits[0].title.lower()
    assert "cost baseline" in hits[0].title.lower()


# ---------------------------------------------------------------------------
# done_action_language_unaddressed
#
# A done card whose last comment reads as an unaddressed finding (action
# language, NO/EN) and which has no child card at all. Flag-only, mirrors
# _rule_review_intent_untagged's low-intervention level.
# ---------------------------------------------------------------------------


def test_done_action_language_fires_on_english_recommendation_no_children():
    now = int(time.time())
    task = _task(status="done")
    comments = [_comment("Investigation done. I recommend building a follow-up rule for X.", ts=now)]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments, graph={"parents": [], "children": []},
    )
    hits = [d for d in diags if d.kind == "done_action_language_unaddressed"]
    assert len(hits) == 1
    assert hits[0].severity == "warning"
    assert "recommend" in hits[0].data["matched_text"].lower()


def test_done_action_language_fires_on_norwegian_recommendation_no_children():
    now = int(time.time())
    task = _task(status="done")
    comments = [_comment("Funn: dette bør bygges som egen regel senere.", ts=now)]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments, graph={"parents": [], "children": []},
    )
    hits = [d for d in diags if d.kind == "done_action_language_unaddressed"]
    assert len(hits) == 1


def test_done_action_language_silent_when_child_exists():
    """A child card already tracks the follow-up — nothing to flag."""
    now = int(time.time())
    task = _task(status="done")
    comments = [_comment("I recommend building a follow-up rule for X.", ts=now)]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments,
        graph={"parents": [], "children": [{"id": "t_child01", "title": "follow-up", "status": "ready"}]},
    )
    assert not [d for d in diags if d.kind == "done_action_language_unaddressed"]


def test_done_action_language_silent_when_not_done():
    now = int(time.time())
    task = _task(status="ready")
    comments = [_comment("I recommend building a follow-up rule for X.", ts=now)]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments, graph={"parents": [], "children": []},
    )
    assert not [d for d in diags if d.kind == "done_action_language_unaddressed"]


def test_done_action_language_silent_when_last_comment_has_no_action_language():
    now = int(time.time())
    task = _task(status="done")
    comments = [_comment("All good, nothing to report here.", ts=now)]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments, graph={"parents": [], "children": []},
    )
    assert not [d for d in diags if d.kind == "done_action_language_unaddressed"]


def test_done_action_language_only_checks_last_comment():
    """An action-language finding buried in an EARLIER comment, superseded by
    a clean final comment, must not fire — only the last comment counts."""
    now = int(time.time())
    task = _task(status="done")
    comments = [
        _comment("I recommend building X.", ts=now - 100, comment_id=1),
        _comment("Never mind, closing this out.", ts=now, comment_id=2),
    ]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments, graph={"parents": [], "children": []},
    )
    assert not [d for d in diags if d.kind == "done_action_language_unaddressed"]


def test_done_action_language_silent_without_comments_or_graph_context():
    """Missing context (no comments/graph passed) must never fire — the rule
    would rather under-report than false-positive."""
    now = int(time.time())
    task = _task(status="done")
    diags = kd.compute_task_diagnostics(task, [], [], now=now)
    assert not [d for d in diags if d.kind == "done_action_language_unaddressed"]


def test_done_action_language_can_be_disabled_via_config():
    now = int(time.time())
    task = _task(status="done")
    comments = [_comment("I recommend building a follow-up rule for X.", ts=now)]
    diags = kd.compute_task_diagnostics(
        task, [], [], now=now, comments=comments, graph={"parents": [], "children": []},
        config={"done_action_language_pattern": ""},
    )
    assert not [d for d in diags if d.kind == "done_action_language_unaddressed"]
