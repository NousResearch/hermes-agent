"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


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








# ---------------------------------------------------------------------------
# review_intent_untagged rule — silent loss of verifier isolation
# ---------------------------------------------------------------------------


def _review_card(**overrides):
    base = {"id": "t_rev001", "title": "Independent review: worker handoff",
            "status": "ready", "assignee": "reviewer", "skills": None}
    base.update(overrides)
    return _task(**base)


def _review_diag(task, **kwargs):
    return [d for d in kd.compute_task_diagnostics(task, [], [], **kwargs)
            if d.kind == "review_intent_untagged"]


def test_review_intent_untagged_fires_on_an_untagged_review_card():
    diags = _review_diag(_review_card())
    assert len(diags) == 1
    assert diags[0].severity == "warning"
    assert diags[0].data["assignee"] == "reviewer"
    # The suggested action must be the one that actually fixes it.
    assert any(a.suggested and "--review" in a.payload.get("command", "") for a in diags[0].actions)


def test_review_intent_untagged_silent_when_the_dispatcher_tag_is_present():
    """The rule's whole job is to warn about MISSING isolation — a card the
    dispatcher will isolate must never be flagged. Reads the dispatcher's own
    tag set so the two cannot drift apart."""
    from hermes_cli.kanban_db_dispatch import REVIEW_TAG_SKILLS

    for tag in sorted(REVIEW_TAG_SKILLS):
        assert _review_diag(_review_card(skills=[tag])) == []
    # Rows straight from sqlite carry skills as a JSON string, not a list.
    assert _review_diag(_review_card(skills=json.dumps(sorted(REVIEW_TAG_SKILLS)[:1]))) == []


def test_review_intent_untagged_ignores_cards_that_are_not_review_roles():
    """A title that merely MENTIONS review is not a review role — over-firing
    here would train operators to ignore the warning."""
    for title in ("Fix review lane deadlock in dispatcher",
                  "Make kanban review-isolation structural: diagnostics + --review flag",
                  "Add regression tests for terminal.env propagation",
                  "Post-merge verification: gateway restart, live probe"):
        assert _review_diag(_review_card(title=title)) == [], title


def test_review_intent_untagged_needs_a_dispatchable_card():
    # No assignee: never dispatches, so nothing can be contaminated.
    assert _review_diag(_review_card(assignee=None)) == []
    assert _review_diag(_review_card(assignee="   ")) == []
    # Terminal: the worker already ran (or never will); the warning is moot.
    for status in ("done", "archived"):
        assert _review_diag(_review_card(status=status)) == []


def test_review_intent_untagged_silent_for_cards_in_the_review_lane():
    """A card in the `review` column is isolated STRUCTURALLY — the dispatcher
    force-adds `sdlc-review` (a REVIEW_TAG_SKILLS member) at spawn — so its
    stored skills being untagged is not a defect. Flagging it would be a pure
    false positive on the one review path that cannot lose isolation."""
    assert _review_diag(_review_card(status="review")) == []


def test_review_intent_untagged_shell_quotes_the_card_title():
    """The suggested command is copy-pasted into a shell and the title is
    card-supplied text, so it must be shell-quoted. An unquoted title carrying
    a quote or a ``;`` would produce a broken — or actively dangerous —
    command in the operator's terminal."""
    import shlex

    nasty = 'Review "x"; rm -rf ~/tmp #'
    diags = _review_diag(_review_card(title=nasty))
    assert len(diags) == 1
    command = [a for a in diags[0].actions if a.suggested][0].payload["command"]
    # The whole title must survive as exactly ONE shell word.
    parsed = shlex.split(command)
    assert nasty in parsed
    assert parsed[:3] == ["hermes", "kanban", "create"]


def test_review_intent_untagged_can_be_disabled_by_config():
    assert _review_diag(_review_card(), config={"review_intent_pattern": ""}) == []


def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True
