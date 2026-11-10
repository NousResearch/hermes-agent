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
# prose_phantom_refs — the advisory flag needs an acknowledgement path
#
# The flag is appended *after* the completion that raises it commits, so the
# only events that can clear it are later ones. Ids that legitimately live on
# another copy of the board (the review-gate matrix's fixture card, a card a
# reviewer filed in a probe copy) land here too, so a comment must clear the
# advisory warning; the error-level hallucinated_cards flag must not.
# ---------------------------------------------------------------------------


_PHANTOM_FLAG = {"phantom_refs": ["t_dead0001"], "source": "completion_summary"}


def _prose_kinds(task, events):
    return [d.kind for d in kd.compute_task_diagnostics(task, events, [], now=10_000)]


def test_prose_phantom_refs_fires_on_unknown_id_in_completion():
    """A post-completion flag on a done card is a warning with a comment action —
    nothing else can acknowledge it, because the completing event predates it."""
    events = [
        _event("created", ts=100),
        _event("completed", ts=200),
        _event("suspected_hallucinated_references", ts=201, **_PHANTOM_FLAG),
    ]
    diags = [d for d in kd.compute_task_diagnostics(_task(status="done"), events, [], now=10_000)
             if d.kind == "prose_phantom_refs"]
    assert len(diags) == 1
    d = diags[0]
    assert d.severity == "warning"
    assert d.data["phantom_refs"] == ["t_dead0001"]
    assert d.actions[0].kind == "comment"
    assert d.actions[0].suggested is True


def test_prose_phantom_refs_clears_on_a_comment():
    """The acknowledgement path: a comment after the flag clears the advisory."""
    events = [
        _event("completed", ts=200),
        _event("suspected_hallucinated_references", ts=201, **_PHANTOM_FLAG),
        _event("commented", ts=300),
    ]
    assert "prose_phantom_refs" not in _prose_kinds(_task(status="done"), events)


def test_prose_phantom_refs_still_clears_on_a_later_completion():
    """The pre-existing clearing event keeps working: widening the set must not
    drop ``completed``/``edited``."""
    for kind in ("completed", "edited"):
        events = [
            _event("suspected_hallucinated_references", ts=200, **_PHANTOM_FLAG),
            _event(kind, ts=300),
        ]
        assert "prose_phantom_refs" not in _prose_kinds(_task(status="done"), events)


def test_hallucinated_cards_is_not_cleared_by_a_comment():
    """A refused completion is an error, not an advisory: a comment must not
    launder it — only a later completion/edit resolves it."""
    blocked = _event("completion_blocked_hallucination", ts=200,
                     phantom_cards=["t_deadbeef1"], reason="hallucinated_cards")
    commented = _event("commented", ts=300)
    task = _task(status="ready")
    assert "hallucinated_cards" in _prose_kinds(task, [blocked, commented])
    assert "hallucinated_cards" not in _prose_kinds(task, [blocked, _event("completed", ts=300)])


def test_prose_phantom_ack_round_trips_through_db(kanban_home):
    """End to end on a real DB: ``complete_task`` appends the flag post-commit,
    the rule surfaces it on the sqlite Row objects the CLI/dashboard pass, and a
    comment written by :func:`kb.add_comment` clears it."""
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="prose phantom", assignee="w")

        def kinds():
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            events = list(conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
            ).fetchall())
            runs = list(conn.execute(
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
            ).fetchall())
            return [(d.kind, d) for d in kd.compute_task_diagnostics(row, events, runs)]

        assert kb.complete_task(
            conn, task_id, summary="done; nothing filed, the probe board holds t_deadbeef1",
        )
        found = dict(kinds())
        assert "prose_phantom_refs" in found
        assert found["prose_phantom_refs"].data["phantom_refs"] == ["t_deadbeef1"]

        kb.add_comment(
            conn, task_id, "w",
            "t_deadbeef1 is a scratch-fixture id on another board copy, not this board",
        )
        assert "prose_phantom_refs" not in dict(kinds())
    finally:
        conn.close()
