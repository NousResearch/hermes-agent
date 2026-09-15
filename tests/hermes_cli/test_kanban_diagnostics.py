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
# repeated_crashes — a streak that is over is reported as historical
# ---------------------------------------------------------------------------

# The board's real shape (t_2d283056 / t_6db2248b / t_8afe5089, 2026-09-14):
# five crashed runs days old, and a NEWER run that ended `unreported` (the
# deliberate "exited cleanly without a terminal kanban call" outcome) or
# `blocked`. The page said "The last 5 runs ended with outcome=crashed" about
# those cards in the present tense and paged them `critical`.
_PROTOCOL_ERR = (
    "worker exited cleanly (rc=0) without calling kanban_complete or "
    "kanban_block — protocol violation."
)
_LAST_CRASH_END = 1789234174        # 2026-09-12 17:29Z
_NEWEST_END = 1789346851            # 2026-09-14 00:47Z


def _run_at(outcome, run_id, started_at, ended_at, error=None):
    return {
        "id": run_id, "outcome": outcome, "error": error,
        "started_at": started_at, "ended_at": ended_at,
    }


def _crash_rule_diag(task, runs):
    """The one repeated_crashes diagnostic, or a readable failure."""
    found = [d for d in kd.compute_task_diagnostics(task, [], runs)
             if d.kind == "repeated_crashes"]
    assert len(found) == 1, f"expected one repeated_crashes finding, got {found}"
    return found[0]


def _five_old_crashes():
    """Five consecutive crashes, newest first, all on 2026-09-12."""
    return [
        _run_at("crashed", 5, 1789233932, _LAST_CRASH_END, error=_PROTOCOL_ERR),
        _run_at("crashed", 4, 1789233691, 1789233932, error=_PROTOCOL_ERR),
        _run_at("crashed", 3, 1789233508, 1789233691, error=_PROTOCOL_ERR),
        _run_at("crashed", 2, 1789233327, 1789233508, error=_PROTOCOL_ERR),
        _run_at("crashed", 1, 1789233145, 1789233327, error=_PROTOCOL_ERR),
    ]


def test_repeated_crashes_finished_streak_is_historical_not_current():
    """A streak the card has since left must not be asserted in the present
    tense, and must not page at the critical floor the board-health page uses.
    """
    task = _task(status="ready")
    runs = [_run_at("unreported", 6, 1789346236, _NEWEST_END, error=_PROTOCOL_ERR)]
    runs += _five_old_crashes()

    d = _crash_rule_diag(task, runs)

    # Pre-fix this was "critical" with a present-tense claim.
    assert d.severity == "warning"
    assert "historical" in d.title
    assert "2026-09-12 17:29Z" in d.title
    assert "2026-09-12 17:29Z" in d.detail
    assert "not a current crash" in d.detail
    assert "2026-09-14 00:47Z" in d.detail and "outcome=unreported" in d.detail
    assert "crashed 5x" in d.title
    # The basis is inspectable: which runs were read, in which order, and what
    # the newest one ended as.
    assert d.data["is_current"] is False
    assert d.data["newest_outcome"] == "unreported"
    assert d.data["newest_run_id"] == 6
    assert d.data["last_crash_run_id"] == 5
    assert d.data["last_crash_at"] == _LAST_CRASH_END
    assert [r["id"] for r in d.data["runs_read"]] == [6, 5, 4, 3, 2, 1]
    assert [r["outcome"] for r in d.data["runs_read"]] == [
        "unreported", "crashed", "crashed", "crashed", "crashed", "crashed",
    ]


def test_repeated_crashes_finished_streak_behind_a_blocked_run_is_historical():
    """Same shape with the newest run `blocked` (t_6db2248b): the streak is
    over, so the finding dates it instead of claiming a live crash."""
    task = _task(status="ready")
    runs = [_run_at("blocked", 6, 1789346111, 1789346472, error=None)]
    runs += _five_old_crashes()

    d = _crash_rule_diag(task, runs)

    assert d.severity == "warning"
    assert d.data["is_current"] is False
    assert d.data["newest_outcome"] == "blocked"
    assert "outcome=blocked" in d.detail
    assert "not a current crash" in d.detail


def test_repeated_crashes_finished_streak_behind_a_live_run_is_historical():
    """A newest run with no outcome yet (in flight while the card is not
    `running`) is not a crash: the streak reads as historical, not current."""
    task = _task(status="blocked")
    runs = [_run_at(None, 6, 1789346800, None, error=None)]
    runs += _five_old_crashes()

    d = _crash_rule_diag(task, runs)

    assert d.severity == "warning"
    assert d.data["is_current"] is False
    assert "is still in flight (no outcome yet)" in d.detail


def test_repeated_crashes_current_streak_pages_exactly_as_before():
    """The other direction: when the newest run IS a crash the finding pages
    exactly as before — same title, same detail, same critical severity. Those
    three assertions pass pre-fix on purpose (they are the no-regression
    contract); the basis keys asserted below them are the new ones.
    """
    task = _task(status="ready")
    runs = _five_old_crashes()

    d = _crash_rule_diag(task, runs)

    assert d.severity == "critical"
    assert d.title == f"Agent crashed 5x: {_PROTOCOL_ERR}"
    assert d.detail == (
        f"The last 5 runs ended with outcome=crashed. Full last error:\n\n{_PROTOCOL_ERR}"
    )
    assert d.data["is_current"] is True
    assert d.data["newest_outcome"] == "crashed"


def test_repeated_crashes_pid_not_alive_mode_still_pages():
    """The `pid <n> not alive` crash mode is a live incident and keeps paging
    critical exactly as before — the fix must not mute it."""
    task = _task(status="ready")
    runs = [
        _run_at("crashed", 4, 1789233508, _LAST_CRASH_END, error="pid 364463 not alive"),
        _run_at("crashed", 3, 1789233327, 1789233508, error=_PROTOCOL_ERR),
        _run_at("crashed", 2, 1789233145, 1789233327, error=_PROTOCOL_ERR),
        _run_at("crashed", 1, 1789233000, 1789233145, error=_PROTOCOL_ERR),
    ]

    d = _crash_rule_diag(task, runs)

    assert d.severity == "critical"
    assert d.title == "Agent crashed 4x: pid 364463 not alive"
    assert d.data["is_current"] is True


def test_repeated_crashes_threshold_and_vocabulary_are_untouched():
    """The fix distinguishes; it does not suppress. One crash below the default
    threshold still fires nothing, and `unreported` is reported by name rather
    than dropped from the vocabulary."""
    task = _task(status="ready")
    below = [d for d in kd.compute_task_diagnostics(
        task, [], [_run_at("crashed", 1, 1789233145, _LAST_CRASH_END, error=_PROTOCOL_ERR)],
    ) if d.kind == "repeated_crashes"]
    assert below == []

    runs = [_run_at("unreported", 6, 1789346236, _NEWEST_END, error=_PROTOCOL_ERR)]
    runs += _five_old_crashes()
    d = _crash_rule_diag(task, runs)
    assert "unreported" in d.detail
    assert d.count == 5
