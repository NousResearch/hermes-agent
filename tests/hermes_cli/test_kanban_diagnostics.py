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
from hermes_cli import kanban_db_workspace as kbw
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
# worktree_without_checkout_root
#
# Mirrors the raise in kanban_db_workspace._resolve_worktree_workspace: a
# worktree task resolves against its own workspace_path or the board's
# default_workdir, and with neither the dispatcher refuses to guess.
# ---------------------------------------------------------------------------


# Explicit "this board has no fallback root", so the rule never reads real
# board metadata off disk during the unit tests.
_NO_BOARD_WORKDIR = {"board_default_workdir": ""}


def _worktree_task(**overrides):
    base = dict(workspace_kind="worktree", workspace_path=None, project_id=None,
                created_at=1_000, created_by="orchestrator")
    base.update(overrides)
    return _task(**base)


def _wt_diags(task, config=None):
    diags = kd.compute_task_diagnostics(
        task, [], [], now=2_000,
        config=_NO_BOARD_WORKDIR if config is None else config)
    return [d for d in diags if d.kind == "worktree_without_checkout_root"]


def test_worktree_without_checkout_root_fires_on_a_board_with_no_default():
    diags = _wt_diags(_worktree_task())
    assert len(diags) == 1
    assert diags[0].severity == "error"
    assert diags[0].data["project_id"] is None
    assert "--project" in diags[0].actions[0].payload["command"]


def test_worktree_without_checkout_root_escalates_once_it_is_failing():
    """0 failures = mis-created; >0 = actively burning dispatch attempts."""
    diags = _wt_diags(_worktree_task(consecutive_failures=3))
    assert len(diags) == 1
    assert diags[0].severity == "critical"
    assert diags[0].data["consecutive_failures"] == 3


def test_worktree_without_checkout_root_silent_when_board_has_a_default():
    """A single-repo board sets default_workdir, so the task resolves."""
    assert _wt_diags(_worktree_task(),
                     config={"board_default_workdir": "/srv/repo"}) == []


def test_worktree_without_checkout_root_silent_when_task_has_its_own_path():
    assert _wt_diags(_worktree_task(workspace_path="/srv/repo")) == []


@pytest.mark.parametrize("kind", ["scratch", "dir"])
def test_worktree_without_checkout_root_ignores_other_workspace_kinds(kind):
    """scratch resolves under the board root; dir has its own guard."""
    assert _wt_diags(_worktree_task(workspace_kind=kind)) == []


@pytest.mark.parametrize("status", ["done", "archived"])
def test_worktree_without_checkout_root_ignores_terminal_tasks(status):
    """Never dispatched again, so a missing root is history, not a fault."""
    assert _wt_diags(_worktree_task(status=status)) == []


def test_worktree_without_checkout_root_fires_when_a_project_gave_no_path():
    """project_id is not what the resolver reads.

    create_task derives <repo>/.worktrees/<id> from a project, but only at
    creation and only when the project has a primary repo. A row that got a
    project without a path is still undispatchable, so the rule follows the
    resolver rather than trusting the link.
    """
    diags = _wt_diags(_worktree_task(project_id="p_abc123"))
    assert len(diags) == 1
    assert diags[0].data["project_id"] == "p_abc123"
    # Different cause, different remedy: the project exists, its repo doesn't.
    assert diags[0].actions[0].payload["command"] == "hermes project list"


def test_read_board_default_workdir_normalizes_unset():
    class _Stub:
        def __init__(self, meta):
            self.meta = meta

        def read_board_metadata(self, board=None):
            return self.meta

    assert kd.read_board_default_workdir(_Stub({"default_workdir": None})) == ""
    assert kd.read_board_default_workdir(_Stub({})) == ""
    assert kd.read_board_default_workdir(_Stub({"default_workdir": "/srv/r"})) == "/srv/r"


def test_worktree_without_checkout_root_matches_the_resolver(kanban_home):
    """Pin the rule to the code it predicts.

    The rule is only useful if it fires exactly when resolve_workspace
    refuses to dispatch, so assert both against one real task instead of
    trusting a hand-built fixture.
    """
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn, title="fix something", assignee="demo", workspace_kind="worktree")
        task = kb.get_task(conn, task_id)
        with pytest.raises(ValueError, match="default_workdir"):
            kbw.resolve_workspace(task)
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert _wt_diags(row)
    finally:
        conn.close()
