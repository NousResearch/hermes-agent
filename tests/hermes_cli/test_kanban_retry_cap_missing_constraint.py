"""Rule 4 (no-infinite-retry) invariant tests: dispatch-time escalation to a
decision-hud ``missing_constraint`` card when a task's ``consecutive_failures``
reaches N=3, and a fail-closed dispatch-side gate on that specific task_id
until the PO resolves it.

Written against the REAL code paths: real ``dispatch_once()``, real
``_record_task_failure()``, real board.json on real disk, real decision-hud
``db.py`` loaded via the real ``hermes_cli.plugin_bridges.decision_hud``
bridge — matching the pattern in
``test_dispatch_gate_and_board_metadata_integration.py`` (F1's own test
file). No mocking of the mechanism under test itself.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

_REAL_DECISION_HUD_DB = Path.home() / ".hermes" / "plugins" / "decision-hud" / "db.py"


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Same shape as F1's ``fresh_home`` fixture (see
    test_dispatch_gate_and_board_metadata_integration.py): isolated
    HERMES_HOME, Path.home() patched, and a real staged copy of
    decision-hud's db.py so the bridge loads the SAME code the real feature
    calls in production, just rooted under a temp HERMES_HOME.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    if _REAL_DECISION_HUD_DB.is_file():
        staged_dir = home / "plugins" / "decision-hud"
        staged_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REAL_DECISION_HUD_DB, staged_dir / "db.py")
    sys.modules.pop("_hermes_decision_hud_db_bridge", None)
    return home


def _require_decision_hud_installed():
    if not _REAL_DECISION_HUD_DB.is_file():
        pytest.skip(
            f"decision-hud plugin not installed on this host "
            f"({_REAL_DECISION_HUD_DB} not found) — Rule 4 tests need a real "
            f"copy of its db.py to stage into the isolated HERMES_HOME"
        )


def _no_spawn(*args, **kwargs):
    raise AssertionError(
        f"spawn_fn was called despite a gate that should have blocked this "
        f"tick (args={args!r} kwargs={kwargs!r})"
    )


def _stub_spawn(task, workspace, board=None):
    return 424242


def _missing_constraint_rows(project: str, task_id: str):
    from hermes_cli.plugin_bridges import decision_hud as bridge

    db = bridge._load_decision_hud_db()
    conn = db.connect()
    try:
        proj = db._resolve_project(project)
        rows = conn.execute(
            "SELECT * FROM decisions WHERE project_id = ? AND card_type = 'missing_constraint' "
            "AND json_extract(card_payload_json, '$.task_id') = ?",
            (proj["id"], task_id),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _resolve_constraint(project: str, task_id: str, choice: str = "resolved"):
    from hermes_cli.plugin_bridges import decision_hud as bridge

    db = bridge._load_decision_hud_db()
    conn = db.connect()
    try:
        row = db.get_missing_constraint(conn, project_id=project, task_id=task_id)
        assert row is not None, "test setup: missing_constraint row must exist before resolving"
        token = db.issue_actor_token("test-po")
        return db.resolve_decision(conn, row["id"], choice, actor_token=token)
    finally:
        conn.close()


def _fail_n_times(conn, task_id: str, n: int, *, failure_limit: int = 10):
    """Drive ``_record_task_failure`` ``n`` times without tripping the
    UNRELATED existing breaker (``failure_limit`` kept comfortably above the
    Rule 4 threshold so the task stays 'ready' throughout)."""
    for i in range(n):
        tripped = kbd._record_task_failure(
            conn, task_id, f"attempt {i} failed",
            outcome="crashed", failure_limit=failure_limit,
            release_claim=False, end_run=False,
        )
        assert not tripped, (
            f"the existing breaker tripped on failure {i + 1}/{n} — raise "
            f"failure_limit/max_retries in the test so Rule 4's own "
            f"threshold (3) is what's being exercised"
        )


class TestRule4MissingConstraintEscalation:
    def _setup(self, project="proj-r4"):
        kb.create_board(project)
        # decision-hud's push/require_constraint_resolved validate project_id
        # against the real hermes_cli.projects_db store (its `_resolve_project`
        # matches by id OR slug) — a board slug with no matching Project row
        # would fail closed on every push, never exercising the gate under
        # test. Create a Project whose slug equals the board slug so the
        # bridge's `project=board_slug` call resolves.
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            if pdb.get_project(pconn, project) is None:
                pdb.create_project(pconn, name=project, slug=project)
        with kbc.connect(board=project) as conn:
            task_id = kb.create_task(
                conn, title="flaky task", assignee="default", max_retries=10, board=project,
            )
        return project, task_id

    def test_third_consecutive_failure_pushes_exactly_one_missing_constraint(self, fresh_home):
        _require_decision_hud_installed()
        project, task_id = self._setup()
        with kbc.connect(board=project) as conn:
            _fail_n_times(conn, task_id, 3)
            # Three separate dispatch ticks while still unresolved must not
            # duplicate the escalation card.
            for _ in range(3):
                kbd.dispatch_once(conn, board=project, spawn_fn=_no_spawn)

        rows = _missing_constraint_rows(project, task_id)
        assert len(rows) == 1, (
            f"expected exactly one missing_constraint row for task {task_id!r}, "
            f"got {len(rows)}: {rows!r}"
        )

    def test_dispatch_refuses_to_spawn_while_constraint_unresolved(self, fresh_home):
        _require_decision_hud_installed()
        project, task_id = self._setup()
        with kbc.connect(board=project) as conn:
            _fail_n_times(conn, task_id, 3)
            result = kbd.dispatch_once(conn, board=project, spawn_fn=_no_spawn)
            after = kb.get_task(conn, task_id)

        assert result.spawned == []
        assert task_id not in [t for t, _, _ in result.spawned]
        assert any(t == task_id for t, _ in result.retry_cap_blocked), (
            f"expected task {task_id!r} in retry_cap_blocked, got {result.retry_cap_blocked!r}"
        )
        # The task must not be silently blocked/consumed by this mechanism —
        # it stays exactly where the existing breaker left it (ready), just
        # un-dispatchable, mirroring F1's "refuse to claim/spawn" contract.
        assert after.status == "ready"

    def test_resolving_constraint_reenables_dispatch(self, fresh_home):
        _require_decision_hud_installed()
        project, task_id = self._setup()
        with kbc.connect(board=project) as conn:
            _fail_n_times(conn, task_id, 3)
            kbd.dispatch_once(conn, board=project, spawn_fn=_no_spawn)  # pushes the card

        _resolve_constraint(project, task_id)

        with kbc.connect(board=project) as conn:
            result = kbd.dispatch_once(conn, board=project, spawn_fn=_stub_spawn)
            after = kb.get_task(conn, task_id)

        spawned_ids = [t for t, _, _ in result.spawned]
        assert task_id in spawned_ids, (
            f"expected task {task_id!r} to be dispatchable again after PO "
            f"resolution, spawned={result.spawned!r} "
            f"retry_cap_blocked={result.retry_cap_blocked!r}"
        )
        assert after.status == "running"

    def test_one_or_two_failures_not_gated_by_rule4(self, fresh_home):
        _require_decision_hud_installed()
        project, task_id = self._setup()
        with kbc.connect(board=project) as conn:
            _fail_n_times(conn, task_id, 2)
            result = kbd.dispatch_once(conn, board=project, spawn_fn=_stub_spawn)
            after = kb.get_task(conn, task_id)

        spawned_ids = [t for t, _, _ in result.spawned]
        assert task_id in spawned_ids
        assert result.retry_cap_blocked == []
        assert after.status == "running"
        assert _missing_constraint_rows(project, task_id) == []

    def test_fails_closed_when_decision_hud_unreachable(self, fresh_home):
        # Do NOT stage decision-hud's db.py: the bridge must be unable to
        # import it, exactly the "plugin not installed" fail-closed case.
        project, task_id = self._setup()
        db_path = fresh_home / "plugins" / "decision-hud" / "db.py"
        if db_path.is_file():
            db_path.unlink()
        with kbc.connect(board=project) as conn:
            _fail_n_times(conn, task_id, 3)
            result = kbd.dispatch_once(conn, board=project, spawn_fn=_no_spawn)
            after = kb.get_task(conn, task_id)

        assert result.spawned == []
        assert any(t == task_id for t, _ in result.retry_cap_blocked)
        assert after.status == "ready"

    def test_zero_regression_existing_breaker_still_trips_below_rule4_threshold(self, fresh_home):
        """The pre-existing consecutive_failures/_record_task_failure breaker
        must behave exactly as before for a task using the default
        (unelevated) failure_limit — this new mechanism must not interfere
        with it."""
        _require_decision_hud_installed()
        kb.create_board("proj-regress")
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            if pdb.get_project(pconn, "proj-regress") is None:
                pdb.create_project(pconn, name="proj-regress", slug="proj-regress")
        with kbc.connect(board="proj-regress") as conn:
            task_id = kb.create_task(conn, title="normal task", assignee="default", board="proj-regress")
            # DEFAULT_FAILURE_LIMIT is 2: the second failure must trip the
            # breaker into 'blocked', same as pre-Rule-4 behaviour.
            tripped = kbd._record_task_failure(
                conn, task_id, "boom", outcome="crashed",
                release_claim=False, end_run=False,
            )
            assert not tripped
            task = kb.get_task(conn, task_id)
            assert task.status == "ready"
            assert task.consecutive_failures == 1

            tripped = kbd._record_task_failure(
                conn, task_id, "boom again", outcome="crashed",
                release_claim=False, end_run=False,
            )
            assert tripped
            task = kb.get_task(conn, task_id)
            assert task.status == "blocked"
            assert task.consecutive_failures == 2
        assert _missing_constraint_rows("proj-regress", task_id) == []
