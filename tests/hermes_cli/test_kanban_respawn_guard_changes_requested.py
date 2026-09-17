"""Respawn guard: a reviewer verdict releases the ``active_pr`` guard.

The canonical review cycle is worker -> PR -> ``request_review`` -> reviewer
``request_changes`` -> back to ``ready`` for the SAME implementer to push more
commits to the SAME PR. Such a card ALWAYS carries a PR-URL comment, so the
``active_pr`` rule would otherwise fire on every dispatch tick forever and the
rework would never spawn.

Contract pinned here:

* latest ended run outcome ``changes_requested`` -> ``active_pr`` must NOT fire.
* no reviewer verdict -> ``active_pr`` still fires (the guard's real case).
* the release survives the review handoff run and the verdict run sharing an
  ``ended_at`` second (the latest-run query needs an ``id`` tiebreak).

Every task here is driven through the REAL lifecycle API (create -> claim ->
request_review -> claim_review_task -> request_changes), not hand-inserted rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_COMMENT = "Opened https://github.com/example/repo/pull/123 for review."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task_returned_by_reviewer(conn, title: str) -> str:
    """Real lifecycle: implementer PRs -> requests review -> reviewer returns it."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    impl = kb.claim_task(conn, tid)
    assert impl is not None
    kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
    assert kb.request_review(
        conn, tid, summary="PR ready", reviewer="reviewer",
        expected_run_id=impl.current_run_id,
    )
    review_run = kb.claim_review_task(conn, tid)
    assert review_run is not None
    ok, implementer = kb.request_changes(
        conn, tid, reason="please add a test",
        expected_run_id=review_run.current_run_id,
    )
    assert ok, implementer
    assert kb.get_task(conn, tid).status == "ready"
    return tid


def test_changes_requested_releases_active_pr_guard(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        reworked = _task_returned_by_reviewer(conn, "rework after review")
        # Reviewer verdict is an explicit re-queue onto the SAME PR.
        assert kbd.check_respawn_guard(conn, reworked, lane="ready") is None

        # Sibling with the same PR comment but no review verdict: the guard's
        # real case (a prior worker's PR, no re-queue signal) still fires.
        no_verdict = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, no_verdict, author="worker", body=PR_COMMENT)
        assert kbd.check_respawn_guard(conn, no_verdict, lane="ready") == "active_pr"


def test_changes_requested_releases_guard_when_runs_share_ended_at(
    kanban_home: Path,
) -> None:
    """Same-second handoff: ``review_requested`` and ``changes_requested`` end
    within one unix second, so ordering by ``ended_at`` alone can surface the
    older run and shadow the verdict."""
    with kbc.connect() as conn:
        tid = _task_returned_by_reviewer(conn, "same-second review handoff")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET ended_at = ("
                "  SELECT MAX(ended_at) FROM task_runs"
                "   WHERE task_id = ? AND ended_at IS NOT NULL"
                ") WHERE task_id = ? AND ended_at IS NOT NULL",
                (tid, tid),
            )
        ends = [
            r["ended_at"] for r in conn.execute(
                "SELECT ended_at FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL",
                (tid,),
            ).fetchall()
        ]
        assert len(ends) >= 2 and len(set(ends)) == 1

        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None
