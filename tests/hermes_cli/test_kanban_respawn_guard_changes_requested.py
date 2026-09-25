"""``changes_requested`` must release the ``active_pr`` respawn guard.

The canonical review cycle is:

    worker opens a PR -> request_review -> reviewer request_changes
    -> task returns to ``ready`` for the SAME implementer to push more
       commits to the SAME PR.

That card ALWAYS carries a recent comment holding an OPEN PR URL, so the
``active_pr`` rule used to fire on every dispatcher tick forever and the
rework never spawned.  These tests pin the bypass AND its negative
controls: the guard must still fire for every other shape of open PR.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


PR_COMMENT = "Opened https://github.com/example/repo/pull/123 for review."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Deterministic: never shell out to `gh` from a test. The PR is OPEN.
    monkeypatch.setattr(kb, "_active_pr_guard_applies", lambda _url: (True, None))
    kb.init_db()
    return home


def _review_handoff(conn, *, title: str = "ship it") -> str:
    """Create a task, open a PR on it, and hand it to review."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
    assert kb.request_review(
        conn, tid, summary="PR ready", reviewer="reviewer",
        expected_run_id=claimed.current_run_id,
    )
    return tid


def _reviewer_requests_changes(conn, tid: str) -> None:
    claimed = kb.claim_review_task(conn, tid)
    assert claimed is not None, "review claim failed"
    ok, detail = kb.request_changes(conn, tid, reason="please add tests")
    assert ok is True, detail


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_changes_requested_releases_active_pr_guard(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = _review_handoff(conn)
        _reviewer_requests_changes(conn, tid)

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready", "rework lands back in the ready lane"

        latest = kb.latest_run(conn, tid)
        assert latest is not None and latest.outcome == "changes_requested"

        # The open-PR comment is still there and still open...
        assert kb.check_respawn_guard(conn, tid) is None, (
            "changes_requested rework must not be held by active_pr"
        )


def test_latest_run_tiebreak_when_runs_end_in_the_same_second(
    kanban_home: Path,
) -> None:
    """Same-second ``ended_at`` must not let the older run shadow the newer.

    The review-handoff run and the reviewer's verdict run routinely end in
    the same unix second; without the ``id DESC`` tiebreak the guard reads a
    non-deterministic outcome.
    """
    with kb.connect() as conn:
        tid = _review_handoff(conn)
        _reviewer_requests_changes(conn, tid)

        # Force every run on this task to share one ended_at second.
        pinned = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET ended_at = ? "
                "WHERE task_id = ? AND ended_at IS NOT NULL",
                (pinned, tid),
            )
        rows = conn.execute(
            "SELECT id, outcome FROM task_runs WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
        assert len(rows) >= 2, "need at least the handoff + verdict runs"
        assert rows[-1]["outcome"] == "changes_requested"

        assert kb.check_respawn_guard(conn, tid) is None


# ---------------------------------------------------------------------------
# Negative controls — active_pr must still fire everywhere else
# ---------------------------------------------------------------------------


def test_open_pr_without_reviewer_verdict_still_guards(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_completed_latest_run_with_open_pr_still_guards(kanban_home: Path) -> None:
    """A ``completed`` latest run + open PR is NOT a rework request."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="done with a PR open", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
        assert kb.complete_task(conn, tid, result="ok")

        latest = kb.latest_run(conn, tid)
        assert latest is not None and latest.outcome == "completed"

        # Age the success past the 1h recent_success window so rule 3 does
        # NOT short-circuit: we want to land on rule 4 for real. The PR
        # comment stays inside the 24h PR window.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET ended_at = ? WHERE task_id = ?",
                (int(time.time()) - 7200, tid),
            )

        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_review_lane_behaviour_unchanged(kanban_home: Path) -> None:
    """The review lane still skips active_pr, and ready still guards."""
    with kb.connect() as conn:
        review_id = _review_handoff(conn, title="review me")
        ready_id = kb.create_task(conn, title="ready with PR", assignee="worker")
        kb.add_comment(conn, ready_id, author="worker", body=PR_COMMENT)

        assert kb.check_respawn_guard(conn, review_id, lane="review") is None
        assert kb.check_respawn_guard(conn, ready_id) == "active_pr"


def test_rate_limit_cooldown_still_wins_over_the_bypass(kanban_home: Path) -> None:
    """Rules 1/2/3 run BEFORE the rule-4 bypass, so a later rate-limited run
    still defers even on a card that once had changes requested."""
    with kb.connect() as conn:
        tid = _review_handoff(conn)
        _reviewer_requests_changes(conn, tid)
        now = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'worker', 'rate_limited', "
                "'rate_limited', ?, ?)",
                (tid, now, now + 5),
            )
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
