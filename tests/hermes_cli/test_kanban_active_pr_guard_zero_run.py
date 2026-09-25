"""``active_pr`` respawn guard: a card that never produced a run cannot have
opened the PR its comments cite.

Live shape (FORGE board card ``t_b0219ec5``, a sibling of #85663): a
dispatcher-created *review* card sits ``ready`` with its routing comments
citing the PR under review and no run of its own. The ready lane matched the
URL, answered ``active_pr`` on every tick, and the card was never spawned for
the whole 24h PR window. A PR URL on a zero-run card is an *input* (the
artifact to review or repair), never an output the card could duplicate, so
step 4 of :func:`check_respawn_guard` requires a run of the card's own before
a PR-URL comment may hold it. Steps 1-3, the review-lane early return and the
post-comment handoff exceptions are untouched: an implementer that has run
stays guarded exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_COMMENT = "Review https://github.com/example/repo/pull/145 and post a verdict."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _backdate_comments(conn, tid, seconds=60):
    """Second-granularity timestamps: make the PR comment older than the
    handoff that follows it in the same test."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, tid),
        )


def _backdate_runs(conn, tid, seconds):
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET started_at = started_at - ?, ended_at = ended_at - ? "
            "WHERE task_id = ?",
            (seconds, seconds, tid),
        )


def _run_count(conn, tid) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,),
    ).fetchone()[0]


def _implementer_card_with_pr(conn, *, assignee="dev"):
    """A card that ran to completion, then cited its PR. The run is older than
    the success window so ``recent_success`` cannot mask step 4; the card is
    dragged back to ``ready`` the way a done->ready re-run would leave it."""
    tid = kb.create_task(conn, title="implement it", assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    assert kb.complete_task(conn, tid, summary="opened the PR") is True
    _backdate_runs(conn, tid, 2 * kbd._RESPAWN_GUARD_SUCCESS_WINDOW)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    kb.add_comment(conn, tid, author=assignee, body=PR_COMMENT)
    _backdate_comments(conn, tid)
    assert _run_count(conn, tid) == 1
    return tid


def test_zero_run_review_card_citing_a_pr_is_not_guarded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) The live ``t_b0219ec5`` shape: created ready, assigned to a reviewer,
    one routing comment citing the PR under review, never claimed. The guard
    must not answer ``active_pr`` and the ready lane must offer the spawn."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Review PR 145", assignee="reviewer-codex")
        kb.add_comment(conn, tid, author="dispatcher", body=PR_COMMENT)
        assert _run_count(conn, tid) == 0
        assert kb.get_task(conn, tid).current_run_id is None

        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]
        assert tid not in dict(res.respawn_guarded)


def test_card_with_completed_run_and_later_pr_url_stays_guarded(
    kanban_home: Path,
) -> None:
    """(b) A card that has run and then cited a PR, with no handoff after the
    comment, is the duplicate-PR risk the guard exists for."""
    with kbc.connect() as conn:
        tid = _implementer_card_with_pr(conn)
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_changes_requested_after_the_pr_comment_still_lifts_the_guard(
    kanban_home: Path,
) -> None:
    """(c) The existing post-comment handoff exception is untouched by the
    has-run precondition: the same card as (b) plus a ``changes_requested``
    event strictly after the comment spawns."""
    with kbc.connect() as conn:
        tid = _implementer_card_with_pr(conn)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "changes_requested", {"reason": "fix tests"})
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_card_with_started_run_and_later_pr_url_stays_guarded(
    kanban_home: Path,
) -> None:
    """(d) A run that started is enough: the card can produce output, so a PR
    URL posted after the spawn is guarded even with no completed run."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="implement it", assignee="dev")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None and claimed.current_run_id is not None
        with kb.write_txn(conn):
            kb._append_event(
                conn, tid, "spawned", {"pid": 4242}, run_id=claimed.current_run_id,
            )
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND outcome = 'completed'",
            (tid,),
        ).fetchone()[0] == 0
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"
