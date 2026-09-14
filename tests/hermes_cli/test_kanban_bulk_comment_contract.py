"""Contract tests for kanban's ``_commented()`` helper (shared by ``block``,
``schedule``, and ``unblock``): the ``PREFIX: reason`` comment is recorded
only after the wrapped per-task operation succeeds, and a bulk call keeps
processing the remaining ids after one id fails.

This mirrors the ordering already used by ``reopen-review`` (comment only
after a successful transition) and pins it for ``block``/``schedule``/
``unblock``. It also pins a related bulk-loop contract: when a bulk verb is
given an unknown task id alongside a valid one, the unknown id is reported
and skipped (no comment, no exception) while every other id in the same
call is still processed — an unknown id never aborts the whole batch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_failed_unblock_leaves_no_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="not blocked", assignee="builder")

    out = kc.run_slash(f'unblock {tid} --reason "should not persist"')
    assert "cannot unblock" in out

    with kbc.connect() as conn:
        assert kb.list_comments(conn, tid) == []
        assert kb.get_task(conn, tid).status == "ready"


def test_successful_unblock_still_writes_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="will be blocked", assignee="builder")
        assert kb.block_task(conn, tid, reason="setup")

    out = kc.run_slash(f'unblock {tid} --reason "resuming now"')
    assert "Unblocked" in out

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert comments[0].body == "UNBLOCK: resuming now"
        assert kb.get_task(conn, tid).status == "ready"


def test_failed_block_leaves_no_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already done", assignee="builder")
        assert kb.complete_task(conn, tid, result="finished")

    out = kc.run_slash(f'block {tid} "should not persist"')
    assert "cannot block" in out

    with kbc.connect() as conn:
        assert kb.list_comments(conn, tid) == []
        assert kb.get_task(conn, tid).status == "done"


def test_successful_block_still_writes_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="blockable", assignee="builder")

    out = kc.run_slash(f'block {tid} "waiting on a decision"')
    assert "Blocked" in out

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert comments[0].body == "BLOCKED: waiting on a decision"
        assert kb.get_task(conn, tid).status == "blocked"


def test_failed_schedule_leaves_no_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already done", assignee="builder")
        assert kb.complete_task(conn, tid, result="finished")

    out = kc.run_slash(f'schedule {tid} "should not persist"')
    assert "cannot schedule" in out

    with kbc.connect() as conn:
        assert kb.list_comments(conn, tid) == []
        assert kb.get_task(conn, tid).status == "done"


def test_successful_schedule_still_writes_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="schedulable", assignee="builder")

    out = kc.run_slash(f'schedule {tid} "waiting on a cron window"')
    assert "Scheduled" in out

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert comments[0].body == "SCHEDULED: waiting on a cron window"
        assert kb.get_task(conn, tid).status == "scheduled"


def test_bulk_unblock_skips_unknown_id_but_still_unblocks_the_rest(kanban_home: Path) -> None:
    """An unknown id in a bulk ``unblock`` call must not abort the batch: the
    valid blocked id after it still gets unblocked (and commented), the
    unknown id gets no comment, and the unknown id is reported with the
    same "cannot unblock" line used for any other unblock failure — not a
    raised/propagated "unknown task" error."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="will be blocked", assignee="builder")
        assert kb.block_task(conn, tid, reason="setup")
    bad_id = "t_does_not_exist"

    out = kc.run_slash(f'unblock {bad_id} {tid} --reason "resuming now"')

    assert f"cannot unblock {bad_id} (not blocked/scheduled?)" in out
    assert f"Unblocked {tid}" in out

    with kbc.connect() as conn:
        assert kb.list_comments(conn, bad_id) == []
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert comments[0].body == "UNBLOCK: resuming now"
        assert kb.get_task(conn, tid).status == "ready"
