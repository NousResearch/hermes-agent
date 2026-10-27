"""The local unblock handoff preserves upstream's mixed-time PR guard."""
from datetime import datetime, timezone
from pathlib import Path
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    with kbc.connect(tmp_path / "kanban.db") as db:
        yield db


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


@pytest.mark.parametrize("encode", [int, _iso])
@pytest.mark.parametrize("offset, expected", [(-1, "active_pr"), (0, None), (1, None)])
def test_unblock_compares_parsed_times(conn, encode, offset, expected):
    now = int(time.time())
    tid = kb.create_task(conn, title="PR handoff", assignee="worker")
    kb.add_comment(conn, tid, author="worker", body="https://github.com/example/repo/pull/77")
    conn.execute("UPDATE task_comments SET created_at = ? WHERE task_id = ?", (encode(now - 10), tid))
    conn.execute("INSERT INTO task_events (task_id, kind, created_at) VALUES (?, 'unblocked', ?)",
                 (tid, encode(now - 10 + offset)))
    conn.commit()
    assert kbd.check_respawn_guard(conn, tid) == expected


@pytest.mark.parametrize("stamp", ["not-a-time", "2000-01-01T00:00:00+00:00"])
def test_unreadable_or_old_unblock_cannot_lift_guard(conn, stamp):
    tid = kb.create_task(conn, title="PR remains guarded", assignee="worker")
    kb.add_comment(conn, tid, author="worker", body="https://github.com/example/repo/pull/77")
    conn.execute("INSERT INTO task_events (task_id, kind, created_at) VALUES (?, 'unblocked', ?)", (tid, stamp))
    conn.commit()
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_unblock_between_pr_comments_does_not_lift_newest_guard(conn):
    now = int(time.time())
    tid = kb.create_task(conn, title="Newest PR owns guard", assignee="worker")
    for offset in (-30, -10):
        cid = kb.add_comment(conn, tid, author="worker", body="https://github.com/example/repo/pull/77")
        conn.execute("UPDATE task_comments SET created_at = ? WHERE id = ?", (_iso(now + offset), cid))
    conn.execute("INSERT INTO task_events (task_id, kind, created_at) VALUES (?, 'unblocked', ?)", (tid, _iso(now - 20)))
    conn.commit()
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


@pytest.mark.parametrize("kind", ["assigned", "changes_requested", "review_reopened"])
def test_same_second_non_unblock_handoff_stays_guarded(conn, kind):
    now = int(time.time())
    tid = kb.create_task(conn, title="Strict non-unblock handoff", assignee="worker")
    kb.add_comment(conn, tid, author="worker", body="https://github.com/example/repo/pull/77")
    conn.execute("UPDATE task_comments SET created_at = ? WHERE task_id = ?", (_iso(now), tid))
    conn.execute("INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                 (tid, kind, '{"from":"worker","assignee":"other"}', _iso(now)))
    conn.commit()
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
