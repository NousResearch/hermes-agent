"""Per-row fault isolation in the ready/review dispatch pass.

One poison row must skip ONE card, not abort the board's whole tick. Before
this, a single ready row whose column values a reader could not parse raised
out of ``_dispatch_lane_task`` and ended the pass, so every other ready/review
card on the board was skipped with it (live: 1203 ``tick failed on board ops``
lines, and 43-46 ready cards on the ops board never attempted while it failed).

The exception surface stays curated (``kbd._ROW_ISOLATION_ERRORS``) rather than
a bare ``except Exception``: an exception raised OUTSIDE the per-row unit —
lane enumeration, reclaim, promotion, budget — still fails the tick loudly.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# ``profile_exists("default")`` is True without any filesystem fixture, so the
# cards route through the REAL profile lookup — nothing on this path is mocked.
ASSIGNEE = "default"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


class _Spawner:
    """Stand-in worker spawn; records every card it was asked to start."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, task, workspace_path, board=None):
        self.calls.append(getattr(task, "id", task))
        return 4242


def _card(conn, title, *, priority=0):
    """A card sitting in the ready lane, ready for this tick to claim.

    ``create_task`` only opens cards as ``running``/``blocked``, so the status
    is set directly: routing through the reclaim pass would add its own
    bookkeeping (and its own failure stamps) that these assertions are not
    about.
    """
    task_id = kb.create_task(conn, title=title, assignee=ASSIGNEE, priority=priority)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    conn.commit()
    return task_id


def _poison_run_column(conn, task_id):
    """Leave a ``task_runs`` row whose INTEGER column holds TEXT.

    SQLite columns are dynamically typed, so an out-of-tree writer can leave
    ``ended_at = 'soon'``. The respawn guard's ``int(ended_at)`` then raises
    ``ValueError`` from inside the per-row unit — the same class as the
    BLOB-comment-body row that first aborted the ops board.
    """
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, 'failed', ?, ?, 'rate_limited')",
        (task_id, int(time.time()) - 60, "soon"),
    )
    conn.commit()


@pytest.fixture(autouse=True)
def _rate_limit_cooldown(monkeypatch):
    """Pin the cooldown so the guard's rate-limit branch is always taken."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")


def _events(conn, task_id):
    return [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        )
    ]


def test_poisoned_row_skips_only_that_card(conn):
    """The poison row errors; every other ready card spawns in the SAME pass."""
    poison = _card(conn, "poison", priority=9)  # priority: poison is seen FIRST
    sibling = _card(conn, "sibling")
    _poison_run_column(conn, poison)
    failures_before = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (poison,)
    ).fetchone()["consecutive_failures"]

    spawner = _Spawner()
    result = kbd.dispatch_once(conn, spawn_fn=spawner)

    assert spawner.calls == [sibling], "a poisoned row must not stop the pass"
    assert [tid for tid, _who, _ws in result.spawned] == [sibling]
    assert [(tid, err.split("(", 1)[0]) for tid, err in result.row_errors] == [
        (poison, "ValueError")
    ], result.row_errors

    row = conn.execute(
        "SELECT status, consecutive_failures, last_failure_error FROM tasks WHERE id = ?",
        (poison,),
    ).fetchone()
    assert row["last_failure_error"].startswith("dispatch_error:"), row["last_failure_error"]
    assert "ValueError" in row["last_failure_error"]
    assert row["consecutive_failures"] == failures_before + 1, "row error must count"
    assert result.auto_blocked == [], "below the limit the card is retried, not blocked"
    assert "tick_row_error" in _events(conn, poison), "the row error must be visible"


def test_poisoned_row_auto_blocks_at_the_failure_limit(conn):
    """Retry accounting: a permanently toxic row parks instead of looping."""
    poison = _card(conn, "poison")
    sibling = _card(conn, "sibling")
    _poison_run_column(conn, poison)

    spawner = _Spawner()
    result = kbd.dispatch_once(conn, spawn_fn=spawner, failure_limit=1)

    assert spawner.calls == [sibling]
    assert result.auto_blocked == [poison]
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (poison,)
    ).fetchone()["status"] == "blocked"


def test_row_error_stamp_is_not_read_as_a_quota_wall(conn):
    """A ``dispatch_error:`` stamp must not be mistaken for an auth/quota block.

    Otherwise a row error whose text happens to contain "403"/"auth" gets
    parked by ``blocker_auth`` on the next tick and never counts toward
    auto-block — a silent skip, the exact failure this card removes.
    """
    stamped = _card(conn, "row-error-stamped", priority=9)
    genuine = _card(conn, "genuine-403")
    conn.execute(
        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
        ("dispatch_error: ValueError: bad value near '403' auth field", stamped),
    )
    conn.execute(
        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
        ("HTTP 403 forbidden: quota exhausted", genuine),
    )
    conn.commit()

    result = kbd.dispatch_once(conn, spawn_fn=_Spawner())

    assert (genuine, "blocker_auth") in result.respawn_guarded, "guard must still work"
    assert not [
        entry for entry in result.respawn_guarded if entry[0] == stamped
    ], "a row error is not a quota wall"


def test_blob_comment_body_is_still_not_a_poison_row(conn):
    """Composition with PR #34: the incident's own vector must never abort.

    A BLOB comment body used to crash the guard's regex and end the pass. On
    this base PR #34 routed that reader through ``_db_text``, so the card
    dispatches clean. If that reader ever regresses — or on a line that predates
    #34 — the row is contained as a ``row_errors`` entry instead. Either way one
    card cannot take the board down, which is the invariant this test pins.
    """
    card = _card(conn, "blob-comment")
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
        (card, "out-of-tree-writer", sqlite3.Binary(b"\xff\xfe not utf8 \x80"), int(time.time())),
    )
    conn.commit()

    spawner = _Spawner()
    result = kbd.dispatch_once(conn, spawn_fn=spawner)

    if result.row_errors:
        assert result.row_errors[0][0] == card, result.row_errors
        assert spawner.calls == [], "a contained row error must not also spawn"
    else:
        assert spawner.calls == [card], "the defused reader must still dispatch"


def test_board_level_exception_still_aborts_the_tick(conn, monkeypatch):
    """Outside the per-row unit, a failure is still loud — not swallowed."""
    _card(conn, "any")

    def _boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kbd, "_lane_rows", _boom)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        kbd.dispatch_once(conn, spawn_fn=_Spawner())


def test_cli_dispatch_exits_zero_with_a_poison_row(conn, monkeypatch, capsys):
    """``hermes kanban dispatch --once`` reports the row error and exits 0."""
    from hermes_cli import kanban_ops as kb_ops

    poison = _card(conn, "poison")
    _poison_run_column(conn, poison)
    monkeypatch.setattr(kbd, "_default_spawn", _Spawner())

    rc = kb_ops._cmd_dispatch(
        argparse.Namespace(dry_run=False, max=None, failure_limit=2, json=False)
    )

    assert rc == 0, "a poisoned row must not turn the CLI tick into a failure"
    out = capsys.readouterr().out
    assert "Row errors:" in out, out
    assert poison in out, out
