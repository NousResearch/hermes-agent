"""An explicit ``board=`` must never be resolved through the caller's OWN pin.

``HERMES_KANBAN_DB`` is injected into every dispatched worker and identifies
*that worker's* board. ``hermes_cli.kanban_db_connect.connect()`` used to resolve
an explicit ``board=<sibling>`` through that pin, so ``kanban_create(board="B")``
returned ``ok`` while writing the card into board A — a silent mis-delivery (third
site of the ``_board_path()`` pin collapse behind 6fdaa43d0). The rule under test,
mirroring ``test_kanban_phantom_refs.py::test_pinned_db_env_resolves_cross_board_reference``:

* the pin is authoritative only for ITS OWN board (the caller's);
* a board-scoped caller (worker / delegate child) asking for another board gets a
  loud ``BoardPinConflict`` — never a row in its own DB;
* a board-agnostic caller (gateway / dashboard / interactive / top-level cron)
  gets the board it asked for.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    kb.init_db()
    return home


def _pin_worker_to(monkeypatch, board: str) -> Path:
    """Env of a dispatcher-spawned worker scoped to ``board`` (see the spawn path:
    ``HERMES_KANBAN_DB`` = canonical file, ``HERMES_KANBAN_BOARD`` = slug)."""
    pinned = kb.kanban_db_path(board=board)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_pinned0001")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return pinned


def _sql(db_path: Path, query: str, params: tuple = ()) -> list:
    """Read a board DB straight off disk, bypassing every env pin."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return conn.execute(query, params).fetchall()


def _titles(db_path: Path) -> list[str]:
    return [r[0] for r in _sql(db_path, "SELECT title FROM tasks ORDER BY id")]


def _opened_file(conn) -> Path:
    return Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()


def test_pinned_worker_create_on_sibling_board_is_refused_not_mis_delivered(
    kanban_home, monkeypatch,
):
    """The live repro: ``kanban_create(board="sibling")`` from a pinned worker.

    Acceptance criterion 1 — a clear refusal, and in NO case a row in the
    caller's own (pinned) DB for a ``board="sibling"`` request.
    """
    kb.create_board("sibling")
    pinned = _pin_worker_to(monkeypatch, "default")
    sibling = kb.board_db_path_unpinned("sibling")

    from tools.kanban_tools import _handle_create

    payload = json.loads(_handle_create({
        "board": "sibling", "title": "must not land anywhere",
        "assignee": "yummi", "body": "addressed to another board",
    }))

    assert "error" in payload and "ok" not in payload, payload
    # Loud, actionable, and names both boards — no silent success.
    assert "out of scope" in payload["error"]
    assert "'sibling'" in payload["error"] and str(pinned) in payload["error"]

    assert _titles(pinned) == []          # not on the caller's own board...
    assert _titles(sibling) == []         # ...and not on the requested one either


def test_board_pin_conflict_is_a_value_error(kanban_home, monkeypatch):
    """``ValueError`` so the tool/CLI surfaces render it without a traceback."""
    kb.create_board("sibling")
    _pin_worker_to(monkeypatch, "default")

    assert issubclass(kb.BoardPinConflict, ValueError)
    with pytest.raises(kb.BoardPinConflict) as excinfo:
        kbc.connect(board="sibling")
    assert "'sibling'" in str(excinfo.value)
    assert "HERMES_KANBAN_DB" in str(excinfo.value)


def test_pinned_worker_show_and_comment_on_sibling_board_refuse_loudly(
    kanban_home, monkeypatch,
):
    """Acceptance criterion 2 — ``unknown task`` was misleading; now it is scoped.

    Both tools resolve the pinned DB today, so a sibling id reads as missing
    ("task t_… not found"). The pin conflict must be reported instead.
    """
    kb.create_board("sibling")
    other = kbc.connect(board="sibling")
    remote = kb.create_task(other, title="lives elsewhere", assignee="sysadmin")
    other.close()

    _pin_worker_to(monkeypatch, "default")

    from tools.kanban_tools import _handle_comment, _handle_show

    shown = json.loads(_handle_show({"task_id": remote, "board": "sibling"}))
    assert "out of scope" in shown["error"]
    assert "not found" not in shown["error"]

    commented = json.loads(_handle_comment({
        "task_id": remote, "board": "sibling", "body": "meant for the sibling board",
    }))
    assert "out of scope" in commented["error"]
    assert "not found" not in commented["error"]

    assert _sql(kb.board_db_path_unpinned("sibling"),
                "SELECT COUNT(*) FROM task_comments")[0][0] == 0


def test_pinned_worker_own_board_still_resolves_through_the_pin(
    kanban_home, monkeypatch,
):
    """Regression guard: the pin stays authoritative for the caller's OWN board.

    ``test_pinned_db_env_resolves_cross_board_reference`` depends on this, as do
    the dispatcher workers themselves.
    """
    pinned = _pin_worker_to(monkeypatch, "default")

    conn = kbc.connect(board="default")
    try:
        assert _opened_file(conn) == pinned.resolve()
        assert kb.create_task(conn, title="own work", assignee="coder")
    finally:
        conn.close()

    conn = kbc.connect()  # no explicit board -> whole legacy chain, pin wins
    try:
        assert _opened_file(conn) == pinned.resolve()
    finally:
        conn.close()

    assert _titles(pinned) == ["own work"]


def test_pinned_delegate_child_cannot_reach_a_sibling_board(kanban_home, monkeypatch):
    """A descendant of a worker is scoped too (read-only already; now also loud)."""
    kb.create_board("sibling")
    _pin_worker_to(monkeypatch, "default")
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    with pytest.raises(kb.BoardPinConflict):
        kbc.connect(board="sibling")

    # Its own board still opens (read-only, as descendants get).
    conn = kbc.connect(board="default")
    conn.close()


def test_board_agnostic_caller_with_a_pin_reaches_the_named_board(
    kanban_home, monkeypatch,
):
    """Acceptance criterion 1(a) — a gateway/dashboard/interactive caller (no
    worker identity) gets the board its ``board=`` argument names, not the file
    some HERMES_KANBAN_DB in its env happens to point at."""
    kb.create_board("sibling")
    pinned = kb.kanban_db_path(board="default")
    sibling = kb.board_db_path_unpinned("sibling")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    assert kb._is_board_scoped_process() is False

    conn = kbc.connect(board="sibling")
    try:
        assert _opened_file(conn) == sibling.resolve()
        tid = kb.create_task(conn, title="routed to the named board", assignee="sysadmin")
    finally:
        conn.close()

    assert [r[0] for r in _sql(sibling, "SELECT id FROM tasks ORDER BY id")] == [tid]
    assert _titles(pinned) == []  # nothing collapsed onto the pinned file
