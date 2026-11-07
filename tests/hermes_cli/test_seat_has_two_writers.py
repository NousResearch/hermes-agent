"""A seat is two writers, and nothing in either says so.

The board claims a card: a **task-scoped** ``claim_lock`` in the board database.
The seat's own chat admits a turn: a **session-scoped** admission in the
profile's ``state.db``. Neither lock domain consults the other, and neither row
carries a reference to the other, so a dispatched worker and an interactive
session can write one seat at the same instant — which is what happened on
``oto-url-api#738``: the same entry pressed twice, once by a card claimed at
17:35:16Z and once by ``oto-site``'s interactive session, on the same verdict.

This test demonstrates the collision, not the absence of a check: both writers
proceed, both are live, and the two rows cannot see each other. If a seat lock is
ever added, this test is the one that must change.
"""

from __future__ import annotations

from contextlib import closing

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch

SEAT = "oto-site"


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def _card(conn, title):
    """``create_task`` returns the id; only 'blocked' or 'running' are mintable, so
    the ready state that a claim requires is set here, as the dispatcher's own
    promotion would."""
    task_id = kb.create_task(conn, title=title, assignee=SEAT, body="")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    return task_id


def _seat_admits_a_turn(tmp_path):
    """What `hermes -p oto-site chat` does: admit input into the profile's own store."""
    db = SessionDB(tmp_path / "oto-site-state.db")
    db.create_session("interactive", "cli")
    epoch = begin_runtime_epoch(db, instance_id="interactive-cli")
    admission = admit_session_input(
        db,
        epoch=epoch,
        principal_id="human",
        session_id="interactive",
        request_id="human:press-the-merge",
        payload={"input": "press the merge on #738"},
    )
    return db, admission


def test_a_dispatched_worker_and_an_interactive_session_write_one_seat(board, tmp_path):
    card_id = _card(board, "press the merge on oto-url-api#738")
    claimed = kb.claim_task(board, card_id, claimer="oto-server:1")
    assert claimed is not None and claimed.claim_lock == "oto-server:1"

    # The seat's other writer, live at the same instant.
    seat_db, admission = _seat_admits_a_turn(tmp_path)
    with closing(seat_db):
        assert admission["status"] in ("queued", "started")

        # Neither row mentions the other: there is no cross-reference to violate.
        task_row = board.execute("SELECT * FROM tasks WHERE id=?", (card_id,)).fetchone()
        task_values = " ".join(str(v) for v in task_row if v is not None)
        assert "interactive" not in task_values

        admission_row = seat_db._read_one(
            "SELECT * FROM session_admissions WHERE admission_id=?", (admission["admission_id"],)
        )
        assert admission_row is not None
        admission_values = " ".join(str(v) for v in dict(admission_row).values() if v is not None)
        assert "oto-server:1" not in admission_values

        # Both writers are still live: nothing refused, nothing waited.
        still_claimed = kb.get_task(board, card_id)
        assert still_claimed is not None and still_claimed.status == "running"
        still_admitted = seat_db._read_one(
            "SELECT status FROM session_admissions WHERE admission_id=?", (admission["admission_id"],)
        )
        assert still_admitted is not None and still_admitted["status"] == admission["status"]


def test_two_cards_for_one_seat_are_claimed_simultaneously(board):
    """The board's lock is per TASK: one seat can hold two dispatched workers at once."""
    first_id = _card(board, "the first thing this seat is told to do")
    second_id = _card(board, "the second thing, dispatched in the same tick")
    assert kb.claim_task(board, first_id, claimer="oto-server:1") is not None
    assert kb.claim_task(board, second_id, claimer="oto-server:1") is not None
    running = {
        row["id"]
        for row in board.execute("SELECT id FROM tasks WHERE assignee=? AND status='running'", (SEAT,))
    }
    assert {first_id, second_id} <= running


def test_the_board_still_locks_one_card_against_a_second_claim(board):
    """The lock that DOES exist: the same card cannot be claimed twice."""
    card_id = _card(board, "one card, one claimant")
    assert kb.claim_task(board, card_id, claimer="oto-server:1") is not None
    assert kb.claim_task(board, card_id, claimer="oto-server:2") is None
