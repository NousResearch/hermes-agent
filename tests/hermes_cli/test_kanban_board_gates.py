"""Board gates: five-field cards and debrief-before-done (opt-in).

The gates are opt-in via ``HERMES_KANBAN_BOARD_GATES`` / ``kanban.board_gates.enabled``
so stock installs and the wider test suite keep the permissive behaviour. This module
exercises both gates with the flag forced on, plus the default-off path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


FIVE_FIELD_BODY = (
    "**Intent:** ship the gate.\n"
    "**End state:** tests pass.\n"
    "**Constraints:** none.\n"
    "**Resources and limits:** 10m.\n"
    "**Report:** a debrief.\n"
)

DEBRIEF = (
    "What was supposed to happen: ship the gate.\n"
    "What actually happened: shipped it, tests pass (evidence: pytest).\n"
    "Why the difference: instruction was clear; execution met it; situation unchanged.\n"
    "Keep: the DB-layer gate. Improve: error copy. Undo: none needed."
)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def gates_on(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BOARD_GATES", "1")


def _event_kinds(conn, tid) -> list[str]:
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()]


def _ready_task(conn, body: str = FIVE_FIELD_BODY) -> str:
    tid = kb.create_task(conn, title="demo", body=body, assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


# --- create gate ---


def test_missing_fields_sent_back(conn, gates_on):
    tid = kb.create_task(conn, title="demo", body="do the thing", assignee="coder")
    assert kb.get_task(conn, tid).status == "triage"
    kinds = _event_kinds(conn, tid)
    assert "card_sent_back_missing_fields" in kinds
    bodies = [c.body for c in kb.list_comments(conn, tid)]
    assert any("Intent" in b and "End state" in b for b in bodies)


def test_missing_end_state_names_only_that_field(conn, gates_on):
    tid = kb.create_task(conn, title="demo", body="**Intent:** ship it", assignee="coder")
    assert kb.get_task(conn, tid).status == "triage"
    ev = [r for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'card_sent_back_missing_fields'",
        (tid,),
    ).fetchall()]
    assert ev
    assert kb._json_or(ev[0]["payload"])["missing_fields"] == ["End state"]


def test_well_formed_card_passes(conn, gates_on):
    tid = kb.create_task(conn, title="demo", body=FIVE_FIELD_BODY, assignee="coder")
    assert kb.get_task(conn, tid).status == "ready"


def test_gates_off_by_default_leave_missing_card_ready(conn):
    tid = kb.create_task(conn, title="demo", body="do the thing", assignee="coder")
    assert kb.get_task(conn, tid).status == "ready"


# --- debrief gate ---


def test_complete_without_debrief_refused(conn, gates_on):
    tid = _ready_task(conn)
    with pytest.raises(kb.MissingDebriefError):
        kb.complete_task(conn, tid, summary="did the thing")
    assert kb.get_task(conn, tid).status == "running"
    kinds = _event_kinds(conn, tid)
    assert "completion_blocked_missing_debrief" in kinds
    assert "completed" not in kinds


def test_complete_with_debrief_succeeds(conn, gates_on):
    tid = _ready_task(conn)
    assert kb.complete_task(conn, tid, summary=DEBRIEF) is True
    assert kb.get_task(conn, tid).status == "done"


def test_review_approval_exempt_from_debrief_gate(conn, gates_on):
    tid = _ready_task(conn)
    assert kb.request_review(conn, tid, summary="please look") is True
    assert kb.complete_task(conn, tid) is True
    assert kb.get_task(conn, tid).status == "done"


def test_gates_off_by_default_allow_debriefless_complete(conn):
    tid = _ready_task(conn, body=FIVE_FIELD_BODY)
    assert kb.complete_task(conn, tid, summary="did the thing") is True
    assert kb.get_task(conn, tid).status == "done"


# --- counters ---


def test_gate_counters_queryable(conn, gates_on):
    tid = kb.create_task(conn, title="bad", body="nope", assignee="coder")
    assert kb.get_task(conn, tid).status == "triage"
    good = _ready_task(conn)
    with pytest.raises(kb.MissingDebriefError):
        kb.complete_task(conn, good, summary="no debrief")
    counters = kb.gate_counters(conn)
    assert counters["cards_sent_back_missing_fields"] == 1
    assert counters["completions_refused_missing_debrief"] == 1
