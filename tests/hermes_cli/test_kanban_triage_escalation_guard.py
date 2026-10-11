"""Tests for the triage escalation guard (card t_5fe6220a).

``block_task`` routes a card that keeps blocking for the same cause to ``triage``
for a HUMAN (``block_loop_detected``). The auto-decompose hook then saw a triage
card, asked the auxiliary LLM to break it up, and fanned it out — defeating the
escalation and manufacturing a fresh graph of children off a card whose state the
human had never touched. These tests pin the guard on EVERY promotion path out of
``triage``, and pin the loop-breaker payload's record of the triage round-trip
(``recurrences`` alone reads as N consecutive honest blocks).

Both directions are covered: a parked card refuses, and an ordinary triage card
still promotes, so the guard cannot pass by simply breaking promotion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as kbg
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as specify
from hermes_cli.kanban_db import BLOCK_RECURRENCE_LIMIT


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim_running(conn, tid: str) -> None:
    """Return a task to the pool and re-claim it, so ``block_task`` can fire again."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _park_in_triage(conn, *, kind: str = "needs_input", title: str = "looping card") -> str:
    """Drive a card through the REAL ladder until the breaker parks it in ``triage``."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    _claim_running(conn, tid)
    for _ in range(BLOCK_RECURRENCE_LIMIT + 1):
        kb.block_task(conn, tid, reason="same cause", kind=kind)
        if kb.get_task(conn, tid).status == "triage":
            return tid
        assert kb.unblock_task(conn, tid)
        _claim_running(conn, tid)
    raise AssertionError("the breaker never parked the card in triage")


def _children() -> list[dict]:
    return [
        {"title": "child a", "body": "do a", "assignee": "worker", "parents": []},
        {"title": "child b", "body": "do b", "assignee": "worker", "parents": [0]},
    ]


def _escape_triage_by_hand(conn, tid: str) -> None:
    """The operator's escape hatch: edit the column directly.

    Neither promotion path will move an escalated card — that IS the guard — so a
    human disposes of it with a direct status write, the same shape the dashboard and
    ``hermes kanban`` use. ``block_recurrences`` is deliberately left alone:
    ``unblock_task`` already refuses to reset it, because that amnesia is what let the
    block loop run unbounded.
    """
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))


# ---------------------------------------------------------------------------
# The guard: a card parked by the breaker refuses on every promotion path
# ---------------------------------------------------------------------------

def test_decompose_triage_task_refuses_a_parked_card(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        before = len(kb.list_tasks(conn, limit=500))

        refusal = kbg.decompose_triage_task(
            conn, tid, root_assignee=None, children=_children(), author="decomposer",
        )

        assert not refusal, "the guard must refuse a card parked by the loop breaker"
        assert not isinstance(refusal, list)
        assert refusal.task_id == tid
        assert "block_loop_detected" in refusal.detail
        assert tid in refusal.detail
        assert str(refusal.event_id) in refusal.detail

        # Nothing was created and the card is still the human's escalation.
        assert kb.get_task(conn, tid).status == "triage"
        assert len(kb.list_tasks(conn, limit=500)) == before
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "decomposed"]


def test_specify_triage_task_refuses_the_same_card(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        original = kb.get_task(conn, tid)

        ok = kb.specify_triage_task(conn, tid, title="still stuck", body="fresh words")

        assert not ok
        assert ok.cause == "block_loop_escalation"
        assert "block_loop_detected" in ok.detail
        assert tid in ok.detail

        after = kb.get_task(conn, tid)
        assert after.status == "triage"
        assert after.title == original.title, "a refused specify must not edit the card"
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "specified"]


def test_decompose_task_refuses_before_spending_the_aux_call(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)

    aux = MagicMock(return_value=None)
    with patch("hermes_cli.kanban_decompose._call_aux", aux):
        outcome = decomp.decompose_task(tid, author="decomposer")

    assert not outcome.ok
    assert aux.call_count == 0, "a parked card must not reach the LLM"
    assert "block_loop_detected" in outcome.reason
    assert tid in outcome.reason
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_specify_task_refuses_before_spending_the_aux_call(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)

    aux = MagicMock(return_value=None)
    with patch("hermes_cli.kanban_specify._call_aux", aux):
        outcome = specify.specify_task(tid, author="specifier")

    assert not outcome.ok
    assert aux.call_count == 0, "a parked card must not reach the LLM"
    assert "block_loop_detected" in outcome.reason
    assert tid in outcome.reason


def test_a_parked_card_is_refused_on_every_later_attempt(kanban_home: Path) -> None:
    """Sticky: nothing in the promotion paths clears the escalation."""
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        for _ in range(3):
            assert not kb.specify_triage_task(conn, tid, title="t", body="b")
            assert not kbg.decompose_triage_task(
                conn, tid, root_assignee=None, children=_children(),
            )
        assert kb.get_task(conn, tid).status == "triage"


# ---------------------------------------------------------------------------
# The guard does not break ordinary triage promotion (the other direction)
# ---------------------------------------------------------------------------

def test_ordinary_triage_card_still_decomposes(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="vague idea", triage=True)
        child_ids = kbg.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=_children(), author="decomposer",
        )
        assert isinstance(child_ids, list) and len(child_ids) == 2
        assert kb.get_task(conn, tid).status == "todo"


def test_ordinary_triage_card_still_specifies(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="vague idea", triage=True)
        assert kb.specify_triage_task(conn, tid, title="Sharp idea", body="now actionable")
        task = kb.get_task(conn, tid)
        assert task.status in {"todo", "ready"}
        assert task.title == "Sharp idea"


def test_a_blocked_but_not_escalated_card_is_not_refused(kanban_home: Path) -> None:
    """Only the breaker's park refuses — one block is not an escalation."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="one-off blocker", assignee="worker")
        _claim_running(conn, tid)
        kb.block_task(conn, tid, reason="waiting on a human answer", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"
        # The card is not in triage, so the guard has no say; move it there directly
        # (a board column edit) and confirm it STILL promotes: a single block with a
        # human-typed reason is exactly what the specifier exists to fix.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
        assert kb.specify_triage_task(conn, tid, title="Answered", body="answer included")
        assert kb.get_task(conn, tid).status in {"todo", "ready"}


def test_guard_predicate_ignores_a_non_block_triage_card(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="vague idea", triage=True)
        assert kb.triage_escalation_refusal(conn, tid) is None
        assert kb.triage_round_trips(conn, tid) == 0


# ---------------------------------------------------------------------------
# The refusal's own words: only the exits that actually clear the park
# ---------------------------------------------------------------------------

def _load_dashboard_plugin():
    """``plugins/kanban/dashboard/plugin_api.py`` by path, as the sibling dashboard tests do."""
    pytest.importorskip("fastapi")
    import importlib.util
    import sys

    path = (
        Path(__file__).resolve().parents[2]
        / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location("kanban_plugin_triage_guard_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _status(conn, tid: str) -> str:
    """The card's status, asserting it still exists (reviewer-facing: no silent None)."""
    task = kb.get_task(conn, tid)
    assert task is not None, f"no task {tid} on this board"
    return task.status


def test_the_refusal_names_only_the_exits_that_clear_the_park(kanban_home: Path) -> None:
    """The detail line is the operator's instruction; a verb that does nothing is a false path.

    The kernel refuses ``unblock`` on a triage card (triage is neither blocked nor
    scheduled), refuses ``complete``, and ``reassign`` succeeds *while leaving the card
    parked* — so naming those three as ways to dispose of it sent the operator down dead
    ends to reach the one that works.
    """
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        refusal = kb.triage_escalation_refusal(conn, tid)
        assert refusal is not None, "the card is parked, so there is a refusal to read"
        detail = refusal.detail

    assert "archive" in detail, "archive must be named as a disposal"
    assert "move it out of the triage column" in detail, (
        "the board move out of Triage must be named as the other disposal"
    )
    assert "unblock / complete / reassign do not clear the park" in detail, (
        "verbs that leave the card parked must be named as non-disposals, never offered"
    )


def test_the_board_move_out_of_triage_disposes_of_the_park(kanban_home: Path) -> None:
    """Drive the move the message names — the dashboard drag out of the column — and prove
    the card is out of the park and promotable again."""
    dash = _load_dashboard_plugin()
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        assert _status(conn, tid) == "triage"
        refusal = kb.triage_escalation_refusal(conn, tid)
        assert refusal is not None, "the card IS parked when the operator reads the message"

        # The same write a drag out of the Triage column performs.
        assert dash._set_status_direct(conn, tid, "todo") is True

        assert _status(conn, tid) == "todo", "the board move leaves triage"
        kb.recompute_ready(conn)
        assert _status(conn, tid) == "ready", "and the card is promotable again"


def test_archive_disposes_of_the_park(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        assert kb.triage_escalation_refusal(conn, tid) is not None
        assert kb.archive_task(conn, tid) is True
        assert _status(conn, tid) == "archived"


def test_unblock_complete_and_reassign_do_not_dispose_of_a_triage_card(
    kanban_home: Path,
) -> None:
    """Every verb the old message offered as a disposal, measured on a parked card."""
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)

        assert kb.unblock_task(conn, tid) is False, "unblock only exits blocked/scheduled"
        assert _status(conn, tid) == "triage"

        completed = kb.complete_task(conn, tid, result="closing it")
        # The refusal's TYPE is deliberately not pinned: a triage card is refused by whatever
        # falsy refusal the kernel returns, and the class that models it is another card's
        # change, not part of this guard.
        assert not completed, "complete refuses a triage card"
        assert _status(conn, tid) == "triage"

        # reassign is the sneakiest of the three: it returns True (the assignee really did
        # change) and the card is still the escalation.
        assert kb.reassign_task(conn, tid, "someone-else") is True
        task = kb.get_task(conn, tid)
        assert task is not None and task.assignee == "someone-else"
        assert _status(conn, tid) == "triage", "reassign does not clear the park"

        # ...and after all three the guard still stands.
        assert not kb.specify_triage_task(conn, tid, title="t", body="b")
        assert _status(conn, tid) == "triage"


# ---------------------------------------------------------------------------
# The payload records the triage round-trip
# ---------------------------------------------------------------------------

def test_first_escalation_payload_reports_no_round_trip(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        payload = [
            e.payload for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"
        ][-1]
        assert payload["recurrences"] == BLOCK_RECURRENCE_LIMIT
        assert payload["triage_round_trips"] == 0


def test_re_escalated_card_reports_its_round_trip(kanban_home: Path) -> None:
    """A card a human pulled out of triage and re-blocked carries the round-trip.

    Without this the payload reads as N consecutive honest blocks, and the re-block
    looks like a fresh cause.
    """
    with kbc.connect_closing() as conn:
        tid = _park_in_triage(conn)
        _escape_triage_by_hand(conn, tid)
        _claim_running(conn, tid)
        kb.block_task(conn, tid, reason="same cause", kind="needs_input")

        escalations = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
        assert len(escalations) == 2
        assert kb.get_task(conn, tid).status == "triage"
        payload = escalations[-1].payload
        assert payload["triage_round_trips"] == 1
        assert payload["recurrences"] == BLOCK_RECURRENCE_LIMIT + 1


# ---------------------------------------------------------------------------
# Field report: an already-decomposed root parked again (cable-span t_40f7472e)
#
# The production loop: a root that had ALREADY been decomposed once (it carries a
# ``decomposed`` event) later blocked needs_input twice -> ``block_loop_detected`` ->
# triage. ``decompose_triage_task`` returned None on every later tick because of the
# ``decomposed`` event, so the fan-out path was never the loop. The auxiliary LLM
# answered ``fanout=false`` instead, and ``_apply_single -> specify_triage_task``
# wrote ``specified`` + ``promoted`` and moved the card to ``todo`` -- 17 times, about
# every 3 minutes.
# ---------------------------------------------------------------------------

def _park_decomposed_root(conn) -> str:
    """A root that fanned out once, then blocked needs_input to the breaker's limit."""
    tid = kb.create_task(conn, title="already-decomposed root", triage=True)
    child_ids = kbg.decompose_triage_task(
        conn, tid, root_assignee="worker", children=_children(), author="decomposer",
    )
    assert isinstance(child_ids, list) and len(child_ids) == 2
    assert [e for e in kb.list_events(conn, tid) if e.kind == "decomposed"]
    # The children finish, the root wakes and runs again, then blocks.
    for cid in child_ids:
        _claim_running(conn, cid)
        assert kb.complete_task(conn, cid, result="done")
    kb.recompute_ready(conn)
    _claim_running(conn, tid)
    for _ in range(BLOCK_RECURRENCE_LIMIT + 1):
        kb.block_task(conn, tid, reason="needs a human answer", kind="needs_input")
        if kb.get_task(conn, tid).status == "triage":
            break
        assert kb.unblock_task(conn, tid)
        _claim_running(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"
    assert [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
    return tid


def _promotion_event_count(conn, tid: str) -> int:
    return len([e for e in kb.list_events(conn, tid) if e.kind in {"specified", "promoted"}])


_FANOUT_FALSE = (
    '{"fanout": false, "title": "respecified root", "body": "rewritten by the aux LLM"}'
)


def test_already_decomposed_root_parked_again_is_not_respecified(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_decomposed_root(conn)
        before = _promotion_event_count(conn, tid)

        # Path 1: the specify sink the loop actually went through.
        ok = kb.specify_triage_task(conn, tid, title="respecified root", body="rewritten")
        assert not ok
        assert "block_loop_detected" in ok.detail

    # Path 2: the auto-decompose tick with the aux LLM answering fanout=false,
    # repeated as the tick did in production.
    aux = MagicMock(return_value=(_FANOUT_FALSE, ""))
    with patch("hermes_cli.kanban_decompose._call_aux", aux):
        for _ in range(3):
            outcome = decomp.decompose_task(tid, author="decomposer")
            assert not outcome.ok
            assert "block_loop_detected" in outcome.reason

    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.get_task(conn, tid).title == "already-decomposed root"
        assert _promotion_event_count(conn, tid) == before, (
            "no new specified/promoted events may be written for a parked root"
        )


def test_already_decomposed_root_single_path_guard_holds_without_preflight(
    kanban_home: Path,
) -> None:
    """Same incident, with the pre-aux preflight disabled: the in-txn guard inside
    ``specify_triage_task`` alone must stop ``_apply_single`` (fanout=false)."""
    with kbc.connect_closing() as conn:
        tid = _park_decomposed_root(conn)
        before = _promotion_event_count(conn, tid)

    aux = MagicMock(return_value=(_FANOUT_FALSE, ""))
    with patch("hermes_cli.kanban_decompose._call_aux", aux), \
            patch("hermes_cli.kanban_decompose._promotion_refusal", return_value=None):
        outcome = decomp.decompose_task(tid, author="decomposer")

    assert aux.call_count == 1, "preflight disabled: the aux call happens"
    assert not outcome.ok
    assert "block_loop_detected" in outcome.reason
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert _promotion_event_count(conn, tid) == before
