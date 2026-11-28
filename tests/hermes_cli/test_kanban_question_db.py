"""Tests for kanban_db_graph.question_triage_task — the DB-layer atomic park
from the triage column to blocked/needs_input (the decomposer's question-card
path). LLM-free by design."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as kbg


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        triage=True,
    )


def test_question_parks_triage_as_blocked_needs_input(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="Shipping + escrow", body="entry", assignee="default")
    with kbc.connect() as conn:
        ok = kbg.question_triage_task(
            conn,
            tid,
            title="Quel prestataire de paiement ?",
            body="**Question**\nQuel prestataire ?\n**Options**\n- Stripe Connect",
            reason="payment-provider premise not decided",
            author="auto-decomposer",
        )
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        # A question card is never routed to a worker.
        assert task.assignee is None
        assert task.block_recurrences == 1
        assert task.title == "Quel prestataire de paiement ?"
        assert "**Options**" in (task.body or "")
        # Audit trail: one 'blocked' event + one comment by the author.
        events = kb.list_events(conn, tid)
        assert events and events[-1].kind == "blocked"
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert "Question card" in comments[0].body
        assert comments[0].author == "auto-decomposer"


def test_question_park_has_no_recompute_ready_side_effect(kanban_home):
    """A parked question stays parked: unlike specify_triage_task there is
    deliberately no recompute_ready — nothing may flip it out of blocked."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="q")
        kbg.question_triage_task(conn, tid, title="Q", body="b")
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
        kb.recompute_ready(conn)
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_question_refuses_non_triage_task(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ready work")  # ready, not triage
        assert kbg.question_triage_task(conn, tid, title="Q", body="b") is False


def test_question_rejects_blank_title_or_body(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="q")
    with kbc.connect() as conn, pytest.raises(ValueError):
        kbg.question_triage_task(conn, tid, title="   ", body="ok")
    with kbc.connect() as conn, pytest.raises(ValueError):
        kbg.question_triage_task(conn, tid, title="Q", body="   ")


def test_question_same_kind_repark_counts_recurrences(kanban_home):
    """Same recurrence arithmetic as _route_block: a same-kind re-park counts
    toward the block-loop breaker; a fresh park starts at 1."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="q")
        kbg.question_triage_task(conn, tid, title="Q1", body="b")
        assert kb.get_task(conn, tid).block_recurrences == 1
        # A human moved the card back to Triage without clearing block_kind
        # (the same shape _route_block's loop breaker leaves behind).
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
        kbg.question_triage_task(conn, tid, title="Q2", body="b")
        assert kb.get_task(conn, tid).block_recurrences == 2
        assert kb.get_task(conn, tid).status == "blocked"


def test_question_without_author_writes_no_comment(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="q")
        kbg.question_triage_task(conn, tid, title="Q", body="b")
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.list_comments(conn, tid) == []
