"""Tests for kb.specify_triage_task — the DB-layer atomic promotion
from the triage column to todo. LLM-free by design."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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


def test_specify_promotes_triage_to_todo(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="rough idea")
        assert kb.get_task(conn, tid).status == "triage"
    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="Refined: rough idea",
            body="**Goal**\nDo the thing.",
            author="specifier-bot",
        )
    assert ok is True
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    # No parents → recompute_ready should have flipped it past todo to ready.
    assert task.status == "ready"
    assert task.title == "Refined: rough idea"
    assert "**Goal**" in (task.body or "")


def test_specify_with_open_parent_lands_in_todo_not_ready(kanban_home):
    # Parent-gated specified tasks must not jump the dispatcher — they go
    # to todo and wait for parent completion like any other gated task.
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent work")
        child = _create_triage(conn, title="child idea")
        kb.link_tasks(conn, parent, child)
        # After linking with an open parent, triage status should still be
        # 'triage' (linking doesn't touch triage tasks).
        assert kb.get_task(conn, child).status == "triage"
    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            child,
            body="full spec",
            author="specifier",
        )
    assert ok is True
    with kb.connect() as conn:
        t = kb.get_task(conn, child)
    # Parent still open → specified child sits in 'todo', not 'ready'.
    assert t.status == "todo"


def test_specify_refuses_non_triage_task(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="normal task")
        assert kb.get_task(conn, tid).status == "ready"
    with kb.connect() as conn:
        ok = kb.specify_triage_task(conn, tid, body="won't apply")
    assert ok is False
    with kb.connect() as conn:
        # Status unchanged.
        assert kb.get_task(conn, tid).status == "ready"


def test_specify_returns_false_for_unknown_id(kanban_home):
    with kb.connect() as conn:
        ok = kb.specify_triage_task(conn, "t_does_not_exist", body="x")
    assert ok is False


def test_specify_rejects_blank_title(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kb.connect() as conn, pytest.raises(ValueError):
        kb.specify_triage_task(conn, tid, title="   ", body="ok")


def test_specify_emits_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kb.connect() as conn:
        kb.specify_triage_task(
            conn, tid, title="new", body="b", author="ace"
        )
    with kb.connect() as conn:
        events = kb.list_events(conn, tid)
    kinds = [e.kind for e in events]
    assert "specified" in kinds
    # The specified event records which fields actually changed as a
    # JSON payload under task_events.payload.
    spec_ev = next(e for e in events if e.kind == "specified")
    assert spec_ev.payload is not None
    fields = spec_ev.payload.get("changed_fields") or []
    assert "title" in fields
    assert "body" in fields


def test_specify_records_audit_comment_only_when_author_given(kanban_home):
    # With author → comment added.
    with kb.connect() as conn:
        tid1 = _create_triage(conn, title="a")
        kb.specify_triage_task(
            conn, tid1, title="A-spec", body="b", author="ace"
        )
        comments1 = kb.list_comments(conn, tid1)
    assert len(comments1) == 1
    assert "Specified" in comments1[0].body
    assert comments1[0].author == "ace"

    # Without author → no comment (silent).
    with kb.connect() as conn:
        tid2 = _create_triage(conn, title="b")
        kb.specify_triage_task(conn, tid2, title="B-spec", body="b")
        comments2 = kb.list_comments(conn, tid2)
    assert comments2 == []


def test_specify_skips_comment_when_nothing_changed(kanban_home):
    # Create triage task with title and body already set; pass identical
    # values to specify. Should promote to todo but skip audit comment.
    with kb.connect() as conn:
        tid = _create_triage(conn, title="same", body="same body")
    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="same",
            body="same body",
            author="ace",
        )
    assert ok is True
    with kb.connect() as conn:
        # Promoted.
        assert kb.get_task(conn, tid).status in {"todo", "ready"}
        # No audit comment because neither field changed.
        assert kb.list_comments(conn, tid) == []


def test_specify_with_only_body_preserves_title(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="keep this title")
    with kb.connect() as conn:
        kb.specify_triage_task(conn, tid, body="new body only")
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    assert t.title == "keep this title"
    assert t.body == "new body only"


def test_specify_second_call_noop_false(kanban_home):
    # Promoting twice must not crash and the second call returns False
    # because the task is no longer in triage.
    with kb.connect() as conn:
        tid = _create_triage(conn, title="once")
    with kb.connect() as conn:
        assert kb.specify_triage_task(conn, tid, body="spec") is True
    with kb.connect() as conn:
        assert kb.specify_triage_task(conn, tid, body="spec again") is False
