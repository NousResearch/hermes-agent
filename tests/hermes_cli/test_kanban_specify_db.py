"""Tests for kb.specify_triage_task — the DB-layer atomic promotion
from the triage column to todo. LLM-free by design."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


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
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough idea")
        assert kb.get_task(conn, tid).status == "triage"
    with kbc.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="Refined: rough idea",
            body="**Goal**\nDo the thing.",
            author="specifier-bot",
        )
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # No parents → recompute_ready should have flipped it past todo to ready.
    assert task.status == "ready"
    assert task.title == "Refined: rough idea"
    assert "**Goal**" in (task.body or "")


def test_specify_rejects_blank_title(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kbc.connect() as conn, pytest.raises(ValueError):
        kb.specify_triage_task(conn, tid, title="   ", body="ok")


def test_specify_records_audit_comment_only_when_author_given(kanban_home):
    # With author → audit comment added, plus the preserved original brief.
    with kbc.connect() as conn:
        tid1 = _create_triage(conn, title="a")
        kb.specify_triage_task(
            conn, tid1, title="A-spec", body="b", author="ace"
        )
        comments1 = kb.list_comments(conn, tid1)
    audit1 = [c for c in comments1 if c.author == "ace"]
    assert len(audit1) == 1
    assert "Specified" in audit1[0].body
    # The rewrite replaces non-empty content, so the pre-rewrite brief is
    # snapshotted as a machine-readable comment in the same transaction
    # (contract added in f44e98fcbe — the author's brief is never lost).
    brief1 = [c for c in comments1 if c.author == "specifier"]
    assert len(brief1) == 1
    assert "kanban-original-brief" in brief1[0].body

    # Without author → no audit comment (silent), but the brief snapshot is
    # still written: it protects the author's text, not the audit trail.
    with kbc.connect() as conn:
        tid2 = _create_triage(conn, title="b")
        kb.specify_triage_task(conn, tid2, title="B-spec", body="b")
        comments2 = kb.list_comments(conn, tid2)
    assert [c.author for c in comments2] == ["specifier"]

    # Same title and body → nothing rewritten, so nothing snapshotted and
    # nothing audited.
    with kbc.connect() as conn:
        tid3 = _create_triage(conn, title="same", body="unchanged")
        kb.specify_triage_task(
            conn, tid3, title="same", body="unchanged", author="ace"
        )
        comments3 = kb.list_comments(conn, tid3)
    assert comments3 == []


