"""Tests for HallucinatedResultError in kanban_db.complete_task.

TDD RED phase: these tests must fail before the implementation exists.
"""
from __future__ import annotations

import re
import sqlite3
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


# ---------------------------------------------------------------------------
# A: HallucinatedResultError class exists and is a ValueError subclass
# ---------------------------------------------------------------------------


def test_hallucinated_result_error_is_value_error():
    """HallucinatedResultError must be a ValueError subclass (fail-closed style)."""
    assert issubclass(kb.HallucinatedResultError, ValueError)


def test_hallucinated_result_error_carries_task_id_and_pattern():
    """Error must carry the task_id and the unmatched pattern for structured access."""
    exc = kb.HallucinatedResultError(
        task_id="t_abc123",
        pattern="expected pattern",
        actual="actual result text",
    )
    assert exc.task_id == "t_abc123"
    assert exc.pattern == "expected pattern"
    assert exc.actual == "actual result text"
    assert "t_abc123" in str(exc)


# ---------------------------------------------------------------------------
# A: complete_task without expected_result_pattern — byte-identical behavior
# ---------------------------------------------------------------------------


def test_complete_task_without_pattern_still_works(kanban_home):
    """Omitting expected_result_pattern must not change existing behavior at all."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, result="anything at all") is True
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
        assert task.result == "anything at all"


# ---------------------------------------------------------------------------
# A: complete_task with expected_result_pattern — matching behavior
# ---------------------------------------------------------------------------


def test_complete_task_with_matching_regex_pattern_completes(kanban_home):
    """When pattern matches result, task must be completed normally."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        assert (
            kb.complete_task(
                conn,
                tid,
                result="Implemented feature X in file.py",
                expected_result_pattern=r"Implemented feature",
            )
            is True
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"


def test_complete_task_with_matching_exact_string_completes(kanban_home):
    """Exact-string (non-regex) match against result must also succeed."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        assert (
            kb.complete_task(
                conn,
                tid,
                result="exact expected output",
                expected_result_pattern="exact expected output",
            )
            is True
        )


# ---------------------------------------------------------------------------
# A: complete_task with expected_result_pattern — non-matching must raise
# ---------------------------------------------------------------------------


def test_complete_task_with_nonmatching_pattern_raises(kanban_home):
    """When pattern does NOT match, HallucinatedResultError must be raised."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        with pytest.raises(kb.HallucinatedResultError) as exc_info:
            kb.complete_task(
                conn,
                tid,
                result="some vague unverified summary",
                expected_result_pattern=r"commit [0-9a-f]{7}",
            )
        assert exc_info.value.task_id == tid
        assert "commit" in exc_info.value.pattern


def test_complete_task_with_nonmatching_pattern_does_not_commit(kanban_home):
    """A rejected completion must NOT flip the task to done."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        try:
            kb.complete_task(
                conn,
                tid,
                result="bad result",
                expected_result_pattern=r"REQUIRED_TOKEN",
            )
        except kb.HallucinatedResultError:
            pass
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status != "done", "Task must not be done after pattern rejection"


def test_complete_task_with_nonmatching_pattern_emits_audit_event(kanban_home):
    """Rejected completion must emit an auditable event (never silent)."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        try:
            kb.complete_task(
                conn,
                tid,
                result="fake result",
                expected_result_pattern=r"REQUIRED_TOKEN",
            )
        except kb.HallucinatedResultError:
            pass
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ?",
            (tid,),
        ).fetchall()
        event_kinds = [e["kind"] for e in events]
        assert any(
            "hallucinated" in k or "blocked" in k for k in event_kinds
        ), f"Expected an audit event, got: {event_kinds}"


def test_complete_task_with_none_result_and_pattern_raises(kanban_home):
    """Pattern check against None/empty result must also raise (fail-closed)."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent done")
        kb.claim_task(conn, tid)
        with pytest.raises(kb.HallucinatedResultError):
            kb.complete_task(
                conn,
                tid,
                result=None,
                expected_result_pattern=r"REQUIRED_TOKEN",
            )
