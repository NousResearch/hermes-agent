"""Regression tests for the documented Kanban status model.

The code-level canonical schema lives in ``kanban_db.VALID_STATUSES``.  The
user-facing docs should not describe a narrower status set or a dashboard column
set that makes first-class statuses disappear.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb


DOC = Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "features" / "kanban.md"


def test_kanban_docs_list_every_canonical_task_status():
    text = DOC.read_text()

    expected = " | ".join([
        "triage", "todo", "scheduled", "ready", "running",
        "blocked", "review", "done", "archived",
    ])
    assert set(expected.split(" | ")) == kb.VALID_STATUSES
    assert f"status (`{expected}`)" in text


def test_kanban_docs_dashboard_columns_match_active_statuses():
    text = DOC.read_text()

    expected_active = " | ".join([
        "triage", "todo", "scheduled", "ready", "running",
        "blocked", "review", "done",
    ])
    assert f"one column per active status: `{expected_active}`" in text
