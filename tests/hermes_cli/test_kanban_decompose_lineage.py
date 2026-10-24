"""Decompose-lineage marking — ``decomposed_child_ids`` (#140323).

A decompose child parked back in triage (dashboard bulk move) must stay
recognizable as decomposition output so the gateway's auto-decompose tick can
skip it; hand-written triage cards must not be flagged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task, decomposed_child_ids
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea"):
    return kb.create_task(conn, title=title, triage=True)


def test_decomposed_child_ids_flags_retriaged_children_not_manual_cards(kanban_home):
    with kbc.connect() as conn:
        root = _create_triage(conn)
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orchestrator",
            children=[{"title": "child A", "assignee": "researcher"}],
            author="auto-decomposer",
        )
        assert child_ids
        manual = _create_triage(conn, title="hand-written idea")
        # A bulk dashboard move parks the child back in triage.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (child_ids[0],))

    with kbc.connect() as conn:
        flagged = decomposed_child_ids(conn, [child_ids[0], manual])
    assert flagged == {child_ids[0]}


def test_decomposed_child_ids_covers_row_stamp_without_created_event(kanban_home):
    with kbc.connect() as conn:
        plain = _create_triage(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET created_by = 'auto-decomposer' WHERE id = ?", (plain,))

    with kbc.connect() as conn:
        assert decomposed_child_ids(conn, [plain]) == {plain}
