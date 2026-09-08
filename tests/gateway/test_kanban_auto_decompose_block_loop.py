"""Regression tests for automatic decomposition of block-loop triage cards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.kanban_watchers_dispatcher import _KanbanDispatcher
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _dispatcher(monkeypatch) -> _KanbanDispatcher:
    dispatcher = _KanbanDispatcher(kb, MagicMock())
    monkeypatch.setattr(dispatcher, "_board_slugs", lambda: ["default"])
    return dispatcher


def _single_task_patches():
    payload = json.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Concrete implementation spec.",
        "assignee": "worker",
    })
    routing = decomp._Routing(
        orchestrator="worker",
        default_assignee="worker",
        auto_promote=True,
        roster=[{"name": "worker", "description": "worker", "has_description": True}],
        valid_names={"worker"},
    )
    return (
        patch.object(decomp, "_call_aux", return_value=(payload, "")),
        patch.object(decomp, "_load_routing", return_value=routing),
    )


def test_auto_decompose_keeps_block_loop_escalation_in_triage_but_manual_remains_available(
    kanban_home, monkeypatch,
):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Needs human attention", triage=True)
        conn.execute(
            "UPDATE tasks SET block_recurrences = ? WHERE id = ?",
            (kb.BLOCK_RECURRENCE_LIMIT, task_id),
        )
        conn.commit()

    automatic = MagicMock(
        return_value=decomp.DecomposeOutcome(task_id, False, "automatic call must be skipped")
    )
    with patch.object(decomp, "decompose_task", automatic):
        dispatcher = _dispatcher(monkeypatch)
        assert dispatcher.auto_decompose_tick(3) == 0
        assert dispatcher.auto_decompose_tick(3) == 0

    automatic.assert_not_called()
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        event_kinds = {
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (task_id,)
            )
        }
        runs = kb.list_runs(conn, task_id)
    assert task is not None
    assert task.status == "triage"
    assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
    assert task.current_run_id is None
    assert task.worker_pid is None
    assert not ({"specified", "promoted", "claimed"} & event_kinds)
    assert runs == []

    aux_patch, routing_patch = _single_task_patches()
    with aux_patch, routing_patch:
        outcome = decomp.decompose_task(task_id, author="operator")

    assert outcome.ok is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT


@pytest.mark.parametrize(
    ("recurrences", "preceding_loop_cards"),
    [(0, 1000), (kb.BLOCK_RECURRENCE_LIMIT - 1, 0)],
)
def test_auto_decompose_still_processes_eligible_triage_tasks(
    kanban_home, monkeypatch, recurrences, preceding_loop_cards,
):
    with kbc.connect_closing() as conn:
        loop_ids = [
            kb.create_task(conn, title=f"Block loop {index}", triage=True)
            for index in range(preceding_loop_cards)
        ]
        conn.executemany(
            "UPDATE tasks SET block_recurrences = ? WHERE id = ?",
            [(kb.BLOCK_RECURRENCE_LIMIT, loop_id) for loop_id in loop_ids],
        )
        conn.commit()
        task_id = kb.create_task(conn, title="Automatically decompose me", triage=True)
        conn.execute(
            "UPDATE tasks SET block_recurrences = ? WHERE id = ?",
            (recurrences, task_id),
        )
        conn.commit()

    aux_patch, routing_patch = _single_task_patches()
    with aux_patch, routing_patch, patch.object(
        decomp, "decompose_task", wraps=decomp.decompose_task,
    ) as automatic:
        assert _dispatcher(monkeypatch).auto_decompose_tick(3) == 1

    automatic.assert_called_once_with(task_id, author="auto-decomposer")
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        event_kinds = {
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (task_id,)
            )
        }
    assert task is not None
    assert task.status == "ready"
    assert task.block_recurrences == recurrences
    assert "specified" in event_kinds
