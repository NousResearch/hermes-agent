"""Take v2: one gesture performs budget planning and respects board capacity."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_batch_take as batch
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_decompose import DecomposeOutcome


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_take_triage_decomposes_then_assigns_and_promotes(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="Large task")
        child = kb.create_task(conn, title="Child")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (root,))

    with patch("hermes_cli.kanban_decompose.decompose_task", return_value=DecomposeOutcome(
        root, True, fanout=True, child_ids=[child]
    )), patch("hermes_cli.kanban_batch_take.plan_and_take", wraps=batch.plan_and_take) as plan, patch(
        "agent.auxiliary_client.call_llm",
        return_value=type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": '{\"edges\": []}'})()})()]})(),
    ):
        outcome = batch.take([root])

    assert outcome.ok
    plan.assert_called_once_with([child], timeout=180)
    with kb.connect() as conn:
        taken = kb.get_task(conn, child)
    assert taken.assignee == "default"
    assert taken.status == "ready"


def test_board_agent_limit_caps_dispatch(kanban_home):
    kb.write_board_metadata("default", agent_limit=1)
    with kb.connect() as conn:
        first = kb.create_task(conn, title="one", assignee="default")
        second = kb.create_task(conn, title="two", assignee="default")
        result = kb.dispatch_once(conn, spawn_fn=lambda *_: 1234, dry_run=True)

    assert len(result.spawned) == 1
    assert result.skipped_board_capped == []
    # A second tick sees one running task only when a real spawn claims it.
    with kb.connect() as conn:
        kb.claim_task(conn, first)
        result = kb.dispatch_once(conn, spawn_fn=lambda *_: 1234, dry_run=True)
    assert second in result.skipped_board_capped


def test_board_metadata_defaults_and_validates_agent_limit(kanban_home):
    assert kb.read_board_metadata("default")["agent_limit"] == 10
    assert kb.write_board_metadata("default", agent_limit=3)["agent_limit"] == 3
    with pytest.raises(ValueError, match="positive"):
        kb.write_board_metadata("default", agent_limit=0)
