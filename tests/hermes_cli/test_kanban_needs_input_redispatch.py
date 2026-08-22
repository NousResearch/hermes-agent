"""Regression coverage for needs_input escalation redispatch (#191)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _escalate_needs_input(conn, task_id: str) -> None:
    assert kb.block_task(conn, task_id, reason="answer required", kind="needs_input")
    assert kb.unblock_task(conn, task_id)
    assert kb.block_task(conn, task_id, reason="answer required", kind="needs_input")
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "triage"


def _fake_response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    return response


def test_escalated_needs_input_is_not_auto_decomposed_after_reassignment(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="wait for answer", assignee="worker-a")
        _escalate_needs_input(conn, task_id)
        assert kb.is_block_loop_escalated(conn, task_id)
        conn.execute(
            "UPDATE tasks SET assignee = ? WHERE id = ?",
            ("worker-b", task_id),
        )
        conn.commit()

    assert task_id not in decomp.list_triage_ids()
    with patch("agent.auxiliary_client.call_llm") as call_llm:
        outcome = decomp.decompose_task(
            task_id,
            author=decomp.AUTO_DECOMPOSER_AUTHOR,
        )
    assert outcome.ok is False
    assert "human recovery" in outcome.reason
    call_llm.assert_not_called()

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"
        assert task.assignee == "worker-b"
        assert kb.is_block_loop_escalated(conn, task_id)


def test_manual_decompose_recovers_only_after_success(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs decision", assignee="worker")
        _escalate_needs_input(conn, task_id)

    payload: dict[str, object] = {
        "fanout": False,
        "title": "Decision supplied",
        "body": "Proceed using the operator decision.",
    }
    with (
        patch("agent.auxiliary_client.call_llm", return_value=_fake_response(payload)),
        patch.object(decomp, "_load_config", return_value={}),
        patch.object(decomp, "_build_roster", return_value=([], {"worker"})),
        patch.object(decomp, "_resolve_orchestrator_profile", return_value="worker"),
        patch.object(decomp, "_resolve_default_assignee", return_value="worker"),
    ):
        outcome = decomp.decompose_task(task_id, author="operator")

    assert outcome.ok, outcome.reason
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.block_kind is None
        assert task.block_recurrences == 0
        assert not kb.is_block_loop_escalated(conn, task_id)
        assert any(
            event.kind == "triage_escalation_recovered"
            for event in kb.list_events(conn, task_id)
        )


def test_manual_fanout_recovers_only_after_success(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs routing", assignee="worker")
        _escalate_needs_input(conn, task_id)

    payload: dict[str, object] = {
        "fanout": True,
        "tasks": [
            {
                "title": "Execute operator decision",
                "body": "Proceed with the supplied choice.",
                "assignee": "worker",
                "parents": [],
            }
        ],
    }
    with (
        patch("agent.auxiliary_client.call_llm", return_value=_fake_response(payload)),
        patch.object(decomp, "_load_config", return_value={}),
        patch.object(decomp, "_build_roster", return_value=([], {"worker"})),
        patch.object(decomp, "_resolve_orchestrator_profile", return_value="worker"),
        patch.object(decomp, "_resolve_default_assignee", return_value="worker"),
    ):
        outcome = decomp.decompose_task(task_id, author="operator")

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 1
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.block_kind is None
        assert task.block_recurrences == 0
        assert not kb.is_block_loop_escalated(conn, task_id)
        assert any(
            event.kind == "triage_escalation_recovered"
            for event in kb.list_events(conn, task_id)
        )


def test_failed_manual_decompose_keeps_human_gate(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="still waiting", assignee="worker")
        _escalate_needs_input(conn, task_id)

    with patch(
        "agent.auxiliary_client.call_llm",
        side_effect=RuntimeError("auxiliary unavailable"),
    ):
        outcome = decomp.decompose_task(task_id, author="operator")

    assert outcome.ok is False
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"
        assert kb.is_block_loop_escalated(conn, task_id)
        assert not any(
            event.kind == "triage_escalation_recovered"
            for event in kb.list_events(conn, task_id)
        )


def test_fresh_triage_remains_auto_decomposable(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="fresh idea", triage=True)
    assert task_id in decomp.list_triage_ids()
