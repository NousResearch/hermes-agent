"""Regression coverage for decompose's explicit-assignment boundary.

Decomposition may shape a dependency graph, but starting work requires a
separate take / batch-take action.  Therefore LLM routing hints must not be
persisted as task assignees.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _aux_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_decompose_fanout_leaves_root_and_children_unassigned(kanban_home):
    """CLI/API decomposition records the graph but cannot start work."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Ship feature", triage=True)

    payload = json.dumps({
        "fanout": True,
        "rationale": "independent seams",
        "tasks": [
            {"title": "Research", "body": "Inspect options.", "assignee": "researcher", "parents": []},
            {"title": "Build", "body": "Implement it.", "assignee": "engineer", "parents": [0]},
        ],
    })

    with (
        patch("agent.auxiliary_client.call_llm", return_value=_aux_response(payload)),
        patch("hermes_cli.kanban_decompose._load_config", return_value={
            "kanban": {
                "orchestrator_profile": "orchestrator",
                "default_assignee": "fallback",
            }
        }),
    ):
        outcome = decomp.decompose_task(task_id, author="test")

    assert outcome.ok, outcome.reason
    assert outcome.child_ids is not None
    with kb.connect() as conn:
        root = kb.get_task(conn, task_id)
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]

    assert root is not None
    assert root.status == "todo"
    assert all(child is not None and child.status == "todo" for child in children)
    assert root.assignee is None
    assert all(child is not None and child.assignee is None for child in children)


def test_decompose_single_task_ignores_llm_assignee(kanban_home):
    """The non-fanout path must obey the same no-routing guarantee."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Small change", triage=True)

    payload = json.dumps({
        "fanout": False,
        "rationale": "one unit",
        "title": "Implement small change",
        "body": "Concrete acceptance criteria.",
        "assignee": "engineer",
    })

    with patch("agent.auxiliary_client.call_llm", return_value=_aux_response(payload)):
        outcome = decomp.decompose_task(task_id, author="test")

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "todo"
    assert task.assignee is None
