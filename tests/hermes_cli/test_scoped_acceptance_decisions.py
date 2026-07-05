import json

import pytest

from hermes_cli import kanban_db as kb
from tools import kanban_tools


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    return home


def _green_acceptance(**overrides):
    data = {
        "touched_files": ["tools/kanban_tools.py"],
        "task_scope_files": [
            "tools/kanban_tools.py",
            "tests/hermes_cli/test_scoped_acceptance_decisions.py",
        ],
        "targeted_tests": [
            {"name": "focused gate", "status": "passed"},
        ],
        "required_checks": [
            {"name": "lint", "status": "passed"},
            {"name": "typecheck", "status": "passed"},
            {"name": "diff-check", "status": "passed"},
        ],
    }
    data.update(overrides)
    return data


def test_unrelated_full_suite_failure_completes_with_scoped_caveat(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="scoped fix", assignee="worker")

    payload = json.loads(
        kanban_tools._handle_complete(
            {
                "task_id": task_id,
                "summary": "Focused Task 13 fix is green.",
                "acceptance": _green_acceptance(
                    full_suite_failures=[
                        {
                            "nodeid": "tests/unrelated/test_legacy.py::test_old_contract",
                            "status": "failed",
                            "message": "pre-existing unrelated failure",
                        }
                    ]
                ),
            }
        )
    )

    assert payload["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
    assert task.status == "done"
    assert "Scoped acceptance caveat" in run.summary
    assert run.metadata["scoped_acceptance"]["decision"] == "complete_with_caveat"
    assert run.metadata["scoped_acceptance"]["caveats"][0]["classification"] == "unrelated_full_suite_failure"


def test_related_contract_or_safety_failures_block_scoped_completion(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="scoped fix", assignee="worker")

    payload = json.loads(
        kanban_tools._handle_complete(
            {
                "task_id": task_id,
                "summary": "Attempted scoped completion.",
                "acceptance": _green_acceptance(
                    full_suite_failures=[
                        {
                            "nodeid": "tools/kanban_tools.py::test_touched_contract",
                            "status": "failed",
                            "message": "touched file regressed",
                        }
                    ],
                    contract_failures=["review evidence contract failed"],
                    safety_failures=["completion safety gate failed"],
                ),
            }
        )
    )

    assert "error" in payload
    assert "scoped acceptance blocked" in payload["error"]
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
