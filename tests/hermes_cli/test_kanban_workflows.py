import argparse
import json

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_workflows import (
    TEMPLATE_CODER_QA_PM_REVIEW,
    create_coder_qa_pm_review_workflow,
)


def test_coder_qa_pm_workflow_builds_dependency_gated_pipeline(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        subscribed: list[str] = []

        created = create_coder_qa_pm_review_workflow(
            conn,
            title="Ship the notification fix",
            body="Implement the code path and verify it end to end.",
            coder_assignee="coder-codex",
            qa_assignee="qa-minimax",
            pm_assignee="pm-deepseek",
            tenant="hermes",
            created_by="pm-deepseek",
            subscribe_task=lambda task_id: subscribed.append(task_id) is None,
        )

        root = kb.get_task(conn, created.root_id)
        coder = kb.get_task(conn, created.coder_task_id)
        qa = kb.get_task(conn, created.qa_task_id)
        pm_review = kb.get_task(conn, created.pm_review_task_id)

        assert root.status == "done"
        assert root.workflow_template_id == TEMPLATE_CODER_QA_PM_REVIEW
        assert root.current_step_key == "root"

        assert coder.status == "ready"
        assert coder.assignee == "coder-codex"
        assert coder.workflow_template_id == TEMPLATE_CODER_QA_PM_REVIEW
        assert coder.current_step_key == "coder"
        assert kb.parent_ids(conn, created.coder_task_id) == [created.root_id]

        assert qa.status == "todo"
        assert qa.assignee == "qa-minimax"
        assert qa.workflow_template_id == TEMPLATE_CODER_QA_PM_REVIEW
        assert qa.current_step_key == "qa"
        assert kb.parent_ids(conn, created.qa_task_id) == [created.coder_task_id]

        assert pm_review.status == "todo"
        assert pm_review.assignee == "pm-deepseek"
        assert pm_review.workflow_template_id == TEMPLATE_CODER_QA_PM_REVIEW
        assert pm_review.current_step_key == "pm_review"
        assert kb.parent_ids(conn, created.pm_review_task_id) == [created.qa_task_id]

        assert subscribed == [
            created.coder_task_id,
            created.qa_task_id,
            created.pm_review_task_id,
        ]
        assert created.subscriptions == {
            created.coder_task_id: True,
            created.qa_task_id: True,
            created.pm_review_task_id: True,
        }
    finally:
        conn.close()


def test_coder_qa_pm_workflow_dispatcher_handoff_promotes_next_step(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_coder_qa_pm_review_workflow(
            conn,
            title="Verify dispatcher handoff",
            coder_assignee="coder",
            qa_assignee="qa",
            pm_assignee="pm",
        )

        kb.complete_task(conn, created.coder_task_id, summary="Coder done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.qa_task_id).status == "ready"
        assert kb.get_task(conn, created.pm_review_task_id).status == "todo"

        kb.complete_task(conn, created.qa_task_id, summary="QA passed")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.pm_review_task_id).status == "ready"
    finally:
        conn.close()


def test_coder_qa_pm_workflow_cli_creates_template_graph(
    monkeypatch, tmp_path, capsys
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    kanban_cli.build_parser(sub)
    args = parser.parse_args([
        "kanban",
        "workflow",
        "coder-qa-pm",
        "CLI flow",
        "--coder",
        "coder",
        "--qa",
        "qa",
        "--pm",
        "pm",
        "--json",
    ])

    assert kanban_cli.kanban_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["template_id"] == TEMPLATE_CODER_QA_PM_REVIEW

    conn = kb.connect()
    try:
        assert kb.get_task(conn, payload["coder_task_id"]).status == "ready"
        assert kb.get_task(conn, payload["qa_task_id"]).status == "todo"
        assert kb.get_task(conn, payload["pm_review_task_id"]).status == "todo"
    finally:
        conn.close()
