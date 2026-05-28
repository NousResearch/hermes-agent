import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import project_autopilot
from hermes_cli import kanban_db
from hermes_cli.project_autopilot import bootstrap_project_home, sync_project_home


def _bootstrap_demo_project(tmp_path, *, board_slug="demo-board"):
    project_home = tmp_path / "projects" / "demo"
    bootstrap_project_home(
        slug="demo",
        title="Demo",
        goal="Make demo restartable",
        board_slug=board_slug,
        root_task_id="t_root",
        project_home=project_home,
        repo_org="summation",
        repo_name="Code",
        canonical_checkout=Path("/Users/vsletten/src/summation/Code/main"),
        final_branch="feat/demo-pr",
        source_plan=None,
    )
    return project_home


def test_sync_project_home_rewrites_project_files_from_board_truth(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind, workspace_path, branch_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                "Project root body",
                "codexapp",
                "done",
                0,
                "tester",
                1,
                "scratch",
                None,
                None,
            ),
        )
        first_child = kanban_db.create_task(
            conn,
            title="Implement slice",
            body="Slice body",
            assignee="codexapp",
            created_by="tester",
            workspace_kind="worktree",
            workspace_path="/Users/vsletten/src/summation/Code/feat/demo-pr",
            branch_name="feat/demo-pr",
            parents=["t_root"],
        )
        second_child = kanban_db.create_task(
            conn,
            title="Review slice",
            body="Review body",
            assignee="reviewer",
            created_by="tester",
            parents=[first_child],
        )

        project_home = _bootstrap_demo_project(tmp_path)
        (project_home / "TASKS.md").write_text("stale task cache\n", encoding="utf-8")
        (project_home / "STATUS.md").write_text(
            "# Status: stale\n\n## Next action\n\nstale\n",
            encoding="utf-8",
        )

        doc = sync_project_home(project_home, db_path=db_path)

        saved = json.loads((project_home / "project.json").read_text())
        assert saved["task_graph"]["nodes"][0]["id"] == "t_root"
        assert {edge["parent"] for edge in saved["task_graph"]["edges"]} == {
            "t_root",
            first_child,
        }
        assert saved["workspace_contracts"][first_child] == {
            "workspace_kind": "worktree",
            "workspace_path": "/Users/vsletten/src/summation/Code/feat/demo-pr",
            "branch_name": "feat/demo-pr",
        }
        assert doc["updated_at"] >= doc["created_at"]

        tasks_md = (project_home / "TASKS.md").read_text(encoding="utf-8")
        assert "Root project" in tasks_md
        assert "Implement slice" in tasks_md
        assert "Review slice" in tasks_md
        assert f"`{first_child}`" in tasks_md
        assert "stale task cache" not in tasks_md

        status_md = (project_home / "STATUS.md").read_text(encoding="utf-8")
        assert "Tasks: 3 total" in status_md
        assert "Next executable task" in status_md
        assert f"`{first_child}`" in status_md
        assert "stale" not in status_md

        handoff_md = (project_home / "SESSION-HANDOFF.md").read_text(
            encoding="utf-8"
        )
        assert "Reconcile `TASKS.md` and `STATUS.md` from board truth" in handoff_md
        assert "Task graph snapshot" in handoff_md
    finally:
        conn.close()


def test_sync_project_home_includes_unlinked_live_board_tasks(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                "Project root body",
                "codexapp",
                "done",
                0,
                "tester",
                1,
                "scratch",
            ),
        )
        implementation_id = kanban_db.create_task(
            conn,
            title="Implement board execution slice",
            body="Implementation body",
            assignee="codexapp",
            created_by="tester",
        )
        conn.execute(
            "UPDATE tasks SET status = 'done' WHERE id = ?",
            (implementation_id,),
        )
        review_id = kanban_db.create_task(
            conn,
            title="Review board execution slice",
            body="Review body",
            assignee="reviewer",
            created_by="tester",
            parents=[implementation_id],
        )
        conn.execute(
            "UPDATE tasks SET status = 'review' WHERE id = ?",
            (review_id,),
        )

        project_home = _bootstrap_demo_project(tmp_path)

        sync_project_home(project_home, db_path=db_path)

        saved = json.loads((project_home / "project.json").read_text())
        node_ids = {node["id"] for node in saved["task_graph"]["nodes"]}
        assert {"t_root", implementation_id, review_id} <= node_ids
        assert {"parent": implementation_id, "child": review_id} in saved[
            "task_graph"
        ]["edges"]

        tasks_md = (project_home / "TASKS.md").read_text(encoding="utf-8")
        assert implementation_id in tasks_md
        assert review_id in tasks_md
        assert "Implement board execution slice" in tasks_md
        assert "Review board execution slice" in tasks_md

        status_md = (project_home / "STATUS.md").read_text(encoding="utf-8")
        assert "Tasks: 3 total" in status_md
        assert review_id in status_md
    finally:
        conn.close()


def test_sync_project_home_blocks_when_terminal_non_root_task_missing_report(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                "Project root body",
                "codexapp",
                "done",
                0,
                "tester",
                1,
                "scratch",
            ),
        )
        child_id = kanban_db.create_task(
            conn,
            title="Completed implementation",
            body="Implementation body",
            assignee="codexapp",
            created_by="tester",
            parents=["t_root"],
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (child_id,))

        project_home = _bootstrap_demo_project(tmp_path)

        doc = sync_project_home(project_home, db_path=db_path)

        assert doc["state"] == "BLOCKED_PROCESS"
        assert project_autopilot.terminal_tasks_missing_reports(
            project_home, doc["task_graph"]["nodes"]
        ) == [child_id]
        assert doc["invariant_failures"][-1] == {
            "type": "terminal_tasks_missing_reports",
            "task_ids": [child_id],
        }

        status_md = (project_home / "STATUS.md").read_text(encoding="utf-8")
        assert f"Blocker: missing terminal task status reports: `{child_id}`" in status_md
        assert status_md.index("Blocker:") < status_md.index("## Next action")
    finally:
        conn.close()


def test_sync_project_home_accepts_non_empty_terminal_task_report(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                "Project root body",
                "codexapp",
                "done",
                0,
                "tester",
                1,
                "scratch",
            ),
        )
        child_id = kanban_db.create_task(
            conn,
            title="Completed implementation",
            body="Implementation body",
            assignee="codexapp",
            created_by="tester",
            parents=["t_root"],
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (child_id,))

        project_home = _bootstrap_demo_project(tmp_path)
        (project_home / "status" / f"{child_id}.md").write_text(
            "# Completed implementation\n\nVerified and pushed.\n",
            encoding="utf-8",
        )

        doc = sync_project_home(project_home, db_path=db_path)

        assert project_autopilot.terminal_tasks_missing_reports(
            project_home, doc["task_graph"]["nodes"]
        ) == []
        assert doc["state"] != "BLOCKED_PROCESS"
        assert doc["invariant_failures"] == []
        assert "Blocker: none" in (project_home / "STATUS.md").read_text(
            encoding="utf-8"
        )
    finally:
        conn.close()


def test_sync_project_home_clears_stale_terminal_task_report_failure(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                "Project root body",
                "codexapp",
                "done",
                0,
                "tester",
                1,
                "scratch",
            ),
        )
        child_id = kanban_db.create_task(
            conn,
            title="Completed implementation",
            body="Implementation body",
            assignee="codexapp",
            created_by="tester",
            parents=["t_root"],
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (child_id,))

        project_home = _bootstrap_demo_project(tmp_path)
        first_doc = sync_project_home(project_home, db_path=db_path)
        assert first_doc["state"] == "BLOCKED_PROCESS"
        assert first_doc["invariant_failures"] == [
            {"type": "terminal_tasks_missing_reports", "task_ids": [child_id]}
        ]

        (project_home / "status" / f"{child_id}.md").write_text(
            "# Completed implementation\n\nVerified and pushed.\n",
            encoding="utf-8",
        )

        repaired_doc = sync_project_home(project_home, db_path=db_path)

        assert repaired_doc["state"] == "PLANNED"
        assert repaired_doc["invariant_failures"] == []
        assert "Blocker: none" in (project_home / "STATUS.md").read_text(
            encoding="utf-8"
        )
    finally:
        conn.close()


@pytest.mark.parametrize("schema_version", [None, "legacy-project/v0"])
def test_sync_project_home_upgrades_legacy_project_json_before_validation(
    tmp_path, schema_version
):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                "Project root body",
                "codexapp",
                "ready",
                0,
                "tester",
                1,
                "scratch",
            ),
        )

        project_home = _bootstrap_demo_project(tmp_path)
        legacy_doc = {
            "slug": "demo",
            "title": "Demo",
            "goal": "Make demo restartable",
            "board_slug": "demo-board",
            "root_task_id": "t_root",
            "project_home": str(project_home),
            "project_type": "Hermes feature project",
            "state": "PLANNED",
            "created_at": 123,
            "updated_at": 456,
        }
        if schema_version is not None:
            legacy_doc["schema_version"] = schema_version
        (project_home / "project.json").write_text(
            json.dumps(legacy_doc, indent=2) + "\n",
            encoding="utf-8",
        )

        doc = sync_project_home(project_home, db_path=db_path)

        saved = json.loads((project_home / "project.json").read_text())
        assert saved["schema_version"] == "project-autopilot/v0"
        assert saved["project_mode"] == "stacked-slices-one-pr"
        assert saved["slug"] == "demo"
        assert saved["state"] == "PLANNED"
        assert saved["root_task_id"] == "t_root"
        assert saved["task_graph"]["nodes"][0]["id"] == "t_root"
        assert saved["branch_strategy"]["final_branch"]
        assert saved["repo"]["org"]
        assert saved["repo"]["name"]
        assert doc == saved
    finally:
        conn.close()


def test_project_sync_cli_reconciles_home_from_kanban_db(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, workspace_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t_root",
                "Root project",
                None,
                "codexapp",
                "done",
                0,
                "tester",
                1,
                "scratch",
            ),
        )
        child_id = kanban_db.create_task(
            conn,
            title="CLI synced task",
            assignee="codexapp",
            created_by="tester",
            parents=["t_root"],
        )
    finally:
        conn.close()

    project_home = _bootstrap_demo_project(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(tmp_path / ".hermes"),
            "HERMES_KANBAN_DB": str(db_path),
            "PYTHONPATH": str(Path.cwd()),
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "project", "sync", str(project_home)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "SYNCED demo" in result.stdout
    assert child_id in (project_home / "TASKS.md").read_text(encoding="utf-8")
