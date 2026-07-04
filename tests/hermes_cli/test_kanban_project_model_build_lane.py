import json
import sqlite3

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_project_model as kpm


def _connect_initialized(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path=db_path)
    return conn


def test_build_lane_starter_graph_has_project_thread_starter_metadata(tmp_path):
    with _connect_initialized(tmp_path) as conn:
        root_body = kpm.project_metadata_markers(
            project_hub_slug="build-implementation-lane",
            project_title="Run 3: Validate Build Lane forum and DSR visibility",
            kanban_root_task_id="t_root",
            stage_name="build-lane-root",
        )
        child_body = kpm.project_metadata_markers(
            project_hub_slug="build-implementation-lane",
            project_title="Run 3: Validate Build Lane forum and DSR visibility",
            kanban_root_task_id="t_root",
            stage_name="config-implementation",
        ) + "Run key: run-3-forum-dsr-visibility\n"
        root_id = kb.create_task(conn, title="Build Lane run: Run 3", body=root_body, assignee="antonetta", idempotency_key="root")
        conn.execute("UPDATE tasks SET id=?, status='done' WHERE id=?", ("t_root", root_id))
        child_id = kb.create_task(conn, title="Implement config change", body=child_body, assignee="forge", parents=["t_root"], idempotency_key="child")

        project = kpm.resolve_project_context(conn, child_id, {"stage": "config-implementation"})
        starter = kpm.format_project_thread_starter(project)

    assert project["project_hub_slug"] == "build-implementation-lane"
    assert project["project_title"] == "Run 3: Validate Build Lane forum and DSR visibility"
    assert project["kanban_root_task_id"] == "t_root"
    assert project["run_key"] == "run-3-forum-dsr-visibility"
    assert kpm.project_thread_key(project) == "build-implementation-lane:run-3-forum-dsr-visibility"
    assert child_id in project["task_ids"]
    assert "Project Hub: `build-implementation-lane`" in starter
    assert "Kanban root: `t_root`" in starter
    assert "DSR: project-linked updates with `dsr_visible`/`dsr_include` metadata are eligible" in starter


def test_dsr_project_activity_exposes_structured_build_lane_metadata(tmp_path):
    with _connect_initialized(tmp_path) as conn:
        body = kpm.project_metadata_markers(
            project_hub_slug="build-implementation-lane",
            project_title="Run 3: Validate Build Lane forum and DSR visibility",
            kanban_root_task_id="t_root",
            stage_name="config-implementation",
        ) + "Run key: run-3-forum-dsr-visibility\n"
        task_id = kb.create_task(conn, title="Implement config change", body=body, assignee="forge")
        metadata = {
            "dsr_visible": True,
            "dsr_summary": "Validated Build Lane forum visibility.",
            "changed_files": ["hermes_cli/kanban_project_model.py"],
        }
        conn.execute("UPDATE tasks SET status='done', completed_at=200 WHERE id=?", (task_id,))
        conn.execute(
            "INSERT INTO task_runs(task_id, profile, status, started_at, ended_at, outcome, summary, metadata) VALUES (?, 'forge', 'done', 100, 200, 'completed', 'implementation complete', ?)",
            (task_id, json.dumps(metadata)),
        )

        rows = kpm.dsr_project_activity(conn, 0, 300)

    assert len(rows) == 1
    row = rows[0]
    assert row["project_hub_slug"] == "build-implementation-lane"
    assert row["project_title"] == "Run 3: Validate Build Lane forum and DSR visibility"
    assert row["kanban_root_task_id"] == "t_root"
    assert row["stage_name"] == "config-implementation"
    assert row["run_key"] == "run-3-forum-dsr-visibility"
    assert row["dsr_visible"] is True
    assert row["summary"] == "Validated Build Lane forum visibility."
    assert row["metadata"]["changed_files"] == ["hermes_cli/kanban_project_model.py"]
