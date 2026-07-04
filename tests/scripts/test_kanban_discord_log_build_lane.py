import importlib.util
import json
import sys
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_project_model as kpm


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "kanban_discord_log.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("kanban_discord_log_under_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_lane_body(*, slug, title, root_id, stage, run_key):
    return (
        kpm.project_metadata_markers(
            project_hub_slug=slug,
            project_title=title,
            kanban_root_task_id=root_id,
            stage_name=stage,
        )
        + "\nBuild Lane task metadata:\n"
        + f"Project Hub slug: {slug}\n"
        + f"Project title: {title}\n"
        + f"Kanban root task id: {root_id}\n"
        + f"Kanban stage: {stage}\n"
        + f"Run key: {run_key}\n"
    )


def test_fresh_build_lane_runs_get_distinct_forum_posts_with_dsr_starter_metadata(tmp_path, monkeypatch):
    log = _load_script_module()
    db_path = tmp_path / "kanban.db"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(log, "DB_PATH", db_path)
    monkeypatch.setattr(log, "STATE_PATH", state_path)

    slug = "build-implementation-lane"
    title_a = "Run A: Validate Build Lane forum visibility"
    title_b = "Run B: Validate Build Lane forum visibility"
    with kb.connect(db_path=db_path) as conn:
        root_a = kb.create_task(
            conn,
            title=f"Build Lane run: {title_a}",
            body=_build_lane_body(slug=slug, title=title_a, root_id="root-a", stage="build-lane-root", run_key="run-a"),
            assignee="antonetta",
            created_by="build-lane-start",
            idempotency_key="root-a",
        )
        root_b = kb.create_task(
            conn,
            title=f"Build Lane run: {title_b}",
            body=_build_lane_body(slug=slug, title=title_b, root_id="root-b", stage="build-lane-root", run_key="run-b"),
            assignee="antonetta",
            created_by="build-lane-start",
            idempotency_key="root-b",
        )
        conn.execute("UPDATE tasks SET id=?, status='done' WHERE id=?", ("root-a", root_a))
        conn.execute("UPDATE tasks SET id=?, status='done' WHERE id=?", ("root-b", root_b))
        conn.execute(
            "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) VALUES (?, NULL, 'project_stage_started', ?, 100)",
            (
                "root-a",
                json.dumps(
                    {
                        "project_hub_slug": slug,
                        "project_title": title_a,
                        "kanban_root_task_id": "root-a",
                        "stage": "build-lane-root",
                        "run_key": "run-a",
                        "summary": "Build Lane Kanban graph initialized.",
                        "dsr_visible": True,
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) VALUES (?, NULL, 'project_stage_started', ?, 101)",
            (
                "root-b",
                json.dumps(
                    {
                        "project_hub_slug": slug,
                        "project_title": title_b,
                        "kanban_root_task_id": "root-b",
                        "stage": "build-lane-root",
                        "run_key": "run-b",
                        "summary": "Build Lane Kanban graph initialized.",
                        "dsr_visible": True,
                    }
                ),
            ),
        )

    created_threads = []
    posted_messages = []

    def fake_create_thread(parent_channel_id, name, message, dry_run=False, applied_tags=None):
        thread_id = f"thread-{len(created_threads) + 1}"
        created_threads.append({"parent": parent_channel_id, "name": name, "message": message, "id": thread_id})
        return {"id": thread_id, "parent_id": parent_channel_id, "name": name}

    def fake_post(channel_id, content, dry_run=False, components=None):
        posted_messages.append({"channel_id": channel_id, "content": content, "components": components})
        return {"id": f"msg-{len(posted_messages)}", "channel_id": channel_id}

    monkeypatch.setattr(log, "create_thread", fake_create_thread)
    monkeypatch.setattr(log, "post", fake_post)

    # Start after create_task's mechanical created events; this mirrors the
    # watcher already being online when the starter helper emits the explicit
    # Build Lane project-stage events that should create fresh forum posts.
    state = {"last_event_id": 2, "components": {}, "task_aliases": {}, "red_thread_posts": {}}
    processed = log.run_once(state, "general-channel", "kanban-project-runs", "red-channel", limit=20)

    assert processed == 2
    assert state["project_threads"] == {
        f"{slug}:run-a": "thread-1",
        f"{slug}:run-b": "thread-2",
    }
    assert [thread["parent"] for thread in created_threads] == ["kanban-project-runs", "kanban-project-runs"]
    assert [thread["name"] for thread in created_threads] == [title_a, title_b]
    for thread, run_key, root_id in zip(created_threads, ["run-a", "run-b"], ["root-a", "root-b"]):
        starter = thread["message"]
        assert f"Project Hub: `{slug}`" in starter
        assert f"Kanban root: `{root_id}`" in starter
        assert f"Run key: `{run_key}`" in starter
        assert "DSR: project-linked updates with `dsr_visible`/`dsr_include` metadata are eligible" in starter
    assert any("Project Stage Started: Build Lane Kanban graph initialized." in msg["content"] for msg in posted_messages)
