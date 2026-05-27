from __future__ import annotations

import json

from hermes_cli.loops import (
    block_story,
    close_loop,
    complete_story,
    init_loop,
    loop_dir,
    loop_text,
    run_loop,
    status_loop,
)


def test_init_loop_creates_simple_aaron_style_state(tmp_path):
    result = init_loop("fruit-loop", root=tmp_path, title="Build Fruit-Loop")

    root = tmp_path / ".hermes" / "loops" / "fruit-loop"
    assert result.slug == "fruit-loop"
    assert result.path == root
    assert (root / "prd.json").exists()
    assert (root / "progress.md").exists()
    assert (root / "status.md").exists()
    assert (root / "archive").is_dir()

    prd = json.loads((root / "prd.json").read_text(encoding="utf-8"))
    assert prd["project"] == "Build Fruit-Loop"
    assert prd["userStories"] == []
    assert prd["status"] == "draft"
    assert "Loop: fruit-loop" in result.text
    assert "Stories: 0 total / 0 pending / 0 passed" in result.text


def test_status_loop_reads_prd_and_counts_pending_stories(tmp_path):
    init_loop("rally", root=tmp_path, title="Rally intake")
    path = loop_dir("rally", root=tmp_path) / "prd.json"
    prd = json.loads(path.read_text(encoding="utf-8"))
    prd["userStories"] = [
        {"id": "S1", "title": "one", "priority": 1, "passes": False},
        {"id": "S2", "title": "two", "priority": 2, "passes": True},
        {"id": "S3", "title": "three", "priority": 3},
    ]
    path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")

    result = status_loop("rally", root=tmp_path)

    assert "Loop: rally" in result.text
    assert "Project: Rally intake" in result.text
    assert "Stories: 3 total / 2 pending / 1 passed" in result.text
    assert "Next: /loop run rally" in result.text
    assert (loop_dir("rally", root=tmp_path) / "status.md").read_text(encoding="utf-8") == result.text + "\n"


def test_run_loop_selects_first_pending_story_and_returns_execution_prompt(tmp_path):
    init_loop("ship", root=tmp_path, title="Ship feature")
    path = loop_dir("ship", root=tmp_path) / "prd.json"
    prd = json.loads(path.read_text(encoding="utf-8"))
    prd["description"] = "Make /loop useful everywhere."
    prd["userStories"] = [
        {
            "id": "S2",
            "title": "second",
            "description": "lower priority",
            "priority": 2,
            "passes": False,
            "acceptanceCriteria": ["not first"],
        },
        {
            "id": "S1",
            "title": "first",
            "description": "higher priority",
            "priority": 1,
            "passes": False,
            "acceptanceCriteria": ["prompt includes acceptance"],
        },
    ]
    path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")

    result = run_loop("ship", root=tmp_path)

    assert "Story: S1 — first" in result.text
    assert "prompt includes acceptance" in result.text
    assert "Do one story only" in result.text
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["userStories"][1]["status"] == "running"
    assert "Running: S1 — first" in (loop_dir("ship", root=tmp_path) / "status.md").read_text(encoding="utf-8")


def test_complete_story_marks_running_story_passed_and_updates_progress(tmp_path):
    init_loop("ship", root=tmp_path, title="Ship feature")
    path = loop_dir("ship", root=tmp_path)
    prd_path = path / "prd.json"
    prd = json.loads(prd_path.read_text(encoding="utf-8"))
    prd["userStories"] = [
        {"id": "S1", "title": "first", "priority": 1, "passes": False, "status": "running"},
        {"id": "S2", "title": "second", "priority": 2, "passes": False},
    ]
    prd_path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")

    result = complete_story("ship", root=tmp_path, note="verified with focused tests")

    updated = json.loads(prd_path.read_text(encoding="utf-8"))
    assert updated["userStories"][0]["passes"] is True
    assert updated["userStories"][0]["status"] == "passed"
    assert "completedAt" in updated["userStories"][0]
    assert "Stories: 2 total / 1 pending / 1 passed" in result.text
    progress = (path / "progress.md").read_text(encoding="utf-8")
    assert "### Completed: S1 — first" in progress
    assert "verified with focused tests" in progress


def test_block_story_marks_story_blocked_and_run_skips_it(tmp_path):
    init_loop("ship", root=tmp_path, title="Ship feature")
    path = loop_dir("ship", root=tmp_path)
    prd_path = path / "prd.json"
    prd = json.loads(prd_path.read_text(encoding="utf-8"))
    prd["userStories"] = [
        {"id": "S1", "title": "blocked first", "priority": 1, "passes": False},
        {"id": "S2", "title": "next runnable", "priority": 2, "passes": False},
    ]
    prd_path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")

    blocked = block_story("ship", root=tmp_path, story_id="S1", note="waiting on API key")
    run = run_loop("ship", root=tmp_path)

    updated = json.loads(prd_path.read_text(encoding="utf-8"))
    assert updated["userStories"][0]["status"] == "blocked"
    assert updated["userStories"][0]["blockReason"] == "waiting on API key"
    assert "Blocked: 1" in blocked.text
    assert "Stories: 2 total / 1 pending / 0 passed" in blocked.text
    assert "Story: S2 — next runnable" in run.text
    progress = (path / "progress.md").read_text(encoding="utf-8")
    assert "### Blocked: S1 — blocked first" in progress
    assert "waiting on API key" in progress


def test_close_loop_marks_closed_and_archives_state(tmp_path):
    init_loop("done", root=tmp_path, title="Done loop")

    result = close_loop("done", root=tmp_path)

    path = loop_dir("done", root=tmp_path)
    prd = json.loads((path / "prd.json").read_text(encoding="utf-8"))
    archives = [child for child in (path / "archive").iterdir() if child.is_dir()]
    assert prd["status"] == "closed"
    assert "Status: closed" in result.text
    assert "Archive:" in result.text
    assert len(archives) == 1
    assert (archives[0] / "prd.json").exists()


def test_closed_loop_does_not_run_complete_or_block_stories(tmp_path):
    init_loop("done", root=tmp_path, title="Done loop")
    path = loop_dir("done", root=tmp_path)
    prd_path = path / "prd.json"
    prd = json.loads(prd_path.read_text(encoding="utf-8"))
    prd["userStories"] = [{"id": "S1", "title": "first", "priority": 1}]
    prd_path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")
    close_loop("done", root=tmp_path)

    run = run_loop("done", root=tmp_path)
    completed = complete_story("done", root=tmp_path, story_id="S1", note="should not mutate")
    blocked = block_story("done", root=tmp_path, story_id="S1", note="should not mutate")

    updated = json.loads(prd_path.read_text(encoding="utf-8"))
    assert "Status: closed" in run.text
    assert "Status: closed" in completed.text
    assert "Status: closed" in blocked.text
    assert updated["status"] == "closed"
    assert updated["userStories"][0].get("status") is None
    assert updated["userStories"][0].get("passes") is None


def test_status_loop_reports_not_started_without_creating_state(tmp_path):
    result = status_loop("missing", root=tmp_path)

    assert "Loop: missing" in result.text
    assert "Status: not started" in result.text
    assert "Next: /loop init missing" in result.text
    assert not loop_dir("missing", root=tmp_path).exists()


def test_loop_text_usage_lists_all_v1_subcommands(tmp_path):
    result = loop_text("wat", root=tmp_path)

    assert result == "Usage: /loop [init|run|status|complete|block|close] <slug>"
