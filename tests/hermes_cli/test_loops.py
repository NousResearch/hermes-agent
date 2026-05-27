from __future__ import annotations

import json

from hermes_cli.loops import init_loop, loop_dir, run_loop, status_loop


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


def test_status_loop_reports_not_started_without_creating_state(tmp_path):
    result = status_loop("missing", root=tmp_path)

    assert "Loop: missing" in result.text
    assert "Status: not started" in result.text
    assert "Next: /loop init missing" in result.text
    assert not loop_dir("missing", root=tmp_path).exists()
