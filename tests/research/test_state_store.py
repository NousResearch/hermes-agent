"""Tests for research.state_store."""

import json

import yaml

from research.models import InboxItem, ResearchProject, RunSpec
from research.state_store import ResearchStateStore


def test_init_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Alpha", brief="explore ideas")
    assert project.name == "Alpha"
    assert project.project_id.startswith("proj_")
    layout = store.layout(project.project_id)
    assert layout["project"].exists()
    assert layout["loop_state"].exists()
    assert layout["ideas"].exists()


def test_init_project_custom_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Beta", project_id="proj_custom")
    assert project.project_id == "proj_custom"


def test_init_project_auto_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Gamma", project_id="myslug")
    assert project.project_id == "proj_myslug"


def test_save_and_load_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test", brief="test project")
    project.phase = "training"
    store.save_project(project)
    loaded = store.load_project(project.project_id)
    assert loaded.phase == "training"
    assert loaded.name == "Test"


def test_list_projects_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    assert store.list_projects() == []


def test_list_projects_ordered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    store.init_project(name="First", project_id="proj_a")
    store.init_project(name="Second", project_id="proj_b")
    projects = store.list_projects()
    assert len(projects) == 2
    names = {p.name for p in projects}
    assert names == {"First", "Second"}


def test_latest_project_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    assert store.latest_project_id() is None
    store.init_project(name="P1", project_id="proj_first")
    # latest_project_id returns the first from the sorted list
    assert store.latest_project_id() is not None


def test_append_and_load_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    store.append_jsonl(project.project_id, "ideas", {"title": "Idea 1"})
    store.append_jsonl(project.project_id, "ideas", {"title": "Idea 2"})
    rows = store.load_jsonl(project.project_id, "ideas")
    assert len(rows) == 2
    assert rows[0]["title"] == "Idea 1"
    assert rows[1]["title"] == "Idea 2"


def test_append_jsonl_invalid_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    import pytest
    with pytest.raises(ValueError, match="Unsupported JSONL store"):
        store.append_jsonl(project.project_id, "invalid_collection", {"data": 1})


def test_save_and_load_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    payload = {"title": "Exp1", "method": "sft", "hypothesis_id": "hyp_abc"}
    path = store.save_experiment(project.project_id, "exp_001", payload)
    assert path.exists()
    loaded = store.load_experiment(project.project_id, "exp_001")
    assert loaded["title"] == "Exp1"


def test_list_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    store.save_experiment(project.project_id, "exp_a", {"title": "A"})
    store.save_experiment(project.project_id, "exp_b", {"title": "B"})
    exps = store.list_experiments(project.project_id)
    assert len(exps) == 2


def test_save_and_load_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    run = RunSpec(
        project_id=project.project_id,
        base_model="llama-8b",
        dataset_path="/data.jsonl",
    )
    path = store.save_run(run)
    assert path.exists()
    loaded = store.load_run(project.project_id, run.run_id)
    assert loaded["base_model"] == "llama-8b"


def test_list_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    for i in range(3):
        store.save_run(RunSpec(
            project_id=project.project_id,
            base_model="llama-8b",
            dataset_path="/data.jsonl",
        ))
    runs = store.list_runs(project.project_id)
    assert len(runs) == 3


def test_loop_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    state = store.load_loop_state(project.project_id)
    assert state["project_id"] == project.project_id
    state["custom_key"] = "custom_value"
    store.save_loop_state(project.project_id, state)
    reloaded = store.load_loop_state(project.project_id)
    assert reloaded["custom_key"] == "custom_value"


def test_inbox_item_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    item = InboxItem(project_id=project.project_id, title="Run done", summary="Success")
    path = store.add_inbox_item(project.project_id, item)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Run done" in content
    unread = store.unread_inbox(project.project_id)
    assert len(unread) == 1


def test_write_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    path = store.write_report(project.project_id, "# Report\nAll good.", iteration=1)
    assert path.exists()
    assert path.name == "iteration-1.md"
    assert "All good" in path.read_text(encoding="utf-8")


def test_write_report_auto_increment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    p1 = store.write_report(project.project_id, "First")
    p2 = store.write_report(project.project_id, "Second")
    assert p1.name == "iteration-1.md"
    assert p2.name == "iteration-2.md"


def test_status_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test", brief="testing")
    store.append_jsonl(project.project_id, "ideas", {"title": "Idea"})
    store.append_jsonl(project.project_id, "literature", {"title": "Paper"})
    summary = store.status_summary(project.project_id)
    assert summary["ideas"] == 1
    assert summary["literature_notes"] == 1
    assert summary["run_count"] == 0


def test_update_project_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    store.update_project_runs(project.project_id, ["run_a", "run_b", "run_a"])
    loaded = store.load_project(project.project_id)
    # Deduped, order preserved
    assert loaded.active_run_ids == ["run_a", "run_b"]
