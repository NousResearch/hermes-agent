"""Tests for append-only project continuity records."""

import json


def test_append_project_continuity_record_writes_jsonl_under_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.project_continuity import append_project_continuity_record

    packet = {
        "session_id": "tip",
        "old_session_id": "root",
        "lineage_root_id": "root",
        "cwd": "N:/CodexProjects/example",
        "git_repo_root": "N:/CodexProjects/example",
        "git_branch": "feature/continuity",
        "title": "Project chat",
        "platform": "cli",
        "compression_count": 2,
        "durable_candidates": {
            "decisions": ["keep compaction enabled"],
            "artifacts": ["N:/Hermes/reports/example.json"],
        },
    }

    path = append_project_continuity_record(packet, event="compression", summary="Compacted")

    assert path.is_file()
    assert path.parent == tmp_path / "reports" / "project-continuity"
    assert "N_CodexProjects_example" in path.name
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "compression"
    assert record["summary"] == "Compacted"
    assert record["session_id"] == "tip"
    assert record["lineage_root_id"] == "root"
    assert record["project"]["cwd"] == "N:/CodexProjects/example"
    assert record["project"]["git_repo_root"] == "N:/CodexProjects/example"
    assert record["decisions"] == ["keep compaction enabled"]
    assert record["artifacts"] == ["N:/Hermes/reports/example.json"]


def test_append_project_continuity_record_uses_cwd_when_repo_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.project_continuity import append_project_continuity_record

    path = append_project_continuity_record({
        "session_id": "s1",
        "cwd": "C:/Users/Alexander Semenov/Documents/Loose Project",
        "durable_candidates": {},
    })

    assert path.name.startswith("C_Users_Alexander_Semenov_Documents_Loose_Project")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["project"]["cwd"] == "C:/Users/Alexander Semenov/Documents/Loose Project"


def test_append_project_continuity_record_returns_none_without_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.project_continuity import append_project_continuity_record

    assert append_project_continuity_record({"session_id": "s1"}) is None
    assert not (tmp_path / "reports" / "project-continuity").exists()
