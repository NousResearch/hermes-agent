from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import projects_db
from tools import roadmaps_tools
from tools.registry import registry


def seed(path: Path) -> None:
    conn = projects_db.connect(path)
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES ('p', 'p', 'P', 1)")
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", "Roadmap", None, "in_progress", 1, "agent", "agent", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "validated", "src", None, "agent", 1, None),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "n-ready", None, "step", "Ready", None, "ready", 0, None, None, 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "n-running", None, "step", "Running", None, "in_progress", 10, "agent", None, 1, 1),
    )
    conn.commit()
    conn.close()


def test_read_tools_are_registered_in_the_roadmaps_toolset():
    roadmap_list_entry = registry.get_entry("roadmap_list")
    roadmap_context_entry = registry.get_entry("roadmap_context")
    assert roadmap_list_entry is not None
    assert roadmap_context_entry is not None
    assert roadmap_list_entry.toolset == "roadmaps"
    assert roadmap_context_entry.toolset == "roadmaps"


def test_roadmap_list_reads_the_active_profile_store_without_mutation(tmp_path: Path, monkeypatch):
    path = tmp_path / "projects.db"
    seed(path)
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: tmp_path)
    before = path.read_bytes()

    result = json.loads(roadmaps_tools.roadmap_list("profile", "p"))

    assert result["scope"] == {"profile_id": "profile", "project_id": "p"}
    assert result["roadmaps"][0]["roadmap_id"] == "r"
    assert path.read_bytes() == before


def test_roadmap_context_requires_explicit_complete_scope(tmp_path: Path, monkeypatch):
    path = tmp_path / "projects.db"
    seed(path)
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: tmp_path)

    result = json.loads(roadmaps_tools.roadmap_context("profile", "p", "r"))

    assert result["found"] is True
    assert result["scope"] == {"profile_id": "profile", "project_id": "p", "roadmap_id": "r"}

    with pytest.raises(ValueError, match="profile_id"):
        roadmaps_tools.roadmap_context("", "p", "r")


def test_roadmap_tools_do_not_create_a_missing_store(tmp_path: Path, monkeypatch):
    home = tmp_path / "profile-home"
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: home)

    result = json.loads(roadmaps_tools.roadmap_list("profile"))

    assert result == {"roadmaps": [], "scope": {"profile_id": "profile"}}
    assert not home.exists()


def test_mutation_tools_are_registered_in_the_roadmaps_toolset():
    for name in (
        "roadmap_claim_node",
        "roadmap_update_progress",
        "roadmap_complete_node",
        "roadmap_block_node",
        "roadmap_unblock_node",
    ):
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.toolset == "roadmaps"


def test_claim_mutation_round_trip(tmp_path: Path, monkeypatch):
    path = tmp_path / "projects.db"
    seed(path)
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: tmp_path)

    result = json.loads(
        roadmaps_tools.roadmap_claim_node("profile", "p", "r", "n-ready", "agent-b", 1)
    )

    assert result["success"] is True
    assert result["node"]["state"] == "in_progress"
    assert result["node"]["owner_agent"] == "agent-b"

    context = json.loads(roadmaps_tools.roadmap_context("profile", "p", "r"))
    claimed = [n for v in context["roadmap"]["versions"] for n in v["nodes"] if n["node_id"] == "n-ready"][0]
    assert claimed["state"] == "in_progress"


def test_mutation_rejects_stale_version_without_mutation(tmp_path: Path, monkeypatch):
    path = tmp_path / "projects.db"
    seed(path)
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: tmp_path)
    before = path.read_bytes()

    result = json.loads(
        roadmaps_tools.roadmap_claim_node("profile", "p", "r", "n-ready", "agent-b", 99)
    )

    assert result == {"success": False, "error": "stale_roadmap_version", "detail": result["detail"]}
    assert path.read_bytes() == before


def test_block_and_unblock_mutation_persist_and_clear_reason(tmp_path: Path, monkeypatch):
    path = tmp_path / "projects.db"
    seed(path)
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: tmp_path)

    blocked = json.loads(
        roadmaps_tools.roadmap_block_node("profile", "p", "r", "n-running", "agent", "missing dep", 1)
    )
    assert blocked["node"]["state"] == "blocked"
    assert blocked["node"]["block_reason"] == "missing dep"

    unblocked = json.loads(
        roadmaps_tools.roadmap_unblock_node("profile", "p", "r", "n-running", "agent", 1)
    )
    assert unblocked["node"]["state"] == "in_progress"
    assert unblocked["node"]["block_reason"] is None


def test_progress_mutation_validates_range(tmp_path: Path, monkeypatch):
    path = tmp_path / "projects.db"
    seed(path)
    monkeypatch.setattr(roadmaps_tools, "get_hermes_home", lambda: tmp_path)

    ok = json.loads(
        roadmaps_tools.roadmap_update_progress("profile", "p", "r", "n-running", "agent", 45, 1)
    )
    assert ok["success"] is True
    assert ok["node"]["progress"] == 45

    bad = json.loads(
        roadmaps_tools.roadmap_update_progress("profile", "p", "r", "n-running", "agent", 101, 1)
    )
    assert bad["success"] is False
