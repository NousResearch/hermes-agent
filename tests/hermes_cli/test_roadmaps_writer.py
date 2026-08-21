"""Tests for the authorized, versioned Roadmaps execution writer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import projects_db
from hermes_cli.roadmaps_writer import (
    InvalidRoadmapTransitionError,
    RoadmapNodeNotFoundError,
    RoadmapNotFoundError,
    RoadmapVersionNotFoundError,
    RoadmapsWriter,
    StaleRoadmapVersionError,
)


def seed(path: Path, *, profile: str = "prof", active_version: int | None = 1) -> None:
    conn = projects_db.connect(path)
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES ('p1', 'one', 'One', 1)")
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", "Roadmap", None, "in_progress", active_version, "a", "b", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", 1, "validated", "src", None, "a", 1, None),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", 1, "n-ready", None, "step", "Ready", None, "ready", 0, None, None, 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", 1, "n-planned", None, "step", "Planned", None, "planned", 0, None, None, 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", 1, "n-running", None, "step", "Running", None, "in_progress", 10, "agent-a", None, 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", 1, "n-blocked", None, "step", "Blocked", None, "blocked", 30, "agent-a", "stuck", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, "p1", "r1", 1, "n-done", None, "step", "Done", None, "completed", 100, "agent-a", None, 1, 1),
    )
    conn.commit()
    conn.close()


def node_state(path: Path, node_id: str) -> dict:
    conn = projects_db.connect(path)
    row = conn.execute(
        "SELECT state, progress, owner_agent, block_reason, updated_at FROM roadmap_nodes WHERE node_id=?",
        (node_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def session_rows(path: Path) -> list[dict]:
    conn = projects_db.connect(path)
    rows = [dict(row) for row in conn.execute(
        "SELECT * FROM roadmap_sessions ORDER BY created_at, stored_session_id"
    )]
    conn.close()
    return rows


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "projects.db"
    seed(path)
    return path


def test_claim_node_moves_ready_to_in_progress_and_assigns_owner(db_path: Path):
    writer = RoadmapsWriter(db_path)
    result = writer.claim_node("prof", "p1", "r1", "n-ready", "agent-b", 1)

    assert result["success"] is True
    assert result["node"]["state"] == "in_progress"
    assert result["node"]["owner_agent"] == "agent-b"
    assert node_state(db_path, "n-ready")["state"] == "in_progress"
    assert node_state(db_path, "n-ready")["owner_agent"] == "agent-b"


def test_claim_node_rejects_transition_from_planned(db_path: Path):
    with pytest.raises(InvalidRoadmapTransitionError):
        RoadmapsWriter(db_path).claim_node("prof", "p1", "r1", "n-planned", "agent-b", 1)
    # Nothing changed: planned stays planned.
    assert node_state(db_path, "n-planned")["state"] == "planned"


def test_claim_node_rejects_stale_expected_version(db_path: Path):
    with pytest.raises(StaleRoadmapVersionError):
        RoadmapsWriter(db_path).claim_node("prof", "p1", "r1", "n-ready", "agent-b", 99)
    assert node_state(db_path, "n-ready")["state"] == "ready"


def test_update_progress_requires_in_progress_and_valid_range(db_path: Path):
    writer = RoadmapsWriter(db_path)
    result = writer.update_progress("prof", "p1", "r1", "n-running", "agent-a", 60, 1)
    assert result["node"]["progress"] == 60
    assert node_state(db_path, "n-running")["progress"] == 60

    with pytest.raises(InvalidRoadmapTransitionError):
        writer.update_progress("prof", "p1", "r1", "n-ready", "agent-a", 10, 1)
    for bad in (-1, 101, 1.5, "50"):
        with pytest.raises(ValueError, match="progress"):
            writer.update_progress("prof", "p1", "r1", "n-running", "agent-a", bad, 1)


def test_complete_node_from_in_progress_and_from_blocked(db_path: Path):
    writer = RoadmapsWriter(db_path)
    result = writer.complete_node("prof", "p1", "r1", "n-running", "agent-a", 1)
    assert result["node"]["state"] == "completed"
    assert result["node"]["progress"] == 100

    writer.complete_node("prof", "p1", "r1", "n-blocked", "agent-a", 1)
    assert node_state(db_path, "n-blocked")["state"] == "completed"
    assert node_state(db_path, "n-blocked")["block_reason"] is None


def test_block_node_persists_reason_and_unblock_clears_it(db_path: Path):
    writer = RoadmapsWriter(db_path)
    result = writer.block_node("prof", "p1", "r1", "n-running", "agent-a", "missing dependency", 1)
    assert result["node"]["state"] == "blocked"
    assert result["node"]["block_reason"] == "missing dependency"
    assert node_state(db_path, "n-running")["block_reason"] == "missing dependency"

    with pytest.raises(ValueError, match="reason"):
        writer.block_node("prof", "p1", "r1", "n-running", "agent-a", "   ", 1)

    writer.unblock_node("prof", "p1", "r1", "n-running", "agent-a", 1)
    assert node_state(db_path, "n-running")["state"] == "in_progress"
    assert node_state(db_path, "n-running")["block_reason"] is None


def test_blocking_an_already_completed_node_is_rejected(db_path: Path):
    with pytest.raises(InvalidRoadmapTransitionError):
        RoadmapsWriter(db_path).block_node("prof", "p1", "r1", "n-done", "agent-a", "why", 1)
    assert node_state(db_path, "n-done")["state"] == "completed"


def test_advance_node_moves_planned_to_ready(db_path: Path):
    writer = RoadmapsWriter(db_path)
    result = writer.advance_node("prof", "p1", "r1", "n-planned", "agent-b", 1)

    assert result["success"] is True
    assert result["node"]["state"] == "ready"
    assert node_state(db_path, "n-planned")["state"] == "ready"


def test_advance_node_rejects_transition_from_ready_and_in_progress(db_path: Path):
    writer = RoadmapsWriter(db_path)
    with pytest.raises(InvalidRoadmapTransitionError):
        writer.advance_node("prof", "p1", "r1", "n-ready", "agent-b", 1)
    with pytest.raises(InvalidRoadmapTransitionError):
        writer.advance_node("prof", "p1", "r1", "n-running", "agent-b", 1)
    assert node_state(db_path, "n-ready")["state"] == "ready"
    assert node_state(db_path, "n-running")["state"] == "in_progress"


def test_advance_node_rejects_stale_expected_version(db_path: Path):
    with pytest.raises(StaleRoadmapVersionError):
        RoadmapsWriter(db_path).advance_node("prof", "p1", "r1", "n-planned", "agent-b", 99)
    assert node_state(db_path, "n-planned")["state"] == "planned"


def test_missing_roadmap_node_or_roadmap_raises_structured_errors(db_path: Path):
    writer = RoadmapsWriter(db_path)
    with pytest.raises(RoadmapNodeNotFoundError):
        writer.claim_node("prof", "p1", "r1", "n-missing", "agent-a", 1)
    with pytest.raises(RoadmapNotFoundError):
        writer.claim_node("prof", "p1", "r2", "n-ready", "agent-a", 1)


def test_roadmap_without_active_version_cannot_be_mutated(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path, active_version=None)
    with pytest.raises(RoadmapNotFoundError):
        RoadmapsWriter(path).claim_node("prof", "p1", "r1", "n-ready", "agent-a", 1)


def test_failed_mutation_rolls_back_without_touching_the_store(db_path: Path):
    before = db_path.read_bytes()
    with pytest.raises(StaleRoadmapVersionError):
        RoadmapsWriter(db_path).claim_node("prof", "p1", "r1", "n-ready", "agent-b", 99)
    assert db_path.read_bytes() == before
    assert node_state(db_path, "n-ready")["state"] == "ready"


def test_writer_updates_roadmap_timestamp_and_actor(db_path: Path):
    RoadmapsWriter(db_path).claim_node("prof", "p1", "r1", "n-ready", "agent-b", 1)
    conn = projects_db.connect(db_path)
    row = conn.execute(
        "SELECT updated_by, updated_at FROM roadmaps WHERE roadmap_id='r1'"
    ).fetchone()
    conn.close()
    assert row["updated_by"] == "agent-b"
    assert row["updated_at"] > 1


def test_attach_vision_session_on_fresh_roadmap_uses_expected_zero(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path, active_version=None)

    result = RoadmapsWriter(path).attach_session(
        "prof", "p1", "r1", "stored-vision-1", "pierre", 0
    )

    assert result["success"] is True
    assert result["scope"] == {
        "profile_id": "prof", "project_id": "p1", "roadmap_id": "r1",
    }
    assert result["session"] == {
        "stored_session_id": "stored-vision-1", "kind": "vision",
        "node_id": None, "plan_version": None, "state": "active",
        "actor": "pierre", "created_at": result["session"]["created_at"],
        "updated_at": result["session"]["updated_at"],
    }
    assert "runtime_session_id" not in result["session"]
    assert session_rows(path)[0]["state"] == "active"


def test_attach_vision_session_replacement_closes_previous_atomically(db_path: Path):
    writer = RoadmapsWriter(db_path)
    writer.attach_session("prof", "p1", "r1", "stored-old", "pierre", 1)

    result = writer.attach_session(
        "prof", "p1", "r1", "stored-new", "pierre", 1
    )

    assert result["session"]["stored_session_id"] == "stored-new"
    assert result["session"]["state"] == "active"
    assert {
        row["stored_session_id"]: row["state"] for row in session_rows(db_path)
    } == {"stored-new": "active", "stored-old": "closed"}


def test_attach_same_durable_session_is_idempotent(db_path: Path):
    writer = RoadmapsWriter(db_path)
    first = writer.attach_session(
        "prof", "p1", "r1", "stored-same", "pierre", 1
    )
    second = writer.attach_session(
        "prof", "p1", "r1", "stored-same", "pierre", 1
    )

    assert second["session"] == first["session"]
    assert len(session_rows(db_path)) == 1


def test_attach_session_plan_version_must_exist_in_exact_scope(db_path: Path):
    writer = RoadmapsWriter(db_path)
    result = writer.attach_session(
        "prof", "p1", "r1", "stored-plan", "pierre", 1, plan_version=1
    )
    assert result["session"]["plan_version"] == 1

    with pytest.raises(RoadmapVersionNotFoundError):
        writer.attach_session(
            "prof", "p1", "r1", "stored-missing-plan", "pierre", 1,
            plan_version=99,
        )
    assert [(row["stored_session_id"], row["state"]) for row in session_rows(db_path)] == [
        ("stored-plan", "active")
    ]


def test_attach_session_records_actor_on_roadmap(db_path: Path):
    RoadmapsWriter(db_path).attach_session(
        "prof", "p1", "r1", "stored-actor", "pierre", 1
    )
    conn = projects_db.connect(db_path)
    roadmap = conn.execute(
        "SELECT updated_by, updated_at FROM roadmaps "
        "WHERE profile_id='prof' AND project_id='p1' AND roadmap_id='r1'"
    ).fetchone()
    conn.close()
    assert roadmap["updated_by"] == "pierre"
    assert roadmap["updated_at"] > 1
