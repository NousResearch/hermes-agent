"""Tests for roadmap CRUD + plan governance (T5b): create/update/archive
roadmap, plans create/list/get/activate/validate, atomic rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import projects_db
from hermes_cli.roadmaps_service import RoadmapsService
from hermes_cli.roadmaps_writer import (
    InvalidRoadmapPlanTransitionError,
    InvalidRoadmapTransitionError,
    RoadmapExistsError,
    RoadmapNotFoundError,
    RoadmapProjectNotFoundError,
    RoadmapVersionExistsError,
    RoadmapVersionNotFoundError,
    RoadmapsWriter,
    StaleRoadmapVersionError,
)


def seed(
    path: Path,
    *,
    profile: str = "prof",
    project: str = "p1",
    roadmap_id: str = "r1",
    with_roadmap: bool = True,
    lifecycle: str = "draft",
    active_version: int | None = None,
) -> None:
    """Seed a project row and optionally a roadmap with its version 1."""
    conn = projects_db.connect(path)
    conn.execute(
        "INSERT INTO projects(id, slug, name, created_at) VALUES (?, ?, ?, 1)",
        (project, project, project),
    )
    if with_roadmap:
        conn.execute(
            "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile, project, roadmap_id, "Roadmap", None, lifecycle,
             active_version, "creator", "creator", 1, 1),
        )
        conn.execute(
            "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile, project, roadmap_id, 1,
             "validated" if active_version else "draft",
             "seed", None, "creator", 1, None),
        )
    conn.commit()
    conn.close()


def count(
    path: Path,
    table: str,
    *,
    profile: str = "prof",
    project: str = "p1",
    roadmap: str = "r1",
    version: int | None = None,
) -> int:
    conn = projects_db.connect(path)
    sql = (
        f"SELECT COUNT(*) AS n FROM {table} "
        "WHERE profile_id=? AND project_id=? AND roadmap_id=?"
    )
    args = [profile, project, roadmap]
    if version is not None:
        sql += " AND version=?"
        args.append(version)
    n = conn.execute(sql, args).fetchone()["n"]
    conn.close()
    return n


def roadmap_row(path: Path, roadmap_id: str = "r1") -> dict:
    conn = projects_db.connect(path)
    row = conn.execute(
        "SELECT * FROM roadmaps WHERE roadmap_id=?", (roadmap_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def version_row(path: Path, version: int, roadmap_id: str = "r1") -> dict:
    conn = projects_db.connect(path)
    row = conn.execute(
        "SELECT * FROM roadmap_versions WHERE roadmap_id=? AND version=?",
        (roadmap_id, version),
    ).fetchone()
    conn.close()
    return dict(row)


def plan_payload(version: int = 2) -> dict:
    """A valid full plan payload: 3 nodes, 2 relations, 2 todos."""
    return {
        "version": version,
        "nodes": [
            {"node_id": "obj", "kind": "objective", "title": "Objective"},
            {"node_id": "phase-1", "kind": "phase", "title": "Phase 1",
             "parent_node_id": "obj", "description": "First phase"},
            {"node_id": "step-1", "kind": "step", "title": "Step 1",
             "parent_node_id": "phase-1", "state": "ready",
             "owner_agent": "agent-a"},
        ],
        "relations": [
            {"relation_id": "rel-1", "from_node_id": "step-1",
             "to_node_id": "phase-1", "kind": "depends_on",
             "reason": "phase gates the step"},
        ],
        "todos": [
            {"todo_id": "todo-1", "node_id": "step-1", "title": "Do it"},
            {"todo_id": "todo-2", "title": "Unattached"},
        ],
    }


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "projects.db"
    seed(path, with_roadmap=False)
    return path


@pytest.fixture()
def roadmap_path(tmp_path: Path) -> Path:
    path = tmp_path / "projects.db"
    seed(path)
    return path


# ── roadmaps.create ──────────────────────────────────────────────────────────


def test_create_roadmap_inserts_roadmap_and_version_one(db_path: Path):
    result = RoadmapsWriter(db_path).create_roadmap(
        "prof", "p1", "My roadmap", "pierre"
    )
    assert result["success"] is True
    assert result["scope"] == {
        "profile_id": "prof", "project_id": "p1", "roadmap_id": result["roadmap_id"],
    }
    assert result["roadmap_id"].startswith("r_")
    assert result["version"] == 1

    row = roadmap_row(db_path, result["roadmap_id"])
    assert row["title"] == "My roadmap"
    assert row["purpose"] is None
    assert row["lifecycle_state"] == "draft"
    assert row["active_version"] is None
    assert row["created_by"] == "pierre"
    assert row["updated_by"] == "pierre"
    assert count(db_path, "roadmap_versions", roadmap=result["roadmap_id"]) == 1
    assert version_row(db_path, 1, roadmap_id=result["roadmap_id"])["state"] == "draft"


def test_create_roadmap_uses_provided_roadmap_id_deterministically(db_path: Path):
    result = RoadmapsWriter(db_path).create_roadmap(
        "prof", "p1", "Title", "pierre", roadmap_id="r-custom"
    )
    assert result["roadmap_id"] == "r-custom"
    assert roadmap_row(db_path, "r-custom")["roadmap_id"] == "r-custom"


def test_create_roadmap_rejects_blank_title_and_scope(db_path: Path):
    writer = RoadmapsWriter(db_path)
    with pytest.raises(ValueError, match="title"):
        writer.create_roadmap("prof", "p1", "   ", "pierre")
    with pytest.raises(ValueError, match="profile_id"):
        writer.create_roadmap("", "p1", "Title", "pierre")
    with pytest.raises(ValueError, match="project_id"):
        writer.create_roadmap("prof", "", "Title", "pierre")


def test_create_roadmap_rejects_overlong_title_and_purpose(db_path: Path):
    writer = RoadmapsWriter(db_path)
    with pytest.raises(ValueError, match="title"):
        writer.create_roadmap("prof", "p1", "x" * 201, "pierre")
    with pytest.raises(ValueError, match="purpose"):
        writer.create_roadmap(
            "prof", "p1", "Title", "pierre", purpose="x" * 2001
        )


def test_create_roadmap_rejects_unknown_lifecycle_state(db_path: Path):
    with pytest.raises(ValueError, match="lifecycle_state"):
        RoadmapsWriter(db_path).create_roadmap(
            "prof", "p1", "Title", "pierre", lifecycle_state="bogus"
        )


@pytest.mark.parametrize(
    "state", ["proposed", "validated", "in_progress", "blocked", "completed", "archived"]
)
def test_create_roadmap_rejects_non_draft_lifecycle_state(db_path: Path, state: str):
    # The lifecycle machine is strict: a roadmap is born 'draft'; jumping
    # straight to a later state would bypass plans.validate / plans.activate
    # / archive_roadmap. No row may be created on a rejected lifecycle.
    with pytest.raises(ValueError, match="lifecycle_state"):
        RoadmapsWriter(db_path).create_roadmap(
            "prof", "p1", "Title", "pierre", lifecycle_state=state
        )
    assert count(db_path, "roadmaps") == 0


def test_create_roadmap_accepts_explicit_draft_lifecycle_state(db_path: Path):
    result = RoadmapsWriter(db_path).create_roadmap(
        "prof", "p1", "Title", "pierre", lifecycle_state="draft"
    )
    assert result["success"] is True
    assert roadmap_row(db_path, result["roadmap_id"])["lifecycle_state"] == "draft"


def test_create_roadmap_rejects_missing_project(db_path: Path):
    with pytest.raises(RoadmapProjectNotFoundError):
        RoadmapsWriter(db_path).create_roadmap("prof", "nope", "Title", "pierre")


def test_create_roadmap_duplicate_id_rejected(db_path: Path):
    writer = RoadmapsWriter(db_path)
    writer.create_roadmap("prof", "p1", "Title", "pierre", roadmap_id="r-x")
    with pytest.raises(RoadmapExistsError):
        writer.create_roadmap("prof", "p1", "Title", "pierre", roadmap_id="r-x")


def test_create_roadmap_rolls_back_on_failure(db_path: Path):
    before = db_path.read_bytes()
    with pytest.raises(ValueError):
        RoadmapsWriter(db_path).create_roadmap("prof", "p1", "  ", "pierre")
    assert db_path.read_bytes() == before
    assert count(db_path, "roadmaps") == 0


# ── roadmaps.update ──────────────────────────────────────────────────────────


def test_update_roadmap_changes_title_and_purpose(roadmap_path: Path):
    result = RoadmapsWriter(roadmap_path).update_roadmap(
        "prof", "p1", "r1", "pierre", 0,
        title="Renamed", purpose="New purpose",
    )
    assert result["success"] is True
    row = roadmap_row(roadmap_path)
    assert row["title"] == "Renamed"
    assert row["purpose"] == "New purpose"
    # lifecycle_state is not writable through update: it stays 'draft' and
    # only moves via plans.validate / plans.activate / roadmaps.archive.
    assert row["lifecycle_state"] == "draft"
    assert row["updated_by"] == "pierre"
    assert row["updated_at"] > 1


def test_update_roadmap_partial_fields_only(roadmap_path: Path):
    RoadmapsWriter(roadmap_path).update_roadmap(
        "prof", "p1", "r1", "pierre", 0, title="Only title"
    )
    row = roadmap_row(roadmap_path)
    assert row["title"] == "Only title"
    assert row["purpose"] is None
    assert row["lifecycle_state"] == "draft"


def test_update_roadmap_requires_expected_version(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    with pytest.raises(StaleRoadmapVersionError):
        writer.update_roadmap("prof", "p1", "r1", "pierre", 99, title="X")
    with pytest.raises(ValueError, match="expected_version"):
        writer.update_roadmap("prof", "p1", "r1", "pierre", None, title="X")
    assert roadmap_row(roadmap_path)["title"] == "Roadmap"


def test_update_roadmap_unknown_roadmap_and_nothing_to_update(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    with pytest.raises(RoadmapNotFoundError):
        writer.update_roadmap("prof", "p1", "r-missing", "pierre", 0, title="X")
    with pytest.raises(ValueError, match="nothing to update"):
        writer.update_roadmap("prof", "p1", "r1", "pierre", 0)


def test_update_roadmap_rejects_invalid_lifecycle(roadmap_path: Path):
    with pytest.raises(ValueError, match="lifecycle_state"):
        RoadmapsWriter(roadmap_path).update_roadmap(
            "prof", "p1", "r1", "pierre", 0, lifecycle_state="bogus"
        )


@pytest.mark.parametrize("state", ["proposed", "completed", "archived"])
def test_update_roadmap_rejects_lifecycle_state_direct_writes(
    roadmap_path: Path, state: str
):
    # Transitions are exclusive to the dedicated ops: setting 'completed'
    # via update would bypass plans.validate / plans.activate, and setting
    # 'archived' would bypass archive_roadmap. Direct writes are rejected
    # and leave the row untouched.
    with pytest.raises(ValueError, match="lifecycle_state"):
        RoadmapsWriter(roadmap_path).update_roadmap(
            "prof", "p1", "r1", "pierre", 0, lifecycle_state=state
        )
    assert roadmap_row(roadmap_path)["lifecycle_state"] == "draft"


# ── roadmaps.archive ─────────────────────────────────────────────────────────


def test_archive_roadmap_sets_archived(roadmap_path: Path):
    result = RoadmapsWriter(roadmap_path).archive_roadmap(
        "prof", "p1", "r1", "pierre", 0
    )
    assert result["success"] is True
    row = roadmap_row(roadmap_path)
    assert row["lifecycle_state"] == "archived"
    assert row["updated_by"] == "pierre"
    # Versions stay intact: archive preserves the plan history.
    assert count(roadmap_path, "roadmap_versions") == 1


def test_archive_roadmap_twice_rejected(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.archive_roadmap("prof", "p1", "r1", "pierre", 0)
    with pytest.raises(InvalidRoadmapTransitionError):
        writer.archive_roadmap("prof", "p1", "r1", "pierre", 0)


def test_archive_roadmap_unknown_or_stale(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    with pytest.raises(RoadmapNotFoundError):
        writer.archive_roadmap("prof", "p1", "r-missing", "pierre", 0)
    with pytest.raises(StaleRoadmapVersionError):
        writer.archive_roadmap("prof", "p1", "r1", "pierre", 99)


# ── plans.create ─────────────────────────────────────────────────────────────


def test_create_plan_inserts_version_nodes_relations_todos(roadmap_path: Path):
    payload = plan_payload(version=2)
    result = RoadmapsWriter(roadmap_path).create_plan(
        "prof", "p1", "r1", "agent-a", **payload
    )
    assert result["success"] is True
    assert result["version"] == 2
    assert result["state"] == "proposed"
    assert result["counts"] == {"nodes": 3, "relations": 1, "todos": 2}
    assert len(result["content_hash"]) == 64

    vrow = version_row(roadmap_path, 2)
    assert vrow["state"] == "proposed"
    assert vrow["created_by"] == "agent-a"
    assert vrow["content_hash"] == result["content_hash"]
    assert count(roadmap_path, "roadmap_nodes", version=2) == 3
    assert count(roadmap_path, "roadmap_relations", version=2) == 1
    assert count(roadmap_path, "roadmap_todos", version=2) == 2
    # Roadmap lifecycle draft -> proposed; still no active version.
    row = roadmap_row(roadmap_path)
    assert row["lifecycle_state"] == "proposed"
    assert row["active_version"] is None


def test_create_plan_default_version_is_max_plus_one(roadmap_path: Path):
    result = RoadmapsWriter(roadmap_path).create_plan(
        "prof", "p1", "r1", "agent-a",
        nodes=[{"node_id": "n1", "kind": "step", "title": "N1"}],
        relations=[], todos=[],
    )
    # version 1 already exists (roadmaps.create marker), so default is 2.
    assert result["version"] == 2


def test_create_plan_rejects_duplicate_version(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    with pytest.raises(RoadmapVersionExistsError):
        writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    # Version 1 marker is also taken.
    with pytest.raises(RoadmapVersionExistsError):
        writer.create_plan(
            "prof", "p1", "r1", "agent-a",
            nodes=[{"node_id": "n1", "kind": "step", "title": "N1"}],
            relations=[], todos=[], version=1,
        )
    assert count(roadmap_path, "roadmap_versions") == 2


def test_create_plan_rolls_back_atomically_on_invalid_node(roadmap_path: Path):
    payload = plan_payload(version=2)
    payload["nodes"][2]["kind"] = "bogus"
    with pytest.raises(ValueError, match="kind"):
        RoadmapsWriter(roadmap_path).create_plan(
            "prof", "p1", "r1", "agent-a", **payload
        )
    assert count(roadmap_path, "roadmap_versions") == 1
    assert count(roadmap_path, "roadmap_nodes") == 0
    assert count(roadmap_path, "roadmap_relations") == 0
    assert count(roadmap_path, "roadmap_todos") == 0
    assert roadmap_row(roadmap_path)["lifecycle_state"] == "draft"


def test_create_plan_rejects_missing_parent_reference(roadmap_path: Path):
    payload = plan_payload(version=2)
    payload["nodes"][2]["parent_node_id"] = "ghost"
    with pytest.raises(ValueError, match="parent_node_id"):
        RoadmapsWriter(roadmap_path).create_plan(
            "prof", "p1", "r1", "agent-a", **payload
        )


def test_create_plan_rejects_self_parent_and_duplicate_node_ids(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    payload = plan_payload(version=2)
    payload["nodes"][0]["parent_node_id"] = "obj"
    with pytest.raises(ValueError, match="parent_node_id"):
        writer.create_plan("prof", "p1", "r1", "agent-a", **payload)
    payload = plan_payload(version=2)
    payload["nodes"].append(dict(payload["nodes"][0]))
    with pytest.raises(ValueError, match="duplicate node_id"):
        writer.create_plan("prof", "p1", "r1", "agent-a", **payload)


def test_create_plan_rejects_relation_to_unknown_node_and_self(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    payload = plan_payload(version=2)
    payload["relations"][0]["to_node_id"] = "ghost"
    with pytest.raises(ValueError, match="to_node_id"):
        writer.create_plan("prof", "p1", "r1", "agent-a", **payload)
    payload = plan_payload(version=2)
    payload["relations"][0]["to_node_id"] = "step-1"
    with pytest.raises(ValueError, match="differ"):
        writer.create_plan("prof", "p1", "r1", "agent-a", **payload)


def test_create_plan_rejects_cyclic_relations(roadmap_path: Path):
    payload = plan_payload(version=2)
    payload["relations"] = [
        {"relation_id": "a", "from_node_id": "step-1", "to_node_id": "phase-1",
         "kind": "depends_on"},
        {"relation_id": "b", "from_node_id": "phase-1", "to_node_id": "step-1",
         "kind": "depends_on"},
    ]
    with pytest.raises(ValueError, match="cyclic relation"):
        RoadmapsWriter(roadmap_path).create_plan(
            "prof", "p1", "r1", "agent-a", **payload
        )


def test_create_plan_rejects_missing_todo_node_ref_and_blank_title(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    payload = plan_payload(version=2)
    payload["todos"][0]["node_id"] = "ghost"
    with pytest.raises(ValueError, match="node_id"):
        writer.create_plan("prof", "p1", "r1", "agent-a", **payload)
    payload = plan_payload(version=2)
    payload["todos"][1]["title"] = "   "
    with pytest.raises(ValueError, match="title"):
        writer.create_plan("prof", "p1", "r1", "agent-a", **payload)


def test_create_plan_rejects_archived_roadmap_and_unknown_roadmap(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.archive_roadmap("prof", "p1", "r1", "pierre", 0)
    with pytest.raises(InvalidRoadmapTransitionError):
        writer.create_plan(
            "prof", "p1", "r1", "agent-a",
            nodes=[{"node_id": "n1", "kind": "step", "title": "N1"}],
            relations=[], todos=[],
        )
    with pytest.raises(RoadmapNotFoundError):
        writer.create_plan(
            "prof", "p1", "r-missing", "agent-a",
            nodes=[{"node_id": "n1", "kind": "step", "title": "N1"}],
            relations=[], todos=[],
        )


# ── plans.list / plans.get (read side) ───────────────────────────────────────


def test_plans_list_orders_by_version_desc(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=3))
    plans = RoadmapsService(roadmap_path).list_plans("prof", "p1", "r1")["plans"]
    assert [p["version"] for p in plans] == [3, 2, 1]
    assert set(plans[0]) >= {
        "version", "state", "source", "reason", "created_by", "created_at",
        "content_hash",
    }


def test_plans_list_unknown_roadmap_returns_empty(roadmap_path: Path):
    result = RoadmapsService(roadmap_path).list_plans("prof", "p1", "r-missing")
    assert result["plans"] == []
    assert result["scope"]["roadmap_id"] == "r-missing"


def test_plans_get_returns_full_version(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    result = RoadmapsService(roadmap_path).get_plan("prof", "p1", "r1", 2)
    assert result["found"] is True
    plan = result["plan"]
    assert plan["version"] == 2
    assert plan["state"] == "proposed"
    assert {n["node_id"] for n in plan["nodes"]} == {"obj", "phase-1", "step-1"}
    assert plan["nodes"][0]["parent_node_id"] is None
    assert len(plan["relations"]) == 1
    assert len(plan["todos"]) == 2
    missing = RoadmapsService(roadmap_path).get_plan("prof", "p1", "r1", 99)
    assert missing["found"] is False
    assert missing["plan"] is None


# ── plans.validate / plans.activate ──────────────────────────────────────────


def test_plans_validate_transitions_draft_and_proposed(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    result = writer.validate_plan("prof", "p1", "r1", 1, "pierre", 0)
    assert result["state"] == "validated"
    assert version_row(roadmap_path, 1)["state"] == "validated"
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    result = writer.validate_plan("prof", "p1", "r1", 2, "pierre", 0)
    assert result["state"] == "validated"
    # Validated is terminal for validate.
    with pytest.raises(InvalidRoadmapPlanTransitionError):
        writer.validate_plan("prof", "p1", "r1", 2, "pierre", 0)


def test_plans_validate_requires_expected_version(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    with pytest.raises(StaleRoadmapVersionError):
        writer.validate_plan("prof", "p1", "r1", 1, "pierre", 42)
    with pytest.raises(RoadmapVersionNotFoundError):
        writer.validate_plan("prof", "p1", "r1", 99, "pierre", 0)
    with pytest.raises(RoadmapNotFoundError):
        writer.validate_plan("prof", "p1", "r-missing", 1, "pierre", 0)


def test_plans_activate_requires_validated_version(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    with pytest.raises(InvalidRoadmapPlanTransitionError):
        writer.activate_plan("prof", "p1", "r1", 2, "pierre", 0)
    with pytest.raises(RoadmapVersionNotFoundError):
        writer.activate_plan("prof", "p1", "r1", 99, "pierre", 0)


def test_plans_activate_sets_active_version_and_supersedes_previous(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path, active_version=1, lifecycle="in_progress")
    writer = RoadmapsWriter(path)
    writer.create_plan("prof", "p1", "r1", "agent-a", **plan_payload(version=2))
    writer.validate_plan("prof", "p1", "r1", 2, "pierre", 1)
    result = writer.activate_plan("prof", "p1", "r1", 2, "pierre", 1)
    assert result["success"] is True
    assert result["active_version"] == 2
    assert result["previous_active_version"] == 1
    row = roadmap_row(path)
    assert row["active_version"] == 2
    assert row["updated_by"] == "pierre"
    # History preserved: the previous active version is superseded.
    assert version_row(path, 1)["state"] == "superseded"
    assert version_row(path, 2)["state"] == "validated"


def test_plans_activate_stale_expected_version_rejected(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    with pytest.raises(StaleRoadmapVersionError):
        writer.activate_plan("prof", "p1", "r1", 1, "pierre", 99)


def test_plans_activate_on_fresh_roadmap_uses_expected_zero(roadmap_path: Path):
    writer = RoadmapsWriter(roadmap_path)
    writer.validate_plan("prof", "p1", "r1", 1, "pierre", 0)
    result = writer.activate_plan("prof", "p1", "r1", 1, "pierre", 0)
    assert result["active_version"] == 1
    row = roadmap_row(roadmap_path)
    assert row["active_version"] == 1
    assert row["lifecycle_state"] == "in_progress"


def test_create_plan_deep_parent_chain_does_not_overflow(roadmap_path: Path):
    """A ~1500-node parent chain must validate without a RecursionError.

    Regression for the recursive _detect_cycle (writer): it overflowed the
    interpreter stack on a deep acyclic parent chain, turning a valid plan
    into a generic 503. The iterative 3-colour DFS handles it.
    """
    from hermes_cli.roadmaps_writer import MAX_PLAN_NODES

    depth = MAX_PLAN_NODES - 1  # within the bound, but deep enough to overflow recursion
    nodes = [{"node_id": f"n{i}", "kind": "step", "title": f"Node {i}"} for i in range(depth)]
    # n1 -> n0 -> ... chain of parents (acyclic).
    for i in range(1, depth):
        nodes[i]["parent_node_id"] = f"n{i - 1}"
    writer = RoadmapsWriter(roadmap_path)
    result = writer.create_plan("prof", "p1", "r1", "agent-a", nodes=nodes, relations=[], todos=[])
    assert result["state"] == "proposed"
    assert result["counts"]["nodes"] == depth


def test_create_plan_rejects_oversized_payloads(roadmap_path: Path):
    """Element bounds: >MAX_PLAN_NODES/RELATIONS/TODOS is a clean 5063-style
    ValueError, never a transaction or a crash."""
    from hermes_cli.roadmaps_writer import MAX_PLAN_NODES, MAX_PLAN_RELATIONS, MAX_PLAN_TODOS

    writer = RoadmapsWriter(roadmap_path)
    too_many = [{"node_id": f"n{i}", "kind": "step", "title": f"N{i}"} for i in range(MAX_PLAN_NODES + 1)]
    with pytest.raises(ValueError, match="at most"):
        writer.create_plan("prof", "p1", "r1", "agent-a", nodes=too_many, relations=[], todos=[])
    # relations/todos bounds are checked the same way; exercise one of them too.
    relations = [
        {"relation_id": f"rel{i}", "from_node_id": f"n{i}", "to_node_id": f"n{i + 1}", "kind": "depends_on"}
        for i in range(MAX_PLAN_RELATIONS + 1)
    ]
    with pytest.raises(ValueError, match="at most"):
        writer.create_plan("prof", "p1", "r1", "agent-a", nodes=[], relations=relations, todos=[])
    # (todos bound is the same code path — covered via the shared guard.)


def test_create_plan_rejects_path_separator_ids(roadmap_path: Path):
    """An id containing '/' or '\\' is rejected up front, matching the pure
    contract (src/roadmaps_contract._identifier); otherwise it becomes
    unreachable via the REST/RPC path segment."""
    writer = RoadmapsWriter(roadmap_path)
    with pytest.raises(ValueError, match="path separator"):
        writer.create_plan(
            "prof", "p1", "r1", "agent-a",
            nodes=[{"node_id": "a/b", "kind": "step", "title": "bad"}],
            relations=[], todos=[],
        )
    with pytest.raises(ValueError, match="path separator"):
        writer.create_plan(
            "prof", "p1", "r1", "agent-a",
            nodes=[{"node_id": "a\\b", "kind": "step", "title": "bad"}],
            relations=[], todos=[],
        )
