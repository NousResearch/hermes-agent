from __future__ import annotations

import pytest

from hermes_cli import projects_db
from hermes_state import SessionDB


def test_project_for_path_prefers_innermost_owner(tmp_path):
    conn = projects_db.connect(db_path=tmp_path / "projects.db")
    try:
        outer_id = projects_db.create_project(
            conn, name="Outer", primary_path=str(tmp_path / "workspace")
        )
        inner_id = projects_db.create_project(
            conn, name="Inner", primary_path=str(tmp_path / "workspace" / "inner")
        )

        assert projects_db.project_for_path(
            conn, str(tmp_path / "workspace" / "inner" / "src")
        ).id == inner_id
        assert projects_db.project_for_path(
            conn, str(tmp_path / "workspace" / "elsewhere")
        ).id == outer_id
    finally:
        conn.close()


def test_session_project_affinity_is_idempotent_and_never_moves_cwd(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "project-affinity"
        cwd = str(tmp_path / "workspace" / "subdir")
        root = str(tmp_path / "workspace")
        db.create_session(session_id, source="test", cwd=cwd)

        first = db.update_session_project_affinity(
            session_id,
            project_id="project-1",
            project_root=root,
            project_context_hash="sha256:one",
        )
        second = db.update_session_project_affinity(
            session_id,
            project_id="project-1",
            project_root=root,
            project_context_hash="sha256:one",
        )
        changed = db.update_session_project_affinity(
            session_id,
            project_id="project-1",
            project_root=root,
            project_context_hash="sha256:two",
        )

        row = db.get_session(session_id)
        assert (first, second, changed) == (1, 1, 2)
        assert row["cwd"] == cwd
        assert row["project_id"] == "project-1"
        assert row["project_root"] == root
        assert row["project_affinity_generation"] == 2
        assert row["project_context_hash"] == "sha256:two"

        cleared = db.update_session_project_affinity(
            session_id,
            project_id=None,
            project_root=None,
            project_context_hash=None,
        )
        repeated_clear = db.update_session_project_affinity(
            session_id,
            project_id=None,
            project_root=None,
            project_context_hash=None,
        )
        row = db.get_session(session_id)
        assert (cleared, repeated_clear) == (3, 3)
        assert row["cwd"] == cwd
        assert row["project_id"] is None
        assert row["project_root"] is None
        assert row["project_context_hash"] is None
    finally:
        db.close()


def test_parent_linked_child_inherits_project_affinity(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        root = str(tmp_path / "workspace")
        db.create_session("parent", source="test", cwd=root)
        db.update_session_project_affinity(
            "parent",
            project_id="project-1",
            project_root=root,
            project_context_hash="sha256:context",
        )

        db.create_session("child", source="subagent", parent_session_id="parent")

        child = db.get_session("child")
        assert child["project_id"] == "project-1"
        assert child["project_root"] == root
        assert child["project_affinity_generation"] == 1
        assert child["project_context_hash"] == "sha256:context"
    finally:
        db.close()


def test_parent_inheritance_preserves_explicit_child_affinity_as_a_tuple(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        parent_root = str(tmp_path / "parent")
        child_root = str(tmp_path / "child")
        db.create_session("parent", source="test", cwd=parent_root)
        db.update_session_project_affinity(
            "parent", project_id="parent-project", project_root=parent_root,
            project_context_hash="sha256:parent",
        )
        db.create_session("child", source="test", cwd=child_root)
        db.update_session_project_affinity(
            "child", project_id="child-project", project_root=child_root,
            project_context_hash="sha256:child",
        )
        db.create_session("child", source="test", parent_session_id="parent")

        child = db.get_session("child")
        assert child["project_id"] == "child-project"
        assert child["project_root"] == child_root
        assert child["project_affinity_generation"] == 1
        assert child["project_context_hash"] == "sha256:child"
    finally:
        db.close()


def test_partial_project_affinity_is_rejected_and_not_mixed_during_inheritance(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        parent_root = str(tmp_path / "parent")
        db.create_session("parent", source="test", cwd=parent_root)
        db.update_session_project_affinity(
            "parent", project_id="parent-project", project_root=parent_root,
            project_context_hash="sha256:parent",
        )
        db.create_session("child", source="test")
        with pytest.raises(ValueError, match="requires project_id, project_root, and project_context_hash"):
            db.update_session_project_affinity(
                "child", project_id="child-project", project_root=None, project_context_hash=None,
            )

        # Simulate a legacy/externally-written partial tuple. Parent linking must
        # preserve it exactly rather than mix in the parent's root/hash.
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET project_id = ?, project_affinity_generation = 7 WHERE id = ?",
            ("legacy-project", "child"),
        ))
        db.create_session("child", source="test", parent_session_id="parent")

        child = db.get_session("child")
        assert child["project_id"] == "legacy-project"
        assert child["project_root"] is None
        assert child["project_affinity_generation"] == 7
        assert child["project_context_hash"] is None
    finally:
        db.close()


def test_compression_child_inherits_project_affinity_atomically(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        parent_id = "project-parent"
        child_id = "project-child"
        root = str(tmp_path / "workspace")
        db.create_session(parent_id, source="test", cwd=root)
        assert db.update_session_project_affinity(
            parent_id,
            project_id="project-1",
            project_root=root,
            project_context_hash="sha256:context",
        ) == 1

        db.publish_compression_child(
            parent_session_id=parent_id,
            child_session_id=child_id,
            source="test",
            messages=[{"role": "user", "content": "compacted handoff"}],
            require_compression_lease=False,
        )

        parent = db.get_session(parent_id)
        child = db.get_session(child_id)
        assert parent["end_reason"] == "compression"
        assert child["parent_session_id"] == parent_id
        assert child["project_id"] == parent["project_id"] == "project-1"
        assert child["project_root"] == parent["project_root"] == root
        assert child["project_affinity_generation"] == parent["project_affinity_generation"] == 1
        assert child["project_context_hash"] == parent["project_context_hash"] == "sha256:context"
    finally:
        db.close()


def test_workspace_and_project_affinity_move_is_one_persisted_claim(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        old_cwd = str(tmp_path / "old")
        new_cwd = str(tmp_path / "new")
        db.create_session("move", source="desktop", cwd=old_cwd)

        first = db.update_session_workspace_project_affinity(
            "move", cwd=new_cwd, project_id="project-1", project_root=new_cwd,
            project_context_hash="sha256:one",
        )
        second = db.update_session_workspace_project_affinity(
            "move", cwd=new_cwd, project_id="project-1", project_root=new_cwd,
            project_context_hash="sha256:one",
        )

        row = db.get_session("move")
        assert first == (1, 1)
        assert second == (2, 1)
        assert row["cwd"] == new_cwd
        assert row["git_metadata_generation"] == 2
        assert row["git_branch"] is None
        assert row["git_repo_root"] is None
        assert row["project_id"] == "project-1"
        assert row["project_root"] == new_cwd
        assert row["project_affinity_generation"] == 1
        assert row["project_context_hash"] == "sha256:one"
    finally:
        db.close()
