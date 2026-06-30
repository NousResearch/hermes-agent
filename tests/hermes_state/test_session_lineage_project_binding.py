"""Regression tests for compression lineage project binding.

Physical continuation rows may rotate (`root -> mid -> tip`) when context is
compacted.  The logical conversation must stay attached to its project even if
one physical row is missing workspace metadata.
"""

import time
from pathlib import Path

from hermes_state import SessionDB


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _end_for_compression(db: SessionDB, session_id: str, ended_at: float) -> None:
    db._conn.execute(
        "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
        (ended_at, "compression", session_id),
    )
    db._conn.commit()


def _set_started(db: SessionDB, session_id: str, started_at: float) -> None:
    db._conn.execute(
        "UPDATE sessions SET started_at=? WHERE id=?",
        (started_at, session_id),
    )
    db._conn.commit()


def _make_broken_child(db: SessionDB, *, with_branch: bool = True) -> None:
    t0 = time.time() - 3600
    db.create_session("root", "cli")
    _set_started(db, "root", t0)
    db.update_session_cwd(
        "root",
        "N:/CodexProjects/example",
        git_branch="feature/continuity" if with_branch else None,
        git_repo_root="N:/CodexProjects/example",
    )
    _end_for_compression(db, "root", t0 + 100)
    db.create_session("tip", "cli", parent_session_id="root")
    _set_started(db, "tip", t0 + 101)
    db._conn.execute(
        "UPDATE sessions SET cwd=NULL, git_branch=NULL, git_repo_root=NULL WHERE id='tip'"
    )
    db._conn.commit()


def test_child_session_inherits_parent_workspace_metadata(tmp_path):
    db = _make_db(tmp_path)
    try:
        db.create_session("root", "cli")
        db.update_session_cwd(
            "root",
            "N:/CodexProjects/example",
            git_branch="feature/continuity",
            git_repo_root="N:/CodexProjects/example",
        )
        _end_for_compression(db, "root", time.time())

        db.create_session("tip", "cli", parent_session_id="root")

        tip = db.get_session("tip")
        assert tip["cwd"] == "N:/CodexProjects/example"
        assert tip["git_branch"] == "feature/continuity"
        assert tip["git_repo_root"] == "N:/CodexProjects/example"
    finally:
        db.close()


def test_explicit_child_workspace_metadata_overrides_parent(tmp_path):
    db = _make_db(tmp_path)
    try:
        db.create_session("root", "cli")
        db.update_session_cwd(
            "root",
            "N:/CodexProjects/example",
            git_branch="main",
            git_repo_root="N:/CodexProjects/example",
        )
        _end_for_compression(db, "root", time.time())

        db.create_session(
            "tip",
            "cli",
            parent_session_id="root",
            cwd="N:/CodexProjects/example-worktree",
            git_branch="feature/worktree",
            git_repo_root="N:/CodexProjects/example-worktree",
        )

        tip = db.get_session("tip")
        assert tip["cwd"] == "N:/CodexProjects/example-worktree"
        assert tip["git_branch"] == "feature/worktree"
        assert tip["git_repo_root"] == "N:/CodexProjects/example-worktree"
    finally:
        db.close()


def test_projection_falls_back_to_root_workspace_when_tip_is_legacy_broken(tmp_path):
    db = _make_db(tmp_path)
    try:
        t0 = time.time() - 3600
        db.create_session("root", "cli")
        _set_started(db, "root", t0)
        db.set_session_title("root", "Project chat")
        db.update_session_cwd(
            "root",
            "N:/CodexProjects/example",
            git_branch="feature/continuity",
            git_repo_root="N:/CodexProjects/example",
        )
        db.append_message("root", "user", "root message")
        _end_for_compression(db, "root", t0 + 100)

        db.create_session("tip", "cli", parent_session_id="root")
        _set_started(db, "tip", t0 + 101)
        db.set_session_title("tip", "Project chat #2")
        db.append_message("tip", "user", "tip message")
        db._conn.execute(
            "UPDATE sessions SET cwd=NULL, git_branch=NULL, git_repo_root=NULL WHERE id=?",
            ("tip",),
        )
        db._conn.commit()

        rows = db.list_sessions_rich(source="cli", limit=10)
        projected = next(row for row in rows if row["id"] == "tip")

        assert projected["_lineage_root_id"] == "root"
        assert projected["cwd"] == "N:/CodexProjects/example"
        assert projected["git_branch"] == "feature/continuity"
        assert projected["git_repo_root"] == "N:/CodexProjects/example"
    finally:
        db.close()


def test_projection_prefers_non_empty_tip_workspace_metadata(tmp_path):
    db = _make_db(tmp_path)
    try:
        t0 = time.time() - 3600
        db.create_session("root", "cli")
        _set_started(db, "root", t0)
        db.update_session_cwd(
            "root",
            "N:/CodexProjects/example",
            git_branch="main",
            git_repo_root="N:/CodexProjects/example",
        )
        _end_for_compression(db, "root", t0 + 100)

        db.create_session("tip", "cli", parent_session_id="root")
        _set_started(db, "tip", t0 + 101)
        db.update_session_cwd(
            "tip",
            "N:/CodexProjects/example-worktree",
            git_branch="feature/worktree",
            git_repo_root="N:/CodexProjects/example-worktree",
        )

        rows = db.list_sessions_rich(source="cli", limit=10)
        projected = next(row for row in rows if row["id"] == "tip")

        assert projected["cwd"] == "N:/CodexProjects/example-worktree"
        assert projected["git_branch"] == "feature/worktree"
        assert projected["git_repo_root"] == "N:/CodexProjects/example-worktree"
    finally:
        db.close()


def test_project_cwd_filter_includes_projected_logical_tip(tmp_path):
    db = _make_db(tmp_path)
    try:
        _make_broken_child(db)

        rows = db.list_sessions_rich(
            source="cli",
            cwd_prefix="N:/CodexProjects/example",
            limit=10,
        )

        assert [row["id"] for row in rows] == ["tip"]
        assert rows[0]["_lineage_root_id"] == "root"
    finally:
        db.close()


def test_compression_tip_ignores_delegate_child_for_projected_lineage(tmp_path):
    db = _make_db(tmp_path)
    try:
        t0 = time.time() - 3600
        db.create_session("root", "cli")
        _set_started(db, "root", t0)
        db.update_session_cwd("root", "N:/CodexProjects/example")

        db.create_session(
            "delegate",
            "cli",
            parent_session_id="root",
            model_config={"_delegate_from": "root"},
        )
        _set_started(db, "delegate", t0 + 50)
        _end_for_compression(db, "root", t0 + 100)

        db.create_session("tip", "cli", parent_session_id="root")
        _set_started(db, "tip", t0 + 101)

        assert db.get_compression_tip("root") == "tip"
        rows = db.list_sessions_rich(source="cli", limit=10)
        ids = [row["id"] for row in rows]
        assert "tip" in ids
        assert "delegate" not in ids
    finally:
        db.close()


def test_find_weak_lineage_project_bindings_reports_repairable_child(tmp_path):
    db = _make_db(tmp_path)
    try:
        _make_broken_child(db)

        weak = db.find_weak_lineage_project_bindings()

        assert len(weak) == 1
        assert weak[0]["id"] == "tip"
        assert weak[0]["ancestor_id"] == "root"
        assert weak[0]["target_cwd"] == "N:/CodexProjects/example"
        assert weak[0]["target_git_repo_root"] == "N:/CodexProjects/example"
        assert weak[0]["target_git_branch"] == "feature/continuity"
        assert "missing" in weak[0]["reason"]
    finally:
        db.close()


def test_find_weak_lineage_project_bindings_ignores_real_different_cwd(tmp_path):
    db = _make_db(tmp_path)
    try:
        t0 = time.time() - 3600
        db.create_session("root", "cli")
        _set_started(db, "root", t0)
        db.update_session_cwd(
            "root",
            "N:/CodexProjects/example",
            git_branch="main",
            git_repo_root="N:/CodexProjects/example",
        )
        _end_for_compression(db, "root", t0 + 100)
        db.create_session(
            "tip",
            "cli",
            parent_session_id="root",
            cwd="N:/CodexProjects/other",
            git_branch="other",
            git_repo_root="N:/CodexProjects/other",
        )
        _set_started(db, "tip", t0 + 101)

        assert db.find_weak_lineage_project_bindings() == []
    finally:
        db.close()


def test_find_weak_lineage_project_bindings_flags_home_cwd(tmp_path):
    db = _make_db(tmp_path)
    try:
        _make_broken_child(db, with_branch=False)
        db._conn.execute(
            "UPDATE sessions SET cwd=?, git_repo_root=NULL WHERE id=?",
            ("C:/Users/Alexander Semenov", "tip"),
        )
        db._conn.commit()

        weak = db.find_weak_lineage_project_bindings()

        assert len(weak) == 1
        assert weak[0]["id"] == "tip"
        assert weak[0]["target_cwd"] == "N:/CodexProjects/example"
    finally:
        db.close()


def test_repair_weak_lineage_project_bindings_check_only_does_not_mutate(tmp_path):
    db = _make_db(tmp_path)
    try:
        _make_broken_child(db)

        result = db.repair_weak_lineage_project_bindings(apply=False)

        assert result["checked"] == 1
        assert result["updated"] == 0
        assert db.get_session("tip")["cwd"] is None
    finally:
        db.close()


def test_repair_weak_lineage_project_bindings_apply_backfills_metadata(tmp_path):
    db = _make_db(tmp_path)
    try:
        _make_broken_child(db)

        result = db.repair_weak_lineage_project_bindings(apply=True)

        tip = db.get_session("tip")
        assert result["checked"] == 1
        assert result["updated"] == 1
        assert tip["cwd"] == "N:/CodexProjects/example"
        assert tip["git_branch"] == "feature/continuity"
        assert tip["git_repo_root"] == "N:/CodexProjects/example"
        assert db.find_weak_lineage_project_bindings() == []
    finally:
        db.close()
