"""Tests for compaction boundary packet construction."""

import time
from pathlib import Path

from hermes_state import SessionDB


BUCKETS = (
    "project_facts",
    "decisions",
    "open_threads",
    "procedures",
    "artifacts",
    "do_not_carry",
)


def test_build_compaction_boundary_packet_includes_project_metadata(tmp_path):
    from agent.compaction_state import build_compaction_boundary_packet

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", "cli")
        db.set_session_title("root", "Durable project chat")
        db.update_session_cwd(
            "root",
            "N:/CodexProjects/example",
            git_branch="feature/continuity",
            git_repo_root="N:/CodexProjects/example",
        )
        db.end_session("root", "compression")
        db.create_session("tip", "cli", parent_session_id="root")

        packet = build_compaction_boundary_packet(
            session_db=db,
            session_id="tip",
            old_session_id="root",
            in_place=False,
            compression_count=3,
            platform="cli",
            boundary_at=12345.0,
        )

        assert packet["session_id"] == "tip"
        assert packet["old_session_id"] == "root"
        assert packet["lineage_root_id"] == "root"
        assert packet["in_place"] is False
        assert packet["compression_count"] == 3
        assert packet["platform"] == "cli"
        assert packet["boundary_at"] == 12345.0
        assert packet["cwd"] == "N:/CodexProjects/example"
        assert packet["git_repo_root"] == "N:/CodexProjects/example"
        assert packet["git_branch"] == "feature/continuity"
        assert packet["title"] == "Durable project chat"
        assert set(packet["durable_candidates"]) == set(BUCKETS)
        assert all(packet["durable_candidates"][bucket] == [] for bucket in BUCKETS)
    finally:
        db.close()


def test_build_compaction_boundary_packet_supports_in_place_boundary(tmp_path):
    from agent.compaction_state import build_compaction_boundary_packet

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("same", "cli")
        db.set_session_title("same", "In place chat")
        db.update_session_cwd("same", "N:/CodexProjects/example")

        packet = build_compaction_boundary_packet(
            session_db=db,
            session_id="same",
            old_session_id="same",
            in_place=True,
            compression_count=1,
            platform="telegram",
        )

        assert packet["session_id"] == "same"
        assert packet["old_session_id"] == "same"
        assert packet["lineage_root_id"] == "same"
        assert packet["in_place"] is True
        assert packet["platform"] == "telegram"
        assert packet["cwd"] == "N:/CodexProjects/example"
        assert isinstance(packet["boundary_at"], float)
        assert packet["boundary_at"] <= time.time()
    finally:
        db.close()


def test_build_compaction_boundary_packet_is_best_effort_without_db():
    from agent.compaction_state import build_compaction_boundary_packet

    packet = build_compaction_boundary_packet(
        session_db=None,
        session_id="tip",
        old_session_id="root",
        in_place=False,
        compression_count=1,
        platform="cli",
        boundary_at=100.0,
    )

    assert packet["session_id"] == "tip"
    assert packet["old_session_id"] == "root"
    assert packet["lineage_root_id"] == "root"
    assert packet["cwd"] == ""
    assert packet["git_repo_root"] == ""
    assert packet["title"] == ""
