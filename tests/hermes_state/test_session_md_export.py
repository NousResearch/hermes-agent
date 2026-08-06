import time

import hermes_state
from hermes_state import SessionDB


def test_export_candidates_via_prune_filters_ended_old_sessions(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_state.time, "time", lambda: 2_000_000.0)
    try:
        db.create_session("old_cli", source="cli")
        db.end_session("old_cli", "done")
        db._conn.execute("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_000_000.0, 1_000_010.0, "old_cli"))

        db.create_session("new_cli", source="cli")
        db.end_session("new_cli", "done")
        db._conn.execute("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_990_000.0, 1_990_010.0, "new_cli"))

        db.create_session("old_active", source="cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (1_000_000.0, "old_active"))
        db._conn.commit()

        # Export uses the shared prune/archive candidate selection.
        candidates = db.list_prune_candidates(
            started_before=2_000_000.0 - 5 * 86400, archived=None
        )
        assert [c["id"] for c in candidates] == ["old_cli"]
    finally:
        db.close()


def test_export_candidates_via_prune_ands_source_filter(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_state.time, "time", lambda: 2_000_000.0)
    try:
        for sid, source in [("old_cli", "cli"), ("old_telegram", "telegram")]:
            db.create_session(sid, source=source)
            db.end_session(sid, "done")
            db._conn.execute("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_000_000.0, 1_000_010.0, sid))
        db._conn.commit()

        candidates = db.list_prune_candidates(
            started_before=2_000_000.0 - 5 * 86400,
            source="telegram",
            archived=None,
        )
        assert [c["id"] for c in candidates] == ["old_telegram"]
    finally:
        db.close()


def test_get_compression_lineage_returns_only_compression_chain(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli")
        db.end_session("root", "compression")
        db.create_session("child", source="cli", parent_session_id="root")
        db.end_session("child", "compression")
        db.create_session("tip", source="cli", parent_session_id="child")
        db.create_session("branch", source="cli", parent_session_id="root", model_config={"_branched_from": "root"})

        assert db.get_compression_lineage("tip") == ["root", "child", "tip"]
        assert db.get_compression_lineage("branch") == ["branch"]
    finally:
        db.close()


def test_export_session_lineage_combines_segments(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli", model="m1")
        db.append_message("root", "user", "before compression")
        db.end_session("root", "compression")
        db.create_session("tip", source="cli", parent_session_id="root", model="m1")
        db.append_message("tip", "assistant", "after compression")

        exported = db.export_session_lineage("tip")
        assert exported["id"] == "tip"
        assert exported["lineage_session_ids"] == ["root", "tip"]
        assert [s["id"] for s in exported["segments"]] == ["root", "tip"]
        assert exported["message_count"] == 2
    finally:
        db.close()
