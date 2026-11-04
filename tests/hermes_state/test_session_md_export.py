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




def test_get_compression_lineage_returns_only_compression_chain(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli")
        db.end_session("root", "compression")
        db.create_session("child", source="cli", parent_session_id="root")
        db.end_session("child", "compression")
        db.create_session("tip", source="cli", parent_session_id="child")
        db.create_session("branch", source="cli", parent_session_id="root", model_config={"_branched_from": "root"})
        db.create_session("delegate", source="delegate", parent_session_id="child", model_config={"_delegate_from": "child"})
        db.create_session("tool", source="tool", parent_session_id="child")

        assert db.get_compression_lineage("tip") == ["root", "child", "tip"]
        assert db.get_compression_lineage("branch") == ["branch"]
        assert db.get_compression_lineage("delegate") == ["delegate"]
        assert db.get_compression_lineage("tool") == ["tool"]
    finally:
        db.close()


def test_fork_children_created_before_continuation_do_not_hijack_lineage(tmp_path):
    # Regression: the forward walk used to accept any non-branch child as the
    # compression continuation. A delegate/tool child spawned BEFORE the real
    # continuation row (the common runtime ordering — the subagent exists
    # before compression rotates the session) was picked as the successor,
    # so lineage and session .md export followed the subagent's transcript.
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli")
        db.append_message("root", role="user", content="root msg")
        db.create_session(
            "delegate",
            source="delegate",
            parent_session_id="root",
            model_config={"_delegate_from": "root"},
        )
        db.append_message("delegate", role="user", content="delegate private msg")
        db.end_session("root", "compression")
        db.create_session("continuation", source="cli", parent_session_id="root")
        db.append_message("continuation", role="user", content="continuation msg")

        db.create_session("root2", source="cli")
        db.create_session("toolchild", source="tool", parent_session_id="root2")
        db.end_session("root2", "compression")
        db.create_session("cont2", source="cli", parent_session_id="root2")

        assert db.get_compression_lineage("root") == ["root", "continuation"]
        assert db.get_compression_lineage("continuation") == ["root", "continuation"]
        assert db.get_compression_lineage("root2") == ["root2", "cont2"]

        exported = db.export_session_lineage("root")
        assert exported is not None
        assert exported["lineage_session_ids"] == ["root", "continuation"]
        contents = [
            m.get("content")
            for seg in exported["segments"]
            for m in (seg.get("messages") or [])
        ]
        assert contents == ["root msg", "continuation msg"]
    finally:
        db.close()

