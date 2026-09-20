"""Persistent tags belong to a profile store and a compression conversation."""
import sqlite3

import pytest

from hermes_state import SessionDB


def test_tags_survive_reopen_and_follow_only_compression_lineage(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("root", source="desktop")
        assert db.set_session_tag("root", "  Research  ", True) == ["Research"]
        db.end_session("root", "compression")
        db.create_session("middle", source="desktop", parent_session_id="root")
        db.end_session("middle", "compression")
        db.create_session("tip", source="desktop", parent_session_id="middle")
        for sid, config, source in [
            ("branch", {"_branched_from": "root"}, "desktop"),
            ("delegate", {"_delegate_from": "root"}, "subagent"),
            ("reset", {"_reset_from": "root"}, "desktop"),
            ("tool", {}, "tool"),
        ]:
            db.create_session(sid, source=source, parent_session_id="root", model_config=config)
            assert db.get_session_tags(sid) == []
        assert db.get_session_tags("tip") == ["Research"]
        assert db.set_session_tag("middle", "Todo", True) == ["Research", "Todo"]
        assert db.set_session_tag("tip", "Research", False) == ["Todo"]
        assert db.get_session_tags("root") == ["Todo"]
    finally:
        db.close()
    db = SessionDB(path)
    try:
        assert db.list_session_tags() == ["Research", "Todo"]
        rows = db.list_sessions_rich(include_children=True)
        assert {r["id"]: r["tags"] for r in rows}["middle"] == ["Todo"]
        projected = db.list_sessions_rich()
        assert next(r for r in projected if r["id"] == "tip")["tags"] == ["Todo"]
        assert db.set_session_tag("tip", "Todo", False) == []
        assert db.list_session_tags() == ["Research", "Todo"]
    finally:
        db.close()


@pytest.mark.parametrize("marker,source", [
    ("_branched_from", "desktop"), ("_reset_from", "desktop"),
    ("_delegate_from", "subagent"),
])
def test_published_compression_inherits_tags_not_ancestral_fork_tags(tmp_path, marker, source):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("ancestor", source="desktop")
        db.set_session_tag("ancestor", "Ancestor", True)
        db.end_session("ancestor", "compression")
        config = {marker: "ancestor"}
        db.create_session("fork", source=source, parent_session_id="ancestor", model_config=config)
        assert db.get_session_tags("fork") == []
        db.set_session_tag("fork", "Fork", True)
        parent = "fork"
        for child in ("compressed", "tip"):
            db.publish_compression_child(
                parent_session_id=parent, child_session_id=child, source=source,
                model_config=config, messages=[{"role": "user", "content": "summary"}],
                require_compression_lease=False)
            assert db.get_session_tags(child) == ["Fork"]
            parent = child
        assert db.set_session_tag("tip", "New", True) == ["Fork", "New"]
        assert db.get_session_tags("fork") == ["Fork", "New"]
        assert db.get_session_tags("ancestor") == ["Ancestor"]


def test_tag_tables_upgrade_old_store_and_read_only_is_safe(tmp_path):
    path = tmp_path / "old.db"
    with SessionDB(path) as db:
        db.create_session("old", source="desktop")
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE session_tags")
        conn.execute("DROP TABLE session_tag_catalog")
    with SessionDB(path, read_only=True) as db:
        assert db.list_session_tags() == []
        assert db.get_session_tags("old") == []
        assert db.list_sessions_rich()[0]["tags"] == []
    with SessionDB(path) as db:
        assert db.set_session_tag("old", "Upgraded", True) == ["Upgraded"]
    with SessionDB(path, read_only=True) as db:
        assert db.get_session_tags("old") == ["Upgraded"]


def test_tag_validation_idempotence_and_no_count_cap(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", source="desktop")
        for value in ["", "   ", "x" * 65, "a\nb", "\ttag", "a\x00b", "a\x7fb", "a\x85b", None, 7]:
            with pytest.raises(ValueError):
                db.set_session_tag("s", value, True)
        assert db.list_session_tags() == []
        with pytest.raises(ValueError):
            db.set_session_tag("missing", "Ghost", True)
        assert db.list_session_tags() == []
        for i in range(75):
            db.set_session_tag("s", f"tag-{i:03}", True)
        assert len(db.get_session_tags("s")) == 75
        assert len(db.set_session_tag("s", "tag-000", True)) == 75
        assert len(db.set_session_tag("s", "unknown", False)) == 75
        assert "unknown" not in db.list_session_tags()
        assert "é" * 64 in db.set_session_tag("s", "é" * 64, True)
    finally:
        db.close()
