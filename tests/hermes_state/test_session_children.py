"""Regression tests for parent/child session classification and grouping."""

from __future__ import annotations

from hermes_state import SessionDB


def test_session_children_group_focused_before_read_only_subagents(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        db.create_session(
            "focused",
            source="cli",
            parent_session_id="parent",
            model_config={
                "_focused_continuation_of": "parent",
                "_child_kind": "focused_continuation",
            },
        )
        db.append_message("focused", "assistant", "focused continuation")
        db.create_session(
            "branch",
            source="cli",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        db.append_message("branch", "user", "branch work")
        db.create_session(
            "active_subagent",
            source="subagent",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.append_message("active_subagent", "assistant", "watch me")
        db.create_session(
            "done_subagent",
            source="subagent",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.append_message("done_subagent", "assistant", "done")
        db.end_session("done_subagent", "completed")
        db.create_session(
            "stale_subagent",
            source="subagent",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.append_message("stale_subagent", "assistant", "failed")
        db.end_session("stale_subagent", "timeout")

        grouped = db.get_session_children("parent")

        assert grouped["parent_session_id"] == "parent"
        assert [s["id"] for s in grouped["focused"]] == ["focused"]
        assert [s["id"] for s in grouped["branches"]] == ["branch"]
        assert [s["id"] for s in grouped["subagents"]["active"]] == ["active_subagent"]
        assert [s["id"] for s in grouped["subagents"]["completed"]] == ["done_subagent"]
        assert grouped["subagents"]["stale_count"] == 1
        assert grouped["subagents"]["stale"] == []
        assert [s["id"] for s in grouped["ordered_children"]] == [
            "focused",
            "branch",
            "active_subagent",
            "done_subagent",
        ]
    finally:
        db.close()


def test_session_children_can_include_stale_subagents(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        db.create_session(
            "stale_subagent",
            source="subagent",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.end_session("stale_subagent", "failed")

        grouped = db.get_session_children("parent", include_stale=True)

        assert grouped["subagents"]["stale_count"] == 1
        assert [s["id"] for s in grouped["subagents"]["stale"]] == ["stale_subagent"]
        assert [s["id"] for s in grouped["ordered_children"]] == ["stale_subagent"]
    finally:
        db.close()
