"""One rule decides whether a child row continues its parent's conversation.

A fork marker (``_branched_from``, ``_delegate_from``, ``_reset_from``) makes a child a fork only
when it names the child's own parent. Rotating compaction (``compression.in_place: false``)
publishes each continuation with the agent's init ``model_config``, so every segment of a delegated
child carries ``_delegate_from=<delegator>``; the worker compress verb copies every marker the same
way. Every reader (tip, chain, resume target, session list, turn lease, resume lineage, the
subagent transcript lookup) must see those segments as one conversation.
"""

import pytest

from hermes_state import SessionDB
from tools.delegate_tool_sessions import SUBAGENT_ID_KEY, stored_child_session


def _rotate(db: SessionDB, parent_id: str, child_id: str, model_config: dict, source: str) -> None:
    holder = f"holder-{parent_id}"
    assert db.try_acquire_compression_lock(parent_id, holder, ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent_id, child_session_id=child_id, source=source,
        messages=[{"role": "user", "content": f"summary of {parent_id}"}],
        model_config=dict(model_config), compression_lock_holder=holder,
    )
    db.release_compression_lock(parent_id, holder)


def _listed(db: SessionDB, **kwargs) -> list[tuple[str, list[str]]]:
    return sorted((row["id"], row.get("_lineage_ids") or [row["id"]])
                  for row in db.list_sessions_rich(limit=50, **kwargs))


def test_rotated_delegate_child_is_one_conversation_to_every_reader(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("delegator", source="cli")
        db.append_message("delegator", "user", "delegate a task")
        child_config = {"max_iterations": 50, "_delegate_from": "delegator", SUBAGENT_ID_KEY: "sa-1"}
        db.create_session("child-0", source="subagent", parent_session_id="delegator",
                          model_config=child_config)
        db.append_message("child-0", "user", "task")
        _rotate(db, "child-0", "child-1", child_config, "subagent")
        _rotate(db, "child-1", "child-2", child_config, "subagent")
        db.append_message("child-2", "user", "newest turn")
        _rotate(db, "delegator", "delegator-1", {"max_iterations": 90}, "cli")

        lineage = ["child-0", "child-1", "child-2"]
        assert db._resume_lineage_ids("child-2") == lineage
        assert db._session_turn_lease_key("child-2") == "child-0"
        assert db.get_compression_chain("child-0") == lineage
        assert db.get_compression_tip("child-0") == "child-2"
        assert db.resolve_resume_session_id("child-0") == "child-2"
        # The delegator's own first hop names it, so its tip never follows the delegate child.
        assert db.get_compression_tip("delegator") == "delegator-1"
        assert stored_child_session(db, "sa-1", ["delegator"]) == "child-2"
        assert _listed(db, include_subagents=True) == [
            ("child-2", lineage), ("delegator-1", ["delegator", "delegator-1"])]
    finally:
        db.close()


@pytest.mark.parametrize(("marker", "boundary"), [("_reset_from", "session_reset"), ("_branched_from", "branched")])
def test_rotated_reset_or_branch_child_keeps_one_listed_row(tmp_path, marker, boundary):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("before", source="cli")
        db.append_message("before", "user", "old conversation")
        db.end_session("before", boundary)
        db.create_session("after-0", source="cli", parent_session_id="before", model_config={marker: "before"})
        db.append_message("after-0", "user", "new conversation")
        _rotate(db, "after-0", "after-1", {marker: "before"}, "cli")

        assert db._resume_lineage_ids("after-1") == ["after-0", "after-1"]
        assert db.get_compression_tip("after-0") == "after-1"
        assert db.resolve_resume_session_id("after-0") == "after-1"
        assert db.get_compression_tip("before") == "before"
        assert _listed(db) == [("after-1", ["after-0", "after-1"]), ("before", ["before"])]
    finally:
        db.close()


def test_rotated_branch_still_owns_delegates_spawned_before_rotation(tmp_path):
    from hermes_state_delegate_children import delegate_descendant_ids

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("origin", source="cli")
        db.end_session("origin", "branched")
        db.create_session("branch-0", source="cli", parent_session_id="origin",
                          model_config={"_branched_from": "origin"})
        db.append_message("branch-0", "user", "branch work")
        db.create_session("worker", source="subagent", parent_session_id="branch-0",
                          model_config={"_delegate_from": "branch-0"})
        _rotate(db, "branch-0", "branch-1", {"_branched_from": "origin"}, "cli")

        assert delegate_descendant_ids(db, "branch-1") == ["worker"]
        assert delegate_descendant_ids(db, "origin") == []
    finally:
        db.close()
