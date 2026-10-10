"""A compressed ``--source tool`` conversation stays one conversation.

Regression for #112550 / #113826: the compression child now inherits the parent's ``tool`` source, but
every lineage predicate treated any ``tool`` child as a separate conversation. The rotated tip was
invisible, so the turn lease re-keyed mid-turn (the turn ended ``session_persistence_failed:turn_lease``
and saved nothing), stale-agent recovery reopened the compressed-away parent as an orphan, and resume
landed on the parent. Each follow-up message compressed again from the parent and failed the same way.
"""

from contextlib import closing
import os
import time

import pytest

from hermes_state import SessionDB
from hermes_state_common import _tool_fork_sql


def _rotate(db: SessionDB, parent: str, child: str, source: str = "tool") -> None:
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source=source,
        messages=[{"role": "user", "content": "[summary]"}], require_compression_lease=False,
    )


@pytest.fixture
def db(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as database:
        database.create_session("root", source="tool")
        database.append_message("root", "user", "automated repair prompt")
        yield database


def test_turn_lease_key_survives_rotation_of_a_tool_conversation(db):
    holder = f"pid={os.getpid()}:turn=live"
    assert db.try_acquire_session_turn_lease("root", holder, ttl_seconds=5)

    _rotate(db, "root", "tip")

    assert db._session_turn_lease_key("tip") == "root"
    assert db.refresh_session_turn_lease("tip", holder, ttl_seconds=5)
    assert not db.try_acquire_session_turn_lease("tip", f"pid={os.getpid()}:turn=other", ttl_seconds=5)
    db.release_session_turn_lease("tip", holder)


def test_tool_continuation_is_the_lineage_tip_and_resume_target(db):
    _rotate(db, "root", "mid")
    _rotate(db, "mid", "tip")

    assert db.get_compression_tip("root") == "tip"
    assert db.get_compression_lineage("mid") == ["root", "mid", "tip"]
    assert db.resolve_resume_session_id("root") == "tip"
    assert db.is_explicit_fork_child("tip") is False


def test_compressed_tool_parent_is_not_reopened_as_an_orphan(db):
    _rotate(db, "root", "tip")

    assert db.find_live_compression_child("root")["id"] == "tip"
    assert db.reopen_orphaned_compression_session("root") is False
    assert db.get_session("root")["end_reason"] == "compression"


def test_prune_keeps_the_compressed_start_of_a_live_tool_conversation(db):
    old = time.time() - 120 * 86400
    db._conn.execute("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = 'root'", (old, old))
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = 'root'", (old,))
    db._conn.commit()
    _rotate(db, "root", "tip")
    db._conn.execute("UPDATE sessions SET ended_at = ?, last_activity_at = ? WHERE id = 'root'", (old + 60, old + 60))
    db._conn.commit()
    db.append_message("tip", "user", "still running today")

    assert "root" not in {c["id"] for c in db.list_prune_candidates(older_than_days=90, whole_lineages=True)}
    db.prune_sessions(older_than_days=90)
    assert db.get_session("root") is not None


def test_unmarked_tool_children_stay_separate_conversations(db):
    """Only the published continuation carries the parent-bound marker: an integration's own ``tool`` row
    under the same parent, or a continuation that inherited a marker naming another session, is not one."""
    db.create_session("side-run", source="tool", parent_session_id="root")
    _rotate(db, "root", "tip")
    db.create_session(
        "inherited", source="tool", parent_session_id="root", model_config={"_compressed_from": "elsewhere"},
    )

    assert db.get_compression_lineage("side-run") == ["side-run"]
    assert db.get_compression_lineage("inherited") == ["inherited"]
    assert db.is_explicit_fork_child("side-run") is True
    assert db.get_compression_tip("root") == "tip"
    assert db.find_live_compression_child("root")["id"] == "tip"


def test_tool_child_of_a_non_tool_parent_is_unchanged(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("chat", source="cli")
        db.create_session("integration", source="tool", parent_session_id="chat")
        _rotate(db, "chat", "chat-2", source="cli")

        assert db.get_compression_tip("chat") == "chat-2"
        assert db.get_compression_lineage("integration") == ["integration"]
        assert db.get_session("chat-2")["model_config"] is None


def test_inherited_marker_is_restamped_with_the_new_parent(db):
    """A rotated agent passes its own ``model_config`` (marker naming the grand-parent): publish re-binds it."""
    _rotate(db, "root", "mid")
    db.publish_compression_child(
        parent_session_id="mid", child_session_id="tip", source="tool", model_config={"_compressed_from": "root"},
        messages=[{"role": "user", "content": "[summary]"}], require_compression_lease=False,
    )

    assert db.get_compression_lineage("tip") == ["root", "mid", "tip"]
    assert db._session_turn_lease_key("tip") == "root"


@pytest.mark.parametrize("source", ["tool", "cli"])
@pytest.mark.parametrize("config", [
    None, {}, {"_compressed_from": "P"}, {"_compressed_from": "other"}, {"_compressed_from": "P", "_branched_from": "P"},
    {"_compressed_from": "P", "_delegate_from": "P"}, {"_compressed_from": "P", "_reset_from": "P"},
    {"_branched_from": "other"}, {"_delegate_from": "P"},
])
def test_python_fork_predicate_matches_the_sql_continuation_filter(tmp_path, source, config):
    """``_is_explicit_fork_child_row`` and ``_NON_CONTINUATION_CHILD_FILTER_SQL`` must agree on every row shape."""
    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("P", source="cli")
        db.create_session("C", source=source, parent_session_id="P", model_config=config)
        db.create_session("orphan", source=source, model_config=config)
        sql_continues = db._conn.execute(
            "SELECT 1 FROM sessions WHERE parent_session_id = ?" + db._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias=""),
            ("P",) * 4).fetchone() is not None
        assert sql_continues is not db._is_explicit_fork_child_row(db.get_session("C"), include_reset=True)
        orphan_is_tool_fork = db._conn.execute(
            f"SELECT 1 FROM sessions WHERE id = 'orphan' AND {_tool_fork_sql()}").fetchone() is not None
        assert orphan_is_tool_fork is (source == "tool")
        assert db._is_explicit_fork_child_row(db.get_session("orphan"), include_reset=True) is (
            source == "tool" or bool(config and any(k != "_compressed_from" for k in config)))
