"""Production lifecycle coverage for opt-in session topic segmentation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.session_topics import (
    TOPIC_SESSION_CONFIG_KEY,
    initialize_topic_segmentation,
    match_existing_topic,
    merge_topic_prompt_context,
    parse_topic_signal,
    prepare_topic_turn,
    process_turn_topic,
    refresh_topic_segmentation,
)
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path):
    store = SessionDB(tmp_path / "state.db")
    store.create_session("session", source="test", model="test")
    yield store
    store.close()


def _agent(db: SessionDB, *, enabled: bool = True, session_id: str = "session"):
    return SimpleNamespace(
        _session_db=db,
        session_id=session_id,
        _topic_segmentation_enabled=enabled,
        _topic_segmentation_default=enabled,
        _active_topic_id=None,
        _persist_user_message_idx=None,
        _active_session_turn_lease_holder=None,
        _session_init_model_config={},
        context_compressor=SimpleNamespace(),
    )


def test_fresh_schema_and_legacy_column_reconciliation(tmp_path: Path):
    legacy = tmp_path / "legacy.db"
    original = SessionDB(legacy)
    original.close()
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        DROP INDEX idx_messages_topic_active;
        DROP INDEX idx_session_topics_conversation;
        ALTER TABLE messages DROP COLUMN topic_id;
        DROP TABLE session_topics;
        """
    )
    conn.close()

    store = SessionDB(legacy)
    try:
        with sqlite3.connect(legacy) as check:
            tables = {row[0] for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            columns = {row[1] for row in check.execute("PRAGMA table_info(messages)")}
        assert "session_topics" in tables
        assert "topic_id" in columns
    finally:
        store.close()


def test_parse_and_conservative_topic_matching():
    assert parse_topic_signal("answer\nTOPIC: Git Merge") == "Git Merge"
    assert parse_topic_signal("TOPIC:\n") is None
    assert parse_topic_signal("TOPIC: old\nbody\nTOPIC: final") == "final"
    topics = [{"id": 1, "title": "git"}, {"id": 2, "title": "cooking"}]
    git_merge = match_existing_topic("git-merge", topics)
    git_exact = match_existing_topic("GIT", topics)
    assert git_merge is not None and git_merge["id"] == 1
    assert git_exact is not None and git_exact["id"] == 1
    assert match_existing_topic("deployment", topics) is None
    specific = match_existing_topic(
        "git-rebase", [{"id": 1, "title": "git"}, {"id": 2, "title": "git-rebase"}]
    )
    assert specific is not None and specific["id"] == 2


def test_initialize_respects_session_override_and_refresh(db: SessionDB):
    db.patch_session_model_config("session", {TOPIC_SESSION_CONFIG_KEY: False})
    agent = _agent(db)
    initialize_topic_segmentation(
        agent, {"session": {"topic_segmentation": {"enabled": True}}}
    )
    assert agent._topic_segmentation_enabled is False
    assert agent._session_init_model_config[TOPIC_SESSION_CONFIG_KEY] is False

    db.patch_session_model_config("session", {TOPIC_SESSION_CONFIG_KEY: True})
    refresh_topic_segmentation(agent)
    assert agent._topic_segmentation_enabled is True


def test_prepare_creates_topic_adopts_legacy_rows_and_filters_history(db: SessionDB):
    db.append_message("session", role="user", content="legacy question")
    db.append_message("session", role="assistant", content="legacy answer")
    agent = _agent(db)
    messages = db.get_messages_as_conversation(
        "session", include_row_ids=True
    ) + [{"role": "user", "content": "continue legacy question"}]

    rebuilt, user_idx, history = prepare_topic_turn(
        agent, messages, len(messages) - 1, "continue legacy question"
    )

    topic = db.get_active_topic("session")
    assert topic is not None
    assert agent._active_topic_id == topic["id"]
    assert agent.context_compressor._active_topic_id == topic["id"]
    assert rebuilt[user_idx]["_topic_id"] == topic["id"]
    assert [row["content"] for row in history] == ["legacy question", "legacy answer"]
    assert all(row["_topic_id"] == topic["id"] for row in history)


def test_process_retags_only_current_turn_and_next_turn_loads_selected_topic(db: SessionDB):
    first = db.ensure_session_topic("session", "git")
    db.append_message("session", role="user", content="old git", topic_id=first["id"])
    db.append_message("session", role="assistant", content="old answer", topic_id=first["id"])
    current_id = db.append_message(
        "session", role="user", content="make ramen", topic_id=first["id"]
    )
    prefix = db.get_messages_as_conversation(
        "session", include_row_ids=True, topic_id=first["id"]
    )[:-1]
    current = {
        "role": "user", "content": "make ramen", "_row_id": current_id,
        "_topic_id": first["id"],
    }
    assistant = {"role": "assistant", "content": "Boil water.\nTOPIC: cooking"}
    messages = [*prefix, current, assistant]
    agent = _agent(db)
    agent._active_topic_id = first["id"]
    agent._persist_user_message_idx = len(prefix)

    process_turn_topic(agent, messages, assistant["content"])
    db.append_message(
        "session", role="assistant", content=assistant["content"],
        topic_id=int(assistant["_topic_id"]),
    )

    cooking = db.get_active_topic("session")
    assert cooking and cooking["title"] == "cooking"
    assert agent._active_topic_id == cooking["id"]
    assert messages[len(prefix)]["_topic_id"] == cooking["id"]
    assert [m["content"] for m in db.get_messages_as_conversation(
        "session", topic_id=first["id"]
    )] == ["old git", "old answer"]
    assert [m["content"] for m in db.get_messages_as_conversation(
        "session", topic_id=cooking["id"]
    )] == ["make ramen", "Boil water.\nTOPIC: cooking"]


def test_invalid_switch_is_atomic_and_title_is_untouched(db: SessionDB):
    topic = db.ensure_session_topic("session", "git")
    db.set_session_title("session", "User chosen title")

    assert db.set_active_topic("session", 999999) is False
    active = db.get_active_topic("session")
    assert active is not None and active["id"] == topic["id"]
    db.create_topic("session", "cooking")
    assert db.get_session_title("session") == "User chosen title"


def test_topic_scoped_compaction_preserves_other_topic_rows_and_counters(db: SessionDB):
    git = db.ensure_session_topic("session", "git")
    db.append_message("session", role="user", content="git q", topic_id=git["id"])
    db.append_message("session", role="assistant", content="git a", topic_id=git["id"])
    cooking_id = db.create_topic("session", "cooking")
    db.append_message("session", role="user", content="cook q", topic_id=cooking_id)
    db.append_message("session", role="assistant", content="cook a", topic_id=cooking_id)

    count = db.archive_and_compact(
        "session", [{"role": "user", "content": "git summary"}], topic_id=git["id"]
    )

    assert count == 3
    assert [m["content"] for m in db.get_messages_as_conversation(
        "session", topic_id=git["id"]
    )] == ["git summary"]
    assert [m["content"] for m in db.get_messages_as_conversation(
        "session", topic_id=cooking_id
    )] == ["cook q", "cook a"]
    session = db.get_session("session")
    assert session is not None and session["message_count"] == 3


def test_topic_scoped_compaction_combines_held_coverage_with_topic_isolation(
    db: SessionDB,
):
    """A proved held set preserves unseen rows without matching duplicates in other topics."""
    git = db.ensure_session_topic("session", "git")
    git_shared_id = db.append_message(
        "session", role="user", content="shared", topic_id=git["id"]
    )
    db.append_message(
        "session", role="assistant", content="unseen git", topic_id=git["id"]
    )
    git_held_id = db.append_message(
        "session", role="user", content="held git", topic_id=git["id"]
    )
    cooking_id = db.create_topic("session", "cooking")
    db.append_message(
        "session", role="user", content="shared", topic_id=cooking_id
    )

    count = db.archive_and_compact(
        "session",
        [{"role": "user", "content": "git summary"}],
        watermark=db.get_active_message_watermark("session"),
        covered_ids=[git_held_id],
        unresolved_held=[{
            "role": "user", "content": "shared", "_row_id": git_shared_id,
            "_db_persisted": True,
        }],
        topic_id=git["id"],
    )

    assert count == 3
    assert [m["content"] for m in db.get_messages_as_conversation(
        "session", topic_id=git["id"]
    )] == ["git summary", "unseen git"]
    assert [m["content"] for m in db.get_messages_as_conversation(
        "session", topic_id=cooking_id
    )] == ["shared"]
    session = db.get_session("session")
    assert session is not None and session["message_count"] == 3


def test_topics_follow_compression_lineage(db: SessionDB):
    topic = db.ensure_session_topic("session", "git")
    db.end_session("session", "compression")
    db.create_session(
        "continuation", source="test", model="test", parent_session_id="session"
    )
    db.append_message(
        "continuation", role="user", content="continued", topic_id=topic["id"]
    )

    active = db.get_active_topic("continuation")
    assert active is not None and active["id"] == topic["id"]
    assert [m["content"] for m in db.get_messages_as_conversation(
        "continuation", include_ancestors=True, topic_id=topic["id"]
    )] == ["continued"]


def test_deleting_session_removes_unreferenced_topics(db: SessionDB):
    db.ensure_session_topic("session", "git")

    assert db.delete_session("session") is True
    with db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_topics").fetchone()[0] == 0


def test_deleting_compression_root_rehomes_topics_to_surviving_child(db: SessionDB):
    topic = db.ensure_session_topic("session", "git")
    db.end_session("session", "compression")
    db.create_session(
        "continuation", source="test", model="test", parent_session_id="session"
    )
    db.append_message(
        "continuation", role="user", content="continued", topic_id=topic["id"]
    )

    assert db.delete_session("session") is True
    active = db.get_active_topic("continuation")
    assert active is not None and active["id"] == topic["id"]


def test_empty_session_cleanup_removes_topic_metadata(db: SessionDB):
    db.ensure_session_topic("session", "unused")

    assert db.delete_session_if_empty("session") is True
    with db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_topics").fetchone()[0] == 0


def test_cross_profile_move_remaps_topic_ids(tmp_path: Path):
    source = SessionDB(tmp_path / "source.db")
    target = SessionDB(tmp_path / "target.db")
    try:
        source.create_session("moved", source="test", model="test")
        topic = source.ensure_session_topic("moved", "git")
        source.append_message("moved", role="user", content="status", topic_id=topic["id"])
        payload = source.export_session_for_move("moved")
        assert payload is not None and payload["topics"]

        target.create_session("resident", source="test", model="test")
        resident = target.ensure_session_topic("resident", "unrelated")
        assert target.import_moved_session(payload, profile_name="default") == "imported"

        active = target.get_active_topic("moved")
        messages = target.get_messages_as_conversation("moved", include_row_ids=True)
        assert active is not None and active["title"] == "git"
        assert active["id"] != resident["id"]
        assert messages[0]["_topic_id"] == active["id"]
    finally:
        source.close()
        target.close()


def test_prompt_context_is_user_tail_only(db: SessionDB):
    topic = db.ensure_session_topic("session", "git")
    agent = _agent(db)
    agent._active_topic_id = topic["id"]

    merged = merge_topic_prompt_context("plugin note", agent)

    assert merged.startswith("plugin note\n\n[SESSION TOPICS")
    assert "TOPIC: <short-name>" in merged
