from copy import deepcopy
from types import SimpleNamespace

import pytest

from acp_adapter.session import SessionManager
from conversation_store import ConversationConflictError
from hermes_state import SessionDB
from tests.plugins.conversation_store_mutation_fixture import MutationStore


@pytest.fixture
def db_store(tmp_path):
    store = MutationStore()
    db = SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store)
    db.create_session("s1", source="cli")
    try:
        yield db, store
    finally:
        db.close()


def _append(db, messages):
    rows = [deepcopy(message) for message in messages]
    db.append_messages_batch("s1", rows)
    return rows


def test_stale_replace_cannot_overwrite_newer_external_history(db_store):
    db, store = db_store
    original = _append(db, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    observed_ids = [row["_row_id"] for row in original]
    store.append_messages(
        "s1", [{"role": "user", "content": "concurrent"}],
        expected_revision=store.get_revision("s1"),
    )
    before = deepcopy(store.messages["s1"])

    with pytest.raises(ConversationConflictError):
        db.replace_messages(
            "s1", [{"role": "user", "content": "stale"}],
            active_only=True, archive_dropped=True,
            expected_active_ids=observed_ids,
        )

    assert store.messages["s1"] == before
    assert db._read_one("SELECT COUNT(*) FROM messages")[0] == 0


def test_replace_rebinds_ids_and_archives_dropped_rows(db_store):
    db, store = db_store
    original = _append(db, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    replacement = [{"role": "user", "content": "one"}]
    db.replace_messages(
        "s1", replacement, active_only=True, archive_dropped=True,
        expected_active_ids=[row["_row_id"] for row in original],
    )

    assert isinstance(replacement[0]["_row_id"], int)
    assert replacement[0]["_row_id"] not in {row["_row_id"] for row in original}
    assert sum(1 for row in store.messages["s1"] if row.get("active", True)) == 1
    assert sum(1 for row in store.messages["s1"] if not row.get("active", True)) == 2


def test_external_rewind_preserves_result_shape(db_store):
    db, store = db_store
    rows = _append(db, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ])
    expected_ids = [row["_row_id"] for row in rows]
    result = db.rewind_to_message(
        "s1", rows[2]["_row_id"], expected_active_ids=expected_ids,
        expected_target_content="three",
    )

    assert result["rewound_count"] == 2
    assert result["target_message"]["content"] == "three"
    assert store.active_message_ids("s1") == expected_ids[:2]
    assert db._read_one("SELECT COUNT(*) FROM messages")[0] == 0


def test_reactions_and_display_metadata_are_canonical(db_store):
    db, store = db_store
    row = _append(db, [{"role": "assistant", "content": "hello"}])[0]
    assert db.set_latest_matching_message_display_kind(
        "s1", role="assistant", content="hello", display_kind="notice",
        display_metadata={"kind": "test"},
    )
    assert store.messages["s1"][0]["display_kind"] == "notice"

    reactions = db.set_message_reaction("s1", row["_row_id"], "👍")
    assert reactions and reactions[0]["emoji"] == "👍"
    assert db.get_message_reactions("s1", row["_row_id"])[0]["author"] == "user"
    assert db.set_message_reaction("s1", row["_row_id"], "👍") == []


def test_visible_state_and_branch_seed_live_in_provider(db_store):
    db, store = db_store
    parent_rows = _append(db, [{"role": "user", "content": "seed"}])
    assert db.set_session_title("s1", "Parent")
    assert db.set_session_archived("s1", True)
    assert db.set_session_hidden("s1", True)
    assert db.set_session_pinned("s1", True)
    assert store.conversations["s1"]["title"] == "Parent"
    assert store.conversations["s1"]["archived"] is True
    assert store.conversations["s1"]["pinned"] is True
    assert store.conversations["s1"]["hidden"] is False

    db.create_session("child", source="cli", parent_session_id="s1")
    copied = [{"role": row["role"], "content": row["content"]} for row in parent_rows]
    db.append_messages_batch("child", copied)
    assert db.set_session_title("child", "Branch")
    assert store.conversations["child"]["parent_session_id"] == "s1"
    assert [row["content"] for row in store.messages["child"]] == ["seed"]
    assert store.conversations["child"]["title"] == "Branch"
    assert db._read_one("SELECT COUNT(*) FROM messages")[0] == 0

def test_acp_restore_keeps_external_snapshot_ids_out_of_model_history(tmp_path):
    store = MutationStore()
    db = SessionDB(db_path=tmp_path / "acp-shadow.db", conversation_store=store)
    db.create_session("acp-1", source="acp", model="fake", model_config={"cwd": "."})
    rows = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    db.append_messages_batch("acp-1", rows)
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="fake"), db=db)
    try:
        state = manager.get_session("acp-1")
        assert state is not None
        assert state.observed_active_ids == [row["_row_id"] for row in rows]
        assert all("_row_id" not in message for message in state.history)

        store.append_messages(
            "acp-1", [{"role": "user", "content": "concurrent"}],
            expected_revision=store.get_revision("acp-1"),
        )
        before = deepcopy(store.messages["acp-1"])
        manager.save_session("acp-1")
        assert store.messages["acp-1"] == before
    finally:
        db.close()
