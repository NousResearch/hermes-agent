"""Command receipts survive UI reloads without joining the model transcript."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sqlite3
import threading
import uuid

import pytest

import hermes_state_display
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


def event(db, sid="chat", output="receipt", command="report", event_id=None):
    return db.append_display_event(sid, event_id or str(uuid.uuid4()), command, output)


def test_durable_event_is_not_model_history_or_fts(db):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "original question", api_content="original question plus context")
    db.append_message("chat", "assistant", "original answer")
    messages = db.get_messages("chat")
    model = db.get_messages_as_conversation("chat")
    resumed = db.get_resume_conversations("chat")
    record = event(db, output="DISPLAYONLYSENTINEL")
    assert record["id"].startswith("display:")
    assert record["content"] == "slash:/report\nDISPLAYONLYSENTINEL"
    assert record["role"] == "system"
    assert record["display_kind"] == "command_result"
    assert isinstance(record["timestamp"], float)
    assert db.get_messages("chat") == messages
    assert db.get_messages_as_conversation("chat") == model
    assert db.get_resume_conversations("chat") == resumed
    assert db.get_session("chat")["message_count"] == 2
    assert db.search_messages("DISPLAYONLYSENTINEL") == []
    with SessionDB(db.db_path, read_only=True) as reader:
        assert reader.get_display_events("chat") == [record]
        assert reader.get_display_messages("chat") == [*messages, record]


def test_uuid_retry_is_immutable_and_owner_bound(db):
    db.create_session("chat", source="desktop")
    db.create_session("other", source="desktop")
    key = str(uuid.uuid4())
    first = event(db, event_id=key)
    assert event(db, command="/report", event_id=key) == first
    for args in (("chat", "report", "changed"), ("chat", "other", "receipt"), ("other", "report", "receipt")):
        with pytest.raises(ValueError, match="different owner or payload"):
            db.append_display_event(args[0], key, args[1], args[2])
    assert db.get_display_events("chat") == [first]
    assert db.get_display_events("other") == []
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._execute_write(lambda conn: conn.execute(
            "UPDATE session_display_events SET output = 'changed' WHERE event_id = ?", (key,)))
    assert db.get_display_events("chat") == [first]


@pytest.mark.parametrize("command", ["report arg", "report\nsecret", "//report", "", "/"])
def test_command_must_not_contain_raw_arguments(db, command):
    db.create_session("chat", source="desktop")
    with pytest.raises(ValueError, match="token"):
        event(db, command=command)
    assert db.get_display_events("chat") == []


def test_invalid_uuid_and_missing_session_do_not_create_state(db):
    with pytest.raises(ValueError):
        event(db, event_id="not-a-uuid")
    with pytest.raises(sqlite3.IntegrityError):
        event(db)
    assert db.get_session("chat") is None
    assert db.get_display_events("chat") == []


def test_union_pagination_happens_after_merge_and_ties_are_stable(db, monkeypatch):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "first", timestamp=10)
    monkeypatch.setattr(hermes_state_display, "time", SimpleNamespace(time=lambda: 20.0))
    first = event(db, output="first event")
    second = event(db, output="second event")
    db.append_message("chat", "assistant", "tied message", timestamp=20)
    db.append_message("chat", "user", "last", timestamp=30)
    real = db.get_messages("chat")
    expected = [real[0], real[1], first, second, real[2]]
    assert db.get_display_messages("chat") == expected
    for offset in range(7):
        for limit in (None, 0, 1, 2, 8):
            assert db.get_display_messages("chat", limit=limit, offset=offset) == expected[offset:][:limit]
            assert db.get_display_messages("chat", limit=limit, offset=offset, latest=True) == (
                expected[::-1][offset:][:limit][::-1])
    assert db.get_display_messages("missing") == []
    with pytest.raises(TypeError):
        db.get_display_messages("chat", after_id=1)
    with pytest.raises(ValueError):
        db.get_display_messages("chat", offset=-1)


def test_compaction_dedup_and_rewind_rules_survive_union(db):
    db.create_session("chat", source="desktop")
    for role, content in (("user", "first"), ("assistant", "answer"), ("user", "next"), ("assistant", "tail")):
        db.append_message("chat", role, content)
    history = db.get_messages_as_conversation("chat")
    db.archive_and_compact("chat", [{"role": "user", "content": "summary"}] + history[-2:], tail_count=2)
    receipt = event(db)
    real = db.get_messages("chat", include_compacted=True)
    projected = db.get_display_messages("chat", include_compacted=True)
    assert [row for row in projected if row["id"] != receipt["id"]] == real
    assert projected.count(receipt) == 1
    assert db.get_display_messages("chat", limit=2, offset=1, latest=True, include_compacted=True) == projected[-3:-1]
    active_before = db.get_messages("chat")
    doomed = db.append_message("chat", "user", "undone")
    db.rewind_to_message("chat", doomed)
    assert db.get_messages("chat") == active_before
    assert not any(row["content"] == "undone" for row in db.get_display_messages("chat", include_compacted=True))


def test_command_only_list_opt_in_counts_and_empty_cleanup(db):
    for sid in ("chat", "empty", "ordinary"):
        db.create_session(sid, source="desktop")
    receipt = event(db)
    db.append_message("ordinary", "user", "hello")
    db.end_session("chat", "tui_shutdown")
    db.end_session("empty", "tui_shutdown")
    assert db.get_session("chat")["message_count"] == 0
    assert not db.delete_session_if_empty("chat")
    assert db.count_empty_sessions() == 1
    assert db.delete_empty_sessions() == 1
    assert db.get_display_events("chat") == [receipt]
    assert [row["id"] for row in db.list_sessions_rich(min_message_count=1)] == ["ordinary"]
    assert db.session_count(min_message_count=1) == 1
    for activity_order in (False, True):
        listed = db.list_sessions_rich(min_message_count=1, include_display_events=True,
                                       order_by_last_active=activity_order)
        assert {row["id"] for row in listed} == {"chat", "ordinary"}
        assert next(row for row in listed if row["id"] == "chat")["message_count"] == 0
    assert db.session_count(min_message_count=1, include_display_events=True) == 2
    assert db.session_count(min_message_count=2, include_display_events=True) == 0
    db.set_session_hidden("chat", True)
    assert [row["id"] for row in db.list_sessions_rich(min_message_count=1, include_display_events=True)] == ["ordinary"]


@pytest.mark.parametrize("bulk", [False, True])
def test_explicit_delete_cascades_display_events(db, bulk):
    db.create_session("chat", source="desktop")
    db.create_session("other", source="desktop")
    event(db)
    other = event(db, sid="other")
    if bulk:
        assert db.delete_sessions(["chat"]) == 1
    else:
        assert db.delete_session("chat")
    assert db.get_display_events("chat") == []
    assert db.get_display_events("other") == [other]


def test_concurrent_retry_uses_one_database_transaction(db):
    db.create_session("chat", source="desktop")
    key = str(uuid.uuid4())
    barrier = threading.Barrier(2)
    with SessionDB(db.db_path) as sibling:
        def append(database):
            barrier.wait(timeout=10)
            return event(database, event_id=key)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = list(pool.map(append, (db, sibling)))
    assert first == second
    assert db.get_display_events("chat") == [first]
    assert db.get_session("chat")["message_count"] == 0


def test_old_readonly_db_falls_back_without_creating_table(db):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "old message")
    db._execute_write(lambda conn: conn.execute("DROP TABLE session_display_events"))
    with SessionDB(db.db_path, read_only=True) as reader:
        assert reader.get_display_events("chat") == []
        for compacted in (False, True):
            assert reader.get_display_messages("chat", include_compacted=compacted) == reader.get_messages(
                "chat", include_compacted=compacted)
        assert reader.session_count(min_message_count=1, include_display_events=True) == 1
        assert len(reader.list_sessions_rich(min_message_count=1, include_display_events=True)) == 1
        assert reader.count_empty_sessions() == 0
        assert not reader._display_events_available()
        with pytest.raises(sqlite3.OperationalError):
            event(reader)


def test_readonly_legacy_compaction_fallback_preserves_dedup(db):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "original", timestamp=10)
    history = db.get_messages_as_conversation("chat")
    db.archive_and_compact("chat", history, tail_count=1)
    receipt = event(db)
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_order = NULL, display_identity = NULL"))
    with SessionDB(db.db_path, read_only=True) as reader:
        expected = reader.get_messages("chat", include_compacted=True) + [receipt]
        assert reader.get_display_messages("chat", include_compacted=True) == expected
        assert reader.get_display_messages("chat", limit=1, latest=True, include_compacted=True) == [receipt]


def test_compression_tip_recovers_ancestor_events_but_explicit_branch_does_not(db):
    db.create_session("root", source="desktop")
    ancestor = event(db, sid="root", output="before compression")
    db.append_message("root", "user", "original turn")
    db.end_session("root", "compression")
    db.create_session("tip", source="desktop", parent_session_id="root")
    db.append_message("tip", "assistant", "continued turn")
    current = event(db, sid="tip", output="after compression")
    db.create_session("branch", source="desktop", parent_session_id="root",
                      model_config={"_branched_from": "root"})
    branch = event(db, sid="branch", output="branch receipt")
    with SessionDB(db.db_path, read_only=True) as reader:
        assert reader.resolve_resume_session_id("root") == "tip"
        assert reader.get_display_events("tip") == [current]
        assert reader.get_display_events("tip", include_ancestors=True) == [ancestor, current]
        for compacted in (False, True):
            page = reader.get_display_messages("tip", include_compacted=compacted)
            assert [row for row in page if row.get("display_kind") == "command_result"] == [ancestor, current]
            assert reader.get_display_messages("tip", limit=1, offset=0, include_compacted=compacted) == [ancestor]
            assert reader.get_display_messages("tip", limit=1, latest=True, include_compacted=compacted) == [current]
            assert reader.get_display_messages("branch", include_compacted=compacted) == [branch]
        assert reader.get_display_events("branch", include_ancestors=True) == [branch]
        assert reader.get_display_events("root") == [ancestor]
        assert not any(row.get("display_kind") == "command_result"
                       for row in reader.get_messages_as_conversation("tip"))


def test_compressed_branch_only_inherits_its_own_branch_events(db):
    db.create_session("root", source="desktop")
    event(db, sid="root", output="must not cross branch")
    db.create_session("branch", source="desktop", parent_session_id="root",
                      model_config={"_branched_from": "root"})
    own = event(db, sid="branch", output="branch-local")
    db.end_session("branch", "compression")
    db.create_session("branch-tip", source="desktop", parent_session_id="branch")
    assert db.get_display_events("branch-tip") == []
    assert db.get_display_events("branch-tip", include_ancestors=True) == [own]
    assert db.get_display_messages("branch-tip") == [own]
