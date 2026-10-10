"""Atomic pre-delete identity capture for plugins; independent follow-up to #124988."""

import os
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionActiveWriteGuardError


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "profile" / "state.db")
    yield database
    database.close()


def test_receipt_preserves_only_routing_identifiers(db):
    db.create_session("session-a", source="example", chat_id="chat-a", thread_id="thread-a")
    db.append_message("session-a", role="user", content="must never enter deletion payload")
    assert db.delete_session("session-a", deletion_origin="user_rpc") is True
    entries = db.list_session_deletion_receipts()
    assert len(entries) == 1
    context = entries[0]["deletion"]
    assert context["reason"] == "explicit_user"
    assert context["surface"] == "session_rpc"
    assert context["profile_home"] == str(db.db_path.resolve().parent)
    assert context["store_id"] == str(db.db_path.resolve())
    assert context["identities"] == [{
        "id": "session-a", "source": "example", "session_key": None,
        "chat_id": "chat-a", "chat_type": None, "thread_id": "thread-a", "parent_session_id": None,
    }]
    assert "must never" not in str(context)
    assert db.get_session_deletion_receipt(context["operation_id"]) == context
    assert db.get_session_deletion_receipt("unknown") is None
    assert db.list_session_deletion_receipts(after_sequence=entries[0]["sequence"]) == []


@pytest.mark.parametrize("bulk", [False, True])
def test_rollback_removes_prepared_receipt(db, monkeypatch, bulk):
    db.create_session("rollback", source="example")
    def fail_after_delete(conn):
        raise sqlite3.IntegrityError("synthetic transaction failure")
    monkeypatch.setattr(db, "_delete_unreferenced_system_prompts", fail_after_delete)
    with pytest.raises(sqlite3.IntegrityError):
        if bulk:
            db.delete_sessions(["rollback"], deletion_origin="user_rest")
        else:
            db.delete_session("rollback", deletion_origin="user_rpc")
    assert db.get_session("rollback") is not None
    assert db.list_session_deletion_receipts() == []


def test_noops_and_snapshot_fences_do_not_prepare(db):
    assert db.delete_session("absent", deletion_origin="user_rpc") is False
    assert db.delete_sessions(["absent"], deletion_origin="user_rest") == 0
    db.create_session("fenced", source="example")
    assert db.delete_session("fenced", expected_delete_ids=["wrong"], deletion_origin="user_rpc") is False
    assert db.delete_session("fenced", expected_display_messages={"fenced": [{"content": "drift"}]},
                             deletion_origin="user_rpc") is False
    assert db.list_session_deletion_receipts() == []


@pytest.mark.parametrize("origin", ["unknown", "prune", "sweep", "compensation", "profile_move", "typo", "USER_RPC"])
def test_non_user_origins_never_authorize_external_cleanup(db, origin):
    db.create_session("internal", source="example")
    assert db.delete_session("internal", deletion_origin=origin) is True
    assert db.list_session_deletion_receipts() == []


def test_chain_and_recursive_delegate_identities_are_exact(db):
    db.create_session("root", source="example", thread_id="thread-a")
    db.end_session("root", "compression")
    db.create_session("tip", source="example", parent_session_id="root")
    db.create_session("child", source="subagent", parent_session_id="tip", model_config={"_delegate_from": "tip"})
    db.create_session("grandchild", source="subagent", parent_session_id="child", model_config={"_delegate_from": "child"})
    db.create_session("branch", source="example", parent_session_id="root",
                      model_config={"_branched_from": "root"})
    db.create_session("other", source="example")
    assert db.delete_sessions(["tip", "tip", "missing"], include_compression_chain=True,
                              deletion_origin="user_rest") == 1
    context = db.list_session_deletion_receipts()[0]["deletion"]
    assert [row["id"] for row in context["identities"]] == ["child", "grandchild", "root", "tip"]
    assert db.get_session("branch")["parent_session_id"] is None
    assert db.get_session("other") is not None


def test_active_guards_prevent_preparation_and_bulk_records_only_deleted_rows(db):
    db.create_session("active", source="example")
    db.create_session("idle", source="example")
    holder = f"pid={os.getpid()}:turn=receipt"
    assert db.try_acquire_session_turn_lease("active", holder, ttl_seconds=300)
    with pytest.raises(SessionActiveWriteGuardError):
        db.delete_session("active", deletion_origin="user_rpc", exclude_active_write_guards=True)
    assert db.list_session_deletion_receipts() == []
    skipped = []
    assert db.delete_sessions(["active", "idle"], deletion_origin="user_rest",
                              exclude_active_write_guards=True, skipped_ids=skipped) == 1
    assert skipped == ["active"]
    assert [row["id"] for row in db.list_session_deletion_receipts()[0]["deletion"]["identities"]] == ["idle"]


@pytest.mark.parametrize("guard", ["turn", "compression"])
def test_guard_on_chain_delegate_prevents_receipt_for_whole_conversation(db, guard):
    db.create_session("root", source="example")
    db.end_session("root", "compression")
    db.create_session("tip", source="example", parent_session_id="root")
    db.create_session("delegate", source="example", parent_session_id="tip", model_config={"_delegate_from": "tip"})
    holder = f"pid={os.getpid()}:{guard}=receipt"
    acquire = db.try_acquire_session_turn_lease if guard == "turn" else db.try_acquire_compression_lock
    assert acquire("delegate", holder, ttl_seconds=300)
    skipped = []
    assert db.delete_sessions(["tip"], include_compression_chain=True, exclude_active_write_guards=True,
                              skipped_ids=skipped, deletion_origin="user_rest") == 0
    assert skipped == ["tip"]
    assert db.list_session_deletion_receipts() == []
    assert all(db.get_session(sid) is not None for sid in ("root", "tip", "delegate"))


def test_same_session_id_in_other_profile_is_not_receipted(tmp_path):
    with SessionDB(tmp_path / "one" / "state.db") as one, SessionDB(tmp_path / "two" / "state.db") as two:
        for database in (one, two):
            database.create_session("same", source="example")
        assert one.delete_session("same", deletion_origin="user_rest")
        context = one.list_session_deletion_receipts()[0]["deletion"]
        assert two.get_session_deletion_receipt(context["operation_id"]) is None
        assert two.list_session_deletion_receipts() == []
        assert two.get_session("same") is not None


def test_real_maintenance_and_profile_move_paths_never_prepare(db):
    db.create_session("pruned", source="example")
    db.end_session("pruned", "done")
    assert db.prune_sessions(older_than_days=None) == 1
    db.create_session("moved", source="example")
    assert db.delete_moved_session("moved")
    db.create_session("empty", source="example")
    assert db.delete_session_if_empty("empty")
    assert db.list_session_deletion_receipts() == []


def test_existing_store_reconciles_receipt_table_without_changing_history(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(path) as database:
        database.create_session("existing", source="example")
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE session_deletion_receipts")
    with SessionDB(path) as database:
        assert database.get_session("existing") is not None
        assert database.list_session_deletion_receipts() == []
        assert database.delete_session("existing", deletion_origin="user_rpc")
        assert len(database.list_session_deletion_receipts()) == 1


def test_receipt_reclamation_is_explicit_and_never_reuses_cursor_sequences(db):
    db.create_session("first", source="example")
    assert db.delete_session("first", deletion_origin="user_rpc")
    first = db.list_session_deletion_receipts()[0]
    assert db.forget_session_deletion_receipts(through_sequence=first["sequence"]) == 1
    assert db.get_session_deletion_receipt(first["deletion"]["operation_id"]) is None
    db.create_session("second", source="example")
    assert db.delete_session("second", deletion_origin="user_rpc")
    assert db.list_session_deletion_receipts(after_sequence=first["sequence"])[0]["sequence"] > first["sequence"]
    with pytest.raises(ValueError):
        db.forget_session_deletion_receipts(through_sequence=-1)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 1001}, {"after_sequence": -1}])
def test_receipt_replay_is_bounded(db, kwargs):
    with pytest.raises(ValueError):
        db.list_session_deletion_receipts(**kwargs)
