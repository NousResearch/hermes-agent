"""Prompt/identity pairing across real SQLite writes, branch copies and atomic compression."""

import sqlite3

import pytest

from agent.context_file_state import ContextFileSnapshot
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


@pytest.fixture
def db(tmp_path):
    with SessionDB(tmp_path / "state.db") as store:
        store.create_session("parent", source="cli", model="test-model")
        yield store


def manifest(number):
    return ContextFileSnapshot(identities={("inode", 1, number)}).serialize()


def test_prompt_and_identities_update_as_one_snapshot(db):
    saved = manifest(11)
    db.update_system_prompt("parent", "first prompt", saved)
    row = db.get_session("parent")
    assert row["system_prompt"] == "first prompt" and row["context_file_identities"] == saved
    assert row["system_prompt_hash"]
    db.update_system_prompt("parent", "first prompt")
    assert db.get_session("parent")["context_file_identities"] == saved
    db.update_system_prompt("parent", "different prompt")
    assert db.get_session("parent")["context_file_identities"] is None
    db.update_system_prompt("parent", "last prompt", manifest(12))
    db.update_system_prompt("parent", None)
    row = db.get_session("parent")
    assert row["system_prompt"] is None and row["context_file_identities"] is None


def test_branch_copy_inherits_only_the_identical_prompt_manifest(db):
    saved = manifest(21)
    db.update_system_prompt("parent", "frozen", saved)
    db.create_session("same", source="cli", parent_session_id="parent", system_prompt="frozen")
    db.create_session("different", source="cli", parent_session_id="parent", system_prompt="new")
    assert db.get_session("same")["context_file_identities"] == saved
    assert db.get_session("different")["context_file_identities"] is None
    db.create_session("same", source="cli", system_prompt="do not replace", context_file_identities=manifest(22))
    assert db.get_session("same")["system_prompt"] == "frozen"
    assert db.get_session("same")["context_file_identities"] == saved


@pytest.mark.parametrize("new_snapshot", [False, True])
def test_atomic_compression_child_publishes_the_matching_manifest(db, new_snapshot):
    old, new = manifest(31), manifest(32)
    db.update_system_prompt("parent", "frozen", old)
    kwargs = {"context_file_identities": new} if new_snapshot else {}
    prompt = "rebuilt" if new_snapshot else "frozen"
    db.publish_compression_child(
        parent_session_id="parent", child_session_id="child", source="cli",
        messages=[{"role": "user", "content": "summary"}], system_prompt=prompt,
        require_compression_lease=False, **kwargs)
    child = db.get_session("child")
    assert child["system_prompt"] == prompt
    assert child["context_file_identities"] == (new if new_snapshot else old)
    assert db.get_session("parent")["end_reason"] == "compression"


def test_failed_compression_publication_rolls_back_prompt_and_identities(db, monkeypatch):
    saved = manifest(41)
    db.update_system_prompt("parent", "frozen", saved)

    def fail_insert(*_args, **_kwargs):
        raise RuntimeError("injected transaction failure")

    monkeypatch.setattr(db, "_insert_message_rows", fail_insert)
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="cli",
            messages=[{"role": "user", "content": "summary"}], system_prompt="rebuilt",
            context_file_identities=manifest(42), require_compression_lease=False)
    assert db.get_session("child") is None
    parent = db.get_session("parent")
    assert parent["ended_at"] is None
    assert parent["system_prompt"] == "frozen" and parent["context_file_identities"] == saved


def test_compact_session_lists_do_not_leak_identity_metadata(db):
    db.update_system_prompt("parent", "frozen", manifest(51))
    rows = db.list_sessions_rich(compact_rows=True)
    assert rows and all("context_file_identities" not in row for row in rows)
    assert db.get_session("parent")["context_file_identities"] == manifest(51)


def test_legacy_schema_keeps_prompt_and_adds_nullable_manifest(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL.replace("    context_file_identities TEXT,\n", ""))
        conn.execute("INSERT INTO sessions(id,source,system_prompt,started_at) VALUES('legacy','cli','old bytes',1)")
    with SessionDB(path) as db:
        row = db.get_session("legacy")
        assert row["system_prompt"] == "old bytes"
        assert row["context_file_identities"] is None
