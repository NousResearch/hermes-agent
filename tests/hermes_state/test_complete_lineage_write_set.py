"""Complete routing/recovery write sets and ambiguous legacy reads (#149132)."""
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionIdOwnedElsewhereError, SessionLineageAmbiguousError
from hermes_state_local import POLICY_PREFIX
from hermes_state_local_migration import LEGACY_PREFIX


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    local, sibling = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    try:
        yield local, sibling
    finally:
        local.close()
        sibling.close()


def _snapshot(db):
    return {table: [tuple(row) for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("sessions", "messages", "state_meta", "gateway_routing")}


def _chain(db):
    for sid, parent, reason in (("root", None, "compression"), ("middle", "root", "compression"),
                                ("tip", "middle", "ws_orphan_reap")):
        db._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id,archived,ended_at,end_reason) "
                         "VALUES (?, 'cli', 1, ?, 1, 2, ?)", (sid, parent, reason))
    db._conn.commit()


def _foreign(db, sid):
    db._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES (?,'cli',1,'science')", (sid,))
    db._conn.commit()


def _peer(db, sid="tip"):
    db.record_gateway_session_peer(sid, source="telegram", session_key="agent:main:telegram:dm:fixture",
                                   include_compression_ancestors=True)


@pytest.mark.parametrize("operation", ["peer", "recover-root", "recover-middle", "recover-tip"])
@pytest.mark.parametrize("contested", ["root", "middle", "tip"])
def test_every_written_compression_segment_requires_its_owner(stores, operation, contested):
    local, sibling = stores
    _chain(local)
    _foreign(sibling, contested)
    before = _snapshot(local), _snapshot(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        if operation == "peer":
            _peer(local)
        else:
            local.unarchive_recoverable_session(operation.removeprefix("recover-"))
    assert (_snapshot(local), _snapshot(sibling)) == before


def _receipt(db):
    db._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason,title) "
                     "VALUES ('owner','cli',0,1,'session_reset','Conversation title')")
    db._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                     "VALUES ('earlier','cli',0,1,'session_reset')")
    receipt = {"entry": {"session_id": "tip"}, "route": "agent:main:local:dm:owner",
               "lineage": ["owner", "earlier", "root", "middle", "tip"]}
    db._conn.executemany("INSERT INTO state_meta(key,value) VALUES (?,?)",
                         [(POLICY_PREFIX + "owner", json.dumps(receipt)), (LEGACY_PREFIX + "owner", "tip")])
    db._conn.commit()


@pytest.mark.parametrize("contested", ["owner", "earlier"])
@pytest.mark.parametrize("operation", ["peer", "boot-heal"])
def test_receipt_handoff_fences_older_reset_segments_and_title_owner(stores, contested, operation):
    local, sibling = stores
    _chain(local)
    _receipt(local)
    if operation == "boot-heal":
        local._conn.execute("UPDATE sessions SET source='telegram',session_key='agent:main:telegram:dm:fixture' WHERE id='tip'")
        local._conn.commit()
    _foreign(sibling, contested)
    from hermes_state_runtime import begin_runtime_epoch
    epoch = begin_runtime_epoch(local, instance_id="fixture")
    before = _snapshot(local), _snapshot(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        if operation == "peer":
            _peer(local)
        else:
            from gateway.session_local_recovery import release_handed_off_receipts
            release_handed_off_receipts(local, epoch=epoch)
    assert (_snapshot(local), _snapshot(sibling)) == before


def test_uncontested_handoff_moves_receipt_segments_and_title_atomically(stores):
    local, sibling = stores
    _chain(local)
    _receipt(local)
    sibling_before = _snapshot(sibling)
    _peer(local)
    assert {row[0] for row in local._conn.execute("SELECT source FROM sessions")} == {"telegram"}
    assert local.get_session("root")["title"] == "Conversation title"
    assert local.get_session("owner")["title"] is None
    assert local._conn.execute("SELECT 1 FROM state_meta WHERE key IN (?,?)",
                               (POLICY_PREFIX + "owner", LEGACY_PREFIX + "owner")).fetchone() is None
    assert _snapshot(sibling) == sibling_before


@pytest.mark.parametrize("operation", ["peer", "recover"])
def test_complete_lineage_sql_failure_rolls_back_all_rows_and_receipts(stores, operation):
    local, sibling = stores
    _chain(local)
    _receipt(local)
    statement = ("BEFORE DELETE ON state_meta WHEN OLD.key='" + POLICY_PREFIX + "owner'"
                 if operation == "peer" else "BEFORE UPDATE OF ended_at ON sessions WHEN OLD.id='tip' AND NEW.ended_at IS NULL")
    local._conn.execute(f"CREATE TRIGGER reject_change {statement} BEGIN SELECT RAISE(ABORT,'injected abort'); END")
    local._conn.commit()
    before = _snapshot(local), _snapshot(sibling)
    with pytest.raises(sqlite3.IntegrityError, match="injected abort"):
        _peer(local) if operation == "peer" else local.unarchive_recoverable_session("root")
    assert (_snapshot(local), _snapshot(sibling)) == before


@pytest.mark.parametrize("operation", ["peer", "recover"])
@pytest.mark.parametrize("marker", ["_branched_from", "_reset_from"])
def test_explicit_fork_boundary_does_not_guard_or_rewrite_its_parent(stores, operation, marker):
    local, sibling = stores
    _chain(local)
    local._conn.execute("UPDATE sessions SET model_config=? WHERE id='middle'", (json.dumps({marker: "root"}),))
    local._conn.commit()
    _foreign(sibling, "root")
    parent = dict(local.get_session("root"))
    sibling_before = _snapshot(sibling)
    if operation == "peer":
        _peer(local)
        assert local.get_session("middle")["source"] == "telegram"
    else:
        assert local.unarchive_recoverable_session("middle")
        assert not local.get_session("tip")["archived"]
    assert local.get_session("root") == parent
    assert _snapshot(sibling) == sibling_before


def test_ambiguous_unindexed_history_uses_read_only_projection_but_writes_refuse(stores):
    local, sibling = stores
    local.create_session("legacy", "cli")
    local.append_message("legacy", "user", "Preserved legacy question")
    local.append_message("legacy", "assistant", "Preserved legacy answer")
    local._conn.execute("UPDATE messages SET display_order=NULL,display_identity=NULL WHERE session_id='legacy'")
    local._conn.commit()
    _foreign(sibling, "legacy")
    before = _snapshot(local), _snapshot(sibling)
    assert [row["content"] for row in local.get_messages("legacy", include_compacted=True)] == [
        "Preserved legacy question", "Preserved legacy answer"]
    assert (_snapshot(local), _snapshot(sibling)) == before
    with pytest.raises(SessionLineageAmbiguousError):
        local.append_message("legacy", "user", "Must refuse")
    assert (_snapshot(local), _snapshot(sibling)) == before


@pytest.mark.parametrize("requested", ["root", "middle", "tip"])
def test_uncontested_three_segment_recovery_survives_reopen(stores, requested):
    local, sibling = stores
    _chain(local)
    before_sibling = _snapshot(sibling)
    assert local.unarchive_recoverable_session(requested)
    with SessionDB(local.db_path) as reopened:
        assert not any(row[0] for row in reopened._conn.execute("SELECT archived FROM sessions"))
        assert reopened.get_session("tip")["ended_at"] is None
        reopened.append_message("tip", "user", "Recovered continuation")
    assert _snapshot(sibling) == before_sibling


def test_gateway_receipt_expansion_after_discovery_rechecks_before_writing(stores, monkeypatch):
    local, sibling = stores
    _chain(local)
    _receipt(local)
    _foreign(sibling, "late")
    execute = local._execute_write
    snapshots = []

    def scheduled(write, *args, **kwargs):
        if not snapshots:
            local._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                                 "VALUES ('late','cli',0,1,'session_reset')")
            raw = local._conn.execute("SELECT value FROM state_meta WHERE key=?", (POLICY_PREFIX + "owner",)).fetchone()[0]
            receipt = json.loads(raw)
            receipt["lineage"].insert(1, "late")
            local._conn.execute("UPDATE state_meta SET value=? WHERE key=?", (json.dumps(receipt), POLICY_PREFIX + "owner"))
            local._conn.commit()
            snapshots.append((_snapshot(local), _snapshot(sibling)))
        return execute(write, *args, **kwargs)

    monkeypatch.setattr(local, "_execute_write", scheduled)
    with pytest.raises(SessionIdOwnedElsewhereError):
        _peer(local)
    assert (_snapshot(local), _snapshot(sibling)) == snapshots[0]
