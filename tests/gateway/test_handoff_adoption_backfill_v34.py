"""v34: a handoff onto a markerless subagent row, made before the route guard read ``created_source``, keeps its chat.

The old code moved the route without stamping ``_chat_adopted``; the guard now reads such a row as a
subagent's capture and evicts the chat. A completed handoff is the one durable record of a user
choice that the capture path (peer stamp, completion re-pin) never writes, so v34 adopts those rows
and nothing else. Real ``SessionStore`` on a real SQLite store, rows written the way the old code
left them, then reopened through the schema migration.
"""

from __future__ import annotations

from gateway.config import GatewayConfig, Platform
from gateway.session_ingress_media import native_inputs_root
from gateway.session import SessionSource, SessionStore
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import SCHEMA_VERSION
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session

_SOURCE = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")


def _old_route_onto_markerless_worker(tmp_path, *, handed_off: bool):
    """The route, peer and row as the pre-guard code left them: no ``_chat_adopted`` anywhere."""
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store.persistence
    entry = store.get_or_create_session(_SOURCE)
    worker = entry.session_id + "-worker"
    seed_session(db, worker, "subagent", parent_session_id=entry.session_id)
    # The old switch_session: route moved and peer stamped, no adoption (the row had no marker).
    store._adopt_delegate_row = lambda *_: False
    assert store.switch_session(entry.session_key, worker) is not None
    if handed_off:
        db.coordination.request_handoff(session_id=worker, platform="telegram")
        db.coordination.claim_handoff(session_id=worker)
        db.coordination.complete_handoff(session_id=worker)
    assert db.sessions.get_model_config_value(worker, "_chat_adopted") is None
    return store, entry, worker


def _reopen_after_upgrade(store, tmp_path):
    """Close the store, mark it as written by v33, and reopen it so the migration chain runs."""
    path = raw_holder(store.persistence).db_path
    store.persistence.close()
    old = StoreHolder(db_path=path)
    old._conn.execute("UPDATE schema_version SET version = 33")
    old._conn.commit()
    old.close()
    upgraded = SessionStore(tmp_path / "sessions", GatewayConfig())
    with raw_holder(upgraded.persistence)._read_ctx() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    return upgraded


def test_a_handed_off_markerless_worker_keeps_the_chat_after_upgrade(tmp_path):
    store, entry, worker = _old_route_onto_markerless_worker(tmp_path, handed_off=True)
    upgraded = _reopen_after_upgrade(store, tmp_path)
    try:
        assert not upgraded._is_session_ended_in_db(worker, entry.session_key)
        assert upgraded.get_or_create_session(_SOURCE).session_id == worker
        assert upgraded.persistence.sessions.get_model_config_value(worker, "_chat_adopted") is True
    finally:
        upgraded.persistence.close()


def test_a_capture_without_a_handoff_is_still_evicted_after_upgrade(tmp_path):
    # Same route, peer and row as the #141940 capture: nothing proves the user chose the worker.
    store, entry, worker = _old_route_onto_markerless_worker(tmp_path, handed_off=False)
    upgraded = _reopen_after_upgrade(store, tmp_path)
    try:
        assert upgraded.persistence.sessions.get_model_config_value(worker, "_chat_adopted") is None
        assert upgraded._is_session_ended_in_db(worker, entry.session_key)
        assert upgraded.get_or_create_session(_SOURCE).session_id != worker
    finally:
        upgraded.persistence.close()


def test_one_malformed_model_config_does_not_stop_the_upgrade(tmp_path):
    # v34 raises every OperationalError but a missing function or column, so a json_set over one
    # malformed model_config would fail every open; json_valid keeps that row out of the UPDATE.
    store, entry, worker = _old_route_onto_markerless_worker(tmp_path, handed_off=True)
    db = store.persistence
    damaged = seed_session(db, worker + "-damaged", "subagent", parent_session_id=entry.session_id)
    db.coordination.request_handoff(session_id=damaged, platform="telegram")
    db.coordination.claim_handoff(session_id=damaged)
    db.coordination.complete_handoff(session_id=damaged)
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET model_config = '{' WHERE id = ?", (damaged,)))
    upgraded = _reopen_after_upgrade(store, tmp_path)
    try:
        assert upgraded.persistence.sessions.get_model_config_value(worker, "_chat_adopted") is True
    finally:
        upgraded.persistence.close()


def test_a_handed_off_row_that_compressed_adopts_nothing(tmp_path):
    # The handed-off row spawned a markerless worker, then compressed. The worker has the same
    # parent, created_source='subagent' and no _delegate_from as the continuation: the capture shape.
    # The compressed row itself stays unadopted: recovery would reopen it and cut it off its tip.
    store, _entry, handed = _old_route_onto_markerless_worker(tmp_path, handed_off=True)
    db = store.persistence
    spawned = seed_session(db, handed + "-spawned", "subagent", parent_session_id=handed)
    tip = _compress(db, handed)
    upgraded = _reopen_after_upgrade(store, tmp_path)
    try:
        sessions = upgraded.persistence.sessions
        assert sessions.get_model_config_value(handed, "_chat_adopted") is None
        assert sessions.get_model_config_value(spawned, "_chat_adopted") is None
        assert sessions.get_model_config_value(tip, "_chat_adopted") is None
    finally:
        upgraded.persistence.close()


def _compress(db, session_id):
    tip = session_id + "-tip"
    assert db.coordination.try_acquire_compression_lock(session_id=session_id, holder="fixture")
    try:
        db.lineage.publish_compression_child(
            session_id, CompressionChild(session_id=tip, source="subagent"),
            [NewMessage.from_live({"role": "user", "content": "carried"})], compression_lock_holder="fixture")
    finally:
        db.coordination.release_compression_lock(session_id=session_id, holder="fixture")
    return tip


def test_a_handed_off_row_whose_continuation_was_deleted_is_adopted(tmp_path):
    # A completed handoff keeps the older row recoverable whatever its end_reason, so once its newest
    # segment is deleted the chat reopens it, as main does for a handed-off chat. Its own marked
    # delegate is not a continuation and must not block the adoption.
    store, entry, handed = _old_route_onto_markerless_worker(tmp_path, handed_off=True)
    db = store.persistence
    db.sessions.touch_activity(handed)  # the handoff's synthetic turn, after the switch
    seed_session(db, handed + "-delegate", "subagent", parent_session_id=handed,
                 model_config={"_delegate_from": handed})
    tip = _compress(db, handed)
    assert store.switch_session(entry.session_key, tip) is not None
    assert db.sessions.delete(tip, media_root=str(native_inputs_root()))
    upgraded = _reopen_after_upgrade(store, tmp_path)
    try:
        assert upgraded.persistence.sessions.get_model_config_value(handed, "_chat_adopted") is True
        assert upgraded.get_or_create_session(_SOURCE).session_id == handed
    finally:
        upgraded.persistence.close()
