"""Native closeout route changes linearize against explicit restore in the owning profile DB."""
import json
import logging
import threading
import uuid

from hermes_cli.conversation_closeout import CloseoutStore

logger = logging.getLogger(__name__)


def publish_native_route(store, source, closeout_id, *, epoch=None):
    """Wait for conflicting lookups before checking the operation; never hold a DB over await.

    The existing save lock orders this scoped routing upsert with whole-index snapshots. The
    profile transaction excludes unarchive/replacement through the actual route mutation, including
    when the authoritative routing index lives in a different profile database.
    """
    route = store._generate_session_key(source)
    slot, owner = store._claim_session_flight(route, ('closeout', closeout_id))
    if not owner:
        slot.event.wait()
        if slot.error is not None:
            raise slot.error
        return slot.result
    try:
        slot.result = _publish(store, source, route, closeout_id, epoch)
        return slot.result
    except BaseException as exc:
        slot.error = exc
        raise
    finally:
        with store._inflight_lock:
            if store._inflight_sessions.get(route) is slot:
                store._inflight_sessions.pop(route)
        slot.event.set()


def _publish(store, source, route, closeout_id, epoch):
    db, routing_db = store._db_for_key(route), store._routing_db
    if db is None or routing_db is None:
        raise ValueError('closeout_routing_store_unavailable')
    receipts = CloseoutStore(db, epoch=epoch)
    # Loading may do I/O; finish it before entering the guarded route mutation.
    store.lookup_by_session_key(route)
    candidate, create = _candidate(store, source, route, receipts.read(closeout_id)['next_create_request_id'])
    serialized = json.dumps(candidate.to_dict())
    same_db = routing_db.db_path.resolve() == db.db_path.resolve()
    scope = store._routing_scope()
    with store._lazy('_save_lock', threading.Lock):
        previous = routing_db._read_one('SELECT entry_json FROM gateway_routing WHERE scope=? AND session_key=?',
                                        (scope, route))
        def transition(conn):
            record = receipts.read_current(closeout_id, conn=conn)
            if record is None or record['status'] != 'preserved' or not record['archived']:
                return None
            with store._lock:
                observed = store._entries.get(route)
            if observed is None or not _old_route(conn, observed.session_id, record):
                return record, None
            from hermes_state_worker_lifecycle import insert_session_row_in_transaction
            insert_session_row_in_transaction(db, conn, **create)
            routing_db.save_gateway_routing_entry(route, serialized, scope=scope,
                                                 **({'conn': conn} if same_db else {}))
            saved = (conn.execute('SELECT entry_json FROM gateway_routing WHERE scope=? AND session_key=?',
                                  (scope, route)).fetchone() if same_db else
                     routing_db._read_one('SELECT entry_json FROM gateway_routing WHERE scope=? AND session_key=?',
                                          (scope, route)))
            if saved is None or saved['entry_json'] != serialized:
                raise ValueError('closeout_route_not_persisted')
            return record, candidate
        try:
            result = receipts._write(transition)
        except BaseException:
            if not same_db:
                _restore_route(routing_db, scope, route, serialized, previous)
            raise
        if result is None or result[1] is None:
            if not same_db:
                _restore_route(routing_db, scope, route, serialized, previous)
            return result
        # No callback retry or failed commit may expose a provisional candidate in memory.
        with store._lock:
            store._entries[route] = candidate
            revision = store._next_routing_generation_locked()
            store._lazy('_fast_persisted_entries', dict)[route] = (revision, serialized)
            snapshot = store._entries_as_dicts()
        if getattr(store, '_write_sessions_json', True):
            try:
                store._save_sessions_json(snapshot)
            except OSError:
                logger.warning('Closeout routing committed; legacy routing mirror is unavailable', exc_info=False)
    store._announce_route(route, candidate)
    return result


def _restore_route(db, scope, route, provisional, previous):
    """A failed cross-DB checkpoint may undo only its own exact provisional routing value."""
    def restore(conn):
        if previous is None:
            conn.execute('DELETE FROM gateway_routing WHERE scope=? AND session_key=? AND entry_json=?',
                         (scope, route, provisional))
        else:
            conn.execute('UPDATE gateway_routing SET entry_json=? WHERE scope=? AND session_key=? AND entry_json=?',
                         (previous['entry_json'], scope, route, provisional))
    db._execute_write(restore)


def _old_route(conn, session_id, record):
    from hermes_state_sessions import _LINEAGE_CTE_SQL
    stored = record['old']['stored_session_id']
    return conn.execute(_LINEAGE_CTE_SQL + ' SELECT 1 FROM lineage WHERE id=?',
                        (stored, stored, session_id)).fetchone() is not None


def _candidate(store, source, route, request_id):
    from gateway.session import SessionEntry
    from gateway.session_identity import transport_profile_of
    from gateway.session_lifecycle import _now
    now = _now()
    session_id = 'new-' + uuid.uuid5(uuid.NAMESPACE_URL, route + '\n' + request_id).hex
    candidate = SessionEntry(session_key=route, session_id=session_id, created_at=now, updated_at=now,
        origin=source, display_name=source.chat_name, platform=source.platform, chat_type=source.chat_type,
        transport_profile=transport_profile_of(source))
    create = store._session_create_kwargs(session_id=session_id, session_key=route, origin=source,
        source_value=source.platform.value, display_name=source.chat_name, parent_session_id=None)
    return candidate, create
