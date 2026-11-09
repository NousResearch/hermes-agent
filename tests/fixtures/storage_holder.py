"""Raw SQL beside the storage interface, for rows no interface verb writes or reads.

This is the one test helper that names the SQLite backend: the holder is ``SQLiteStore.holder``.
"""


def _sqlite_store(store):
    """The SQLite backend's own store for *store*: a ``Persistence`` handle's bound store (a handle forwards only the
    storage interface), else *store* itself."""
    from persistence_assembly import Persistence

    return store._bound_store() if isinstance(store, Persistence) else store


def raw_holder(store):
    """The SQLite backend's holder for the handle's store. ``_execute_write(fn)`` runs ``fn(conn)`` in one write
    transaction and returns its result; ``with _read_ctx() as conn`` gives a read connection. Fetching it opens the
    store, so a test that asserts the store is closed never calls it."""
    return _sqlite_store(store).holder()


def holder_type():
    """The SQLite backend's holder class, for a test that wraps every holder's writes."""
    from persistence.sqlite._holder import StoreHolder
    return StoreHolder


def session_repository_type():
    """The SQLite backend's ``sessions`` repository class, for a test that patches its ``create(self, spec)`` on
    every store handle."""
    from persistence.sqlite.conversation.sessions import SQLiteSessionRepository
    return SQLiteSessionRepository


def policy_key(session_id=""):
    """The ``state_meta`` key of *session_id*'s creation receipt (with no id, the key prefix), for a raw write or read
    of a receipt no verb writes (a partial, old-shaped or undecodable one)."""
    from persistence.sqlite.conversation._store.local import POLICY_PREFIX
    return POLICY_PREFIX + session_id


def api_binding_key(suffix=""):
    """The ``state_meta`` key of an API binding row (with no suffix, the key prefix)."""
    from persistence.sqlite.conversation._store.local import API_BINDING_PREFIX
    return API_BINDING_PREFIX + suffix


def terminal_admission_key(suffix=""):
    """The ``state_meta`` key of a terminal admission row (with no suffix, the key prefix)."""
    from persistence.sqlite.conversation._store.terminal import ADMISSION_PREFIX
    return ADMISSION_PREFIX + suffix


def admission_result_key(admission_id=""):
    """The ``state_meta`` key of *admission_id*'s stored result (with no id, the key prefix)."""
    from persistence.sqlite.conversation._store.terminal import RESULT_PREFIX
    return RESULT_PREFIX + admission_id


def retired_session_key(session_id=""):
    """The ``state_meta`` key of *session_id*'s retirement marker (with no id, the key prefix)."""
    from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
    return RETIRED_PREFIX + session_id


def retired_route_key(suffix=""):
    """The ``state_meta`` key of a retired-route fence, *suffix* being ``<profile>:<session id>`` (with no suffix,
    the key prefix)."""
    from persistence.sqlite.gateway._routes import ROUTE_FENCE_PREFIX
    return ROUTE_FENCE_PREFIX + suffix


def history_binding_key(suffix=""):
    """The ``state_meta`` key of a history-control binding row (with no suffix, the key prefix)."""
    from persistence.sqlite.authority._binding_history import BINDING_PREFIX
    return BINDING_PREFIX + suffix


def hosted_owner_key(room_id=""):
    """The ``state_meta`` key of hosted room *room_id*'s owner stamp (with no id, the key prefix)."""
    from persistence.sqlite.gateway.hosted_rooms import OWNER_PREFIX
    return OWNER_PREFIX + room_id


def retired_media_key(suffix=""):
    """The ``state_meta`` key of a retired-media candidate (with no suffix, the key prefix)."""
    from persistence.sqlite.conversation._store.media import PREFIX
    return PREFIX + suffix


def encode_result(result):
    """A stored admission result as the backend encodes it before writing it."""
    from persistence.sqlite.conversation._store.terminal import encode_result
    return encode_result(result)


def load_result(text):
    """A stored admission result row decoded as the backend reads it."""
    from persistence.sqlite.conversation._store.terminal import load_result
    return load_result(text)


def result_decoder():
    """The backend's stored-result decoder itself, for a spy that wraps it where production looks it up: take it
    before the patch (``load_result`` above imports at call time, so it would reach the spy)."""
    from persistence.sqlite.conversation._store.terminal import load_result
    return load_result


def terminal_api_admissions(conn, request_id):
    """The backend's scan, on the admission write's connection *conn*, of the retired terminal admissions under the
    API *request_id*, for a spy wrapping the scan where the admission write looks it up."""
    from persistence.sqlite.authority._runtime import terminal_api_admissions
    return terminal_api_admissions(conn, request_id)


def create_conflict(store, session_id):
    """``(code, message)`` the store refuses a create of the explicit *session_id* with (a live row, or a deleted
    id), else None, in one read that creates nothing."""
    from persistence.sqlite.conversation._store.retirement import create_conflict
    with raw_holder(store)._read_ctx() as conn:
        return create_conflict(conn, session_id)


def retired_result_fields():
    """The keys of a stored result's ``result`` that a session's retirement drops."""
    from persistence.sqlite.conversation._store.retirement import _TRANSCRIPT_FIELDS
    return _TRANSCRIPT_FIELDS


def reclaim_tombstones(store, older_than=0):
    """Reclaim the deletion tombstones retired more than *older_than* seconds ago, as the prune's own reclaim does
    (0: as if the retry horizon had passed); returns what the reclaim returns."""
    from persistence.sqlite.housekeeping._retirement import prune_retired_tombstones
    return raw_holder(store)._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=older_than))


def reclaim_expired(store, prefix, cutoff):
    """Drop the sweep (``expires``) markers under the key *prefix* retired before *cutoff*, as the route save's
    reclaim does."""
    from persistence.sqlite.gateway._routes import _reclaim_expired
    raw_holder(store)._execute_write(lambda conn: _reclaim_expired(conn, prefix, cutoff))


def expiring_retirements(store, session_ids):
    """The ids among *session_ids* whose own retirement marker is a sweep's (ages out at the horizon)."""
    from persistence.sqlite.authority._runtime_retirement import expiring_retirements
    return expiring_retirements(raw_holder(store), session_ids)


def retire_routes(store, session_ids, *, expiring=()):
    """Write the retirement markers and drop the routes of *session_ids*, as a delete's retirement does."""
    from persistence.sqlite.conversation._store.retirement import retire_routes
    raw_holder(store)._execute_write(lambda conn: retire_routes(conn, session_ids, expiring=expiring))


def guarded_ids(store, ids):
    """The ids among *ids* a live turn lease or compression lock guards, as a delete or prune sees them (expired or
    dead-holder guards are reclaimed on the way)."""
    from persistence.sqlite.conversation._store.write_guards import _guarded_ids
    holder = raw_holder(store)
    with holder._read_ctx() as conn:
        return _guarded_ids(holder, conn, ids)


def search_slow_ms():
    """The search duration, in milliseconds, past which the store logs a slow search, for the profile in scope."""
    from persistence.sqlite.conversation._store.search import _search_slow_ms
    return _search_slow_ms()


def chain_cap():
    """How many compression steps one walk of a session's compression chain follows before it stops."""
    from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
    return _CHAIN_CAP


def holder_open_seam():
    """``(registry, name)``: the backend's store registry module and the name of its per-path holder open
    (``name(path)`` builds the holder on a ``state.db`` the process has not opened yet, running the schema setup),
    for a spy patched where the registry looks it up."""
    from persistence.sqlite import _registry
    return _registry, "_open_holder"


def read_busy_timeout_seam():
    """``(module, name)``: the backend's holder module and the name of the lock wait, in seconds, a read-only handle
    opens its connection with, for a test that shortens it where the holder looks it up."""
    from persistence.sqlite import _holder
    return _holder, "_READ_BUSY_TIMEOUT_S"


def store_copy_seam():
    """``(module, name)``: the backend's backup module and the name of the WAL-safe file copy the store admin's
    ``backup_to`` runs (``name(src, dst, *, timeout_seconds)``, False when no snapshot could be made), for a stub
    patched where it is looked up."""
    from persistence.sqlite.admin import _backup
    return _backup, "_safe_copy_db"


def compaction_write_seam():
    """``(module, compact, insert)``: the backend's transcript module and the names of its in-place compaction body
    (``compact(holder, conn, session_id, compacted_messages, ...)``, run inside the compaction's write transaction)
    and of its message-row insert (``insert(holder, conn, session_id, messages)``), for a hook patched where the
    compaction looks them up."""
    from persistence.sqlite.conversation._store import messages
    return messages, "_archive_and_compact_on_conn", "_insert_message_rows"


def rebuild_fts(store):
    """Rebuild the full-text indexes in place, as the backend's rebuild does under its cross-process admission
    (0 when another process holds it), without the search verb's skip while other processes hold the file; returns
    the indexes rebuilt."""
    from persistence.sqlite.conversation._store.search import rebuild_fts
    return rebuild_fts(raw_holder(store))


def optimize_fts(store):
    """Merge each full-text index's segments (FTS5 ``'optimize'``), as the backend's vacuum does first; returns the
    indexes optimized."""
    from persistence.sqlite.admin._maintenance import _optimize_fts
    return _optimize_fts(raw_holder(store))


def hosted_upload(store, placement, room_id, upload_id):
    """The attachment row of *room_id*'s upload *upload_id* as a dict, in the hosted-room file *placement* names
    (``'profile'`` the profile's ``state.db``, ``'shared'`` the install's ``shared-state.db``), or None when that
    file holds no such upload or no attachment table."""
    from persistence.sqlite.gateway._hosted_store import HostedFiles

    def read(conn):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_attachments'"
                        ).fetchone() is None:
            return None
        cursor = conn.execute("SELECT * FROM hosted_room_attachments WHERE room_id=? AND upload_id=?",
                              (room_id, upload_id))
        row = cursor.fetchone()
        return None if row is None else dict(zip([column[0] for column in cursor.description], row))
    return HostedFiles(_sqlite_store(store))._hosted_read(placement, read)


def hosted_attachment_setup_seam():
    """``(module, name)``: the backend's attachment module and the name of its set of files whose attachment setup
    (tables, reconciliation, prune) already ran in this process, for a test that stands for a new process by
    replacing it with an empty set."""
    from persistence.sqlite.gateway import hosted_room_attachments
    return hosted_room_attachments, "_STARTED"


def header_probe_fds():
    """``(lock, fds)``: the backend's open ``state.db`` header probes by path (``fds[path]`` is ``(fd, dev, ino)``)
    and the lock guarding them, for a fixture that closes the probes its own test opened."""
    import hermes_state_dbfile as dbfile
    return dbfile._HEADER_PROBE_LOCK, dbfile._HEADER_PROBE_FDS


def state_db_replaced_error():
    """The error the backend raises on a read or write of a ``state.db`` replaced under the open store."""
    from hermes_state_errors import StateDbReplacedError
    return StateDbReplacedError


def admission_store_operation(name):
    """The backend's admission-store operation *name* itself (``admit_session_input``, ``claim_session_input``,
    ``settle_session_input``, ``cancel_session_input``, ``close_discarded_turn``), for a spy patched where production
    looks it up (``persistence.sqlite.authority._admission_store.<name>``)."""
    from persistence.sqlite.authority import _admission_store
    return getattr(_admission_store, name)


def idle_guard():
    """The backend's busy-session guard ``require_idle(holder, conn, ids)`` itself, for a spy patched where production
    looks it up (``persistence.sqlite.authority._runtime_guards.require_idle``)."""
    from persistence.sqlite.authority._runtime_guards import require_idle
    return require_idle


def runtime_session_write():
    """The backend's runtime mutation write ``mutate_runtime_session`` itself, for a spy patched where production looks
    it up (``persistence.sqlite.authority._mutation.mutate_runtime_session``)."""
    from persistence.sqlite.authority._mutation import mutate_runtime_session
    return mutate_runtime_session


def idle_local_session_end():
    """The backend's end-of-conversation stamp ``end_idle_local_session`` itself, for a spy patched where production
    looks it up (``persistence.sqlite.authority._local.end_idle_local_session``)."""
    from persistence.sqlite.authority._local import end_idle_local_session
    return end_idle_local_session


def hosted_room_reload():
    """The backend's hosted-room re-read after a write (``_reload``) itself, for a fault patched where production looks
    it up (``persistence.sqlite.gateway.hosted_rooms._reload``) and the restore after it."""
    from persistence.sqlite.gateway.hosted_rooms import _reload
    return _reload


def rewind_store_operation(name):
    """The backend's rewind operation *name* itself (``rewind_to_message`` the write, ``get_messages_as_conversation``
    the read), for a spy patched where production looks it up (``persistence.sqlite.conversation._store.rewind.<name>``)."""
    from persistence.sqlite.conversation._store import rewind
    return getattr(rewind, name)
def schema_sql():
    """The store's declared schema DDL, for a fixture that builds the declared tables or compares against them."""
    from persistence.sqlite._schema_ddl import SCHEMA_SQL
    return SCHEMA_SQL


def schema_version():
    """The schema version the store stamps on a store it creates or upgrades."""
    from persistence.sqlite._schema_ddl import SCHEMA_VERSION
    return SCHEMA_VERSION


def legacy_inline_fts_sql():
    """The DDL of the older search index layout that keeps message text inline (both indexes), for a fixture that
    builds a store optimize-storage still has to rebuild."""
    from persistence.sqlite._schema_ddl import LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL
    return LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL


def fts_vtable_sql(table):
    """The stored definition (``sqlite_master.sql``) of the current layout of the search index *table*
    (``messages_fts`` or ``messages_fts_trigram``)."""
    from persistence.sqlite._schema_ddl import FTS_SQL, FTS_TRIGRAM_SQL, _expected_vtable_sql
    return _expected_vtable_sql({"messages_fts": FTS_SQL, "messages_fts_trigram": FTS_TRIGRAM_SQL}[table], table)
def is_fts_write_corruption(exc):
    """Whether the store's write-repair gate reads the SQLite error *exc* as corruption scoped to the search index."""
    from persistence.sqlite._schema import _is_fts_write_corruption_error
    return _is_fts_write_corruption_error(exc)


def browse_integrity_scan():
    """The backend's pre-lock browse integrity scan ``_integrity_problem(path, budget)`` itself (None for a sound
    store, else its first problems), for a spy patched where production looks it up
    (``persistence.sqlite.admin._reconcile._integrity_problem``)."""
    from persistence.sqlite.admin._reconcile import _integrity_problem
    return _integrity_problem


def browse_budgets():
    """``(integrity, reconcile)``: the seconds a browse's pre-lock integrity scan and its reconcile may each take, as
    the backend reads them (``BROWSE_INTEGRITY_BUDGET_S``, which a test patches at
    ``persistence.sqlite.admin._reconcile.BROWSE_INTEGRITY_BUDGET_S``, and ``BROWSE_RECONCILE_BUDGET_S``)."""
    from persistence.sqlite.admin import _reconcile
    return _reconcile.BROWSE_INTEGRITY_BUDGET_S, _reconcile.BROWSE_RECONCILE_BUDGET_S


def forget_slow_scans():
    """Forget the stores whose browse integrity scan ran out of time, so the next browse of each scans it again."""
    from persistence.sqlite.admin._reconcile import _scan_too_slow
    _scan_too_slow.clear()


def recovery_orphan_cleanup(conn):
    """Run the backend's recovery cleanup of references to unsalvageable sessions on the raw ``state.db`` connection
    *conn*, as a recovery does on the file it rebuilt: dangling parent, prompt and tool-pin references are cleared and
    rows still orphaned are removed; returns the counts it reports."""
    from persistence.sqlite.admin._recovery import _cleanup_partial_orphans
    return _cleanup_partial_orphans(conn)
