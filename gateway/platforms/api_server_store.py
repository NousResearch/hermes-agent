"""Runtime-bound API storage selection; borrowed stores are never adapter-owned."""
from pathlib import Path
import sqlite3


def selected_session_db(adapter, home):
    from gateway.session_authorities import authority_for_home
    # Only a home this runtime reserved has a store here; a multiplex route alone is not proof.
    authority = authority_for_home(adapter.gateway_runner, home)
    if authority is None:
        return None
    with adapter._session_db_cache_lock:
        if adapter._session_db_cache_closed:
            return None
    return authority.db


def _exact_durable_store(store, path):
    """Return whether *store* is the live durable handle for exactly *path*."""
    try:
        configured = Path(store._db_path).resolve()
        actual = Path(store._conn.execute('PRAGMA database_list').fetchone()[2]).resolve()
    except (AttributeError, OSError, TypeError, sqlite3.Error):
        return False
    return bool(store.durable and configured == path and actual == path)


def selected_run_idempotency_store(adapter, home):
    """The API receipt store bound to the registry owner for *home*.

    The launch store is constructed with the adapter.  A secondary store is
    created only for a home this process really owns and serves, then pinned to
    that exact authority/epoch/instance.  Registry replacement therefore
    refuses instead of silently reusing another owner's receipts.
    """
    from gateway.session_authorities import authority_for_home
    from hermes_constants import hermes_home_key

    home = Path(home).resolve()
    runner = getattr(adapter, 'gateway_runner', None)
    authority = authority_for_home(runner, home) if runner is not None else None
    if authority is None or Path(authority.profile_id).resolve() != home:
        return None
    path = home / 'runs_idempotency.db'
    launch = getattr(adapter, '_run_idempotency_store', None)
    if authority is getattr(runner, 'session_authority', None) and _exact_durable_store(launch, path):
        return launch

    from gateway.runtime_ownership import process_ownership
    if not process_ownership.owns(home):
        return None
    owner_epoch, owner_instance_id = authority.epoch, authority.instance_id
    key = hermes_home_key(home)
    lock = getattr(adapter, '_run_idempotency_store_lock', None)
    if lock is None:
        return None
    with lock:
        stores = getattr(adapter, '_profile_run_idempotency_stores', None)
        if stores is None:
            return None
        current = authority_for_home(runner, home)
        if (current is not authority or getattr(current, 'epoch', None) != owner_epoch
                or getattr(current, 'instance_id', None) != owner_instance_id
                or not process_ownership.owns(home)):
            return None
        existing = stores.get(key)
        if existing is not None:
            owner, epoch, instance_id, store = existing
            if (owner is not authority or epoch != owner_epoch
                    or instance_id != owner_instance_id
                    or not _exact_durable_store(store, path)):
                return None
            return store
        from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
        store = RunIdempotencyStore(str(path))
        if not _exact_durable_store(store, path):
            store.close()
            return None
        # Opening SQLite may block.  The registry slot and process reservation are
        # the publication authority, so re-read both after construction while the
        # release path is excluded by this same lock.
        current = authority_for_home(runner, home)
        if (current is not authority or getattr(current, 'epoch', None) != owner_epoch
                or getattr(current, 'instance_id', None) != owner_instance_id
                or not process_ownership.owns(home)):
            store.close()
            return None
        stores[key] = (authority, owner_epoch, owner_instance_id, store)
        return store


def close_profile_run_idempotency_stores(adapter):
    """Close only secondary stores; the adapter closes its launch store."""
    lock = getattr(adapter, '_run_idempotency_store_lock', None)
    if lock is None:
        return
    with lock:
        stores = getattr(adapter, '_profile_run_idempotency_stores', {})
        values = list(stores.values())
        stores.clear()
    for _owner, _epoch, _instance_id, store in values:
        store.close()


def release_profile_run_idempotency_store(adapter, authority):
    """Retire the secondary receipt handle only for its exact withdrawn owner."""
    from hermes_constants import hermes_home_key

    lock = getattr(adapter, '_run_idempotency_store_lock', None)
    if lock is None:
        return
    key = hermes_home_key(authority.profile_id)
    with lock:
        stores = getattr(adapter, '_profile_run_idempotency_stores', {})
        existing = stores.get(key)
        if existing is None or existing[0] is not authority:
            return
        store = stores.pop(key)[-1]
        retained = getattr(adapter, '_run_receipt_stores', {})
        for run_id in [run_id for run_id, candidate in retained.items() if candidate is store]:
            retained.pop(run_id, None)
    store.close()
