"""Runtime-bound API storage selection; borrowed stores are never adapter-owned. Without a runtime the adapter caches
one storage interface handle per profile home and closes it. An authority's handle for the API and webhook doors:
:func:`authority_store`."""
import contextlib


def selected_session_db(adapter, home):
    from gateway.session_authorities import authority_for_home
    # Only a home this runtime reserved has a store here; a multiplex route alone is not proof.
    authority = authority_for_home(adapter.gateway_runner, home)
    if authority is None:
        return None
    with adapter._session_db_cache_lock:
        if adapter._session_db_cache_closed:
            return None
    return authority.persistence


def adapter_persistence(adapter, home=None):
    """The storage interface handle for the store ``_ensure_store`` answers for ``home`` (the active profile home
    when None): the reserving authority's under a runtime, else the adapter's cached handle, or its explicit
    ``_session_db`` override when that is a handle; :func:`close_cached_stores` closes the adapter's. None without a
    store or beside an override that is not a handle."""
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    if adapter.gateway_runner is not None:
        from gateway.session_authorities import authority_for_home
        authority = authority_for_home(adapter.gateway_runner, home)
        with adapter._session_db_cache_lock:
            if adapter._session_db_cache_closed:
                return None
        return getattr(authority, "persistence", None)
    override = adapter._session_db
    if override is None:
        with adapter._session_db_cache_lock:
            return None if adapter._session_db_cache_closed else adapter._home_stores.get(str(home))
    from persistence_assembly import Persistence
    with adapter._session_db_cache_lock:
        if adapter._session_db_cache_closed:
            return None
    return override if isinstance(override, Persistence) else None


def authority_store(authority):
    """*authority*'s storage interface handle (``authority_persistence``); without one, the store failure
    (``StoreUnavailable``) its callers already answer for an unreadable store."""
    from gateway.session_persistence import authority_persistence
    store = authority_persistence(authority)
    if store is None:
        from persistence import StoreUnavailable
        raise StoreUnavailable("the authority has no storage interface handle")
    return store


def delete_session(store, session_id, sessions_dir) -> bool:
    """Hard-delete *session_id* with its local lineage and delegate children (``SessionWriteGuarded`` while a target
    holds a live lease), then run the post-delete work: hook, retired media, transcript files under *sessions_dir*.
    True when a row was removed."""
    from gateway.session_ingress_media import native_inputs_root
    from session_retention import after_sessions_removed
    removed = store.sessions.delete(session_id, refuse_guarded=True, media_root=str(native_inputs_root()))
    after_sessions_removed(store, store.scope.profile_dir, removed, reason="delete", primary=session_id,
                           sessions_dir=sessions_dir)
    return bool(removed)


def close_cached_stores(adapter) -> None:
    """Close the storage interface handles ``adapter``'s profile cache holds."""
    with adapter._session_db_cache_lock:
        adapter._session_db_cache_closed = True
        handles = list(adapter._home_stores.values())
        adapter._home_stores.clear()
    for handle in handles:
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
