"""Runtime-bound API storage selection; borrowed stores are never adapter-owned. Without a runtime the adapter caches
one storage interface handle per profile home and closes it. An authority's handle for the API and webhook doors:
:func:`authority_store`."""
from pathlib import Path


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
    when None): the reserving authority's under a runtime, else the adapter's cached handle, or one opened beside its
    explicit ``_session_db`` override; :func:`close_cached_stores` closes the adapter's. None without a store, beside an
    override without a file path, or when the handle cannot open (the next call tries again)."""
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
    path = getattr(override, "db_path", None)
    if not isinstance(path, (str, Path)):
        return None
    home = Path(path).parent
    from persistence_beside import open_beside
    key = str(home)
    with adapter._session_db_cache_lock:
        if adapter._session_db_cache_closed:
            return None
        handles = adapter.__dict__.setdefault("_persistence_handles", {})
        handle = handles.get(key)
        if handle is None and (handle := open_beside(Path(home))) is not None:
            handles[key] = handle
        return handle


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
    """Close the storage interface handles ``adapter``'s profile cache holds and those opened beside its override."""
    from persistence_beside import close_beside
    with adapter._session_db_cache_lock:
        adapter._session_db_cache_closed = True
        handles = [*adapter._home_stores.values(), *adapter.__dict__.pop("_persistence_handles", {}).values()]
        adapter._home_stores.clear()
    for handle in handles:
        close_beside(handle)
