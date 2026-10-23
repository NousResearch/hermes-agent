"""Runtime-bound API storage selection; borrowed stores are never adapter-owned."""
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
    return authority.db


def remove_profile_routes(store, session_id, db):
    """Retire routing only in the same profile as the selected transcript database."""
    from gateway.session import _session_key_namespace
    from hermes_constants import profile_name_for_home

    namespace = _session_key_namespace(profile_name_for_home(Path(db.db_path).parent))
    return store.remove_by_session_id(session_id, namespace=namespace)
