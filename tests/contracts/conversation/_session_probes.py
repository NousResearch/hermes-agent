"""Session facts the contract suites observe through more than one public verb."""

from persistence import RetiredSessionError

from tests.contracts._scopes import MEDIA_ROOT


def is_retired(backend, scope, session_id) -> bool:
    """Whether ``session_id`` is retired: no row, and an exclusive create of it refuses as deleted. A probe row that lands is discarded without retiring the id."""
    store = backend.open(scope)
    if store.sessions.get(session_id) is not None:
        return False
    try:
        store.sessions.create_exclusive(session_id, source="probe")
    except RetiredSessionError:
        return True
    store.sessions.discard_unadmitted(session_id, media_root=MEDIA_ROOT)
    return False
