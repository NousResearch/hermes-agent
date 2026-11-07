"""`hermes sessions archive --include-open` on a SQLite backend store: ``session_retention.archive_sessions`` (reversible) may reach never-ended sessions, and the maintenance repository's prune refuses ``include_open`` (#139597)."""
import pytest

from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from session_retention import archive_sessions
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _archive(store, **kwargs):
    return archive_sessions(store, store.holder().db_path.parent, **kwargs)


def test_archive_include_open_reaches_never_ended_sessions(db, store):
    """Desktop sessions never set ended_at, so the shared ended-only selector hid them from
    bulk archive entirely; archive is reversible, so include_open lifts the guard (#139597)."""
    create_session(db, session_id="never-ended", source="desktop")
    create_session(db, session_id="ended", source="desktop")
    end_session(db, "ended", "completed")

    assert _archive(store, source="desktop") == 1
    assert get_session(db, "ended")["archived"] == 1
    assert get_session(db, "never-ended")["archived"] == 0

    assert _archive(store, source="desktop", criteria=PruneCriteria(include_open=True)) == 1
    assert get_session(db, "never-ended")["archived"] == 1
    assert store.maintenance.count_open_prune_matches(
        source="desktop", criteria=PruneCriteria(archived=False, include_open=True)) == 0


def test_prune_rejects_include_open_fail_closed(db, store):
    """include_open is the reversible-archive escape hatch; the delete path must refuse it
    rather than silently drop the ended-session guard."""
    create_session(db, session_id="open", source="cli")
    with pytest.raises(ValueError, match="archive-only"):
        store.maintenance.prune_sessions(criteria=PruneCriteria(include_open=True), media_root=MEDIA_ROOT)
    assert get_session(db, "open") is not None
