"""Placeholder-first creation on a SQLite backend store must not hide real CLI sessions (#138831)."""
import pytest

from hermes_cli.gateway_session_store import SurfaceLabelReader
from hermes_cli.session_listing import query_session_listing
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.conversation._store.titles import set_session_title
from persistence.sqlite.conversation._store.usage import update_token_counts
from persistence_assembly import open_persistence_reader


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def test_placeholder_creator_remains_visible_after_gateway_routing(db, store):
    surface = "cli"
    update_token_counts(db, "previous", input_tokens=5)
    create_session(db, "previous", source=surface)
    set_session_title(db, "previous", "Previous conversation")
    store.routing.record_session_peer(
        "previous", source="local", session_key="local:previous", chat_id="previous",
    )
    with open_persistence_reader(db.db_path) as reader:
        rows = query_session_listing(SurfaceLabelReader(reader), source=surface)
    assert [row["id"] for row in rows] == ["previous"]
    assert get_session(db, "previous")["created_source"] == surface
    assert get_session(db, "previous")["input_tokens"] == 5


def test_routing_label_local_is_not_creation_provenance(db):
    """An upsert carrying the routing label ``local`` never stamps it, and a row that already holds
    ``local`` (or the ``unknown`` placeholder) is repaired by the real creator."""
    update_token_counts(db, "placeholder", input_tokens=1)
    create_session(db, "placeholder", source="local")
    assert get_session(db, "placeholder")["created_source"] == "unknown"
    create_session(db, "frozen", source="local")
    assert get_session(db, "frozen")["created_source"] == "local"
    for sid in ("placeholder", "frozen"):
        create_session(db, sid, source="cli")
        assert get_session(db, sid)["created_source"] == "cli", sid
