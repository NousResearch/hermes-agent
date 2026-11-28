"""A resume of a just-deleted session id must not materialize the row back.

``session.create`` mints the stored key but writes no state.db row until the first prompt
(no "Untitled" litter). A client that deletes that conversation still holds the id, and its
next resume reaches ``_resume_materialize_minted``, which creates the row through the legacy
upsert. That upsert ignored the deletion fence, so the deleted conversation came back as an
empty "Untitled" row, could be deleted again, and came back again — reported upstream
(#129439) and reproduced on v0.22.2.

The retired marker (``gateway.retired_session.v1.<id>``) is the fence: the canonical create
already refuses a retired id by name. The upsert now answers ``not_found`` instead of sweeping
a row in beside the marker, so the resume falls through to its existing 4007 "session not
found" — while a draft that was never deleted still materializes as before.
"""
import pytest

from persistence.conversation.sessions import NewSession, SessionId
from persistence.core.errors import AuthorityRefusal, RetiredSessionError
from persistence.sqlite.conversation._store.sessions import create_session, get_session


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


def _create(store, session_id: str, source: str = "cli") -> None:
    store.sessions.create(NewSession(session_id=SessionId(session_id), source=source))


def test_a_resumed_deleted_session_is_not_materialized(store, tmp_path):
    _create(store, "deleted-conv")
    assert store.sessions.delete("deleted-conv", media_root=str(tmp_path)) == ("deleted-conv",)
    assert get_session(store.holder(), "deleted-conv") is None

    # This is exactly what the desktop client's resume does for an id it still holds.
    with pytest.raises(RetiredSessionError) as refused:
        _create(store, "deleted-conv")
    assert refused.value.reason == "not_found"
    assert get_session(store.holder(), "deleted-conv") is None, (
        "the deleted row was materialized again beside its deletion fence")


def test_the_legacy_upsert_refuses_a_retired_id(store, tmp_path):
    _create(store, "deleted-2")
    store.sessions.delete("deleted-2", media_root=str(tmp_path))

    with pytest.raises(AuthorityRefusal) as refused:
        create_session(store.holder(), "deleted-2", "cli")
    assert refused.value.reason == "not_found"
    assert get_session(store.holder(), "deleted-2") is None


def test_a_draft_that_was_never_deleted_still_materializes(store):
    # The minted-but-unpersisted path this guard must not break (#96793).
    _create(store, "live-draft")
    row = get_session(store.holder(), "live-draft")
    assert row is not None and row["id"] == "live-draft"
