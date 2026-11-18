import pytest

from persistence.sqlite.conversation._store.sessions import (
    create_session, get_session, list_sessions_rich, set_session_hidden,
)


@pytest.fixture
def holder(backend, scope_a):
    return backend.open(scope_a).holder()


def test_hidden_excluded_by_default_included_on_request(holder):
    create_session(holder, "visible", source="cli")
    create_session(holder, "secret", source="cli")
    # Give both a message so the default min_message_count filter keeps them.
    for sid in ("visible", "secret"):
        holder._write_sql(
            "UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,)
        )

    # Flip the hidden flag on one session.
    assert set_session_hidden(holder, "secret", True) is True
    assert get_session(holder, "secret")["hidden"] == 1
    assert get_session(holder, "visible")["hidden"] == 0

    # Default listing drops the hidden row; include_hidden=True surfaces it.
    default_ids = {s["id"] for s in list_sessions_rich(holder, min_message_count=1)}
    assert default_ids == {"visible"}

    all_ids = {
        s["id"]
        for s in list_sessions_rich(holder, min_message_count=1, include_hidden=True)
    }
    assert all_ids == {"visible", "secret"}

    # Unhiding brings it back into the default listing.
    assert set_session_hidden(holder, "secret", False) is True
    assert get_session(holder, "secret")["hidden"] == 0
    unhidden_ids = {s["id"] for s in list_sessions_rich(holder, min_message_count=1)}
    assert unhidden_ids == {"visible", "secret"}
