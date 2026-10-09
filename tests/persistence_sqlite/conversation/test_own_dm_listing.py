"""``own_dm_of``: the Telegram Mini App's non-admin tier lists, counts and checks only its own DM sessions."""

import pytest

from persistence.conversation.sessions import (
    NewSession, SessionCountQuery, SessionQuery, SessionWindow, is_own_dm_session)

ROWS = {
    "mine": dict(source="telegram", user_id="42", chat_id="42", chat_type="dm"),
    "theirs": dict(source="telegram", user_id="99", chat_id="99", chat_type="dm"),
    # A group chat id can collide numerically with a user id: chat_type is part of the rule.
    "group-same-id": dict(source="telegram", chat_id="42", chat_type="group"),
    # Rows from before chat ids were captured cannot prove ownership.
    "legacy": dict(source="telegram", user_id="42", chat_type="dm"),
    "other-platform": dict(source="cli", chat_id="42", chat_type="dm"),
}


@pytest.fixture()
def sessions(backend, scope_a):
    store = backend.open(scope_a).sessions
    for sid, fields in ROWS.items():
        store.create(NewSession(sid, **fields))
    return store


def _ids(sessions, **query):
    page = sessions.list_sessions(SessionQuery(**query), SessionWindow(limit=50))
    return {entry.session.session_id for entry in page}


def test_listing_keeps_only_the_requesters_dm(sessions):
    assert _ids(sessions, own_dm_of="42") == {"mine"}


def test_count_matches_the_listing(sessions):
    assert sessions.count_sessions(SessionCountQuery(own_dm_of="42")) == 1


def test_no_filter_lists_everything(sessions):
    assert _ids(sessions) == set(ROWS)


def test_pinned_backfill_excludes_other_users_pins(sessions):
    sessions.set_pinned("theirs", True)
    assert _ids(sessions, own_dm_of="42", include_pinned=True) == {"mine"}


def test_blank_user_id_is_refused_not_unfiltered(sessions):
    with pytest.raises(ValueError):
        sessions.list_sessions(SessionQuery(own_dm_of=""), SessionWindow(limit=50))
    with pytest.raises(ValueError):
        sessions.count_sessions(SessionCountQuery(own_dm_of=""))


@pytest.mark.parametrize("sid", sorted(ROWS))
def test_row_predicate_agrees_with_the_sql_filter(sessions, sid):
    """The listing's SQL clause and the single-row check (the dashboard's per-session ownership) must agree."""
    row = sessions.get(sid).as_session_row()
    assert is_own_dm_session(row, "42") == (sid in _ids(sessions, own_dm_of="42"))


@pytest.mark.parametrize("row,user_id,owned", [
    ({"source": "telegram", "chat_type": "dm", "chat_id": "42"}, "42", True),
    ({"source": "telegram", "chat_type": "dm", "chat_id": 42}, "42", True),  # stored type must not matter
    ({"source": "telegram", "chat_type": "dm", "chat_id": "42"}, 42, True),
    ({"source": "telegram", "chat_type": "dm", "chat_id": "99"}, "42", False),
    ({"source": "telegram", "chat_type": "dm", "chat_id": None}, "42", False),
    ({"source": "telegram", "chat_type": "dm", "chat_id": ""}, "42", False),
    ({"source": "telegram", "chat_type": "dm", "chat_id": "42"}, None, False),
    ({"source": "telegram", "chat_type": "dm", "chat_id": "42"}, "", False),
])
def test_row_predicate_fails_closed(row, user_id, owned):
    assert is_own_dm_session(row, user_id) is owned
