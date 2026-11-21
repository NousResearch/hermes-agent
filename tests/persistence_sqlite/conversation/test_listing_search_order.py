"""The SQLite listing's text filter is active only for recency-ordered session listings."""

from unittest.mock import patch

import pytest

from persistence.conversation.sessions import NewSession, SessionOrder, SessionQuery, SessionWindow


@pytest.mark.parametrize("order, expected", [(SessionOrder.STARTED_AT, ["other", "needle"]), (SessionOrder.LAST_ACTIVE, ["needle"])])
def test_search_follows_the_listing_order(backend, scope_a, order, expected):
    sessions = backend.open(scope_a).sessions
    for index, sid in enumerate(("needle", "other")):
        with patch("time.time", return_value=1700000000 + index):
            sessions.create(NewSession(sid, "cli"))
    page = sessions.list_sessions(SessionQuery(order=order, search_text="needle"), SessionWindow(limit=10))
    assert [row.session.session_id for row in page] == expected
