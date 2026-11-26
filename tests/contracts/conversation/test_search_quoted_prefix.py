"""A quoted run keeps a trailing prefix operator."""

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


@pytest.mark.parametrize("query, expected", [
    ('"quick bro"*', ["quick brood fox", "quick brown fox"]),
    ('"quick bro"**', ["quick brood fox", "quick brown fox"]),
    ('"zeb"*', ["zeb quick", "zebra quick"]),
    ('"zeb"* AND quick', ["zeb quick", "zebra quick"]),
    ('"zebra*"', ["zebra quick"]),
])
def test_quoted_prefix_keeps_the_store_match_set(backend, scope_a, query, expected):
    store_a = backend.open(scope_a)
    sid = SessionId("quoted-prefix")
    store_a.sessions.create(NewSession(sid, "cli"))
    bodies = ["quick brown fox", "quick brood fox", "zebra quick", "zeb quick"]
    identities = {}
    for i, body in enumerate(bodies):
        row_id = store_a.messages.append(sid, NewMessage("user", body, 1700000000.0 + i))
        identities[row_id] = body
    hits = store_a.search.search_messages(query)
    assert sorted(identities[h.message_id] for h in hits) == expected
