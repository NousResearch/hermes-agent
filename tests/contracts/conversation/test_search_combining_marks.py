"""A combining mark after a CJK run stays part of the searched text."""

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


@pytest.mark.parametrize("query, expected", [("山水湖i\u0307", [0, 1]), ("山水湖i\u0307*", [0, 1]), ("山水湖i", [0, 1, 3]), ('"山水湖i\u0307"', [0, 1])])
def test_combining_mark_does_not_match_unmarked_cjk_suffix(backend, scope_a, query, expected):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("marks", "cli"))
    positions = {}
    for index, text in enumerate(["山水湖I\u0307", "山水湖i\u0307", "山水湖İ", "山水湖i"]):
        row_id = store_a.messages.append("marks", NewMessage("user", text, 1700000000.0 + index))
        positions[row_id] = index
    hits = store_a.search.search_messages(query)
    assert sorted(positions[hit.message_id] for hit in hits) == expected
