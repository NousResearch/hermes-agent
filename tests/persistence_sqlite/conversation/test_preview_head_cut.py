"""The SQLite listing preview trims leading whitespace before the head cut, so every cut preview carries the ``...``
marker and the shaped text is the first 60 characters of the message's text."""

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, RecentBrowseQuery

KEY = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"


@pytest.mark.parametrize("lead", ["", "\n\n\t  ", " " * 20])
def test_recent_preview_cuts_after_the_leading_whitespace(backend, scope_a, lead):
    store = backend.open(scope_a)
    text = f"{'x' * 24} {KEY} is the key"
    store.sessions.create(NewSession("s_cut", "cli"))
    store.messages.append("s_cut", NewMessage(role="user", content=lead + text))
    store.sessions.create(NewSession("s_dots", "cli"))
    store.messages.append("s_dots", NewMessage(role="user", content="wait..."))

    previews = {entry.session_id: entry.preview
                for entry in store.sessions.list_recent_sessions(RecentBrowseQuery(limit=10))}

    assert previews["s_cut"] == text[:60] + "..."
    assert previews["s_dots"] == "wait..."
