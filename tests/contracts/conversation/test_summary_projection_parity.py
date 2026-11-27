"""Session listings and the single-row summary share the hybrid row and preview."""

from datetime import UTC, datetime

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow


@pytest.mark.parametrize("include_children", [False, True])
def test_summary_and_listing_share_projected_records_and_previews(backend, scope_a, include_children):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("root", "cli"))
    store_a.sessions.end("root", "compression")
    store_a.sessions.create(NewSession("tip", "cli", parent_session_id="root"))
    store_a.messages.append("tip", NewMessage("user", "preview", timestamp=datetime(2026, 9, 28, tzinfo=UTC)))
    root = store_a.sessions.get("root")
    tip = store_a.sessions.get("tip")
    entries = store_a.sessions.list_sessions(SessionQuery(include_children=include_children), SessionWindow(limit=10))
    by_id = {e.session.session_id: e for e in entries}
    assert by_id["tip"].preview == "preview"
    assert by_id["tip"].session.message_count == 1
    if include_children:
        assert set(by_id) == {"root", "tip"}
        assert by_id["root"].session == root
        assert by_id["tip"].session == tip
        assert all(e.lineage_root_id is None for e in entries)
    else:
        assert set(by_id) == {"tip"}
        assert by_id["tip"].session.started_at == root.started_at
        assert by_id["tip"].session.parent_session_id is None
        assert by_id["tip"].lineage_root_id == "root"
    direct = store_a.sessions.summary("tip")
    assert direct.session == tip and direct.preview == "preview"
    assert direct.lineage_root_id is None
