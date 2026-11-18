"""Reference listings retain the source admission and projection rules."""

from datetime import UTC, datetime
from unittest.mock import patch

from persistence.conversation.sessions import NewSession, SessionCountQuery, SessionOrder, SessionQuery, SessionWindow
from tests.contracts._reference import open_reference_backend
from tests.contracts._scopes import profile_scope


def test_counts_keep_raw_children_and_project_only_the_filter_target():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        for sid, source, parent in [("root", "slack", None), ("parent", "telegram", None), ("child", "discord", "parent"), ("hidden", "cli", None)]:
            backend.open(scope).sessions.create(NewSession(sid, source, parent_session_id=parent))
        backend.open(scope).sessions.end("root", "compression")
        backend.open(scope).sessions.create(NewSession("tip", "slack", parent_session_id="root"))
        backend.open(scope).sessions.set_title("tip", "tip title")
        backend.open(scope).sessions.set_hidden("hidden", True)
        # Counts admit hidden rows and children unless told to exclude children; the compression tip is a child of its root.
        assert backend.open(scope).sessions.count_sessions(SessionCountQuery(exclude_children=True)) == 3
        assert dict(backend.open(scope).sessions.count_by_source(exclude_children=True)) == {"cli": 1, "slack": 1, "telegram": 1}
        assert backend.open(scope).sessions.count_sessions(SessionCountQuery()) == 5
        assert backend.open(scope).sessions.count_at_least(5) and not backend.open(scope).sessions.count_at_least(6)
        query = SessionQuery(search_text="tip title", order=SessionOrder.LAST_ACTIVE)
        assert [row.session.session_id for row in backend.open(scope).sessions.list_sessions(query, SessionWindow(limit=10))] == ["tip"]
        assert {row.session.session_id for row in backend.open(scope).sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))} == {"parent", "tip"}


def test_projected_listing_row_keeps_root_parent_and_config():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("root", "cli", model_config={"temperature": 0.1}))
        backend.open(scope).sessions.end("root", "compression")
        backend.open(scope).sessions.create(NewSession("tip", "cli", parent_session_id="root"))
        rows = backend.open(scope).sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
        assert len(rows) == 1
        assert rows[0].session.session_id == "tip"
        assert rows[0].session.parent_session_id is None
        assert rows[0].session.model_config == {"temperature": 0.1}


def test_id_search_uses_ascii_case_folding_for_candidate_membership():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        for sid in ("ÉCHO", "écho-low", "x-écho-tail"):
            backend.open(scope).sessions.create(NewSession(sid, "cli"))
        rows = backend.open(scope).sessions.search_by_id("écho", limit=2)
        assert [row.session.session_id for row in rows] == ["écho-low", "x-écho-tail"]


def test_pinned_rows_ride_every_window_they_are_not_in():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        for sid in ("first", "second", "pinned"):
            backend.open(scope).sessions.create(NewSession(sid, "cli"))
        backend.open(scope).sessions.set_pinned("pinned", True)
        query = SessionQuery(include_pinned=True)
        first = backend.open(scope).sessions.list_sessions(query, SessionWindow(limit=1))
        assert [row.session.session_id for row in first] == ["pinned"]
        # An offset window back-fills every pin it missed, so the pin follows the slice on later windows.
        second = backend.open(scope).sessions.list_sessions(query, SessionWindow(limit=1, offset=1))
        assert [row.session.session_id for row in second] == ["second", "pinned"]


def test_started_listing_ignores_chain_search_text():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        for sid in ("match", "other"):
            backend.open(scope).sessions.create(NewSession(sid, "cli"))
        query = SessionQuery(search_text="match", order=SessionOrder.STARTED_AT)
        rows = backend.open(scope).sessions.list_sessions(query, SessionWindow(limit=10))
        assert {row.session.session_id for row in rows} == {"match", "other"}


def test_roster_latest_local_chat_ties_keep_the_first_stored_session():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        with patch("tests.contracts._reference.conversation.sessions._now", return_value=datetime.fromtimestamp(1700000000, UTC)):
            for sid in ("local-a", "local-z"):
                backend.open(scope).sessions.create(NewSession(sid, "cli", chat_id="local-chat"))
        latest = backend.open(scope).sessions.roster_sessions(owner_user_id=None, owner=None).last_session
        assert latest is not None and latest.id == "local-a"
