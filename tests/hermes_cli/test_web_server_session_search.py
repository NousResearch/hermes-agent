import asyncio

from hermes_cli import web_server


class _FakeSessionDB:
    """Fake backing the /api/sessions/search endpoint.

    The endpoint surfaces direct session-id matches first, then FTS message
    matches, deduping both by compression lineage root. This fake has no
    compression chains (get_session returns no parent), so each session is its
    own lineage root.
    """

    closed = False

    def search_sessions_by_id(self, query, limit=20, include_archived=True):
        assert query == "20260603"
        assert include_archived is True
        return [
            {
                "id": "20260603_090200_exact",
                "preview": "ID match preview",
                "source": "cli",
                "model": "claude",
                "started_at": 100,
            }
        ]

    def search_sessions_by_title(self, query, limit=20, include_archived=True):
        assert query == "20260603"
        assert include_archived is True
        return []

    def search_messages(self, query, limit=20):
        assert query == "20260603*"
        return [
            {
                "session_id": "20260603_090200_exact",
                "snippet": "duplicate content hit should not replace ID hit",
                "role": "user",
                "source": "cli",
                "model": "claude",
                "session_started": 100,
            },
            {
                "session_id": "content_session",
                "snippet": "content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 200,
            },
        ]

    def get_session(self, session_id):
        # No compression chains in this fixture — every session is its own root.
        return {"id": session_id, "parent_session_id": None}

    def get_compression_tip(self, session_id):
        return session_id

    def close(self):
        self.closed = True


def test_desktop_session_search_merges_id_matches_before_content_matches(monkeypatch):
    monkeypatch.setattr("hermes_state.SessionDB", _FakeSessionDB)

    response = asyncio.run(web_server.search_sessions(q="20260603", limit=2))

    # ID match surfaces first; the content hit on the SAME session is deduped
    # by lineage root (not double-listed); the unrelated content hit follows.
    assert response == {
        "results": [
            {
                "session_id": "20260603_090200_exact",
                "lineage_root": "20260603_090200_exact",
                "snippet": "ID match preview",
                "role": None,
                "source": "cli",
                "model": "claude",
                "session_started": 100,
            },
            {
                "session_id": "content_session",
                "lineage_root": "content_session",
                "snippet": "content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 200,
            },
        ]
    }


class _FakeTitleSessionDB:
    """Fake proving title matches rank above content matches (and dedupe)."""

    def search_sessions_by_id(self, query, limit=20, include_archived=True):
        return []

    def search_sessions_by_title(self, query, limit=20, include_archived=True):
        assert query == "portal"
        return [
            {
                "id": "titled_session",
                "title": "DNS Portal Investigation",
                "preview": "first user message",
                "source": "discord",
                "model": "claude",
                "started_at": 300,
            }
        ]

    def search_messages(self, query, limit=20):
        assert query == "portal*"
        return [
            # Content hit on the SAME session the title lane already surfaced —
            # must dedupe by lineage, not double-list.
            {
                "session_id": "titled_session",
                "snippet": "portal mentioned in content",
                "role": "user",
                "source": "discord",
                "model": "claude",
                "session_started": 300,
            },
            {
                "session_id": "content_only",
                "snippet": "another portal content hit",
                "role": "assistant",
                "source": "cli",
                "model": "gpt",
                "session_started": 400,
            },
        ]

    def get_session(self, session_id):
        return {"id": session_id, "parent_session_id": None}

    def get_compression_tip(self, session_id):
        return session_id

    def close(self):
        pass


def test_desktop_session_search_ranks_title_matches_before_content_matches(monkeypatch):
    monkeypatch.setattr("hermes_state.SessionDB", _FakeTitleSessionDB)

    response = asyncio.run(web_server.search_sessions(q="portal", limit=5))

    results = response["results"]
    assert [r["session_id"] for r in results] == ["titled_session", "content_only"]
    # The title lane carries the title through to the client.
    assert results[0]["title"] == "DNS Portal Investigation"
    # Title-lane snippet prefers the session preview.
    assert results[0]["snippet"] == "first user message"
    # The duplicate content hit on titled_session did not double-list it.
    assert len(results) == 2
