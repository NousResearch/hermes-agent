import asyncio

from hermes_cli import web_server
import hermes_cli.web_routers.sessions as _rt_sessions


class _FakeSessionDB:
    """Fake backing the /api/sessions/search endpoint.

    The endpoint surfaces direct session-id matches first, then FTS message
    matches, deduping both by compression lineage root. This fake has no
    compression chains (get_session returns no parent), so each session is its
    own lineage root.
    """

    closed = False
    opened_read_only = None
    requested_fields = None

    def __init__(self, *args, **kwargs):
        type(self).opened_read_only = kwargs.get("read_only")

    @staticmethod
    def _source_allowed(row, source=None, sources=None, exclude_sources=None):
        row_source = row.get("source")
        if source and row_source != source:
            return False
        if sources and row_source not in sources:
            return False
        if exclude_sources and row_source in exclude_sources:
            return False
        return True

    def search_sessions_by_id(
        self,
        query,
        limit=20,
        include_archived=True,
        source=None,
        sources=None,
        exclude_sources=None,
    ):
        assert query == "20260603"
        assert include_archived is True
        rows = [
            {
                "id": "20260603_090200_exact",
                "preview": "ID match preview",
                "source": "cli",
                "model": "claude",
                "started_at": 100,
                "last_active": 150,
            }
        ]
        return [
            row
            for row in rows
            if self._source_allowed(
                row, source=source, sources=sources, exclude_sources=exclude_sources
            )
        ][:limit]

    def search_messages(
        self,
        query,
        source_filter=None,
        exclude_sources=None,
        limit=20,
        fields=None,
    ):
        assert query == "20260603*"
        type(self).requested_fields = fields
        rows = [
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
        return [
            row
            for row in rows
            if self._source_allowed(
                row, sources=source_filter, exclude_sources=exclude_sources
            )
        ][:limit]

    def get_session(self, session_id):
        # No compression chains in this fixture — every session is its own root.
        return {"id": session_id, "parent_session_id": None}

    def get_compression_tip(self, session_id):
        return session_id

    def close(self):
        self.closed = True


def test_desktop_session_search_merges_id_matches_before_content_matches(monkeypatch):
    _FakeSessionDB.opened_read_only = None
    _FakeSessionDB.requested_fields = None
    monkeypatch.setattr("hermes_state.SessionDB", _FakeSessionDB)

    response = asyncio.run(_rt_sessions.search_sessions(q="20260603", limit=2))

    assert _FakeSessionDB.requested_fields is not None
    assert "context" not in _FakeSessionDB.requested_fields
    # ID match surfaces first; the content hit on the SAME session is deduped
    # by lineage root (not double-listed); the unrelated content hit follows.
    assert response == {
        "results": [
            {
                "id": "20260603_090200_exact",
                "profile": "default",
                "is_default_profile": True,
                "session_id": "20260603_090200_exact",
                "lineage_root": "20260603_090200_exact",
                "snippet": "ID match preview",
                "role": None,
                "source": "cli",
                "model": "claude",
                "session_started": 100,
                # Row recency rides on id-match rows (sessions table)...
                "last_active": 150,
            },
            {
                "id": "content_session",
                "profile": "default",
                "is_default_profile": True,
                "session_id": "content_session",
                "lineage_root": "content_session",
                "snippet": "content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 200,
                # ...while FTS hits have none and leave it null.
                "last_active": None,
            },
        ]
    }
    assert _FakeSessionDB.opened_read_only is True


def test_desktop_session_search_stamps_the_requested_profile(monkeypatch):
    monkeypatch.setattr(
        _rt_sessions, "_cron_profile_home", lambda profile: (profile, None)
    )
    monkeypatch.setattr(
        _rt_sessions,
        "_open_session_db_for_profile",
        lambda profile, *, read_only: _FakeSessionDB(read_only=read_only),
    )

    response = asyncio.run(
        _rt_sessions.search_sessions(q="20260603", limit=2, profile="worker")
    )

    assert {
        (row["profile"], row["is_default_profile"])
        for row in response["results"]
    } == {("worker", False)}


def test_desktop_session_search_attaches_profile_to_rich_results(monkeypatch):
    class _RichFakeSessionDB(_FakeSessionDB):
        def get_session_rich_row(self, session_id):
            return {
                "id": session_id,
                "source": "cli",
                "model": "claude",
                "title": "Custom Title",
                "started_at": 100,
                "ended_at": None,
                "last_active": 100,
                "message_count": 2,
                "tool_call_count": 0,
                "input_tokens": 10,
                "output_tokens": 20,
                "preview": "Test preview",
                "parent_session_id": None,
                "archived": False,
            }

    monkeypatch.setattr("hermes_state.SessionDB", _RichFakeSessionDB)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    response = asyncio.run(_rt_sessions.search_sessions(q="20260603", limit=1, profile="personal"))
    assert response["results"][0]["profile"] == "personal"
    assert response["results"][0]["title"] == "Custom Title"


class _DeepLineageSessionDB(_FakeSessionDB):
    """A compression lineage deeper than the chain walk's defensive bound.

    Sessions s000..s178 form one in-place-compression lineage (each session's
    parent ended with end_reason='compression'). The forward chain walk is
    bounded; the bug remapped the SEARCH hit via tip(root), which truncates to
    a stale mid id, while the CLI resumes from the matched id and reaches the
    live tip (#125041).
    """

    DEPTH = 1101
    TIP = "s1100"
    MID = "s100"

    def search_sessions_by_id(self, query, limit=20, include_archived=True,
                              source=None, sources=None, exclude_sources=None):
        return []  # pragma: no cover - content-hit path only in this test

    def search_messages(self, query, source_filter=None, exclude_sources=None,
                        limit=20, fields=None):
        return [{
            "session_id": self.MID,
            "snippet": "content hit inside a deep lineage",
            "role": "assistant",
            "source": "desktop",
            "model": "gpt",
            "session_started": 200,
        }][:limit]

    def get_session(self, session_id):
        parent = None
        if session_id != "s000":
            parent = f"s{int(session_id[1:]) - 1:03d}"
        n = int(session_id[1:])
        return {"id": session_id, "parent_session_id": parent,
                "end_reason": "compression",
                "ended_at": n + 1, "started_at": n}

    def get_compression_tip(self, session_id):
        # Faithful simulation of the REAL bounded forward walk in
        # SessionDB.get_compression_chain: at most _CHAIN_CAP hops from the
        # START id, so a lineage deeper than the cap truncates when the walk
        # starts far from the tip. Resolving from the MATCHED id keeps the
        # remaining distance under the cap; the raised bound lets even a
        # root-started walk terminate at the live tip (#125041).
        from hermes_state_compression import _CHAIN_CAP

        idx = int(session_id[1:])
        return f"s{min(idx + _CHAIN_CAP, self.DEPTH - 1):03d}"


def test_deep_lineage_search_resolves_tip_from_matched_id(monkeypatch):
    monkeypatch.setattr("hermes_state.SessionDB", _DeepLineageSessionDB)

    response = asyncio.run(_rt_sessions.search_sessions(q="content", limit=2))

    [row] = response["results"]
    # The resume id MUST be the live tip — not a stale mid the bounded
    # root-started walk truncated to.
    assert row["session_id"] == _DeepLineageSessionDB.TIP
    # Dedupe/lineage bookkeeping stays keyed by the root.
    assert row["lineage_root"] == "s000"
