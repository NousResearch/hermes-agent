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
                "last_active": 240,
            }
        ]

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
                "last_active": 240,
            },
            {
                "session_id": "content_session",
                "snippet": "content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 200,
                "last_active": 225,
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
    monkeypatch.setattr(web_server.time, "time", lambda: 250)
    monkeypatch.setattr(web_server, "_active_session_registry_by_session", lambda: {})

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
                "last_active": 240,
                "last_tool_runtime_event": None,
                "last_activity_age_seconds": 10,
                "queued_steer_count": 0,
                "compression_tip_session_id": "20260603_090200_exact",
                "status_evidence_source": ["db_row", "unavailable"],
            },
            {
                "session_id": "content_session",
                "lineage_root": "content_session",
                "snippet": "content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 200,
                "last_active": 225,
                "last_tool_runtime_event": None,
                "last_activity_age_seconds": 25,
                "queued_steer_count": 0,
                "compression_tip_session_id": "content_session",
                "status_evidence_source": ["db_row", "unavailable"],
            },
        ]
    }


class _CompressedLineageSearchDB:
    """Fake proving search status evidence ages from the surfaced tip row."""

    closed = False

    def search_sessions_by_id(self, query, limit=20, include_archived=True):
        assert query == "oldroot"
        assert include_archived is True
        return []

    def search_messages(self, query, limit=20):
        assert query == "oldroot*"
        return [
            {
                "session_id": "old_root",
                "snippet": "old root content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 100,
            }
        ]

    def get_session(self, session_id):
        return {
            "old_root": {
                "id": "old_root",
                "parent_session_id": None,
                "started_at": 100,
                "last_active": 480,
                "ended_at": 500,
                "end_reason": "compression",
            },
            "fresh_tip": {
                "id": "fresh_tip",
                "parent_session_id": "old_root",
                "started_at": 501,
                "last_active": 900,
                "ended_at": None,
            },
        }.get(session_id)

    def get_compression_tip(self, session_id):
        return "fresh_tip" if session_id == "old_root" else session_id

    def close(self):
        self.closed = True


def test_desktop_session_search_ages_compressed_hit_from_surfaced_tip(monkeypatch):
    monkeypatch.setattr("hermes_state.SessionDB", _CompressedLineageSearchDB)
    monkeypatch.setattr(web_server.time, "time", lambda: 1000)
    monkeypatch.setattr(web_server, "_active_session_registry_by_session", lambda: {})

    response = asyncio.run(web_server.search_sessions(q="oldroot", limit=1))

    assert response == {
        "results": [
            {
                "session_id": "fresh_tip",
                "lineage_root": "old_root",
                "snippet": "old root content hit",
                "role": "assistant",
                "source": "desktop",
                "model": "gpt",
                "session_started": 100,
                "last_active": 900,
                "last_tool_runtime_event": None,
                "last_activity_age_seconds": 100,
                "queued_steer_count": 0,
                "compression_tip_session_id": "fresh_tip",
                "status_evidence_source": [
                    "db_row",
                    "compression_projection",
                    "unavailable",
                ],
            }
        ]
    }


def test_session_status_evidence_surfaces_value_free_active_owner_summary(monkeypatch):
    from hermes_cli import active_sessions

    monkeypatch.setattr(
        active_sessions,
        "active_session_registry_snapshot",
        lambda: [
            {
                "session_id": "session-1",
                "owner_summary": {
                    "pid": 12345,
                    "session_id": "session-1",
                    "surface": "gateway:telegram",
                    "owner_kind": "gateway",
                    "cwd_fingerprint": "abc123def4567890",
                    "private_url": "https://secret.invalid",
                },
            }
        ],
    )

    registry = web_server._active_session_registry_by_session()
    row = {"id": "session-1", "started_at": 100}

    web_server._attach_session_status_evidence(
        row,
        now=105,
        registry_by_session=registry,
    )

    assert row["active_session_owner_summary"] == {
        "pid": 12345,
        "session_id": "session-1",
        "surface": "gateway:telegram",
        "owner_kind": "gateway",
        "cwd_fingerprint": "abc123def4567890",
    }
    assert "secret.invalid" not in str(row["active_session_owner_summary"])
