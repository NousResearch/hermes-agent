import asyncio
from pathlib import Path

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
    # Single-profile responses carry no profile tag (the desktop treats absent
    # as the default profile); only the cross-profile aggregator tags rows.
    assert all("profile" not in r for r in response["results"])


class _ProfileScopedFakeDB:
    """Per-profile fake keyed by the profile home dir it is opened with.

    Backs the ``profile=all`` aggregator: each profile's ``state.db`` yields its
    own single content hit so we can assert merge ordering and profile tagging.
    """

    def __init__(self, db_path=None, read_only=False):
        self.profile = Path(db_path).parent.name

    def search_sessions_by_id(self, query, limit=20, include_archived=True):
        return []

    def search_messages(self, query, limit=20):
        started = {"default": 100, "coder": 200}[self.profile]
        return [
            {
                "session_id": f"{self.profile}_sess",
                "snippet": f"hit in {self.profile}",
                "role": "user",
                "source": "cli",
                "model": "m",
                "session_started": started,
            }
        ]

    def get_session(self, session_id):
        return {"id": session_id, "parent_session_id": None}

    def get_compression_tip(self, session_id):
        return session_id

    def close(self):
        pass


def _profile_info(name, path):
    class _Info:
        pass

    info = _Info()
    info.name = name
    info.path = path
    return info


def test_desktop_session_search_all_profiles_aggregates_and_tags(tmp_path, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    homes = {}
    for name in ("default", "coder"):
        home = tmp_path / name
        home.mkdir()
        (home / "state.db").write_bytes(b"")
        homes[name] = home

    # A profile whose state.db doesn't exist yet must be skipped, not crash the
    # whole cross-profile search.
    ghost = tmp_path / "ghost"
    ghost.mkdir()

    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: [
            _profile_info("default", homes["default"]),
            _profile_info("coder", homes["coder"]),
            _profile_info("ghost", ghost),
        ],
    )
    monkeypatch.setattr("hermes_state.SessionDB", _ProfileScopedFakeDB)

    response = asyncio.run(web_server.search_sessions(q="hit", limit=10, profile="all"))
    results = response["results"]

    # Every hit is tagged with its owning profile; the missing-DB profile is
    # silently skipped. Merged newest-first across profiles (coder 200 > default 100).
    assert [(r["session_id"], r["profile"]) for r in results] == [
        ("coder_sess", "coder"),
        ("default_sess", "default"),
    ]
    assert results[0]["snippet"] == "hit in coder"
    assert results[1]["snippet"] == "hit in default"


def test_desktop_session_search_all_profiles_respects_limit(tmp_path, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    homes = {}
    for name in ("default", "coder"):
        home = tmp_path / name
        home.mkdir()
        (home / "state.db").write_bytes(b"")
        homes[name] = home

    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: [
            _profile_info("default", homes["default"]),
            _profile_info("coder", homes["coder"]),
        ],
    )
    monkeypatch.setattr("hermes_state.SessionDB", _ProfileScopedFakeDB)

    response = asyncio.run(web_server.search_sessions(q="hit", limit=1, profile="all"))

    # Two profiles, one hit each, capped at limit=1 → the newest (coder) wins.
    assert [r["session_id"] for r in response["results"]] == ["coder_sess"]
