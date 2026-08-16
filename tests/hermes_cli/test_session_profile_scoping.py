"""Session listings must respect the shared session store's profile column.

A multiplexed gateway keeps every served profile's sessions in one store,
namespaced ``agent:<profile>:…`` and stamped with ``profile_name`` (see
``website/docs/user-guide/multi-profile-gateways.md``). The sessions API used
to open ``profiles/<name>/state.db`` per profile and then overwrite each row's
``profile`` with the *requested* one — so Career Ops chats handled by the
multiplexer were read out of the default store and relabelled ``default``,
surfacing under the wrong profile in the desktop app.

Filtering only applies to a shared store: a per-profile file is already scoped
by the file itself, and rows written before the ``profile_name`` column existed
are unstamped there, so blanket filtering would hide them.
"""

import pytest

from hermes_state import SessionDB


def _seed(db):
    db.create_session("d1", source="telegram", user_id="u")
    db.create_session("d2", source="telegram", user_id="u")
    db.create_session("c1", source="telegram", user_id="u")
    db.create_session("c2", source="telegram", user_id="u")
    db._conn.execute("UPDATE sessions SET profile_name = NULL WHERE id = 'd1'")
    db._conn.execute("UPDATE sessions SET profile_name = 'default' WHERE id = 'd2'")
    db._conn.execute("UPDATE sessions SET profile_name = 'career-ops' WHERE id IN ('c1','c2')")
    db._conn.commit()


class TestProfileFilteredQueries:
    def test_default_matches_unstamped_and_default_rows(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)

        ids = {
            r["id"] for r in db.list_sessions_rich(limit=50, profile_name="default")
        }
        assert ids == {"d1", "d2"}

    def test_named_profile_matches_only_its_own_rows(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)

        ids = {
            r["id"] for r in db.list_sessions_rich(limit=50, profile_name="career-ops")
        }
        assert ids == {"c1", "c2"}

    def test_no_filter_returns_every_profile(self, tmp_path):
        """Unfiltered stays unfiltered — the per-profile-store path relies on it."""
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)

        ids = {r["id"] for r in db.list_sessions_rich(limit=50)}
        assert ids == {"d1", "d2", "c1", "c2"}

    def test_legacy_main_namespace_counts_as_default(self, tmp_path):
        """The default profile keeps the historical agent:main namespace."""
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)
        db._conn.execute("UPDATE sessions SET profile_name = 'main' WHERE id = 'd2'")
        db._conn.commit()

        ids = {
            r["id"] for r in db.list_sessions_rich(limit=50, profile_name="default")
        }
        assert ids == {"d1", "d2"}

    @pytest.mark.parametrize(
        "profile,expected",
        [("default", 2), ("career-ops", 2), (None, 4)],
    )
    def test_count_agrees_with_the_listing(self, tmp_path, profile, expected):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)

        assert db.session_count(profile_name=profile) == expected
        rows = db.list_sessions_rich(limit=50, profile_name=profile)
        assert len(rows) == expected


class TestFilterSelection:
    def test_no_filter_when_no_multiplexer_is_running(self, monkeypatch):
        """Single-profile installs keep the historical unfiltered behavior."""
        from hermes_cli import web_server

        monkeypatch.setattr(web_server, "_multiplex_host_home", lambda: None)
        assert web_server._session_profile_filter(None) is None
        assert web_server._session_profile_filter("career-ops") is None

    def test_filters_by_requested_profile_under_multiplex(self, monkeypatch, tmp_path):
        from hermes_cli import web_server

        monkeypatch.setattr(web_server, "_multiplex_host_home", lambda: tmp_path)
        monkeypatch.setattr(web_server, "_cron_default_profile", lambda: "default")
        monkeypatch.setattr(
            web_server, "_cron_profile_home", lambda p: (p, tmp_path / p)
        )

        assert web_server._session_profile_filter(None) == "default"
        assert web_server._session_profile_filter("career-ops") == "career-ops"


class TestRowStamping:
    def test_rows_report_their_own_profile_not_the_requested_one(
        self, monkeypatch, tmp_path
    ):
        """The regression: a Career Ops row read from the shared store was
        relabelled 'default' and filed under the default profile."""
        from hermes_cli import web_routers, web_server
        from hermes_cli.web_routers import sessions as sessions_routes

        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)
        db.close()

        monkeypatch.setattr(web_server, "_multiplex_host_home", lambda: tmp_path)
        monkeypatch.setattr(web_server, "_cron_default_profile", lambda: "default")
        monkeypatch.setattr(
            web_server, "_cron_profile_home", lambda p: (p, tmp_path)
        )
        monkeypatch.setattr(
            web_server, "_maybe_auto_archive_for_profile", lambda _p: None
        )
        monkeypatch.setattr(
            web_server,
            "_open_session_db_for_profile",
            lambda _p, read_only: SessionDB(db_path=tmp_path / "state.db"),
        )

        result = sessions_routes.get_sessions(limit=50, offset=0, profile="career-ops")

        by_id = {s["id"]: s for s in result["sessions"]}
        assert set(by_id) == {"c1", "c2"}
        assert all(s["profile"] == "career-ops" for s in by_id.values())
        assert all(s["is_default_profile"] is False for s in by_id.values())

    def test_default_view_excludes_secondary_profile_rows(self, monkeypatch, tmp_path):
        from hermes_cli import web_server
        from hermes_cli.web_routers import sessions as sessions_routes

        db = SessionDB(db_path=tmp_path / "state.db")
        _seed(db)
        db.close()

        monkeypatch.setattr(web_server, "_multiplex_host_home", lambda: tmp_path)
        monkeypatch.setattr(web_server, "_cron_default_profile", lambda: "default")
        monkeypatch.setattr(web_server, "_cron_profile_home", lambda p: (p, tmp_path))
        monkeypatch.setattr(
            web_server, "_maybe_auto_archive_for_profile", lambda _p: None
        )
        monkeypatch.setattr(
            web_server,
            "_open_session_db_for_profile",
            lambda _p, read_only: SessionDB(db_path=tmp_path / "state.db"),
        )

        result = sessions_routes.get_sessions(limit=50, offset=0)

        by_id = {s["id"]: s for s in result["sessions"]}
        assert set(by_id) == {"d1", "d2"}
        assert all(s["is_default_profile"] is True for s in by_id.values())
        assert result["total"] == 2
