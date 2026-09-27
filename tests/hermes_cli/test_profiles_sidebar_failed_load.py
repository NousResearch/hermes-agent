"""A failed profile scan must not look like an empty session list.

A lagged store is healed once. When that read still fails, or the one-shot
heal is already exhausted and the profile contributed no rows, the sidebar
slice is a failed load with a retry signal. ``sessions: []`` is reserved for
a successful read of zero rows. A structurally corrupt store keeps its own
notice and is not offered Retry.
"""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _uncached_sidebar(monkeypatch):
    from hermes_cli.web_routers import profiles as profiles_routes

    monkeypatch.setattr(profiles_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    profiles_routes._sidebar_profile_cache_clear()
    profiles_routes._profile_read_warned.clear()


@pytest.fixture(autouse=True)
def _fresh_heal_latch(monkeypatch):
    from hermes_cli import web_server_sessions as sessions_mod

    monkeypatch.setattr(sessions_mod, "_session_db_heal_exhausted", set())
    monkeypatch.setattr(sessions_mod, "_session_db_heal_warned", set())


@pytest.fixture
def profiles_on_disk(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker": worker_home}


@pytest.fixture
def client(monkeypatch, profiles_on_disk):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    http = TestClient(app)
    http.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return http


def _seed_session(home, session_id, *, source="cli"):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, source=source)
        db.append_message(session_id=session_id, role="user", content="hi")
    finally:
        db.close()


def _sidebar(client, profile):
    return client.get(
        "/api/profiles/sessions/sidebar",
        params={"recents_profile": profile},
    ).json()


def _assert_failed_load(slice_, profile):
    """A failed load is not an empty session list, and it asks for a retry."""
    assert "sessions" not in slice_
    assert slice_["failed"] is True
    assert slice_["retry"] is True
    assert [row["profile"] for row in slice_["errors"]] == [profile]


class TestSidebarFailedLoad:
    def test_read_error_is_a_failed_load_not_an_empty_list(
        self, client, profiles_on_disk, monkeypatch
    ):
        _seed_session(profiles_on_disk["worker"], "worker-chat")
        import hermes_state

        def boom(self, *args, **kwargs):
            raise sqlite3.OperationalError(
                "no such column: s.compression_ineffective_count"
            )

        monkeypatch.setattr(hermes_state.SessionDB, "list_sessions_rich", boom)

        payload = _sidebar(client, "worker")

        assert payload["storage"] == {}
        for key in ("recents", "cron", "messaging"):
            _assert_failed_load(payload[key], "worker")
        assert payload["errors"][0]["profile"] == "worker"

    def test_heal_exhausted_empty_read_is_a_failed_load(
        self, client, profiles_on_disk, monkeypatch
    ):
        home = profiles_on_disk["worker"]
        _seed_session(home, "worker-chat")
        from hermes_cli import web_server_sessions as sessions_mod

        sessions_mod._session_db_heal_exhausted.add(str(home / "state.db"))
        import hermes_state

        monkeypatch.setattr(
            hermes_state.SessionDB,
            "list_sessions_rich",
            lambda self, *args, **kwargs: [],
        )

        payload = _sidebar(client, "worker")

        for key in ("recents", "cron", "messaging"):
            _assert_failed_load(payload[key], "worker")
        assert "heal exhausted" in payload["errors"][0]["error"]

    def test_unfixable_schema_is_healed_once_then_a_failed_load(
        self, client, profiles_on_disk, monkeypatch
    ):
        home = profiles_on_disk["worker"]
        _seed_session(home, "worker-chat")
        from hermes_cli import web_server_sessions as sessions_mod

        monkeypatch.setattr(
            sessions_mod,
            "_session_db_read_probe_statements",
            lambda: ('SELECT "sessions"."not_a_real_column" FROM "sessions" LIMIT 0',),
        )
        import hermes_state

        writable_opens = []
        real_init = hermes_state.SessionDB.__init__

        def counting_init(self, *args, **kwargs):
            if not kwargs.get("read_only", False):
                writable_opens.append(1)
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(hermes_state.SessionDB, "__init__", counting_init)
        real_list = hermes_state.SessionDB.list_sessions_rich

        def list_then_miss(self, *args, **kwargs):
            if str(getattr(self, "_db_path", home / "state.db")):
                raise sqlite3.OperationalError("no such column: s.not_a_real_column")
            return real_list(self, *args, **kwargs)

        monkeypatch.setattr(hermes_state.SessionDB, "list_sessions_rich", list_then_miss)

        first = _sidebar(client, "worker")
        second = _sidebar(client, "worker")

        assert len(writable_opens) == 1
        assert str(home / "state.db") in sessions_mod._session_db_heal_exhausted
        for payload in (first, second):
            _assert_failed_load(payload["recents"], "worker")

    def test_heal_exhausted_with_rows_still_returns_them(self, client, profiles_on_disk):
        home = profiles_on_disk["worker"]
        _seed_session(home, "worker-chat")
        from hermes_cli import web_server_sessions as sessions_mod

        sessions_mod._session_db_heal_exhausted.add(str(home / "state.db"))

        payload = _sidebar(client, "worker")

        assert payload["errors"] == []
        assert payload["recents"].get("retry") is not True
        assert [row["id"] for row in payload["recents"]["sessions"]] == ["worker-chat"]

    def test_successful_heal_still_returns_rows(self, client, profiles_on_disk):
        home = profiles_on_disk["worker"]
        _seed_session(home, "worker-chat")
        legacy = sqlite3.connect(str(home / "state.db"))
        try:
            legacy.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
            legacy.execute("ALTER TABLE sessions DROP COLUMN last_activity_at")
            legacy.commit()
        finally:
            legacy.close()

        payload = _sidebar(client, "worker")

        assert payload["errors"] == []
        assert payload["recents"].get("retry") is not True
        assert [row["id"] for row in payload["recents"]["sessions"]] == ["worker-chat"]

    def test_healthy_empty_store_stays_an_empty_list(self, client, profiles_on_disk):
        from hermes_state import SessionDB

        SessionDB(db_path=profiles_on_disk["worker"] / "state.db").close()

        payload = _sidebar(client, "worker")

        assert payload["errors"] == []
        assert payload["recents"].get("retry") is not True
        assert payload["recents"]["sessions"] == []

    def test_mixed_scan_marks_only_the_failed_profile(
        self, client, profiles_on_disk, monkeypatch
    ):
        _seed_session(profiles_on_disk["default"], "default-chat")
        _seed_session(profiles_on_disk["worker"], "worker-chat")
        import hermes_state

        real_list = hermes_state.SessionDB.list_sessions_rich
        worker_db = (profiles_on_disk["worker"] / "state.db").resolve()

        def explode_worker(self, *args, **kwargs):
            if Path(self.db_path).resolve() == worker_db:
                raise sqlite3.OperationalError("database is locked")
            return real_list(self, *args, **kwargs)

        monkeypatch.setattr(hermes_state.SessionDB, "list_sessions_rich", explode_worker)

        payload = _sidebar(client, "all")

        assert [row["id"] for row in payload["recents"]["sessions"]] == ["default-chat"]
        failed = payload["recents"]["profiles_failed"]["worker"]
        assert failed["failed"] is True
        assert failed["retry"] is True
        assert "worker" not in {
            row["profile"] for row in payload["recents"]["sessions"]
        }
