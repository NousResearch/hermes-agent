"""A failed profile scan must not look like an empty session list.

Browsing never repairs the owner's store: a read that fails (lagged schema, busy
store) makes that profile's sidebar slice a failed load with a retry signal. ``sessions: []`` is reserved for
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

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    http = TestClient(app)
    http.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return http


def _seed_session(home, session_id, *, source="cli"):
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    with open_persistence(profile_scope(home=home)) as store:
        seed_session(store, session_id, source)
        seed_messages(store, session_id, {"role": "user", "content": "hi"})


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
        from persistence.sqlite.conversation import sessions as sqlite_sessions

        def boom(holder, *args, **kwargs):
            raise sqlite3.OperationalError(
                "no such column: s.compression_ineffective_count"
            )

        monkeypatch.setattr(sqlite_sessions, "list_sessions_rich", boom)

        payload = _sidebar(client, "worker")

        assert payload["storage"] == {}
        for key in ("recents", "cron", "messaging"):
            _assert_failed_load(payload[key], "worker")
        assert payload["errors"][0]["profile"] == "worker"

    def test_stale_schema_is_a_failed_load_and_never_opens_a_writer(
        self, client, profiles_on_disk, monkeypatch
    ):
        home = profiles_on_disk["worker"]
        _seed_session(home, "worker-chat")
        from persistence.sqlite.admin._maintenance import MaintenanceAdmin
        from persistence.sqlite.conversation import sessions as sqlite_sessions

        monkeypatch.setattr(MaintenanceAdmin, "schema_drift_reason", lambda _self: "no such column: not_a_real_column")
        import persistence_assembly
        from persistence.sqlite._holder import StoreHolder

        writable_opens = []
        real_init = StoreHolder.__init__

        def counting_init(self, *args, **kwargs):
            if not kwargs.get("read_only", False):
                writable_opens.append(1)
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(StoreHolder, "__init__", counting_init)
        real_writer = persistence_assembly.open_persistence
        monkeypatch.setattr(persistence_assembly, "open_persistence",
                            lambda *a, **kw: writable_opens.append(1) or real_writer(*a, **kw))

        def list_then_miss(holder, *args, **kwargs):
            raise sqlite3.OperationalError("no such column: s.not_a_real_column")

        monkeypatch.setattr(sqlite_sessions, "list_sessions_rich", list_then_miss)

        first = _sidebar(client, "worker")
        second = _sidebar(client, "worker")

        assert writable_opens == []
        for payload in (first, second):
            _assert_failed_load(payload["recents"], "worker")

    def test_healthy_empty_store_stays_an_empty_list(self, client, profiles_on_disk):
        from persistence_assembly import open_persistence, profile_scope

        with open_persistence(profile_scope(home=profiles_on_disk["worker"])) as store:
            store.ready()

        payload = _sidebar(client, "worker")

        assert payload["errors"] == []
        assert payload["recents"].get("retry") is not True
        assert payload["recents"]["sessions"] == []

    def test_mixed_scan_marks_only_the_failed_profile(
        self, client, profiles_on_disk, monkeypatch
    ):
        _seed_session(profiles_on_disk["default"], "default-chat")
        _seed_session(profiles_on_disk["worker"], "worker-chat")
        from persistence.sqlite.conversation import sessions as sqlite_sessions

        real_list = sqlite_sessions.list_sessions_rich
        worker_db = (profiles_on_disk["worker"] / "state.db").resolve()

        def explode_worker(holder, *args, **kwargs):
            if Path(holder.db_path).resolve() == worker_db:
                raise sqlite3.OperationalError("database is locked")
            return real_list(holder, *args, **kwargs)

        monkeypatch.setattr(sqlite_sessions, "list_sessions_rich", explode_worker)

        payload = _sidebar(client, "all")

        assert [row["id"] for row in payload["recents"]["sessions"]] == ["default-chat"]
        failed = payload["recents"]["profiles_failed"]["worker"]
        assert failed["failed"] is True
        assert failed["retry"] is True
        assert "worker" not in {
            row["profile"] for row in payload["recents"]["sessions"]
        }
