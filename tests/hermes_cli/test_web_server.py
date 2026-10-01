"""Tests for hermes_cli.web_server and related config utilities."""

import asyncio
import os
import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import hermes_yaml as yaml

from hermes_cli.config import (
    reload_env,
    redact_key,
    OPTIONAL_ENV_VARS,
)
import gateway.status as _gw_status
import hermes_cli.config as _cfg_mod
import hermes_cli.web_routers.chat_ws as _rt_chat_ws
import hermes_cli.web_server_chat as _web_server_chat
import hermes_cli.web_server_config as _web_server_config
import hermes_cli.web_server_dashboard as _web_server_dashboard
import hermes_cli.web_server_files as _web_server_files
import hermes_cli.web_server_gateway as _web_server_gateway
import hermes_cli.web_server_lifecycle as _web_server_lifecycle
import hermes_cli.web_server_memory as _web_server_memory
import hermes_cli.web_server_messaging as _web_server_messaging
import hermes_cli.web_server_sessions as _web_server_sessions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


# Path to the test-only example-dashboard plugin. Lives under
# tests/fixtures/ so the bundled-plugins directory stays clean — stock
# installs no longer ship a dummy "Example" sidebar tab. Tests that
# depend on its routes opt in via the `_install_example_plugin` fixture
# below.
_EXAMPLE_PLUGIN_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "plugins" / "example-dashboard"
)


@pytest.fixture
def _install_example_plugin(_isolate_hermes_home):
    """Drop the example-dashboard fixture into the per-test HERMES_HOME
    user-plugins directory and force the web_server's dashboard plugin
    cache + API mount to rediscover it.

    The plugin used to live under ``<repo>/plugins/example-dashboard/``
    and was loaded for every install, putting an "Example" tab in every
    user's sidebar. It is now a tests-only fixture: any test that needs
    ``/api/plugins/example/hello`` or ``/dashboard-plugins/example/...``
    requests this fixture so the plugin appears only for that test's
    isolated ``HERMES_HOME``.

    The user-plugin source is preferred over a transient
    ``HERMES_BUNDLED_PLUGINS`` override because the bundled dir is
    resolved per-call (other tests in the suite implicitly rely on the
    real bundled plugins — kanban, hermes-achievements, model providers
    — being available, and globally swapping that root would yank them
    all). User plugins are first in the discovery search order, so
    laying down the fixture here is enough.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    user_plugins_dir = get_hermes_home() / "plugins"
    user_plugins_dir.mkdir(parents=True, exist_ok=True)
    dst = user_plugins_dir / "example-dashboard"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(_EXAMPLE_PLUGIN_FIXTURE, dst)

    # The dashboard now gates user-plugin asset serving + backend import
    # behind the ``plugins.enabled`` allow-list (GHSA-mcfc-hp25-cjv7).
    # An installed-but-not-enabled user plugin has its API mount skipped
    # and its assets 404'd — which is the whole point of the gate. These
    # fixtures exist to exercise the *serving* paths, so opt the example
    # plugin in exactly as a real operator would with `hermes plugins
    # enable example`.
    from hermes_cli.config import load_config, save_config
    _cfg = load_config()
    _plugins_cfg = _cfg.setdefault("plugins", {})
    _enabled = _plugins_cfg.get("enabled")
    if not isinstance(_enabled, list):
        _enabled = []
    if "example" not in _enabled:
        _enabled.append("example")
    _plugins_cfg["enabled"] = _enabled
    save_config(_cfg)

    # Snapshot the existing routes BEFORE mounting so we can:
    #   1. Identify the routes the mount call appends.
    #   2. Restore the original list on teardown — otherwise leftover
    #      ``/api/plugins/example/*`` routes leak into subsequent tests
    #      and start serving requests against a torn-down HERMES_HOME.
    app = web_server.app
    original_routes = list(app.router.routes)

    # Bust the module-level cache and re-discover so the example plugin
    # shows up in `_get_dashboard_plugins()`. `_mount_plugin_api_routes`
    # imports the plugin's `plugin_api.py` and ``include_router``s its
    # FastAPI router under ``/api/plugins/example/*``. The static-asset
    # route at ``/dashboard-plugins/<name>/<path>`` reads the plugins
    # list dynamically per request, so the rescan alone is enough for
    # the static-asset tests; the API auth tests additionally need the
    # route reorder below.
    web_server._dashboard_plugins_cache = None
    web_server._get_dashboard_plugins(force_rescan=True)
    _web_server_dashboard._mount_plugin_api_routes()

    # ``include_router`` appends the new routes to the END of
    # ``app.router.routes``. That works fine at import time — the SPA
    # catch-all ``mount_spa(app)`` registers AFTER the initial mount
    # call — but when we mount mid-flight the catch-all is already in
    # place, so the new ``/api/plugins/example/*`` route loses the
    # match-order race and we get a 404. Move the newly-appended routes
    # to the front of the list so FastAPI matches them first. They're
    # path-prefixed to ``/api/plugins/example/`` and can't shadow
    # anything else.
    new_routes = [r for r in app.router.routes if r not in original_routes]
    for route in new_routes:
        app.router.routes.remove(route)
    for offset, route in enumerate(new_routes):
        app.router.routes.insert(offset, route)

    try:
        yield
    finally:
        # Restore the original route list — drops the example plugin's
        # routes so the next test sees a clean app — and clear the
        # cache for the same reason.
        app.router.routes[:] = original_routes
        web_server._dashboard_plugins_cache = None


# ---------------------------------------------------------------------------
# reload_env tests
# ---------------------------------------------------------------------------


class TestReloadEnv:
    """Tests for reload_env() — re-reads .env into os.environ."""

    def test_adds_new_vars(self, tmp_path):
        """reload_env() adds vars from .env that are not in os.environ."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_RELOAD_VAR=hello123\n", encoding="utf-8")
        with patch.dict(reload_env.__globals__, {"get_env_path": lambda: env_file}):
            os.environ.pop("TEST_RELOAD_VAR", None)
            count = reload_env()
            assert count >= 1
            assert os.environ.get("TEST_RELOAD_VAR") == "hello123"
        os.environ.pop("TEST_RELOAD_VAR", None)


    def test_removes_deleted_known_vars(self, tmp_path):
        """reload_env() removes known Hermes vars not present in .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("")  # empty .env
        # Pick a known key from OPTIONAL_ENV_VARS
        known_key = next(iter(OPTIONAL_ENV_VARS.keys()))
        with patch.dict(reload_env.__globals__, {"get_env_path": lambda: env_file}):
            os.environ[known_key] = "stale_value"
            count = reload_env()
            assert known_key not in os.environ
            assert count >= 1


# ---------------------------------------------------------------------------
# redact_key tests
# ---------------------------------------------------------------------------


class TestRedactKey:
    def test_long_key_shows_prefix_suffix(self):
        result = redact_key("sk-1234567890abcdef")
        assert result.startswith("sk-1")
        assert result.endswith("cdef")
        assert "..." in result

    def test_short_key_fully_masked(self):
        assert redact_key("short") == "***"


class TestSessionTokenInjection:
    """The desktop shell mints HERMES_DASHBOARD_SESSION_TOKEN and signs its
    /api + /api/ws calls with it. The backend must adopt that token, else every
    desktop request 401s ("gateway is offline"). A main-merge once silently
    dropped this read — this guards the contract, not a literal value.
    """

    def test_honors_injected_token(self, monkeypatch):
        import hermes_cli.web_server as ws

        original_app = ws.app
        original_token = ws._SESSION_TOKEN
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-seeded-token")
        assert ws._resolve_session_token() == "desktop-seeded-token"
        # No module reload: the loaded app and its adopted token are untouched.
        assert ws.app is original_app
        assert ws._SESSION_TOKEN == original_token


    def test_session_token_resolution_preserves_loaded_app_auth(self, monkeypatch):
        import hermes_cli.web_server as ws
        from starlette.testclient import TestClient

        original_app = ws.app
        original_header_name = ws._SESSION_HEADER_NAME
        original_token = ws._SESSION_TOKEN
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-seeded-token")
        assert ws._resolve_session_token() == "desktop-seeded-token"
        monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
        with patch.object(ws.secrets, "token_urlsafe", return_value="generated-token"):
            assert ws._resolve_session_token() == "generated-token"

        client = TestClient(original_app)
        client.headers[original_header_name] = original_token
        assert client.get("/api/__session_token_probe").status_code == 404
        assert ws.app is original_app
        assert ws._SESSION_HEADER_NAME == original_header_name
        assert ws._SESSION_TOKEN == original_token


# ---------------------------------------------------------------------------
# web_server tests (FastAPI endpoints)
# ---------------------------------------------------------------------------


class TestWebServerEndpoints:
    """Test the FastAPI REST endpoints using Starlette TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        """Create a TestClient and isolate the state DB under the test HERMES_HOME."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    @pytest.mark.requires_wal
    def test_get_sessions_poll_preserves_pending_wal(self):
        """Repeated GET-only polls must not checkpoint another writer's WAL."""
        import sqlite3

        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        _web_server_sessions._last_auto_archive_check.clear()
        db_path = get_hermes_home() / "state.db"
        wal_path = Path(f"{db_path}-wal")
        writer = SessionDB(db_path=db_path)
        monitor = None
        try:
            writer._conn.execute("PRAGMA wal_autocheckpoint=0")
            writer.create_session("poll-wal", source="cli")
            writer.append_message(
                "poll-wal",
                role="user",
                content="pending writer frame " + ("x" * 65_536),
            )

            monitor = sqlite3.connect(str(db_path), isolation_level=None)
            wal_bytes_before = wal_path.stat().st_size
            data_version_before = monitor.execute(
                "PRAGMA data_version"
            ).fetchone()[0]
            counts_before = monitor.execute(
                "SELECT (SELECT COUNT(*) FROM sessions), "
                "(SELECT COUNT(*) FROM messages)"
            ).fetchone()

            responses = [
                self.client.get(
                    "/api/sessions?limit=50&offset=0&order=created"
                )
                for _ in range(3)
            ]

            wal_bytes_after = wal_path.stat().st_size
            data_version_after = monitor.execute(
                "PRAGMA data_version"
            ).fetchone()[0]
            counts_after = monitor.execute(
                "SELECT (SELECT COUNT(*) FROM sessions), "
                "(SELECT COUNT(*) FROM messages)"
            ).fetchone()

            assert all(response.status_code == 200 for response in responses)
            assert all(response.json()["total"] == 1 for response in responses)
            assert wal_bytes_before > 0
            assert wal_bytes_after == wal_bytes_before
            assert data_version_after == data_version_before
            assert counts_after == counts_before == (1, 1)
        finally:
            if monitor is not None:
                monitor.close()
            writer.close()

    def test_get_sessions_transient_ioerr_is_503(self, monkeypatch):
        """Busy store, not a gone store: the desktop keeps the list it has."""
        import sqlite3


        def boom(*_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", boom)
        assert self.client.get("/api/sessions?limit=1&offset=0").status_code == 503

    def test_get_sessions_non_transient_operational_error_is_500(self, monkeypatch):
        import sqlite3


        def boom(*_args, **_kwargs):
            raise sqlite3.OperationalError("no such table: sessions")

        monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", boom)
        assert self.client.get("/api/sessions?limit=1&offset=0").status_code == 500


    def test_get_sessions_auto_archive_uses_maintenance_writer(self):
        from hermes_cli.config import load_config, save_config
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        db_path = get_hermes_home() / "state.db"
        seed = SessionDB(db_path=db_path)
        try:
            seed.create_session("stale", source="cli")
            seed.create_session("fresh", source="cli")
            seed._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 30 * 86400, "stale"),
            )
        finally:
            seed.close()

        config = load_config()
        config.setdefault("sessions", {}).update(
            {
                "auto_archive": True,
                "auto_archive_days": 3,
                "min_interval_hours": 0,
            }
        )
        save_config(config)
        _web_server_sessions._last_auto_archive_check.clear()

        response = self.client.get("/api/sessions?limit=50&offset=0")

        assert response.status_code == 200
        assert [row["id"] for row in response.json()["sessions"]] == ["fresh"]
        verify = SessionDB(db_path=db_path, read_only=True)
        try:
            assert verify.get_session("stale")["archived"] == 1
            assert verify.get_meta("last_auto_archive")
        finally:
            verify.close()

    def test_get_sessions_fresh_store_returns_empty_list(self):
        response = self.client.get("/api/sessions?limit=50&offset=0")

        assert response.status_code == 200
        assert response.json()["sessions"] == []
        assert response.json()["total"] == 0

    @pytest.mark.parametrize(
        "missing_column", ["archived", "pinned", "last_activity_at"]
    )
    def test_get_sessions_heals_stale_schema_store(self, missing_column):
        import sqlite3

        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        db_path = get_hermes_home() / "state.db"
        seed = SessionDB(db_path=db_path)
        try:
            seed.create_session("stale-schema", source="cli")
        finally:
            seed.close()

        legacy = sqlite3.connect(str(db_path))
        try:
            # SQLite refuses DROP COLUMN while an index references the
            # column; a pre-column legacy store has neither.
            legacy.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
            legacy.execute(f"ALTER TABLE sessions DROP COLUMN {missing_column}")
            legacy.commit()
        finally:
            legacy.close()

        response = self.client.get("/api/sessions?limit=50&offset=0")

        assert response.status_code == 200
        assert [row["id"] for row in response.json()["sessions"]] == [
            "stale-schema"
        ]
        healed = sqlite3.connect(str(db_path))
        try:
            columns = {
                row[1] for row in healed.execute("PRAGMA table_info(sessions)")
            }
        finally:
            healed.close()
        assert missing_column in columns

    def test_profiles_sidebar_heals_stale_schema_store(self):
        """The desktop's batched sidebar route must heal a stale store too.

        The shipped regression (#72424 aftermath): a store predating
        ``sessions.last_activity_at`` made every per-profile read raise
        "no such column", which this endpoint swallowed into its ``errors``
        array — the desktop rendered "No sessions yet" after `hermes update`
        until the user's first message forced a writable open elsewhere.
        """
        import sqlite3

        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        db_path = get_hermes_home() / "state.db"
        seed = SessionDB(db_path=db_path)
        try:
            seed.create_session("sidebar-stale", source="cli")
            seed.append_message(
                session_id="sidebar-stale", role="user", content="hi"
            )
        finally:
            seed.close()

        legacy = sqlite3.connect(str(db_path))
        try:
            legacy.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
            legacy.execute("ALTER TABLE sessions DROP COLUMN last_activity_at")
            legacy.commit()
        finally:
            legacy.close()

        response = self.client.get("/api/profiles/sessions/sidebar")

        assert response.status_code == 200
        payload = response.json()
        assert payload["errors"] == []
        assert [row["id"] for row in payload["recents"]["sessions"]] == [
            "sidebar-stale"
        ]

    def test_startup_eager_reconcile_heals_stale_store(self):
        """The lifespan's eager reconcile brings a stale store current.

        #79531/#80037: after `hermes update` an old-schema state.db used to
        stay stale until the first NEW session forced a writable open —
        every /api/sessions poll 500ed with "no such column" in between.
        The lifespan now schedules one writable open at startup; this
        exercises that worker directly against a store missing
        sessions.last_read_at and asserts the schema is brought current.
        """
        import sqlite3

        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        db_path = get_hermes_home() / "state.db"
        seed = SessionDB(db_path=db_path)
        try:
            seed.create_session("eager-stale", source="cli")
        finally:
            seed.close()

        legacy = sqlite3.connect(str(db_path))
        try:
            legacy.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
            legacy.commit()
        finally:
            legacy.close()

        _web_server_lifecycle._eager_reconcile_own_session_db()

        healed = sqlite3.connect(str(db_path))
        try:
            columns = {
                row[1] for row in healed.execute("PRAGMA table_info(sessions)")
            }
        finally:
            healed.close()
        assert "last_read_at" in columns

        # The healed store serves the full rich listing.
        db = SessionDB(db_path=db_path, read_only=True)
        try:
            rows = db.list_sessions_rich(limit=10, compact_rows=True)
        finally:
            db.close()
        assert [r["id"] for r in rows] == ["eager-stale"]

    def test_startup_eager_reconcile_is_read_only_on_a_healthy_store(self, monkeypatch):
        """A current-schema store gets NO writable open from the dashboard (#107688).

        The gateway owns the writer; a second writable SessionDB from the
        dashboard (close-time checkpoint, possible FTS rebuild) is the
        two-writer corruption vector. Only the stale-schema heal may write.
        """
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        SessionDB(db_path=get_hermes_home() / "state.db").close()

        writable_opens = []
        real_init = SessionDB.__init__

        def spy(self, *args, **kwargs):
            if not kwargs.get("read_only"):
                writable_opens.append(kwargs)
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(hermes_state.SessionDB, "__init__", spy)
        _web_server_lifecycle._eager_reconcile_own_session_db()

        assert writable_opens == []

    def test_startup_eager_reconcile_never_raises(self, monkeypatch):
        """A store the eager reconcile cannot open must not break startup."""
        import sqlite3 as sqlite3_module

        import hermes_state


        def boom(*args, **kwargs):
            raise sqlite3_module.OperationalError("database is locked")

        monkeypatch.setattr(hermes_state, "SessionDB", boom)
        # Must swallow — reads fall back to the per-poll probe heal.
        _web_server_lifecycle._eager_reconcile_own_session_db()

    def test_heal_gives_up_when_reconcile_cannot_fix_the_store(self, monkeypatch):
        """A probe failure reconciliation can't cure must not retry forever.

        The writable heal is a full SessionDB init against a possibly-live
        DB. If the store is STILL behind the probe afterwards (schema problem
        ADD COLUMN can't express), retrying that init on every sidebar poll
        would hammer the DB for nothing: serve reads probe-less instead, warn
        once, and never pay the writable open for that store again.
        """
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        db_path = get_hermes_home() / "state.db"
        seed = SessionDB(db_path=db_path)
        try:
            seed.create_session("unfixable", source="cli")
        finally:
            seed.close()

        # A column no SCHEMA_SQL declares: the heal's writable reconcile
        # cannot add it, so the re-probe keeps failing.
        monkeypatch.setattr(
            _web_server_sessions,
            "_session_db_read_probe_statements",
            lambda: ('SELECT "sessions"."not_a_real_column" FROM "sessions" LIMIT 0',),
        )
        monkeypatch.setattr(_web_server_sessions, "_session_db_heal_exhausted", set())
        monkeypatch.setattr(_web_server_sessions, "_session_db_heal_warned", set())

        writable_opens = []

        import hermes_state

        original_init = hermes_state.SessionDB.__init__

        def counting_init(self, *args, **kwargs):
            if not kwargs.get("read_only", False):
                writable_opens.append(1)
            return original_init(self, *args, **kwargs)

        # web_server imports SessionDB inside the function body, so patching
        # the class on hermes_state covers every open the helper makes.
        monkeypatch.setattr(hermes_state.SessionDB, "__init__", counting_init)

        # First open: probe fails -> one writable heal -> re-probe fails ->
        # exhausted. Still returns a usable read-only handle.
        db = _web_server_sessions._open_session_db_for_profile(None, read_only=True)
        try:
            assert db.list_sessions_rich(limit=10, compact_rows=True)
        finally:
            db.close()
        assert len(writable_opens) == 1
        assert str(db_path) in _web_server_sessions._session_db_heal_exhausted

        # Second open: probe skipped, NO further writable opens.
        db = _web_server_sessions._open_session_db_for_profile(None, read_only=True)
        try:
            assert db.list_sessions_rich(limit=10, compact_rows=True)
        finally:
            db.close()
        assert len(writable_opens) == 1

    def test_generic_corruption_does_not_trigger_writable_heal(
        self, tmp_path, monkeypatch
    ):
        """Unscoped SQLITE_CORRUPT must not escalate a dashboard read to writes."""
        import sqlite3

        import hermes_state

        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"not-empty")
        opens = []

        def corrupt_open(*_args, **kwargs):
            opens.append(kwargs.get("read_only", False))
            raise sqlite3.DatabaseError("database disk image is malformed")

        monkeypatch.setattr(hermes_state, "SessionDB", corrupt_open)

        with pytest.raises(sqlite3.DatabaseError, match="disk image is malformed"):
            _web_server_sessions._open_session_db_at_path(db_path, read_only=True)

        assert opens == [True]

    def test_decode_error_triggers_writable_heal(self, tmp_path, monkeypatch):
        """UnicodeDecodeError — pysqlite failing to decode SQLite's own error
        message over corrupt file bytes (#98924) — must route through the
        same one-writable-open heal as malformed schema."""
        import hermes_state

        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"not-empty")
        opens = []

        class _OkDB:
            _conn = None

            def close(self):
                pass

        def scripted_open(*_args, **kwargs):
            opens.append(kwargs.get("read_only", False))
            if opens == [True]:
                raise UnicodeDecodeError("utf-8", b"\x81", 0, 1, "invalid start byte")
            return _OkDB()

        monkeypatch.setattr(hermes_state, "SessionDB", scripted_open)

        db = _web_server_sessions._open_session_db_at_path(db_path, read_only=True)

        assert isinstance(db, _OkDB)
        assert opens == [True, False, True]

    def test_get_sessions_zero_byte_store_returns_empty_list(self):
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "state.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()

        response = self.client.get("/api/sessions?limit=50&offset=0")

        assert response.status_code == 200
        assert response.json()["sessions"] == []
        assert response.json()["total"] == 0

    def test_concurrent_first_load_reads_all_succeed_on_fresh_store(self):
        from concurrent.futures import ThreadPoolExecutor

        paths = [
            "/api/sessions?limit=50&offset=0",
            "/api/sessions/stats",
            "/api/sessions/empty/count",
            "/api/sessions?limit=10&offset=0&order=recent",
        ] * 2
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(self.client.get, paths))

        assert [response.status_code for response in responses] == [
            200
        ] * len(paths)


    def test_messaging_platforms_profile_scopes_gateway_reads(self, monkeypatch):
        """?profile=<name> must resolve liveness from the profile's own home.

        The gateway status readers resolve process-level paths and ignore the
        HERMES_HOME contextvar override (#56986), so /api/messaging/platforms
        has to pass the profile directory explicitly — otherwise it reports a
        DIFFERENT profile's gateway as this profile's, which hides a real
        outage behind a false "connected" (issue #71211).
        """
        import hermes_cli.web_server as web_server
        from hermes_cli import profiles as profiles_mod

        worker_home = profiles_mod.get_profile_dir("worker")
        worker_home.mkdir(parents=True)
        (worker_home / "config.yaml").touch()  # identity marker: bare dirs are not profiles

        seen = {}

        def _pid(pid_path=None, **kw):
            # The served-profile probe also verifies the DEFAULT home's gateway identity; the
            # contract here is that the worker's OWN pid file is what the scoped rung reads.
            seen.setdefault("pid_paths", []).append(pid_path)
            return None

        def _runtime(path=None):
            seen.setdefault("status_paths", []).append(path)
            return None

        def _runtime_pid(runtime=None, *, expected_home=None):
            seen.setdefault("expected_homes", []).append(expected_home)
            return None

        monkeypatch.setattr(_gw_status, "get_running_pid_cached", _pid)
        monkeypatch.setattr(_gw_status, "get_running_pid", _pid)
        monkeypatch.setattr(_gw_status, "read_runtime_status", _runtime)
        monkeypatch.setattr(_gw_status, "get_runtime_status_running_pid", _runtime_pid)
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)

        resp = self.client.get("/api/messaging/platforms?profile=worker")

        assert resp.status_code == 200
        assert worker_home / "gateway.pid" in seen["pid_paths"]
        assert worker_home / "gateway_state.json" in seen["status_paths"]
        assert worker_home in seen["expected_homes"]


    def test_gateway_drain_bad_action_400(self):
        resp = self.client.post("/api/gateway/drain", json={"action": "explode"})
        assert resp.status_code == 400


    @staticmethod
    def _provider_field_map(payload):
        return {field["key"]: field for field in payload["fields"]}


    def test_openviking_dashboard_persists_typed_recall_values(self):
        from hermes_cli.config import load_config

        resp = self.client.put(
            "/api/memory/providers/openviking/config",
            json={
                "values": {
                    "endpoint": "http://127.0.0.1:1933",
                    "recall_limit": "12",
                    "recall_score_threshold": "0.42",
                    "recall_max_injected_chars": "8000",
                    "profile_token_budget": "7000",
                    "recall_timeout_seconds": "2.5",
                    "recall_request_timeout_seconds": "1.5",
                    "recall_full_read_limit": "5",
                    "recall_prefer_abstract": True,
                    "recall_resources": False,
                }
            },
        )

        assert resp.status_code == 200
        config = load_config()["memory"]["openviking"]
        assert config["recall_limit"] == 12
        assert config["recall_score_threshold"] == 0.42
        assert config["profile_token_budget"] == 7000
        assert config["recall_prefer_abstract"] is True
        assert config["recall_resources"] is False

    def test_openviking_dashboard_rejects_out_of_range_recall_value(self):
        resp = self.client.put(
            "/api/memory/providers/openviking/config",
            json={
                "values": {
                    "endpoint": "http://127.0.0.1:1933",
                    "recall_limit": 101,
                }
            },
        )

        assert resp.status_code == 400

    def test_openviking_dashboard_rejects_blocked_endpoint_before_saving(self):
        from hermes_cli.config import load_config

        resp = self.client.put(
            "/api/memory/providers/openviking/config",
            json={
                "values": {
                    "endpoint": "http://169.254.169.254/latest/meta-data/credential",
                }
            },
        )

        assert resp.status_code == 400
        assert "credential" not in resp.json()["detail"]
        memory_config = load_config().get("memory", {})
        assert "openviking" not in memory_config


    # A user-installed memory provider with a DECLARED config surface (``config_schema.py``, flat
    # ``<home>/<name>/config.json`` storage) and a live ``get_config_schema``/``save_config`` pair.
    # Bundled providers no longer ship a flat-storage declared schema (hindsight moved to the
    # plugin catalog), so the generic router paths are exercised against this synthetic one.
    _FLATPROV_INIT = """
import json
from pathlib import Path
from agent.memory_provider import MemoryProvider


class FlatProvMemoryProvider(MemoryProvider):
    @property
    def name(self):
        return "flatprov"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def get_config_schema(self):
        return [
            {"key": "mode", "label": "Mode", "choices": ["cloud", "local_external"], "default": "cloud"},
            {"key": "api_url", "label": "API URL", "default": ""},
            {"key": "api_key", "label": "API key", "secret": True, "env_var": "FLATPROV_API_KEY"},
            {"key": "bank_id", "label": "Bank", "default": "hermes"},
            {"key": "recall_budget", "label": "Budget", "choices": ["low", "mid", "high"], "default": "mid"},
        ]

    def save_config(self, values, hermes_home):
        path = Path(hermes_home) / "flatprov" / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update(values)
        path.write_text(json.dumps(existing))
"""
    _FLATPROV_SCHEMA = """
from plugins.memory.config_schema import (
    KIND_SECRET, KIND_SELECT, KIND_TEXT, ProviderConfigSchema, ProviderField, ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="flatprov",
    label="Flat Provider",
    fields=(
        ProviderField(key="mode", label="Mode", kind=KIND_SELECT, description="", default="cloud",
                      options=(ProviderFieldOption("cloud", "Cloud"), ProviderFieldOption("local_external", "Local"))),
        ProviderField(key="api_url", label="API URL", kind=KIND_TEXT, description=""),
        ProviderField(key="api_key", label="API key", kind=KIND_SECRET, description="", env_key="FLATPROV_API_KEY"),
    ),
)
"""

    def _install_flatprov(self):
        from hermes_constants import get_hermes_home

        plugin_dir = get_hermes_home() / "plugins" / "flatprov"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "__init__.py").write_text(self._FLATPROV_INIT, encoding="utf-8")
        (plugin_dir / "config_schema.py").write_text(self._FLATPROV_SCHEMA, encoding="utf-8")
        return plugin_dir

    def test_declared_surface_put_writes_config_and_secret(self):
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_env

        self._install_flatprov()
        resp = self.client.put(
            "/api/memory/providers/flatprov/config?surface=declared",
            json={
                "values": {
                    "mode": "local_external",
                    "api_url": "http://localhost:8888",
                    "api_key": "fp-declared-key",
                }
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert load_env()["FLATPROV_API_KEY"] == "fp-declared-key"

        config_path = get_hermes_home() / "flatprov" / "config.json"
        provider_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert provider_config["mode"] == "local_external"
        assert provider_config["api_url"] == "http://localhost:8888"
        assert "api_key" not in provider_config


    def test_post_memory_provider_setup_routes_python_deps_through_pm(self, monkeypatch):
        """Dashboard dependency setup publishes through PM, never direct pip."""
        import subprocess as _subprocess

        import hermes_cli.web_server as web_server
        from hermes_cli import memory_setup

        prepared = []
        monkeypatch.setattr(
            memory_setup,
            "prepare_memory_provider_dependencies",
            lambda name: (prepared.append(name) or ({}, "installed")),
        )

        # Any direct pip/uv subprocess from the memory-provider pip path is
        # a regression; external-dep checks may still run subprocess, so only
        # trip on pip-flavored commands.
        real_run = _subprocess.run

        def guarded_run(command, **kwargs):
            flat = command if isinstance(command, str) else " ".join(map(str, command))
            assert "pip install" not in flat, f"direct pip call leaked: {flat}"
            return real_run(command, **kwargs)

        monkeypatch.setattr(web_server.subprocess, "run", guarded_run)

        resp = self.client.post("/api/memory/providers/honcho/setup", json={"values": {}})

        assert resp.status_code == 200
        data = resp.json()
        pip_rows = [row for row in data["results"] if row["kind"] == "pip"]
        assert pip_rows and pip_rows[0]["status"] == "installed"
        assert pip_rows[0]["command"] == "hermes pm install"
        assert prepared == ["honcho"]


    def test_put_memory_provider_config_writes_config_and_secret(self):
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_config, load_env

        self._install_flatprov()
        resp = self.client.put(
            "/api/memory/providers/flatprov/config",
            json={
                "values": {
                    "mode": "local_external",
                    "api_url": "http://localhost:8888",
                    "api_key": "fp-test-key",
                    "bank_id": "ben-bank",
                    "recall_budget": "high",
                }
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "active": "flatprov"}
        assert load_config()["memory"]["provider"] == "flatprov"
        assert load_env()["FLATPROV_API_KEY"] == "fp-test-key"

        config_path = get_hermes_home() / "flatprov" / "config.json"
        provider_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert provider_config["mode"] == "local_external"
        assert provider_config["api_url"] == "http://localhost:8888"
        assert provider_config["bank_id"] == "ben-bank"
        assert provider_config["recall_budget"] == "high"
        assert "api_key" not in provider_config


    def test_get_memory_provider_config_does_not_return_secret(self):
        self._install_flatprov()
        self.client.put(
            "/api/memory/providers/flatprov/config",
            json={
                "values": {
                    "mode": "cloud",
                    "api_url": "https://api.example.invalid",
                    "api_key": "secret-value",
                    "bank_id": "hermes",
                    "recall_budget": "mid",
                }
            },
        )

        resp = self.client.get("/api/memory/providers/flatprov/config")

        assert resp.status_code == 200
        data = resp.json()
        fields = self._provider_field_map(data)
        assert fields["api_key"]["is_set"] is True
        assert fields["api_key"]["value"] == ""
        assert "secret-value" not in json.dumps(data)


    # ── Memory provider config (Honcho host-block backend) ──────────────

    @pytest.fixture(autouse=True)
    def _isolate_honcho_config(self):
        # Honcho tests write the suite-wide HERMES_HOME honcho.json; snapshot and
        # restore it so provider status/config state never leaks across tests.
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "honcho.json"
        before = path.read_bytes() if path.exists() else None
        yield
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(before)

    @staticmethod
    def _seed_local_honcho(cfg=None):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "honcho.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg if cfg is not None else {}), encoding="utf-8")
        return path


    def test_put_honcho_writes_host_block_root_and_secret(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HONCHO_API_KEY", "guard")
        monkeypatch.delenv("HONCHO_API_KEY")
        self._seed_local_honcho()
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_config, load_env

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={
                "values": {
                    "apiKey": "hch-test-key",
                    "baseUrl": "https://honcho.example.dev",
                    "environment": "local",
                    "workspace": "myws",
                    "peerName": "eri",
                    "aiPeer": "hermes",
                    "sessionStrategy": "per-repo",
                }
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert load_config()["memory"]["provider"] == "honcho"
        assert load_env()["HONCHO_API_KEY"] == "hch-test-key"

        cfg = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))
        # baseUrl is root-scoped; the rest live in the active host block.
        assert cfg["baseUrl"] == "https://honcho.example.dev"
        assert cfg["hosts"]["hermes"]["workspace"] == "myws"
        assert cfg["hosts"]["hermes"]["peerName"] == "eri"
        assert cfg["hosts"]["hermes"]["environment"] == "local"
        assert cfg["hosts"]["hermes"]["sessionStrategy"] == "per-repo"
        # The key lands where the client reads first; GET keeps it write-only.
        assert cfg["hosts"]["hermes"]["apiKey"] == "hch-test-key"


    def test_get_honcho_config_does_not_return_secret(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HONCHO_API_KEY", "guard")
        monkeypatch.delenv("HONCHO_API_KEY")
        self._seed_local_honcho()

        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"apiKey": "secret-value"}},
        )

        resp = self.client.get("/api/memory/providers/honcho/config?surface=declared")

        assert resp.status_code == 200
        data = resp.json()
        fields = self._provider_field_map(data)
        assert fields["apiKey"]["is_set"] is True
        assert fields["apiKey"]["value"] == ""
        assert "secret-value" not in json.dumps(data)


    # ── GET /api/media (remote image display) ───────────────────────────


    def test_get_media_requires_auth(self):
        from hermes_cli.web_server import _SESSION_HEADER_NAME

        resp = self.client.get(
            "/api/media",
            params={"path": "/tmp/x.png"},
            headers={_SESSION_HEADER_NAME: "wrong-token"},
        )
        assert resp.status_code == 401

    # ── GET /api/media/proxy (client-blocked CDN fallback, #74564) ──────


    def test_media_proxy_requires_auth(self):
        from hermes_cli.web_server import _SESSION_HEADER_NAME

        resp = self.client.get(
            "/api/media/proxy",
            params={"url": "https://v3.fal.media/x.png"},
            headers={_SESSION_HEADER_NAME: "wrong-token"},
        )
        assert resp.status_code == 401

    def test_media_proxy_rejects_disallowed_hosts_and_schemes(self):
        for bad in (
            "https://evil.example.com/img.png",
            "https://sub.fal.media.evil.com/img.png",
            "file:///etc/passwd",
            "not a url",
            "",
        ):
            resp = self.client.get("/api/media/proxy", params={"url": bad})
            assert resp.status_code in (400, 403), (bad, resp.status_code)

    def test_media_proxy_fetches_allowlisted_image_and_returns_data_url(self, monkeypatch):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 8

        class _Resp:
            status_code = 200
            headers = {"content-type": "image/png"}
            content = png_bytes

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                assert url == "https://v3.fal.media/media/abc123"
                return _Resp()

        import hermes_cli.web_routers.files as files_router

        monkeypatch.setattr(files_router, "_require_token", lambda request: None, raising=False)
        # The route imports httpx locally; patch the module it resolves from.
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _Client, raising=False)

        resp = self.client.get(
            "/api/media/proxy", params={"url": "https://v3.fal.media/media/abc123"}
        )
        assert resp.status_code == 200
        import base64

        assert resp.json()["data_url"] == (
            "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        )

    def test_media_proxy_rejects_redirect_to_disallowed_host(self, monkeypatch):
        class _Resp:
            status_code = 302
            headers = {"location": "http://127.0.0.1/admin"}

        class _Client:
            def __init__(self, *a, **k):
                assert k.get("follow_redirects") is False
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                self.calls.append(url)
                return _Resp()

        import hermes_cli.web_routers.files as files_router
        import httpx

        monkeypatch.setattr(files_router, "_require_token", lambda request: None, raising=False)
        monkeypatch.setattr(httpx, "AsyncClient", _Client, raising=False)

        resp = self.client.get(
            "/api/media/proxy", params={"url": "https://fal.media/media/abc123"}
        )
        assert resp.status_code == 403

    def test_media_proxy_rejects_non_image_content_type(self, monkeypatch):
        class _Resp:
            status_code = 200
            headers = {"content-type": "text/html"}
            content = b"<html>nope</html>"

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                return _Resp()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _Client, raising=False)

        resp = self.client.get(
            "/api/media/proxy", params={"url": "https://fal.media/media/abc123"}
        )
        assert resp.status_code == 415

    # ── POST /api/chat/image-upload (browser clipboard/drop images) ─────


    # ── Dashboard font override ─────────────────────────────────────────


    def test_import_sessions_endpoint_imports_exported_json(self):
        from hermes_state import SessionDB

        payload = {
            "id": "imported-web-session",
            "source": "cli",
            "title": "Imported from dashboard",
            "started_at": 100.0,
            "ended_at": 110.0,
            "end_reason": "complete",
            "messages": [
                {"role": "user", "content": "hello", "timestamp": 101.0},
                {"role": "assistant", "content": "hi", "timestamp": 102.0},
            ],
        }

        resp = self.client.post("/api/sessions/import", json={"sessions": [payload]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 1
        assert data["skipped"] == 0

        db = SessionDB()
        try:
            session = db.get_session("imported-web-session")
            assert session["title"] == "Imported from dashboard"
            assert session["message_count"] == 2
            assert [m["content"] for m in db.get_messages("imported-web-session")] == [
                "hello",
                "hi",
            ]
        finally:
            db.close()

        duplicate = self.client.post("/api/sessions/import", json={"sessions": [payload]})
        assert duplicate.status_code == 200
        assert duplicate.json()["skipped_ids"] == ["imported-web-session"]

        invalid = self.client.post(
            "/api/sessions/import",
            json={"sessions": [{"source": "cli", "messages": []}]},
        )
        assert invalid.status_code == 400
        errors = invalid.json()["detail"]["errors"]
        assert [e["index"] for e in errors] == [0] and errors[0]["error"]


    def test_latest_descendant_survives_parent_cycle(self):
        """Regression for the #39140 CTE salvage: a corrupted parent chain
        that loops (a -> b -> a) must terminate (UNION dedup) instead of
        recursing forever like UNION ALL would."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="cyc-a", source="cli")
            db.create_session(
                session_id="cyc-b", source="cli", parent_session_id="cyc-a"
            )
            db._conn.execute(
                "UPDATE sessions SET parent_session_id='cyc-b' WHERE id='cyc-a'"
            )
            db._conn.commit()
        finally:
            db.close()

        resp = self.client.get("/api/sessions/cyc-a/latest-descendant")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "cyc-b"

    def test_latest_descendant_never_resumes_into_a_subagent_or_branch_child(self):
        """#115092: after a ws_orphan_reap the dashboard resumes the predecessor's newest descendant. A
        subagent run (``_delegate_from``) or a /branch fork (``_branched_from``) is its own conversation and
        never listed as a continuation, so following it parks the user's chat in a hidden row; only
        compression continuations are followed."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="primary", source="tui")
            db.create_session(session_id="primary-sub", source="tui", parent_session_id="primary",
                              model_config={"_delegate_from": "primary"})
            db.create_session(session_id="primary-fork", source="tui", parent_session_id="primary",
                              model_config={"_branched_from": "primary"})
            db.end_session("primary", "ws_orphan_reap")
            assert self.client.get("/api/sessions/primary/latest-descendant").json()["session_id"] == "primary"

            db._conn.execute("UPDATE sessions SET end_reason='compression' WHERE id='primary'")
            db._conn.commit()
            db.create_session(session_id="primary-cont", source="tui", parent_session_id="primary")
        finally:
            db.close()

        resp = self.client.get("/api/sessions/primary/latest-descendant")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "primary-cont"


    def test_update_hermes_returns_docker_guidance_without_spawning(self, monkeypatch):

        spawned = False

        def fail_spawn(*_args, **_kwargs):
            nonlocal spawned
            spawned = True
            raise AssertionError("docker update guard should not spawn hermes update")

        # Bypass the managed-externally gate so we reach the docker install check.
        monkeypatch.setattr(_web_server_files, "_dashboard_local_update_managed_externally", lambda: False)
        # The shared admission gate (#91277 Phase 3) resolves the install
        # method through hermes_cli.config directly.
        monkeypatch.setattr(
            "hermes_cli.config.detect_install_method", lambda *_a, **_k: "docker"
        )
        monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda _root: "docker")
        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", fail_spawn)
        _web_server_gateway._ACTION_PROCS.pop("hermes-update", None)
        _web_server_gateway._ACTION_RESULTS.pop("hermes-update", None)

        resp = self.client.post("/api/hermes/update")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["name"] == "hermes-update"
        assert data["pid"] is None
        assert data["error"] == "docker_update_unsupported"
        assert spawned is False

        status = self.client.get("/api/actions/hermes-update/status")
        assert status.status_code == 200
        status_data = status.json()
        assert status_data["running"] is False
        assert status_data["exit_code"] == 1
        assert status_data["pid"] is None

    def test_update_hermes_returns_apt_guidance_without_spawning(self, monkeypatch):

        spawned = False

        def fail_spawn(*_args, **_kwargs):
            nonlocal spawned
            spawned = True
            raise AssertionError("APT-managed update guard should not spawn hermes update")

        monkeypatch.setattr(_web_server_files, "_dashboard_local_update_managed_externally", lambda: False)
        # The shared admission gate (#91277 Phase 3) resolves the install
        # method through hermes_cli.config directly, so patch it there (the
        # web_server module alias only feeds the /update/check endpoint).
        monkeypatch.setattr(
            "hermes_cli.config.detect_install_method", lambda *_a, **_k: "apt"
        )
        monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda _root: "apt")
        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", fail_spawn)
        _web_server_gateway._ACTION_PROCS.pop("hermes-update", None)
        _web_server_gateway._ACTION_RESULTS.pop("hermes-update", None)

        resp = self.client.post("/api/hermes/update")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["pid"] is None
        assert data["error"] == "apt_update_required"
        assert data["update_command"]
        assert spawned is False

        check = self.client.get("/api/hermes/update/check")
        assert check.status_code == 200
        check_data = check.json()
        assert check_data["install_method"] == "apt"
        assert check_data["can_apply"] is False
        assert check_data["update_command"] == data["update_command"]

    def test_update_status_recovers_completed_result_after_dashboard_restart(self, monkeypatch, tmp_path):

        action_id = "c" * 32
        (tmp_path / "hermes-update.log").write_text(
            "=== hermes-update started 2026-08-17 11:19:34 ===\n"
            "pulling updates...\n",
            encoding="utf-8",
        )
        (tmp_path / "update.log").write_text(
            "=== hermes update started 2026-08-17T11:19:35 ===\n"
            "✓ Update complete!\n"
            f"=== hermes-update completed {action_id} ===\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path)
        monkeypatch.setattr(_web_server_gateway, "_ACTION_PROCS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_RESULTS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_COMMANDS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_IDS", {})

        status = self.client.get("/api/actions/hermes-update/status?lines=2000")

        assert status.status_code == 200
        data = status.json()
        assert data["running"] is False
        assert data["exit_code"] == 0
        assert data["action_id"] == action_id
        assert f"=== hermes-update completed {action_id} ===" in data["lines"]

    def test_update_hermes_spawns_with_action_id(self, monkeypatch):
        import hermes_cli.web_server as web_server

        class Proc:
            pid = 12345

        calls = []

        def fake_spawn(subcommand, name, *, env_overrides=None):
            calls.append((subcommand, name, env_overrides))
            return Proc()

        monkeypatch.setattr(_web_server_files, "_dashboard_local_update_managed_externally", lambda: False)
        monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda _root: "git")
        monkeypatch.setattr(web_server.secrets, "token_hex", lambda _size: "a" * 32)
        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", fake_spawn)
        _web_server_gateway._ACTION_PROCS.pop("hermes-update", None)
        _web_server_gateway._ACTION_RESULTS.pop("hermes-update", None)

        resp = self.client.post("/api/hermes/update")

        assert resp.status_code == 200
        assert resp.json() == {
            "ok": True,
            "pid": 12345,
            "name": "hermes-update",
            "action_id": "a" * 32,
        }
        assert calls == [
            (["update"], "hermes-update", {"HERMES_ACTION_ID": "a" * 32})
        ]

    def test_update_hermes_reuses_running_action(self, monkeypatch):

        class Proc:
            pid = 24680

            def poll(self):
                return None

        monkeypatch.setattr(_web_server_files, "_dashboard_local_update_managed_externally", lambda: False)
        monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda _root: "git")
        monkeypatch.setattr(
            _web_server_gateway,
            "_spawn_hermes_action",
            lambda *_args, **_kwargs: pytest.fail("must not spawn a duplicate update"),
        )
        _web_server_gateway._ACTION_PROCS["hermes-update"] = Proc()
        _web_server_gateway._ACTION_IDS["hermes-update"] = "b" * 32

        try:
            resp = self.client.post("/api/hermes/update")
        finally:
            _web_server_gateway._ACTION_PROCS.pop("hermes-update", None)
            _web_server_gateway._ACTION_IDS.pop("hermes-update", None)

        assert resp.status_code == 200
        assert resp.json() == {
            "ok": True,
            "pid": 24680,
            "name": "hermes-update",
            "already_running": True,
            "action_id": "b" * 32,
        }


    def test_model_set_maps_unknown_vendor_to_aggregator(self, monkeypatch):
        """A bare vendor name from analytics rows (no billing_provider) is not
        a Hermes provider — keep the user's aggregator instead of writing a
        provider that can never resolve credentials."""
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: None,
        )
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        cfg["model"] = {"provider": "openrouter", "default": "openai/gpt-5.5"}
        save_config(cfg)

        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "moonshotai",  # vendor prefix, not a provider
                "model": "moonshotai/kimi-k2.6",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["provider"] == "openrouter"
        assert data["model"] == "moonshotai/kimi-k2.6"


    def test_model_set_flips_a_stale_setup_record(self, monkeypatch):
        """POST /api/model/set landed a provider on disk; the serve process's boot record
        (``provider_configured: false`` since a failed boot-time mint) must follow at once, with
        the ``setup.ready`` broadcast, or the web chat stays gated on "need setup" until a restart
        (setup.status answers from the record)."""
        from hermes_cli import free_tier_bootstrap as fb

        fb.reset_for_tests()
        monkeypatch.setattr("hermes_cli.model_cost_guard.expensive_model_warning", lambda *_a, **_k: None)
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)
        broadcasts = []
        monkeypatch.setattr(fb, "_broadcast", broadcasts.append)
        with fb._lock:
            fb._record = fb.SetupRecord(provider_configured=False, inference_provider="", free_tier_account=False,
                                        has_identity=False, other_providers=False)
            fb._started = True
            fb._done.set()
        try:
            resp = self.client.post(
                "/api/model/set",
                json={"scope": "main", "provider": "custom", "model": "local-model",
                      "base_url": "http://127.0.0.1:8081/v1", "api_key": "sk-local"},
            )
            assert resp.status_code == 200 and resp.json()["ok"] is True
            record = fb.current_record()
            assert record.provider_configured is True and record.other_providers is True
            assert record.inference_provider == "custom"
            assert broadcasts == [record]
        finally:
            fb.reset_for_tests()


    def test_reveal_env_var(self, tmp_path):
        """POST /api/env/reveal should return the real unredacted value."""
        from hermes_cli.config import save_env_value
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN
        save_env_value("TEST_REVEAL_KEY", "super-secret-value-12345")
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_KEY"},
            headers={_SESSION_HEADER_NAME: _SESSION_TOKEN},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "TEST_REVEAL_KEY"
        assert data["value"] == "super-secret-value-12345"


    def test_reveal_env_var_custom_session_header_ignores_proxy_authorization(self, tmp_path):
        """A valid dashboard session header should coexist with proxy auth."""
        from hermes_cli.config import save_env_value
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

        save_env_value("TEST_REVEAL_PROXY_AUTH", "secret-value")
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_PROXY_AUTH"},
            headers={
                _SESSION_HEADER_NAME: _SESSION_TOKEN,
                "Authorization": "Basic dXNlcjpwYXNz",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["value"] == "secret-value"

    def test_reveal_env_var_legacy_authorization_header_still_works(self, tmp_path):
        """Keep old dashboard bundles working while the new header rolls out."""
        from hermes_cli.config import save_env_value
        from hermes_cli.web_server import _SESSION_TOKEN

        save_env_value("TEST_REVEAL_LEGACY_AUTH", "secret-value")
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_LEGACY_AUTH"},
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
        )

        assert resp.status_code == 200


    def test_messaging_catalog_prefers_plugin_label_over_enum_pseudo_member(self):
        """A plugin platform that leaked into Platform.__members__ as a pseudo-
        member must still render with its plugin label, not a title-cased id.

        Regression: Platform("<plugin id>") caches a pseudo-member in the enum;
        the catalog iterated the enum FIRST and claimed the id with no plugin
        metadata, so bundled plugin platforms (irc, ntfy, photon, …) rendered
        as nameless "Irc"/"Ntfy" cards with empty descriptions.
        """
        from gateway.config import Platform
        from gateway.platform_registry import PlatformEntry, platform_registry

        entry = PlatformEntry(
            name="pseudofake",
            label="Pseudo Fake (plugin label)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
        )
        platform_registry.register(entry)
        try:
            # Materialize the enum pseudo-member the way any earlier config
            # read would (Platform(value) on a registered plugin platform).
            member = Platform("pseudofake")
            assert member.value == "pseudofake"
            assert "PSEUDOFAKE" in Platform.__members__

            resp = self.client.get("/api/messaging/platforms")
            ids = {row["id"]: row for row in resp.json()["platforms"]}
            assert "pseudofake" in ids
            assert ids["pseudofake"]["name"] == "Pseudo Fake (plugin label)"
        finally:
            platform_registry.unregister("pseudofake")
            Platform._value2member_map_.pop("pseudofake", None)
            Platform._member_map_.pop("PSEUDOFAKE", None)


    def test_telegram_onboarding_apply_reports_restart_failure_after_save(
        self, monkeypatch
    ):
        from hermes_cli.config import load_config, load_env

        with _web_server_messaging._telegram_onboarding_lock:
            _web_server_messaging._telegram_onboarding_pairings.clear()

        def fake_request(method, path, *, body=None, bearer_token=None):
            if method == "POST":
                return {
                    "pairing_id": "pair-restart-fails",
                    "poll_token": "poll-secret",
                    "suggested_username": "hermes_pair_restart_fails_bot",
                    "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_pair_restart_fails_bot",
                    "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_pair_restart_fails_bot",
                    "expires_at": "2027-05-18T00:00:00.000Z",
                }
            assert method == "GET"
            assert path == "/v1/telegram/pairings/pair-restart-fails"
            assert bearer_token == "poll-secret"
            return {
                "status": "ready",
                "bot_username": "hermes_pair_restart_fails_bot",
                "owner_user_id": 123456789,
                "token": "123456:SECRET",
            }

        monkeypatch.setattr(_web_server_messaging, "_telegram_onboarding_request_sync", fake_request)
        _web_server_gateway._ACTION_PROCS.pop("gateway-restart", None)

        def fail_spawn_action(subcommand, name):
            # The default home is named explicitly: a bare child would re-read the sticky active_profile.
            assert subcommand == ["-p", "default", "gateway", "restart"]
            assert name == "gateway-restart"
            raise RuntimeError("supervisor unavailable")

        monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", fail_spawn_action)

        start = self.client.post("/api/messaging/telegram/onboarding/start", json={})
        assert start.status_code == 200
        ready = self.client.get("/api/messaging/telegram/onboarding/pair-restart-fails")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

        applied = self.client.post(
            "/api/messaging/telegram/onboarding/pair-restart-fails/apply",
            json={"allowed_user_ids": ["123456789"]},
        )

        assert applied.status_code == 200
        applied_data = applied.json()
        assert applied_data["ok"] is True
        assert applied_data["needs_restart"] is True
        assert applied_data["restart_started"] is False
        assert "supervisor unavailable" in applied_data["restart_error"]
        assert "token" not in applied_data
        env = load_env()
        assert env["TELEGRAM_BOT_TOKEN"] == "123456:SECRET"
        assert env["TELEGRAM_ALLOWED_USERS"] == "123456789"
        assert load_config()["platforms"]["telegram"]["enabled"] is True


    def test_unauthenticated_api_blocked(self):
        """API requests without the session token should be rejected."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app
        # Create a client WITHOUT the dashboard session header
        unauth_client = TestClient(app)
        resp = unauth_client.get("/api/env")
        assert resp.status_code == 401
        resp = unauth_client.get("/api/config")
        assert resp.status_code == 401
        # Public endpoints should still work
        resp = unauth_client.get("/api/status")
        assert resp.status_code == 200
        resp = unauth_client.get("/api/dashboard/plugins")
        assert resp.status_code == 200
        resp = unauth_client.get("/api/dashboard/plugins/rescan")
        assert resp.status_code == 401
        resp = self.client.get("/api/dashboard/plugins/rescan")
        assert resp.status_code == 200


    def test_parse_model_ids_handles_openai_and_bare_shapes(self):
        """Model discovery must tolerate the common /v1/models shapes and
        never raise (so a slightly non-standard local endpoint still works)."""
        from hermes_cli.web_server_profiles import _parse_model_ids

        class FakeResp:
            def __init__(self, payload, ok=True):
                self._payload = payload
                self.is_success = ok

            def json(self):
                if isinstance(self._payload, Exception):
                    raise self._payload
                return self._payload

        # OpenAI / vLLM / llama.cpp shape.
        assert _parse_model_ids(
            FakeResp({"data": [{"id": "llama-3.1-8b"}, {"id": "qwen2.5-7b"}]})
        ) == ["llama-3.1-8b", "qwen2.5-7b"]
        # Bare list of ids.
        assert _parse_model_ids(FakeResp({"data": ["m1", "m2"]})) == ["m1", "m2"]
        # Top-level list.
        assert _parse_model_ids(FakeResp([{"id": "x"}])) == ["x"]
        # Non-success / malformed / exception → [] (never raises).
        assert _parse_model_ids(FakeResp({"data": []}, ok=False)) == []
        assert _parse_model_ids(FakeResp({"nope": 1})) == []
        assert _parse_model_ids(FakeResp(ValueError("bad json"))) == []


    def test_set_model_main_custom_persists_api_key_and_registers_provider(self):
        """A custom endpoint that requires auth must persist model.api_key (where
        the runtime reads it) AND register a named custom_providers entry so the
        endpoint reappears as a ready row in the picker — matching the
        ``hermes model`` custom flow. Regression for the desktop loop where a
        keyed custom endpoint could never be configured from the GUI."""
        from hermes_cli.config import load_config

        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom",
                "model": "gpt-oss-120b",
                "base_url": "https://text.example.com/v1",
                "api_key": "sk-secret",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        cfg = load_config()
        model_cfg = cfg.get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["provider"] == "custom"
        assert model_cfg["base_url"] == "https://text.example.com/v1"
        assert model_cfg["api_key"] == "sk-secret"

        # Registered in custom_providers (dedup by base_url) so the picker shows
        # a proper ready row instead of the "needs setup" dead-end.
        custom = cfg.get("custom_providers") or []
        assert any(
            isinstance(e, dict)
            and e.get("base_url") == "https://text.example.com/v1"
            and e.get("api_key") == "sk-secret"
            and e.get("model") == "gpt-oss-120b"
            for e in custom
        )


    def test_deleting_the_active_custom_endpoint_clears_its_model_mirror(self):
        """Deleting an endpoint must not leave its credential running the agent.

        ``activate`` mirrors the endpoint's base_url + credential reference
        onto ``model``, and that mirror outranks the environment at client
        construction (#62269). Without clearing it the agent keeps
        authenticating to the deleted host, and the credential the operator
        just removed through the dashboard survives the delete.
        """
        from hermes_cli.config import custom_endpoint_key_env, get_env_value, load_config

        self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "acme",
                "name": "Acme",
                "base_url": "https://llm.acme.corp/v1",
                "model": "acme/model-1",
                "api_key": "sk-acme-secret",
            },
        )
        assert self.client.post(
            "/api/providers/custom-endpoints/acme/activate", json={}
        ).status_code == 200

        env_var = custom_endpoint_key_env("acme")
        cfg = load_config()
        assert cfg["model"]["key_env"] == env_var
        assert get_env_value(env_var) == "sk-acme-secret"

        assert self.client.request(
            "DELETE", "/api/providers/custom-endpoints/acme"
        ).status_code == 200

        cfg = load_config()
        assert "acme" not in (cfg.get("providers") or {})
        model_cfg = cfg.get("model") or {}
        assert not model_cfg.get("api_key"), "deleted endpoint's key still in config.yaml"
        assert not model_cfg.get("key_env"), "deleted endpoint's key ref still in config.yaml"
        assert not model_cfg.get("base_url"), "deleted endpoint's host still routed to"
        assert not model_cfg.get("provider")
        assert not get_env_value(env_var), "deleted endpoint's key still in .env"


    def test_numeric_yaml_provider_key_can_be_activated_and_deleted(self):
        """Hand-edited `providers: 2070:` (YAML int key) must still activate.

        PyYAML loads unquoted 2070 as int; string lookup then 404ed, so
        Desktop could list the endpoint but not assign or delete it.
        """
        from hermes_cli.config import get_config_path, load_config

        get_config_path().write_text(
            "model:\n"
            "  provider: 2070\n"
            "  default: Qwen.gguf\n"
            "  base_url: http://127.0.0.1:1/v1\n"
            "providers:\n"
            "  2070:\n"
            "    name: 2070\n"
            "    base_url: http://127.0.0.1:1/v1\n"
            "    model: Qwen.gguf\n",
            encoding="utf-8",
        )

        listed = self.client.get("/api/providers/custom-endpoints")
        assert listed.status_code == 200
        assert "2070" in [e["id"] for e in listed.json()["endpoints"]]

        activate = self.client.post(
            "/api/providers/custom-endpoints/2070/activate", json={}
        )
        assert activate.status_code == 200, activate.text
        assert activate.json()["provider"] == "2070"

        deleted = self.client.request(
            "DELETE", "/api/providers/custom-endpoints/2070"
        )
        assert deleted.status_code == 200, deleted.text
        providers = load_config().get("providers") or {}
        assert 2070 not in providers
        assert "2070" not in providers

    def test_punctuated_provider_key_round_trips_through_activate_edit_and_delete(self):
        """A stored ``providers.<key>`` with dots/colons or mixed case is what the
        list route returns as ``id``; the same spelling must reach the entry on
        activate, save (edit) and delete instead of being slugified into a
        non-existent twin (404 / duplicate row), and delete must still clear
        the model mirror ``switch_model`` wrote for it.
        """
        from urllib.parse import quote

        from hermes_cli.config import get_config_path, load_config

        get_config_path().write_text(
            "model:\n"
            "  provider: openrouter\n"
            "  default: some/model\n"
            "providers:\n"
            "  local-127.0.0.1:8283:\n"
            "    name: Local (127.0.0.1:8283)\n"
            "    base_url: http://127.0.0.1:8283/v1\n"
            "    model: Qwen.gguf\n"
            "  EXllamav3:\n"
            "    name: EXllamav3\n"
            "    base_url: http://127.0.0.1:8290/v1\n"
            "    model: Qwen3-27B\n",
            encoding="utf-8",
        )
        dotted = "local-127.0.0.1:8283"
        listed = [e["id"] for e in self.client.get("/api/providers/custom-endpoints").json()["endpoints"]]
        assert dotted in listed and "EXllamav3" in listed

        # Edit by the listed id updates the entry in place — no slugged twin.
        saved = self.client.post(
            "/api/providers/custom-endpoints",
            json={"id": dotted, "name": "Local (127.0.0.1:8283)",
                  "base_url": "http://127.0.0.1:8283/v1", "model": "Qwen2.gguf"},
        )
        assert saved.status_code == 200, saved.text
        providers = load_config()["providers"]
        assert providers[dotted]["model"] == "Qwen2.gguf"
        assert "local-127-0-0-1-8283" not in providers

        for key in (dotted, "EXllamav3"):
            path = f"/api/providers/custom-endpoints/{quote(key, safe='')}"
            activate = self.client.post(f"{path}/activate", json={})
            assert activate.status_code == 200, activate.text
            assert load_config()["model"].get("base_url"), key
            current = [e["id"] for e in self.client.get("/api/providers/custom-endpoints").json()["endpoints"]
                       if e["is_current"]]
            assert current == [key], f"{key}: list does not mark the endpoint just activated as current: {current}"
            deleted = self.client.request("DELETE", path)
            assert deleted.status_code == 200, deleted.text
            cfg = load_config()
            assert key not in (cfg.get("providers") or {})
            assert not cfg["model"].get("base_url"), f"{key}: deleted endpoint's host still routed to"
            assert not cfg["model"].get("provider"), key

    def test_unslugged_display_name_still_resolves_to_its_slug_key(self):
        """Compatibility fallback: a caller sending the display name reaches the
        dashboard-minted slug key; an unknown id is still a 404."""
        from hermes_cli.config import get_config_path, load_config

        get_config_path().write_text(
            "providers:\n"
            "  local-8000:\n"
            "    name: Local 8000\n"
            "    base_url: http://127.0.0.1:8000/v1\n"
            "    model: m\n",
            encoding="utf-8",
        )
        assert self.client.request("DELETE", "/api/providers/custom-endpoints/nope.nope").status_code == 404
        assert self.client.request("DELETE", "/api/providers/custom-endpoints/Local%208000").status_code == 200
        assert "local-8000" not in (load_config().get("providers") or {})


    def test_custom_endpoint_save_scopes_to_the_requested_profile(self):
        """``?profile=<name>`` must write into that profile's config.yaml.

        The desktop settings UI targets the active profile, so a custom
        endpoint saved while a non-default profile is selected has to land in
        that profile's config — not the dashboard process's default home.
        Before this fix the handlers ran bare ``load_config``/``save_config``,
        so every custom provider silently landed in the default profile and
        never appeared for the profile the user was actually configuring.
        """
        from hermes_cli import profiles as profiles_mod
        from hermes_cli.config import custom_endpoint_key_env
        from hermes_constants import get_hermes_home

        default_home = get_hermes_home()
        worker_home = profiles_mod.get_profile_dir("worker")
        worker_home.mkdir(parents=True)
        (worker_home / "config.yaml").touch()  # identity marker: bare dirs are not profiles

        assert self.client.post(
            "/api/providers/custom-endpoints?profile=worker",
            json={
                "id": "worker-proxy",
                "name": "Worker Proxy",
                "base_url": "https://llm.worker.example/v1",
                "model": "worker/model-1",
                "api_key": "sk-worker-secret",
            },
        ).status_code == 200

        # Assert against the files on disk rather than load_config()/
        # get_env_value(): save_env_value also mirrors the key into the shared
        # os.environ, so a reader-based check can't tell WHICH profile's store
        # actually received the write.
        env_var = custom_endpoint_key_env("worker-proxy")

        worker_cfg = (worker_home / "config.yaml").read_text()
        assert "worker-proxy" in worker_cfg
        assert env_var in worker_cfg
        assert "sk-worker-secret" in (worker_home / ".env").read_text()

        for leaked in (default_home / "config.yaml", default_home / ".env"):
            text = leaked.read_text() if leaked.exists() else ""
            assert "worker-proxy" not in text, f"endpoint leaked into default profile ({leaked.name})"
            assert "sk-worker-secret" not in text, f"credential leaked into default profile ({leaked.name})"

        # And it comes back through the scoped GET, not the unscoped one.
        scoped = self.client.get("/api/providers/custom-endpoints?profile=worker").json()
        assert any(e["id"] == "worker-proxy" for e in scoped["endpoints"])
        default_list = self.client.get("/api/providers/custom-endpoints").json()
        assert not any(e["id"] == "worker-proxy" for e in default_list["endpoints"])


    def test_custom_endpoint_save_keeps_the_api_key_out_of_config(self):
        """The key belongs in .env behind key_env, never in config.yaml (#69449)."""
        from hermes_cli.config import custom_endpoint_key_env, get_env_value, load_config

        self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "proxy",
                "name": "Proxy",
                "base_url": "https://llm.example.com/v1",
                "model": "m",
                "api_key": "sk-super-secret",
                "make_default": True,
            },
        )

        cfg = load_config()
        entry = cfg["providers"]["proxy"]
        env_var = custom_endpoint_key_env("proxy")
        assert entry["key_env"] == env_var
        assert "api_key" not in entry
        assert "api_key" not in cfg["model"]
        assert get_env_value(env_var) == "sk-super-secret"
        assert "sk-super-secret" not in yaml.safe_dump(cfg)


    def test_custom_endpoint_save_pins_api_mode_and_resolves_reasoning_alias(self):
        """Desktop's Custom Endpoints form pins the transport and keeps alias metadata (#93622).

        A Responses-only host 404s on the runtime's Chat Completions default, so the chosen
        ``api_mode`` must land on the providers entry and read back; a discovered reasoning
        alias resolves to its canonical model + ``agent.reasoning_overrides`` instead of being
        saved as a literal upstream model id.
        """
        from hermes_cli.config import load_config

        response = self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "custom-responses", "name": "custom-responses",
                "base_url": "https://responses-gateway.example.com/v1",
                "model": "gpt-5.6-sol-high", "api_mode": "codex_responses", "make_default": True,
                "models": ["gpt-5.6-sol", "gpt-5.6-sol-high"],
                "model_details": [
                    {"id": "gpt-5.6-sol"},
                    {"id": "gpt-5.6-sol-high", "canonical_model": "gpt-5.6-sol", "reasoning_effort": "high"},
                ],
            },
        )
        assert response.status_code == 200
        row = next(e for e in response.json()["endpoints"] if e["id"] == "custom-responses")
        assert row["api_mode"] == "codex_responses"
        assert row["model"] == "gpt-5.6-sol"

        cfg = load_config()
        entry = cfg["providers"]["custom-responses"]
        assert entry["api_mode"] == "codex_responses"
        assert entry["model"] == "gpt-5.6-sol"
        assert entry["models"]["gpt-5.6-sol-high"] == {"canonical_model": "gpt-5.6-sol", "reasoning_effort": "high"}
        assert cfg["model"]["default"] == "gpt-5.6-sol"
        assert cfg["agent"]["reasoning_overrides"]["gpt-5.6-sol"] == "high"

        # An older UI payload (no api_mode) leaves the pinned transport alone; "" clears it.
        self.client.post("/api/providers/custom-endpoints", json={
            "id": "custom-responses", "name": "custom-responses",
            "base_url": "https://responses-gateway.example.com/v1", "model": "gpt-5.6-sol"})
        assert load_config()["providers"]["custom-responses"]["api_mode"] == "codex_responses"
        self.client.post("/api/providers/custom-endpoints", json={
            "id": "custom-responses", "name": "custom-responses", "api_mode": "",
            "base_url": "https://responses-gateway.example.com/v1", "model": "gpt-5.6-sol"})
        listed = self.client.get("/api/providers/custom-endpoints").json()["endpoints"]
        assert next(e for e in listed if e["id"] == "custom-responses")["api_mode"] == ""
        assert "api_mode" not in load_config()["providers"]["custom-responses"]

    def test_custom_endpoint_validate_keeps_model_alias_metadata(self, monkeypatch):
        """``validate`` returns the bare id list older clients read AND ``model_details`` with
        the ``canonical_model`` / ``reasoning_effort`` a gateway advertises (#93622)."""
        import contextlib

        from hermes_cli.web_routers import config_env

        class FakeResp:
            status_code = 200
            is_success = True

            def json(self):
                return {"data": [
                    {"id": "gpt-5.6-sol", "object": "model"},
                    {"id": "gpt-5.6-sol-high", "canonical_model": "gpt-5.6-sol", "reasoning_effort": "high"},
                ]}

        class FakeClient:
            async def get(self, url, headers=None):
                return FakeResp()

            async def post(self, url, json=None, headers=None):
                return FakeResp()

        @contextlib.asynccontextmanager
        async def fake_probe_client(url, timeout):
            yield FakeClient()

        monkeypatch.setattr(config_env, "_endpoint_probe_client", fake_probe_client)
        body = self.client.post("/api/providers/custom-endpoints/validate", json={
            "name": "x", "base_url": "https://responses-gateway.example.com/v1", "model": ""}).json()
        assert body["ok"] is True
        assert body["models"] == ["gpt-5.6-sol", "gpt-5.6-sol-high"]
        assert body["model_details"] == [
            {"id": "gpt-5.6-sol"},
            {"id": "gpt-5.6-sol-high", "canonical_model": "gpt-5.6-sol", "reasoning_effort": "high"},
        ]

    @staticmethod
    def _responses_only_host(monkeypatch, posted):
        """A gateway that lists models on GET /models and serves POST /responses but 404s
        POST /chat/completions — the #93622 reporter's host."""
        import contextlib

        from hermes_cli.web_routers import config_env

        class Resp:
            def __init__(self, status):
                self.status_code, self.is_success = status, status < 400

            def json(self):
                return {"data": [{"id": "gpt-5.6-sol"}]}

        class Client:
            async def get(self, url, headers=None):
                return Resp(200)

            async def post(self, url, json=None, headers=None):
                posted.append((url, json))
                return Resp(400 if url.endswith("/responses") else 404)

        @contextlib.asynccontextmanager
        async def probe_client(url, timeout):
            yield Client()

        monkeypatch.setattr(config_env, "_endpoint_probe_client", probe_client)

    def test_custom_endpoint_validate_fails_when_the_transport_route_is_missing(self, monkeypatch):
       ... [truncated]