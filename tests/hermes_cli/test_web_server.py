"""Tests for hermes_cli.web_server and related config utilities."""

import asyncio
import os
import json
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import yaml

from hermes_cli.config import (
    reload_env,
    redact_key,
    OPTIONAL_ENV_VARS,
    DEFAULT_CONFIG,
)


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
    web_server._mount_plugin_api_routes()

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
        env_file.write_text("TEST_RELOAD_VAR=hello123\n")
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

    def test_empty_key(self):
        result = redact_key("")
        assert "not set" in result.lower() or result == "***" or "\x1b" in result


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

    def test_falls_back_to_random_token(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
        with patch.object(
            ws.secrets, "token_urlsafe", return_value="generated-token"
        ) as token_urlsafe:
            assert ws._resolve_session_token() == "generated-token"
        token_urlsafe.assert_called_once_with(32)

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

        from hermes_cli import web_server
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        web_server._last_auto_archive_check.clear()
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

    def test_get_sessions_auto_archive_uses_maintenance_writer(self):
        from hermes_cli import web_server
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
        web_server._last_auto_archive_check.clear()

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

    @pytest.mark.parametrize("missing_column", ["archived", "pinned"])
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

        seen = {}

        def _pid(pid_path=None, **kw):
            seen["pid_path"] = pid_path
            return None

        def _runtime(path=None):
            seen["status_path"] = path
            return None

        def _runtime_pid(runtime=None, *, expected_home=None):
            seen["expected_home"] = expected_home
            return None

        monkeypatch.setattr(web_server, "get_running_pid_cached", _pid)
        monkeypatch.setattr(web_server, "get_running_pid", _pid)
        monkeypatch.setattr(web_server, "read_runtime_status", _runtime)
        monkeypatch.setattr(web_server, "get_runtime_status_running_pid", _runtime_pid)
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)

        resp = self.client.get("/api/messaging/platforms?profile=worker")

        assert resp.status_code == 200
        assert seen["pid_path"] == worker_home / "gateway.pid"
        assert seen["status_path"] == worker_home / "gateway_state.json"
        assert seen["expected_home"] == worker_home




    def test_gateway_drain_bad_action_400(self):
        resp = self.client.post("/api/gateway/drain", json={"action": "explode"})
        assert resp.status_code == 400




    @staticmethod
    def _provider_field_map(payload):
        return {field["key"]: field for field in payload["fields"]}






    def test_declared_surface_put_writes_config_and_secret(self):
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_env

        resp = self.client.put(
            "/api/memory/providers/hindsight/config?surface=declared",
            json={
                "values": {
                    "mode": "local_external",
                    "api_url": "http://localhost:8888",
                    "api_key": "hs-declared-key",
                }
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert load_env()["HINDSIGHT_API_KEY"] == "hs-declared-key"

        config_path = get_hermes_home() / "hindsight" / "config.json"
        provider_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert provider_config["mode"] == "local_external"
        assert provider_config["api_url"] == "http://localhost:8888"
        assert "api_key" not in provider_config






    def test_post_memory_provider_setup_routes_pip_through_lazy_deps(self, monkeypatch):
        """NS-605: dashboard pip installs must use the environment-aware
        lazy_deps pipeline (durable-target redirect on immutable hosted
        images), never a direct `pip install --python sys.executable`."""
        import subprocess as _subprocess

        import hermes_cli.web_server as web_server
        from tools import lazy_deps as ld

        # honcho declares pip_dependencies: [honcho-ai]; force it missing.
        monkeypatch.setattr(web_server, "_dependency_importable", lambda dep: False)

        installed = []

        def fake_install_specs(specs, *, timeout=300):
            installed.append(tuple(specs))
            return ld.InstallSpecsResult(
                ok=True, command="uv pip install --target /opt/data/lazy-packages honcho-ai",
                stdout="ok", stderr="",
            )

        monkeypatch.setattr(ld, "install_specs", fake_install_specs)

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
        assert "--target /opt/data/lazy-packages" in pip_rows[0]["command"]
        assert installed == [("honcho-ai",)]





    def test_put_memory_provider_config_writes_config_and_secret(self):
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_config, load_env

        resp = self.client.put(
            "/api/memory/providers/hindsight/config",
            json={
                "values": {
                    "mode": "local_external",
                    "api_url": "http://localhost:8888",
                    "api_key": "hs-test-key",
                    "bank_id": "ben-bank",
                    "recall_budget": "high",
                }
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "active": "hindsight"}
        assert load_config()["memory"]["provider"] == "hindsight"
        assert load_env()["HINDSIGHT_API_KEY"] == "hs-test-key"

        config_path = get_hermes_home() / "hindsight" / "config.json"
        provider_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert provider_config["mode"] == "local_external"
        assert provider_config["api_url"] == "http://localhost:8888"
        assert provider_config["bank_id"] == "ben-bank"
        assert provider_config["recall_budget"] == "high"
        assert "api_key" not in provider_config


    def test_get_memory_provider_config_does_not_return_secret(self):
        self.client.put(
            "/api/memory/providers/hindsight/config",
            json={
                "values": {
                    "mode": "cloud",
                    "api_url": "https://api.hindsight.vectorize.io",
                    "api_key": "secret-value",
                    "bank_id": "hermes",
                    "recall_budget": "mid",
                }
            },
        )

        resp = self.client.get("/api/memory/providers/hindsight/config")

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

    # ── POST /api/chat/image-upload (browser clipboard/drop images) ─────





    # ── Dashboard font override ─────────────────────────────────────────




    def _create_session_with_heavy_fields(self, session_id: str) -> None:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(
                session_id=session_id,
                source="cli",
                system_prompt="# SOUL.md\n" + ("prompt body " * 500),
                model_config={"temperature": 0.7, "notes": "x" * 200},
            )
        finally:
            db.close()






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
        assert invalid.json()["detail"]["errors"] == [
            {"index": 0, "error": "session id is required"}
        ]













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















    def test_update_hermes_returns_docker_guidance_without_spawning(self, monkeypatch):
        import hermes_cli.web_server as web_server

        spawned = False

        def fail_spawn(*_args, **_kwargs):
            nonlocal spawned
            spawned = True
            raise AssertionError("docker update guard should not spawn hermes update")

        # Bypass the managed-externally gate so we reach the docker install check.
        monkeypatch.setattr(web_server, "_dashboard_local_update_managed_externally", lambda: False)
        monkeypatch.setattr(web_server, "detect_install_method", lambda _root: "docker")
        monkeypatch.setattr(web_server, "_spawn_hermes_action", fail_spawn)
        web_server._ACTION_PROCS.pop("hermes-update", None)
        web_server._ACTION_RESULTS.pop("hermes-update", None)

        resp = self.client.post("/api/hermes/update")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["name"] == "hermes-update"
        assert data["pid"] is None
        assert data["error"] == "docker_update_unsupported"
        assert "docker pull nousresearch/hermes-agent:latest" in data["message"]
        assert spawned is False

        status = self.client.get("/api/actions/hermes-update/status")
        assert status.status_code == 200
        status_data = status.json()
        assert status_data["running"] is False
        assert status_data["exit_code"] == 1
        assert status_data["pid"] is None
        assert any("docker pull nousresearch/hermes-agent:latest" in line for line in status_data["lines"])








    def _schema_provider_options(self, key):
        resp = self.client.get("/api/config/schema")
        assert resp.status_code == 200
        return resp.json()["fields"][key]["options"]







    def test_platform_scoped_messaging_env_vars_are_channel_managed(self):
        """Platform-scoped vars belong to a Channels card; cross-cutting
        gateway vars belong to the Keys page.

        Uses credentials as the example: the self-configuring knobs
        (*_HOME_CHANNEL, *_ALLOW_ALL_USERS, …) were deliberately dropped from
        the setup cards and handed back to Keys — see
        tests/hermes_cli/test_setup_hidden_env.py.
        """
        from hermes_cli.web_server import (
            _MESSAGING_KEYS_PAGE_KEYS,
            _build_catalog_entry,
            _channel_managed_env_keys,
        )

        discord = _build_catalog_entry("discord")
        assert "DISCORD_BOT_TOKEN" in discord["env_vars"]
        assert "DISCORD_ALLOWED_USERS" in discord["env_vars"]

        managed = _channel_managed_env_keys()
        assert "DISCORD_BOT_TOKEN" in managed
        assert "MATTERMOST_TOKEN" in managed
        assert "GATEWAY_PROXY_URL" not in managed
        assert "GATEWAY_PROXY_URL" in _MESSAGING_KEYS_PAGE_KEYS



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
        import hermes_cli.web_server as ws
        from hermes_cli.config import load_config, load_env

        with ws._telegram_onboarding_lock:
            ws._telegram_onboarding_pairings.clear()

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

        monkeypatch.setattr(ws, "_telegram_onboarding_request_sync", fake_request)
        ws._ACTION_PROCS.pop("gateway-restart", None)

        def fail_spawn_action(subcommand, name):
            assert subcommand == ["gateway", "restart"]
            assert name == "gateway-restart"
            raise RuntimeError("supervisor unavailable")

        monkeypatch.setattr(ws, "_spawn_hermes_action", fail_spawn_action)

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
        from hermes_cli.web_server import _parse_model_ids

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






    def _seed_custom_provider_with_key(self):
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["providers"] = {
            "acme": {
                "name": "Acme",
                "base_url": "https://llm.acme.corp/v1",
                "model": "acme/m1",
                "api_key": "sk-stored-old",
                "models": {"acme/m1": {}},
            }
        }
        save_config(cfg)


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


    def test_custom_endpoint_save_leaves_a_hand_written_env_ref_alone(self, monkeypatch):
        """``api_key: ${MY_KEY}`` is already safe — don't copy it elsewhere.

        load_config() expands env refs, so such an entry looks like a literal
        secret by the time Save sees it. Migrating it would duplicate the
        user's secret into a second env var they never asked for.
        """
        import yaml

        from hermes_cli.config import custom_endpoint_key_env, get_config_path, get_env_value

        monkeypatch.setenv("MY_PROXY_KEY", "sk-user-managed")
        get_config_path().write_text(
            yaml.safe_dump({
                "providers": {
                    "proxy": {
                        "name": "Proxy",
                        "base_url": "https://llm.example.com/v1",
                        "model": "m",
                        "api_key": "${MY_PROXY_KEY}",
                    }
                },
            }),
            encoding="utf-8",
        )

        self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "proxy",
                "name": "Proxy",
                "base_url": "https://llm.example.com/v1",
                "model": "m",
            },
        )

        raw = yaml.safe_load(get_config_path().read_text(encoding="utf-8"))
        assert raw["providers"]["proxy"]["api_key"] == "${MY_PROXY_KEY}"
        assert not get_env_value(custom_endpoint_key_env("proxy"))


    def test_two_endpoints_on_one_host_keep_separate_credentials(self):
        """Two local servers must not share an .env slot.

        Deriving the env var from the hostname collapses ``127.0.0.1:8000``
        and ``:8001`` onto one name, so saving the second silently overwrites
        the first's key.
        """
        from hermes_cli.config import custom_endpoint_key_env, get_env_value

        for port, key in ((8000, "sk-first"), (8001, "sk-second")):
            self.client.post(
                "/api/providers/custom-endpoints",
                json={
                    "id": f"local-{port}",
                    "name": f"Local {port}",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "model": "m",
                    "api_key": key,
                },
            )

        assert get_env_value(custom_endpoint_key_env("local-8000")) == "sk-first"
        assert get_env_value(custom_endpoint_key_env("local-8001")) == "sk-second"

    def test_custom_endpoint_response_reports_a_key_held_in_env(self):
        """has_api_key must follow key_env, not just a plaintext api_key.

        Reading only ``api_key`` made the panel report "no API key" for every
        endpoint whose credential had been moved to .env.
        """
        resp = self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "proxy",
                "name": "Proxy",
                "base_url": "https://llm.example.com/v1",
                "model": "m",
                "api_key": "sk-in-env",
            },
        )

        endpoint = next(e for e in resp.json()["endpoints"] if e["id"] == "proxy")
        assert endpoint["has_api_key"] is True
        assert "sk-in-env" not in (endpoint["api_key_preview"] or "")

    def test_activating_an_endpoint_carries_its_credential_either_way(self):
        """Activate must work for both key_env and pre-#69449 plaintext entries."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["providers"] = {
            "legacy": {
                "name": "Legacy",
                "base_url": "https://llm.legacy.com/v1",
                "model": "m",
                "api_key": "sk-legacy",
                "models": {"m": {}},
            },
            "modern": {
                "name": "Modern",
                "base_url": "https://llm.modern.com/v1",
                "model": "m",
                "key_env": "MODERN_API_KEY",
                "models": {"m": {}},
            },
        }
        save_config(cfg)

        self.client.post("/api/providers/custom-endpoints/modern/activate", json={})
        model_cfg = load_config()["model"]
        assert model_cfg["key_env"] == "MODERN_API_KEY"
        assert "api_key" not in model_cfg

        self.client.post("/api/providers/custom-endpoints/legacy/activate", json={})
        model_cfg = load_config()["model"]
        assert model_cfg["api_key"] == "sk-legacy"

    def test_get_sessions_rejects_negative_limit(self):
        """limit=-1 must be rejected (422), not passed through to SQLite as
        LIMIT -1 (unbounded) — issue #74316."""
        resp = self.client.get("/api/sessions?limit=-1")
        assert resp.status_code == 422

    def test_get_sessions_rejects_negative_offset(self):
        resp = self.client.get("/api/sessions?offset=-1")
        assert resp.status_code == 422

    def test_get_sessions_positive_limit_still_works(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            for i in range(5):
                db.create_session(session_id=f"pos-limit-{i}", source="cli")
                db.append_message(session_id=f"pos-limit-{i}", role="user", content="hi")
        finally:
            db.close()

        resp = self.client.get("/api/sessions?limit=3&offset=0")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["limit"] == 3
        assert len(payload["sessions"]) == 3

    def test_profiles_sessions_rejects_negative_limit(self):
        """Same guard on the cross-profile aggregate route — negative limit
        previously bypassed the per-profile 500-row clamp entirely."""
        resp = self.client.get("/api/profiles/sessions?limit=-1")
        assert resp.status_code == 422

    def test_profiles_sessions_rejects_negative_offset(self):
        resp = self.client.get("/api/profiles/sessions?offset=-1")
        assert resp.status_code == 422

    def test_profiles_sessions_positive_limit_still_works(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            for i in range(5):
                db.create_session(session_id=f"pos-plimit-{i}", source="cli")
                db.append_message(session_id=f"pos-plimit-{i}", role="user", content="hi")
        finally:
            db.close()

        resp = self.client.get("/api/profiles/sessions?limit=3&offset=0")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["limit"] == 3
        assert len(payload["sessions"]) == 3

    def test_get_session_messages_rejects_negative_limit(self):
        """limit=-1 previously bypassed the documented 500-row clamp because
        min(-1, 500) == -1, which SQLite treats as 'no limit'."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="neg-limit-messages", source="cli")
            for i in range(60):
                db.append_message(
                    session_id="neg-limit-messages", role="user", content=f"msg {i}"
                )
        finally:
            db.close()

        resp = self.client.get("/api/sessions/neg-limit-messages/messages?limit=-1")
        assert resp.status_code == 422

    def test_get_session_messages_rejects_negative_offset(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="neg-offset-messages", source="cli")
            db.append_message(session_id="neg-offset-messages", role="user", content="hi")
        finally:
            db.close()

        resp = self.client.get("/api/sessions/neg-offset-messages/messages?offset=-1")
        assert resp.status_code == 422

    def test_get_session_messages_limit_above_500_is_capped_not_rejected(self):
        """A limit above the documented 500-row cap is silently clamped
        (existing ``min(limit, 500)`` behaviour), not rejected — the request
        still succeeds."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="many-messages", source="cli")
            for i in range(60):
                db.append_message(session_id="many-messages", role="user", content=f"msg {i}")
        finally:
            db.close()

        resp = self.client.get("/api/sessions/many-messages/messages?limit=1000")
        assert resp.status_code == 200
        assert resp.json()["pagination"]["limit"] == 500







# ---------------------------------------------------------------------------
# _build_schema_from_config tests
# ---------------------------------------------------------------------------


class TestBuildSchemaFromConfig:


    def test_overrides_applied(self):
        from hermes_cli.web_server import CONFIG_SCHEMA
        # terminal.backend should be a select with options
        if "terminal.backend" in CONFIG_SCHEMA:
            entry = CONFIG_SCHEMA["terminal.backend"]
            assert entry["type"] == "select"
            assert "options" in entry
            assert "local" in entry["options"]
            assert "vercel_sandbox" in entry["options"]
        runtime_entry = CONFIG_SCHEMA["terminal.vercel_runtime"]
        assert runtime_entry["type"] == "select"
        assert "node24" in runtime_entry["options"]
        assert "python3.13" in runtime_entry["options"]
        assert len(runtime_entry["options"]) >= 3



    def test_timezone_field_is_searchable_select(self):
        """timezone must ship as a searchable, clearable select of IANA ids.

        Desktop renders this via SearchableSelect (Popover + cmdk); the old
        free-text input let users type invalid timezone strings (#68970).
        Invariants, not snapshots: valid IANA entries present, sorted, no
        blank entry server-side (the clear item is client-side via
        ``clearable``), and never empty even without tzdata (UTC fallback).
        """
        from hermes_cli.web_server import CONFIG_SCHEMA, _timezone_options

        entry = CONFIG_SCHEMA["timezone"]
        assert entry["type"] == "select"
        assert entry.get("searchable") is True
        assert entry.get("clearable") is True
        options = entry["options"]
        assert len(options) >= 1
        assert options == sorted(options)
        assert "" not in options
        assert "UTC" in options
        # Fallback path: never returns an empty list.
        assert len(_timezone_options()) >= 1

    def test_dynamic_merge_recomputes_memory_provider_options(self, monkeypatch):
        """The per-request schema merge re-discovers memory providers.

        The import-time _SCHEMA_OVERRIDES freezes the list at server start;
        _schema_with_dynamic_provider_options must recompute it so a provider
        installed mid-session is selectable without a restart.
        """
        from hermes_cli import web_server

        monkeypatch.setattr(web_server, "load_config", lambda: {"memory": {"provider": "honcho"}})
        monkeypatch.setattr(
            web_server,
            "_memory_provider_options",
            lambda: ["", "honcho", "hindsight", "freshly_installed"],
        )

        fields = web_server._schema_with_dynamic_provider_options()

        assert "freshly_installed" in fields["memory.provider"]["options"]
        # The entry is copied, not mutated in place, and keeps its select type.
        assert fields["memory.provider"]["type"] == "select"
        assert web_server.CONFIG_SCHEMA["memory.provider"] is not fields["memory.provider"]







    def test_no_single_field_categories(self):
        """After merging, no category should have just 1 field."""
        from hermes_cli.web_server import CONFIG_SCHEMA
        from collections import Counter
        cats = Counter(e["category"] for e in CONFIG_SCHEMA.values())
        for cat, count in cats.items():
            assert count >= 2, f"Category '{cat}' has only {count} field(s) — should be merged"


# ---------------------------------------------------------------------------
# Config round-trip tests
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Verify config survives GET → edit → PUT without data loss."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN






    def test_round_trip_preserves_schema_invisible_nested_keys(self):
        """Nested keys that aren't in CONFIG_SCHEMA must also survive a
        round-trip. Deep-merge is required — a shallow merge would drop
        ``agent.<custom_key>`` when the frontend sends a partial ``agent``
        dict containing only schema-known sub-fields."""
        from hermes_cli.config import load_config, read_raw_config, save_config

        # Seed config with a key under `agent` that isn't in the schema.
        # Use a sentinel name to avoid colliding with future schema fields.
        save_config({
            "agent": {
                "max_turns": 50,
                "x_dashboard_invisible_test_key": {"nested": "value"},
            },
        })

        # PUT only schema-known agent fields, exactly like the dashboard.
        web_config = self.client.get("/api/config").json()
        web_config.setdefault("agent", {})
        web_config["agent"]["max_turns"] = 75
        # Strip our sentinel so we're sending what the schema-driven form
        # would send.
        web_config["agent"].pop("x_dashboard_invisible_test_key", None)

        resp = self.client.put("/api/config", json={"config": web_config})
        assert resp.status_code == 200

        on_disk = read_raw_config()
        assert on_disk.get("agent", {}).get("max_turns") == 75
        assert on_disk.get("agent", {}).get("x_dashboard_invisible_test_key") \
            == {"nested": "value"}, \
            "Shallow-merge regression: agent.x_dashboard_invisible_test_key " \
            "was wiped when the frontend sent a partial agent dict."

    def test_schema_types_match_config_values(self):
        """Every schema field should have a matching-type value in the config."""
        config = self.client.get("/api/config").json()
        schema_resp = self.client.get("/api/config/schema").json()
        schema = schema_resp["fields"]

        def get_nested(obj, path):
            parts = path.split(".")
            cur = obj
            for p in parts:
                if cur is None or not isinstance(cur, dict):
                    return None
                cur = cur.get(p)
            return cur

        mismatches = []
        for key, entry in schema.items():
            val = get_nested(config, key)
            if val is None:
                continue  # not set in user config — fine
            expected = entry["type"]
            if expected in {"string", "select"} and not isinstance(val, str):
                mismatches.append(f"{key}: expected str, got {type(val).__name__}")
            elif expected == "number" and not isinstance(val, (int, float)):
                mismatches.append(f"{key}: expected number, got {type(val).__name__}")
            elif expected == "boolean" and not isinstance(val, bool):
                mismatches.append(f"{key}: expected bool, got {type(val).__name__}")
            elif expected == "list" and not isinstance(val, list):
                mismatches.append(f"{key}: expected list, got {type(val).__name__}")
        assert not mismatches, "Type mismatches:\n" + "\n".join(mismatches)

    def test_desktop_terminal_font_round_trip_preserves_terminal_config(self):
        """The Appearance picker persists a font without replacing sibling settings."""
        from hermes_cli.config import load_config

        web_config = self.client.get("/api/config").json()
        terminal_before = dict(web_config.get("terminal", {}))
        web_config.setdefault("terminal", {})["font_family"] = "MesloLGS NF"

        response = self.client.put("/api/config", json={"config": web_config})

        assert response.status_code == 200
        persisted = load_config()["terminal"]
        assert persisted["font_family"] == "MesloLGS NF"
        for key, value in terminal_before.items():
            if key != "font_family":
                assert persisted[key] == value

        reloaded = self.client.get("/api/config").json()
        assert reloaded["terminal"]["font_family"] == "MesloLGS NF"


# ---------------------------------------------------------------------------
# New feature endpoint tests
# ---------------------------------------------------------------------------


class TestNewEndpoints:
    """Tests for session detail, logs, cron, skills, tools, raw config, analytics."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
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


    # --- Automation Blueprints ---




    # --- Profiles ---



    def test_profiles_create_builder_mcp_auth_is_profile_scoped(
        self, monkeypatch
    ):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        secret = "profile-builder-secret"
        resp = self.client.post(
            "/api/profiles",
            json={
                "name": "builder-auth",
                "mcp_servers": [
                    {
                        "name": "Bearer Server",
                        "url": "https://example.com/mcp",
                        "auth": "header",
                        "bearer_token": f"Bearer {secret}",
                    },
                    {
                        "name": "oauth-server",
                        "url": "https://example.com/oauth-mcp",
                        "auth": "oauth",
                    },
                    {
                        "name": "local-server",
                        "command": "uvx",
                        "args": ["mcp-server", "--debug"],
                        "env": {"API_KEY": "stdio-secret"},
                    },
                    {
                        "name": "missing-token",
                        "url": "https://example.com/bad",
                        "auth": "header",
                    },
                    {
                        "name": "http-with-env",
                        "url": "https://example.com/bad-env",
                        "env": {"NOT_SUPPORTED": "value"},
                    },
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.json()["mcp_written"] == 3

        root = get_hermes_home()
        profile_dir = root / "profiles" / "builder-auth"
        config_text = (profile_dir / "config.yaml").read_text(encoding="utf-8")
        config = yaml.safe_load(config_text)
        servers = config["mcp_servers"]

        assert sorted(servers) == [
            "Bearer Server",
            "local-server",
            "oauth-server",
        ]
        assert servers["Bearer Server"] == {
            "url": "https://example.com/mcp",
            "headers": {
                "Authorization": "Bearer ${MCP_BEARER_SERVER_API_KEY}",
            },
        }
        assert servers["oauth-server"] == {
            "url": "https://example.com/oauth-mcp",
            "auth": "oauth",
        }
        assert servers["local-server"] == {
            "command": "uvx",
            "args": ["mcp-server", "--debug"],
            "env": {"API_KEY": "stdio-secret"},
        }

        assert secret not in config_text
        profile_env = (profile_dir / ".env").read_text(encoding="utf-8")
        assert f"MCP_BEARER_SERVER_API_KEY={secret}" in profile_env
        assert "Bearer Bearer" not in profile_env
        assert not (root / ".env").exists()



    # --- New profiles endpoints: active / description / model / describe-auto ---












    def test_discord_toolsets_read_and_write_discord_platform(self):
        """Platform-restricted toolsets must not be saved as successful CLI no-ops."""
        from hermes_cli.config import load_config

        listing = {t["name"]: t for t in self.client.get("/api/tools/toolsets").json()}
        assert listing["discord"]["platform"] == "discord"
        assert listing["discord"]["platform_label"] == "Discord"
        assert listing["discord"]["enabled"] is False

        resp = self.client.put("/api/tools/toolsets/discord", json={"enabled": True})
        assert resp.status_code == 200
        assert resp.json() == {
            "ok": True,
            "name": "discord",
            "platform": "discord",
            "enabled": True,
        }

        config = load_config()
        assert "discord" in config["platform_toolsets"]["discord"]
        assert "discord" not in config["platform_toolsets"].get("cli", [])

        listing = {t["name"]: t for t in self.client.get("/api/tools/toolsets").json()}
        assert listing["discord"]["enabled"] is True
        assert listing["discord_admin"]["enabled"] is False

        resp = self.client.put(
            "/api/tools/toolsets/discord_admin", json={"enabled": True}
        )
        assert resp.status_code == 200
        config = load_config()
        assert {"discord", "discord_admin"} <= set(
            config["platform_toolsets"]["discord"]
        )


    def test_get_toolset_config_returns_provider_matrix(self):
        """GET .../config returns provider rows with structured env_vars."""
        resp = self.client.get("/api/tools/toolsets/tts/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "tts"
        assert data["has_category"] is True
        assert isinstance(data["providers"], list)
        assert data["providers"], "tts always has at least the built-in providers"
        # active_provider is part of the contract so the GUI can highlight the
        # provider actually written to config (else it falls back to the first
        # keyless one). It's either None or the name of one listed provider.
        assert "active_provider" in data
        names = {p["name"] for p in data["providers"]}
        assert data["active_provider"] is None or data["active_provider"] in names
        for prov in data["providers"]:
            assert "name" in prov
            assert "is_active" in prov
            assert "env_vars" in prov
            assert isinstance(prov["env_vars"], list)
            for ev in prov["env_vars"]:
                assert "key" in ev
                assert "is_set" in ev
        # active_provider summarizes the first provider flagged is_active
        # (some catalogs list two rows backed by the same config value, e.g.
        # Firecrawl cloud + self-hosted both map to web.backend=firecrawl).
        active = [p["name"] for p in data["providers"] if p["is_active"]]
        if active:
            assert data["active_provider"] == active[0]
        else:
            assert data["active_provider"] is None

    def test_get_toolset_config_reports_truthful_provider_status(self, monkeypatch):
        """Each provider row carries a server-computed readiness `status`.

        Regression: the GUI pilled every zero-env-var row "Ready" — including
        logged-out Nous Subscription rows, xAI TTS without Grok OAuth, and
        never-installed KittenTTS/Piper. The endpoint now reports the honest
        state so keyless ≠ ready.
        """
        import hermes_cli.tools_config as tools_config
        from hermes_cli.nous_account import NousPortalAccountInfo

        # Logged out of Nous Portal → managed subscription rows need sign-in.
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_portal_account_info",
            lambda *a, **k: NousPortalAccountInfo(
                logged_in=False, source="none", fresh=False, paid_service_access=None
            ),
        )
        # No xAI credentials → the Grok OAuth-backed row needs sign-in.
        monkeypatch.setattr(tools_config, "_xai_credentials_present", lambda: False)
        # Local TTS engines not installed → their rows need setup.
        monkeypatch.setattr(tools_config, "_module_installed", lambda name: False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        resp = self.client.get("/api/tools/toolsets/tts/config")
        assert resp.status_code == 200
        data = resp.json()
        by_name = {p["name"]: p for p in data["providers"]}

        valid = {"ready", "needs_keys", "needs_auth", "needs_setup"}
        assert all(p["status"] in valid for p in data["providers"])
        # Genuinely-free keyless row stays Ready.
        assert by_name["Microsoft Edge TTS"]["status"] == "ready"
        # Keyless ≠ ready for gated rows:
        assert by_name["Nous Subscription"]["status"] == "needs_auth"
        assert by_name["xAI TTS"]["status"] == "needs_auth"
        assert by_name["KittenTTS"]["status"] == "needs_setup"
        assert by_name["Piper"]["status"] == "needs_setup"
        # Keyed row with the key unset:
        assert by_name["ElevenLabs"]["status"] == "needs_keys"






    def test_select_managed_nous_provider_reports_needs_nous_auth(self, monkeypatch):
        """Selecting a managed Nous row while logged out flags needs_nous_auth.

        Regression: the GUI PUT wrote browser.cloud_provider + use_gateway
        but skipped the Portal entitlement handshake the CLI runs inline
        (ensure_nous_portal_access) — so the row never activated and nothing
        told the user to sign in. The endpoint now reports the entitlement
        gap so the client can drive the existing Nous OAuth flow.
        """
        from hermes_cli.nous_account import NousPortalAccountInfo

        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_portal_account_info",
            lambda *a, **k: NousPortalAccountInfo(
                logged_in=False, source="none", fresh=False, paid_service_access=None
            ),
        )

        resp = self.client.put(
            "/api/tools/toolsets/browser/provider",
            json={"provider": "Nous Subscription (Browser Use cloud)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["needs_nous_auth"] is True
        assert data["feature"] == "browser"
        # The selection is still persisted — activation is what's gated.
        from hermes_cli.config import load_config
        cfg = load_config()
        assert cfg["browser"]["cloud_provider"] == "browser-use"


    # -- Web capability split (search vs extract backends) ------------------



    def test_select_web_search_backend_matches_runtime_resolution(self, monkeypatch):
        """PUT provider with capability=search writes web.search_backend and the
        runtime search dispatcher resolves to it — while extract is untouched."""
        # Make SearXNG available so both the endpoint gate and the runtime
        # availability check agree it's usable.
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8888")
        # Give extract an explicit shared backend so the assertion isn't
        # hostage to whatever creds exist on the machine running the tests.
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        base = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "Firecrawl Self-Hosted"},
        )
        assert base.status_code == 200

        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "SearXNG", "capability": "search"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["capability"] == "search"

        from hermes_cli.config import load_config
        cfg = load_config()
        assert cfg["web"]["search_backend"] == "searxng"
        # The shared backend selected first must be preserved for extract.
        assert cfg["web"]["backend"] == "firecrawl"

        # The REAL runtime resolution — not a parallel reimplementation.
        from tools.web_tools import _get_extract_backend, _get_search_backend
        assert _get_search_backend() == "searxng"
        assert _get_extract_backend() == "firecrawl"

        # And the config endpoint reports the same split.
        data = self.client.get("/api/tools/toolsets/web/config").json()
        assert data["active_search_backend"] == "searxng"
        assert data["active_extract_backend"] == "firecrawl"


    # -- Terminal execution backend picker ---------------------------------





    def test_terminal_ssh_probe_ready_when_configured(self, monkeypatch):
        """SSH host + user in config.yaml -> ready."""
        import hermes_cli.web_server as web_server
        from hermes_cli.config import load_config, save_config

        monkeypatch.setattr(web_server.shutil, "which", lambda name: None)
        config = load_config()
        config.setdefault("terminal", {})
        config["terminal"]["ssh_host"] = "devbox.example.com"
        config["terminal"]["ssh_user"] = "hermes"
        save_config(config)

        body = self.client.get("/api/tools/terminal/backends").json()
        ssh = next(r for r in body["backends"] if r["name"] == "ssh")
        assert ssh["status"] == "ready"
        assert "hermes@devbox.example.com" in ssh["detail"]









    def test_analytics_usage_includes_skill_breakdown(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(
                session_id="skills-analytics-test",
                source="cli",
                model="anthropic/claude-sonnet-4",
            )
            db.update_token_counts(
                "skills-analytics-test",
                input_tokens=120,
                output_tokens=45,
            )
            db.append_message(
                "skills-analytics-test",
                role="assistant",
                content="Loading and updating skills.",
                tool_calls=[
                    {
                        "function": {
                            "name": "skill_view",
                            "arguments": '{"name":"github-pr-workflow"}',
                        }
                    },
                    {
                        "function": {
                            "name": "skill_manage",
                            "arguments": '{"name":"github-code-review"}',
                        }
                    },
                ],
            )
        finally:
            db.close()

        resp = self.client.get("/api/analytics/usage?days=7")
        assert resp.status_code == 200

        data = resp.json()
        assert data["skills"]["summary"] == {
            "total_skill_loads": 1,
            "total_skill_edits": 1,
            "total_skill_actions": 2,
            "distinct_skills_used": 2,
        }
        assert len(data["skills"]["top_skills"]) == 2

        top_skill = data["skills"]["top_skills"][0]
        assert top_skill["skill"] == "github-pr-workflow"
        assert top_skill["view_count"] == 1
        assert top_skill["manage_count"] == 0
        assert top_skill["total_count"] == 1
        assert top_skill["last_used_at"] is not None


# ---------------------------------------------------------------------------
# Model context length: normalize/denormalize + /api/model/info
# ---------------------------------------------------------------------------


class TestModelContextLength:
    """Tests for model_context_length in normalize/denormalize and /api/model/info."""

    def test_normalize_extracts_context_length_from_dict(self):
        """normalize should surface context_length from model dict."""
        from hermes_cli.web_server import _normalize_config_for_web

        cfg = {
            "model": {
                "default": "anthropic/claude-opus-4.6",
                "provider": "openrouter",
                "context_length": 200000,
            }
        }
        result = _normalize_config_for_web(cfg)
        assert result["model"] == "anthropic/claude-opus-4.6"
        assert result["model_context_length"] == 200000

    def test_normalize_bare_string_model_yields_zero(self):
        """normalize should set model_context_length=0 for bare string model."""
        from hermes_cli.web_server import _normalize_config_for_web

        result = _normalize_config_for_web({"model": "anthropic/claude-sonnet-4"})
        assert result["model"] == "anthropic/claude-sonnet-4"
        assert result["model_context_length"] == 0


    def test_denormalize_writes_context_length_into_model_dict(self):
        """denormalize should write model_context_length back into model dict."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        # Set up disk config with model as a dict
        save_config({
            "model": {"default": "anthropic/claude-opus-4.6", "provider": "openrouter"}
        })

        result = _denormalize_config_from_web({
            "model": "anthropic/claude-opus-4.6",
            "model_context_length": 100000,
        })
        assert isinstance(result["model"], dict)
        assert result["model"]["context_length"] == 100000
        assert "model_context_length" not in result  # virtual field removed


class TestDenormalizeProviderSwitch:
    """The flat Config-page Model field carries no provider info. When the
    model string changes to one served by a different provider, the saved
    provider must follow it (issue #14058)."""

    def test_vendor_slug_switches_off_non_aggregator_provider(self):
        """ollama-local + a vendor/model slug → switch to openrouter and drop
        the stale local base_url (the issue's exact repro)."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({
            "model": {
                "default": "llama3.2",
                "provider": "ollama-local",
                "base_url": "http://localhost:11434/v1",
                "api_mode": "chat_completions",
            }
        })

        result = _denormalize_config_from_web({"model": "google/gemini-2.5-flash"})
        model = result["model"]
        assert model["provider"] == "openrouter"
        assert model["default"] == "google/gemini-2.5-flash"
        # The old ollama-local endpoint must not carry over to openrouter.
        assert not model.get("base_url")


    def test_context_length_override_survives_provider_switch(self):
        """An explicit context-length override must persist alongside a
        provider switch."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({"model": {"default": "llama3.2", "provider": "ollama-local"}})

        result = _denormalize_config_from_web({
            "model": "google/gemini-2.5-flash",
            "model_context_length": 128000,
        })
        model = result["model"]
        assert model["provider"] == "openrouter"
        assert model["context_length"] == 128000


class TestModelContextLengthSchema:
    """Tests for model_context_length placement in CONFIG_SCHEMA."""


    def test_schema_model_context_length_after_model(self):
        """model_context_length should appear immediately after model in schema."""
        from hermes_cli.web_server import CONFIG_SCHEMA
        keys = list(CONFIG_SCHEMA.keys())
        model_idx = keys.index("model")
        assert keys[model_idx + 1] == "model_context_length"

    def test_schema_model_context_length_is_number(self):
        from hermes_cli.web_server import CONFIG_SCHEMA
        entry = CONFIG_SCHEMA["model_context_length"]
        assert entry["type"] == "number"
        assert "category" in entry


class TestModelInfoEndpoint:
    """Tests for GET /api/model/info endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app
        self.client = TestClient(app)


    def test_model_info_with_dict_config(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "load_config", lambda: {
            "model": {
                "default": "anthropic/claude-opus-4.6",
                "provider": "openrouter",
                "context_length": 100000,
            }
        })

        with patch("agent.model_metadata.get_model_context_length", return_value=200000):
            resp = self.client.get("/api/model/info")

        data = resp.json()
        assert data["model"] == "anthropic/claude-opus-4.6"
        assert data["provider"] == "openrouter"
        assert data["auto_context_length"] == 200000
        assert data["config_context_length"] == 100000
        assert data["effective_context_length"] == 100000  # override wins


    def test_model_info_graceful_on_metadata_error(self, monkeypatch):
        """Endpoint should return zeros on import/resolution errors, not 500."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "load_config", lambda: {
            "model": "some/obscure-model"
        })

        with patch("agent.model_metadata.get_model_context_length", side_effect=Exception("boom")):
            resp = self.client.get("/api/model/info")

        assert resp.status_code == 200
        data = resp.json()
        assert data["auto_context_length"] == 0


# ---------------------------------------------------------------------------
# Gateway health probe tests
# ---------------------------------------------------------------------------


class TestProbeGatewayHealth:
    """Tests for _probe_gateway_health() — cross-container gateway detection."""


    def test_probe_uses_configured_short_timeout(self, monkeypatch):
        """The HTTP probe must not fall through to the OS TCP timeout."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 0.75)
        timeouts = []

        def mock_urlopen(req, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            raise TimeoutError("mock timeout")

        monkeypatch.setattr(ws.urllib.request, "urlopen", mock_urlopen)

        alive, body = ws._probe_gateway_health()

        assert alive is False
        assert body is None
        assert timeouts == [0.75, 0.75]




    def test_detailed_fails_falls_back_to_simple_health(self, monkeypatch):
        """If /health/detailed fails, falls back to /health."""
        import hermes_cli.web_server as ws
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)

        call_count = [0]

        def mock_urlopen(req, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("detailed failed")
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        monkeypatch.setattr(ws.urllib.request, "urlopen", mock_urlopen)
        alive, body = ws._probe_gateway_health()
        assert alive is True
        assert body["status"] == "ok"
        assert call_count[0] == 2


class TestStatusRemoteGateway:
    """Tests for /api/status with remote gateway health fallback."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_status_falls_back_to_remote_probe(self, monkeypatch):
        """When local PID check fails and remote probe succeeds, gateway shows running."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_probe_gateway_health", lambda: (True, {
            "status": "ok",
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "pid": 999,
        }))

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is True
        assert data["gateway_pid"] == 999
        assert data["gateway_state"] == "running"
        assert data["gateway_health_url"] == "http://gw:8642"


    def test_status_remote_probe_not_attempted_when_local_pid_found(self, monkeypatch):
        """When local PID check succeeds, the remote probe is never called."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
        })
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        probe_called = [False]
        original = ws._probe_gateway_health

        def track_probe():
            probe_called[0] = True
            return original()

        monkeypatch.setattr(ws, "_probe_gateway_health", track_probe)

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        assert not probe_called[0]


    def test_status_remote_running_null_pid(self, monkeypatch):
        """Remote gateway running but PID not in response — pid should be None."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_probe_gateway_health", lambda: (True, {
            "status": "ok",
        }))

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is True
        assert data["gateway_pid"] is None
        assert data["gateway_state"] == "running"


class TestGatewayBusyReadout:
    """Tests for the NAS busy/drainable readout on /api/status.

    Behaviour contracts (not snapshots): assert how gateway_busy / gateway_drainable
    must RELATE to gateway_running + gateway_state + active_agents, and that every
    field degrades to a safe falsy value when the gateway is down or its status
    file is absent. Liveness must key off gateway_running, NEVER gateway_updated_at.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN


    def test_draining_state_is_neither_busy_nor_drainable(self, monkeypatch):
        """While draining, the gateway is not a fresh begin-drain target, and
        busy is False even with a stale active_agents>0 in the file — the state
        gate dominates."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "draining",
            "platforms": {},
            "active_agents": 3,
        })

        data = self.client.get("/api/status").json()
        assert data["gateway_busy"] is False
        assert data["gateway_drainable"] is False


    def test_active_agents_unparseable_in_file_degrades_to_zero(self, monkeypatch):
        """A corrupt active_agents value in the status file must not 500 or
        produce a spurious busy — it degrades to 0/not-busy."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": "garbage",
        })

        data = self.client.get("/api/status").json()
        assert data["active_agents"] == 0
        assert data["gateway_busy"] is False


class TestGatewayUpdatedAtContract:
    """Contract tests for /api/status ``gateway_updated_at``.

    The field is promised to consumers (web/src/lib/api.ts declares
    ``string | null``) as an RFC3339 timestamp or null — NEVER a number.
    Legacy gateways wrote epoch floats into gateway_state.json and the file
    is hand-editable, so the endpoint must normalize whatever it reads.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    @staticmethod
    def _assert_contract(value):
        """gateway_updated_at is None or a tz-aware-parseable ISO string."""
        from datetime import datetime

        assert not isinstance(value, bool), f"bool leaked: {value!r}"
        assert not isinstance(value, (int, float)), f"number leaked: {value!r}"
        if value is not None:
            assert isinstance(value, str)
            parsed = datetime.fromisoformat(value)
            assert parsed.tzinfo is not None, f"naive timestamp leaked: {value!r}"


    def test_local_runtime_valid_epoch_becomes_iso_string(self, monkeypatch):
        """A plausible legacy epoch value is converted, not dropped."""
        from datetime import datetime, timezone
        import hermes_cli.web_server as ws

        epoch = 1750000000
        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 0,
            "updated_at": epoch,
        })

        value = self.client.get("/api/status").json()["gateway_updated_at"]
        assert isinstance(value, str)
        parsed = datetime.fromisoformat(value)
        assert parsed.tzinfo is not None
        assert parsed == datetime.fromtimestamp(epoch, tz=timezone.utc)


    def test_remote_health_numeric_updated_at_normalized(self, monkeypatch):
        """Cross-container path: the remote /health/detailed body is the
        runtime source, and a numeric updated_at from an older gateway build
        must still come out as string|null."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_probe_gateway_health", lambda: (True, {
            "status": "ok",
            "gateway_state": "running",
            "platforms": {},
            "updated_at": 1750000000.25,
            "pid": 999,
        }))

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is True
        self._assert_contract(data["gateway_updated_at"])
        # A plausible epoch is converted, not nulled.
        assert isinstance(data["gateway_updated_at"], str)


# ---------------------------------------------------------------------------
# Dashboard theme normaliser tests
# ---------------------------------------------------------------------------


class TestNormaliseThemeDefinition:
    """Tests for _normalise_theme_definition() — parses YAML theme files."""


    def test_rejects_non_dict(self):
        from hermes_cli.web_server import _normalise_theme_definition
        assert _normalise_theme_definition("string") is None
        assert _normalise_theme_definition(None) is None
        assert _normalise_theme_definition([1, 2, 3]) is None

    def test_loose_colors_shorthand(self):
        """Bare hex strings under `colors` parse as {hex, alpha=1.0}."""
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({
            "name": "loose",
            "colors": {"background": "#000000", "midground": "#ffffff"},
        })
        assert result is not None
        assert result["palette"]["background"] == {"hex": "#000000", "alpha": 1.0}
        assert result["palette"]["midground"] == {"hex": "#ffffff", "alpha": 1.0}
        # foreground falls back to default (transparent white)
        assert result["palette"]["foreground"]["hex"] == "#ffffff"
        assert result["palette"]["foreground"]["alpha"] == 0.0





class TestDiscoverUserThemes:
    """Tests for _discover_user_themes() — scans ~/.hermes/dashboard-themes/."""

    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli import web_server
        assert web_server._discover_user_themes() == []

    def test_loads_and_normalises_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "ocean.yaml").write_text(
            "name: ocean\n"
            "label: Ocean\n"
            "palette:\n"
            "  background:\n"
            "    hex: \"#0a1628\"\n"
            "    alpha: 1.0\n"
            "layout:\n"
            "  density: spacious\n"
        )
        from hermes_cli import web_server
        results = web_server._discover_user_themes()
        assert len(results) == 1
        assert results[0]["name"] == "ocean"
        assert results[0]["label"] == "Ocean"
        assert results[0]["palette"]["background"]["hex"] == "#0a1628"
        assert results[0]["layout"]["density"] == "spacious"
        # defaults filled in
        assert "fontSans" in results[0]["typography"]


    def test_ignores_transient_profile_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "mine.yaml").write_text("name: mine\n")

        other = tmp_path / "other-profile"
        other.mkdir()

        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_cli import web_server

        token = set_hermes_home_override(str(other))
        try:
            results = web_server._discover_user_themes()
        finally:
            reset_hermes_home_override(token)

        assert [r["name"] for r in results] == ["mine"]


class TestThemeBootstrapCSS:
    """Tests for _render_active_theme_bootstrap_css() and its injection
    into index.html via _serve_index() — the critical-CSS shim that kills
    the default-teal first-paint flash for user YAML themes."""

    @staticmethod
    def _write_theme(hermes_home, name="ocean"):
        themes_dir = hermes_home / "dashboard-themes"
        themes_dir.mkdir(exist_ok=True)
        (themes_dir / f"{name}.yaml").write_text(
            f"name: {name}\n"
            "label: Ocean\n"
            "palette:\n"
            "  background:\n"
            "    hex: \"#0a1628\"\n"
            "  midground:\n"
            "    hex: \"#dbe4f0\"\n"
            "typography:\n"
            "  fontSans: \"Inter, sans-serif\"\n"
            "  baseSize: \"17px\"\n",
            encoding="utf-8",
        )

    def test_user_theme_renders_bundle_vars(self, tmp_path, monkeypatch):
        """Active user theme → style block with ONLY variable names the
        bundle actually consumes (layerVars/typographyVars tokens)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_theme(tmp_path)
        from hermes_cli import web_server
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": "ocean"}}
        )
        css = web_server._render_active_theme_bootstrap_css()
        assert css.startswith('<style id="hermes-theme-bootstrap">')
        assert css.endswith("</style>")
        # Real bundle tokens (web/src/themes/context.tsx + index.css).
        assert "--background-base:#0a1628;" in css
        assert "--midground-base:#dbe4f0;" in css
        assert "--theme-font-sans:Inter, sans-serif;" in css
        assert "--theme-base-size:17px;" in css
        # Names that do NOT exist in the bundle must not be emitted.
        for bogus in ("--color-background", "--color-midground",
                      "--font-sans:", "--font-base-size"):
            assert bogus not in css
        # Canvas rule flows through the variables (never goes stale when
        # applyTheme() rewrites them as inline styles at runtime).
        assert "html,body{background-color:var(--background-base);" in css
        assert "font-family:var(--theme-font-sans);" in css
        assert "font-size:var(--theme-base-size);" in css
        # No baked literal values in the html,body rule.
        assert "#0a1628" not in css.split("html,body")[1]






    @staticmethod
    def _mount_spa_client(tmp_path, monkeypatch):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import hermes_cli.web_server as ws

        dist = tmp_path / "web_dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text(
            "<html><head><title>t</title></head><body>SPA</body></html>",
            encoding="utf-8",
        )
        monkeypatch.setattr(ws, "WEB_DIST", dist)
        spa_app = FastAPI()
        ws.mount_spa(spa_app)
        return TestClient(spa_app)

    def test_serve_index_injects_bootstrap_for_user_theme(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_theme(tmp_path)
        import hermes_cli.web_server as ws
        monkeypatch.setattr(
            ws, "load_config", lambda: {"dashboard": {"theme": "ocean"}}
        )
        client = self._mount_spa_client(tmp_path, monkeypatch)
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert '<style id="hermes-theme-bootstrap">' in resp.text
        assert "--background-base:#0a1628;" in resp.text
        # Injected inside <head>, before the closing tag.
        head = resp.text.split("</head>")[0]
        assert "hermes-theme-bootstrap" in head




class TestNormaliseThemeExtensions:
    """Tests for the extended normaliser fields (assets, customCSS,
    componentStyles, layoutVariant) — the surfaces themes use to reskin
    the dashboard without shipping code."""





    def test_custom_css_passthrough_and_capped(self):
        from hermes_cli.web_server import _normalise_theme_definition
        # Small CSS passes through verbatim.
        r = _normalise_theme_definition({
            "name": "t",
            "customCSS": "body { color: red; }",
        })
        assert r["customCSS"] == "body { color: red; }"

        # 40 KiB of CSS gets clipped to the 32 KiB cap.
        huge = "/* x */ " * (40 * 1024 // 8 + 10)
        r2 = _normalise_theme_definition({"name": "t", "customCSS": huge})
        assert len(r2["customCSS"]) <= 32 * 1024


    def test_component_styles_per_bucket(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "t",
            "componentStyles": {
                "card": {
                    "clipPath": "polygon(0 0, 100% 0, 100% 100%, 0 100%)",
                    "boxShadow": "inset 0 0 0 1px red",
                    "bad prop!": "ignored",  # non-alnum prop rejected
                },
                "header": {"background": "linear-gradient(red, blue)"},
                "rogueBucket": {"foo": "bar"},  # not a known bucket — rejected
            },
        })
        assert r["componentStyles"]["card"] == {
            "clipPath": "polygon(0 0, 100% 0, 100% 100%, 0 100%)",
            "boxShadow": "inset 0 0 0 1px red",
        }
        assert r["componentStyles"]["header"]["background"].startswith("linear-gradient")
        assert "rogueBucket" not in r["componentStyles"]




class TestDeleteSessionEndpoint:
    """Tests for ``DELETE /api/sessions/{session_id}`` — the single-row delete
    behind the desktop sidebar's per-session delete.

    The desktop optimistically removes the row, then RESTORES it on any error
    and surfaces the message. So a 404 on a row that is already gone (reaped by
    empty-session hygiene, or removed by a concurrent client — both common amid
    /goal + auto-compression churn that leaves transient empty rows) resurrected
    a ghost row and showed "session not found". DELETE must be idempotent and
    resolve ids like every other session endpoint.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )

        self.auth_client = TestClient(app)
        self.auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _seed(self, ids):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            for sid in ids:
                db.create_session(session_id=sid, source="cli")
        finally:
            db.close()

    def _exists(self, sid) -> bool:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            return db.get_session(sid) is not None
        finally:
            db.close()


    def test_delete_absent_session_is_idempotent(self):
        # PREMISE / regression: deleting a row that no longer exists must NOT
        # 404 — the desktop would resurrect the ghost row and show
        # "session not found". DELETE's contract is "ensure it's gone".
        resp = self.auth_client.delete("/api/sessions/never_existed")
        assert resp.status_code == 200
        assert resp.json().get("ok") is True


class TestBulkDeleteSessionsEndpoint:
    """Tests for ``POST /api/sessions/bulk-delete`` — backs the
    dashboard's "Delete N selected" flow on the sessions page.

    Locks in four things:

    1. Route-ordering: ``/api/sessions/bulk-delete`` must shadow the
       templated ``/api/sessions/{session_id}`` route below it (see
       the block comment in ``hermes_cli/web_server.py``).
    2. Behaviour parity with :meth:`SessionDB.delete_sessions` — real
       deleted count, archive/active sessions deleted on explicit
       selection.
    3. The 500-ID payload cap is enforced.
    4. Auth gating (issue #19533 contract).
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )

        self.client = TestClient(app)
        self.auth_client = TestClient(app)
        self.auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _seed(self, ids):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            for sid in ids:
                db.create_session(session_id=sid, source="cli")
        finally:
            db.close()


    def test_deletes_listed_sessions_only(self):
        from hermes_state import SessionDB

        self._seed(["a", "b", "c"])
        resp = self.auth_client.post(
            "/api/sessions/bulk-delete", json={"ids": ["a", "b"]}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 2}

        db = SessionDB()
        try:
            assert db.get_session("a") is None
            assert db.get_session("b") is None
            assert db.get_session("c") is not None
        finally:
            db.close()




    def test_route_order_not_shadowed_by_session_id(self):
        """Pin the route-ordering contract: ``POST /api/sessions/bulk-delete``
        must hit the bulk handler, not be re-interpreted via the
        templated ``/api/sessions/{session_id}`` family. Concretely the
        response carries our ``ok`` + ``deleted`` keys."""
        resp = self.auth_client.post(
            "/api/sessions/bulk-delete", json={"ids": []}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True
        assert "deleted" in body, (
            "If this assertion fails, /api/sessions/bulk-delete is "
            "being shadowed by /api/sessions/{session_id} — check "
            "registration order in hermes_cli/web_server.py."
        )


class TestDeleteEmptySessionsEndpoint:
    """Tests for ``GET /api/sessions/empty/count`` and
    ``DELETE /api/sessions/empty`` — the bulk-delete endpoints backing
    the dashboard's "Delete empty" button.

    Locks in three things the implementation has to get right:

    1. Route-ordering: the literal ``/api/sessions/empty[/count]`` paths
       must shadow the templated ``/api/sessions/{session_id}`` route
       above them. A regression here would route ``DELETE /api/sessions/
       empty`` to the single-session handler with ``session_id="empty"``
       (which 404s instead of bulk-deleting).
    2. Behaviour parity with :meth:`SessionDB.delete_empty_sessions`:
       active sessions and archived sessions are both preserved.
    3. Auth gating: both routes require the session token like every
       other ``/api/*`` endpoint (issue #19533 contract).
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        # Pin the SessionDB to the isolated HERMES_HOME so each test
        # starts with a clean state.db.
        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )

        self.client = TestClient(app)
        self.auth_client = TestClient(app)
        self.auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _seed(self):
        """Build the standard test corpus:

        * ``empty1`` / ``empty2`` — ended, no messages → should delete
        * ``hasmsg``  — ended, has one message → must survive
        * ``live``    — un-ended, empty → must survive (active)
        * ``archived``— ended, empty, archived → must survive
        """
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="empty1", source="cli")
            db.end_session("empty1", end_reason="done")
            db.create_session(session_id="empty2", source="cli")
            db.end_session("empty2", end_reason="done")

            db.create_session(session_id="hasmsg", source="cli")
            db.append_message("hasmsg", role="user", content="hello")
            db.end_session("hasmsg", end_reason="done")

            db.create_session(session_id="live", source="cli")

            db.create_session(session_id="archived", source="cli")
            db.end_session("archived", end_reason="done")
            db.set_session_archived("archived", True)
        finally:
            db.close()


    def test_delete_endpoint_requires_auth(self):
        """DELETE /api/sessions/empty must 401 without the session token.

        Regression guard for issue #19533 — the bulk-delete is a strictly
        destructive primitive, the middleware must gate it even if a
        future refactor introduces a non-auth path."""
        resp = self.client.delete("/api/sessions/empty")
        assert resp.status_code == 401


    def test_delete_returns_count_and_removes_only_empties(self):
        """DELETE returns the deleted count and removes only the
        empty-ended-unarchived rows — same shape contract as the
        DB-level method's unit tests."""
        from hermes_state import SessionDB

        self._seed()
        resp = self.auth_client.delete("/api/sessions/empty")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 2}

        db = SessionDB()
        try:
            assert db.get_session("empty1") is None
            assert db.get_session("empty2") is None
            # Survivors: hasmsg has a message, live is active, archived
            # is archived. All three must still be there.
            assert db.get_session("hasmsg") is not None
            assert db.get_session("live") is not None
            assert db.get_session("archived") is not None
            # And the count endpoint now reports 0.
            assert db.count_empty_sessions() == 0
        finally:
            db.close()


    def test_route_order_empty_not_shadowed_by_session_id(self):
        """Pin the route-ordering contract: ``DELETE /api/sessions/empty``
        must hit the bulk handler, not the templated single-session
        handler (which would 404 because no session has id 'empty').

        Concretely: a request against the bulk path on an EMPTY corpus
        returns ``{ok: True, deleted: 0}``. If the templated route were
        winning, we'd see 404 ("Session not found") instead.
        """
        resp = self.auth_client.delete("/api/sessions/empty")
        assert resp.status_code == 200
        body = resp.json()
        assert "deleted" in body, (
            "If this assertion fails, the literal /api/sessions/empty "
            "route is being shadowed by the templated /api/sessions/"
            "{session_id} route — check registration order in "
            "hermes_cli/web_server.py."
        )


class TestPluginAPIAuth:
    """Tests that plugin API routes require the session token (issue #19533)."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home, _install_example_plugin):
        """Create a TestClient without the session token header.

        Pulls in ``_install_example_plugin`` so ``test_plugin_route_allows_auth``
        has the ``/api/plugins/example/hello`` endpoint available — the
        example plugin is no longer a bundled plugin, so the fixture
        installs it into the per-test ``HERMES_HOME``.
        """
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.auth_client = TestClient(app)
        self.auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN


    def test_plugin_route_allows_auth(self):
        """Plugin API routes should work with a valid session token.

        Uses ``/api/plugins/example/hello`` from the example-dashboard
        test fixture (installed into HERMES_HOME by the class-level
        ``_install_example_plugin`` fixture) — a stable, side-effect-free
        GET that's only loaded for tests. With a valid token the handler
        should run (200); without one the middleware should 401 before
        the handler is reached.
        """
        # Without auth: middleware blocks before reaching the handler.
        resp = self.client.get("/api/plugins/example/hello")
        assert resp.status_code == 401

        # With auth: handler runs.
        resp = self.auth_client.get("/api/plugins/example/hello")
        assert resp.status_code == 200


    def test_plugin_patch_requires_auth(self):
        """Plugin PATCH routes should return 401 without a valid session token.

        PATCH is the mutation method most commonly used by the dashboard for
        kanban task edits — explicitly cover it so a future middleware
        regression that whitelists non-GET methods can't sneak through.
        """
        resp = self.client.patch(
            "/api/plugins/kanban/tasks/t_fake",
            json={"title": "renamed"},
        )
        assert resp.status_code == 401


    def test_non_kanban_plugin_route_requires_auth(self):
        """Auth must be plugin-agnostic, not kanban-specific.

        The middleware fix is at the gate level (no per-plugin allowlist),
        so any plugin's API surface — kanban, hermes-achievements, future
        plugins — must require the session token. Hit a non-kanban plugin
        path to lock that in.
        """
        # Real plugin path (hermes-achievements is loaded by default).
        resp = self.client.get("/api/plugins/hermes-achievements/overview")
        assert resp.status_code == 401
        # Same for an arbitrary plugin namespace that doesn't even exist —
        # the middleware should 401 before routing decides 404, so an
        # attacker can't fingerprint plugin names by status codes.
        resp = self.client.get("/api/plugins/_definitely_not_a_plugin_/anything")
        assert resp.status_code == 401

    def test_plugin_websocket_unaffected_by_http_middleware(self):
        """The kanban /events WebSocket has its own ``?token=`` check;
        the HTTP middleware change must not start gating WS upgrades.

        Starlette doesn't run HTTP middleware on WebSocket upgrades anyway,
        but pin the behavior so a future refactor that moves auth into a
        shared layer can't silently break the WS auth contract.
        """
        from starlette.websockets import WebSocketDisconnect

        # Without a token the WS endpoint must close the upgrade itself
        # (its own _check_ws_token), NOT 401 from the HTTP middleware.
        try:
            with self.client.websocket_connect(
                "/api/plugins/kanban/events"
            ):
                pass  # if we got here without disconnect, the WS accepted us
        except WebSocketDisconnect:
            pass  # expected — WS endpoint rejected via its own check
        except Exception:
            # The kanban plugin may not be mounted in this test environment,
            # in which case the route doesn't exist at all (3xx/4xx during
            # upgrade). That's fine for this regression — it only matters
            # that the HTTP middleware didn't start intercepting WS upgrades.
            pass


class TestDashboardPluginManifestExtensions:
    """Tests for the extended plugin manifest fields (tab.override,
    tab.hidden, slots) read by _discover_dashboard_plugins()."""

    def _write_plugin(self, tmp_path, name, manifest):
        import json
        plug_dir = tmp_path / "plugins" / name / "dashboard"
        plug_dir.mkdir(parents=True)
        (plug_dir / "manifest.json").write_text(json.dumps(manifest))
        return plug_dir

    def test_override_and_hidden_carried_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_plugin(tmp_path, "skin-home", {
            "name": "skin-home",
            "label": "Skin Home",
            "tab": {"path": "/skin-home", "override": "/", "hidden": True},
            "slots": ["sidebar", "header-left"],
            "entry": "dist/index.js",
        })
        from hermes_cli import web_server
        # Bust the process-level cache so the test plugin is picked up.
        web_server._dashboard_plugins_cache = None
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "skin-home")
        assert entry["tab"]["override"] == "/"
        assert entry["tab"]["hidden"] is True
        assert entry["slots"] == ["sidebar", "header-left"]

    def test_user_plugins_ignore_profile_home_override(self, tmp_path, monkeypatch):
        """Regression: user dashboard extensions are a dashboard-owned asset
        (like theme YAML), so they must stay visible after a context-local
        HERMES_HOME override scopes a request to another profile."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        launch_home = tmp_path / "launch"
        launch_home.mkdir()
        self._write_plugin(launch_home, "skin-home", {
            "name": "skin-home",
            "label": "Skin Home",
            "tab": {"path": "/skin-home"},
            "entry": "dist/index.js",
        })
        other = tmp_path / "other-profile"
        other.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(launch_home))
        from hermes_cli import web_server
        token = set_hermes_home_override(str(other))
        try:
            plugins = web_server._discover_dashboard_plugins()
        finally:
            reset_hermes_home_override(token)
        assert any(p["name"] == "skin-home" for p in plugins)




# ---------------------------------------------------------------------------
# /api/pty WebSocket — terminal bridge for the dashboard "Chat" tab.
#
# These tests drive the endpoint with a tiny fake command (typically ``cat``
# or ``sh -c 'printf …'``) instead of the real ``hermes --tui`` binary.  The
# endpoint resolves its argv through ``_resolve_chat_argv``, so tests
# monkeypatch that hook.
# ---------------------------------------------------------------------------

import sys


skip_on_windows = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="PTY bridge is POSIX-only"
)


@skip_on_windows
class TestPtyWebSocket:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        from starlette.testclient import TestClient

        import hermes_cli.web_server as ws

        # Avoid exec'ing the actual TUI in tests: every test below installs
        # its own fake argv via ``ws._resolve_chat_argv``.
        self.ws_module = ws
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        ws.app.state.pty_active_session_files = {}
        self.token = ws._SESSION_TOKEN
        self.client = TestClient(ws.app)

    def _url(self, token: str | None = None, **params: str) -> str:
        tok = token if token is not None else self.token
        # TestClient.websocket_connect takes the path; it reconstructs the
        # query string, so we pass it inline.
        from urllib.parse import urlencode

        q = {"token": tok, **params}
        return f"/api/pty?{urlencode(q)}"



    def test_tui_python_command_uses_child_path(self, tmp_path):
        """Bare Python commands are resolved from the TUI child's PATH."""
        import hermes_cli.main as main_mod

        command = f"hermes-review-python{Path(sys.executable).suffix}"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        executable = bin_dir / command
        # copy2, not os.link: tmp_path may sit on a different filesystem than
        # the venv (tmpfs /tmp vs disk home) where hard links raise EXDEV.
        shutil.copy2(sys.executable, executable)
        env = {
            "HERMES_CWD": str(tmp_path),
            "HERMES_PYTHON": command,
            "PATH": str(bin_dir),
        }

        main_mod._apply_tui_python_env(env)

        assert env["HERMES_PYTHON"] == command





    def test_resolve_chat_argv_async_uses_worker_thread(self, monkeypatch):
        captured: dict = {}

        def fake_resolve(resume=None, sidecar_url=None, profile=None):
            captured["resume"] = resume
            captured["sidecar_url"] = sidecar_url
            captured["profile"] = profile
            return (["node", "dist/entry.js"], "/tmp/ui-tui", {"NODE_ENV": "production"})

        async def fake_to_thread(fn, *args, **kwargs):
            captured["thread_fn"] = fn
            captured["thread_args"] = args
            captured["thread_kwargs"] = kwargs
            return fn(*args, **kwargs)

        monkeypatch.setattr(self.ws_module, "_resolve_chat_argv", fake_resolve)
        monkeypatch.setattr(self.ws_module.asyncio, "to_thread", fake_to_thread)

        argv, cwd, env = asyncio.run(
            self.ws_module._resolve_chat_argv_async(
                resume="sess-42",
                sidecar_url="ws://127.0.0.1:9119/api/pub?channel=abc",
                profile="worker",
            )
        )

        assert callable(captured["thread_fn"])
        assert captured["thread_args"] == ()
        assert captured["thread_kwargs"] == {
            "resume": "sess-42",
            "sidecar_url": "ws://127.0.0.1:9119/api/pub?channel=abc",
            "profile": "worker",
        }
        assert argv == ["node", "dist/entry.js"]
        assert cwd == "/tmp/ui-tui"
        assert env == {"NODE_ENV": "production"}
        assert captured["resume"] == "sess-42"
        assert captured["sidecar_url"] == "ws://127.0.0.1:9119/api/pub?channel=abc"
        assert captured["profile"] == "worker"


    def _assert_pty_propagates(self, monkeypatch, raising_resolver, *, profile=None, expect_detail=None):
        """Drive /api/pty with a resolver that raises, and assert the error
        propagates through the real _resolve_chat_argv_async -> asyncio.to_thread
        -> lock -> re-raise chain into pty_ws's handler: the "Chat unavailable"
        notice is sent and the socket closes with code 1011 (the stable
        contract — we assert the close code, not the exact notice wording)."""
        from starlette.websockets import WebSocketDisconnect

        # Patch the REAL resolver so the whole wrapper/to_thread/lock chain runs.
        monkeypatch.setattr(self.ws_module, "_resolve_chat_argv", raising_resolver)

        url = self._url(profile=profile) if profile else self._url()
        with self.client.websocket_connect(url) as conn:
            notice = conn.receive_text()
            with pytest.raises(WebSocketDisconnect) as exc:
                conn.receive_text()
        assert "Chat unavailable" in notice
        assert exc.value.code == 1011
        if expect_detail is not None:
            assert expect_detail in notice






    def test_unavailable_platform_closes_with_message(self, monkeypatch):
        from hermes_cli.pty_bridge import PtyUnavailableError

        def _raise(argv, **kwargs):
            raise PtyUnavailableError("pty missing for tests")

        monkeypatch.setattr(
            self.ws_module,
            "_resolve_chat_argv",
            lambda resume=None, sidecar_url=None, profile=None: (["/bin/cat"], None, None),
        )
        # Patch PtyBridge.spawn at the web_server module's binding.
        import hermes_cli.web_server as ws_mod

        monkeypatch.setattr(ws_mod.PtyBridge, "spawn", classmethod(lambda cls, *a, **k: _raise(*a, **k)))

        with self.client.websocket_connect(self._url()) as conn:
            # Expect a final text frame with the error message, then close.
            msg = conn.receive_text()
            assert "pty missing" in msg or "unavailable" in msg.lower() or "pty" in msg.lower()



    def test_pub_broadcasts_to_events_subscribers(self):
        """A frame handed to _broadcast_event is sent verbatim to every
        subscriber registered on that channel — and not to subscribers on
        other channels.

        This drives the broadcast unit directly under asyncio rather than
        round-tripping through Starlette's TestClient WebSocket portal. The
        portal version was flaky under heavy parallel CI load: the broadcast
        had to traverse two nested threaded portals within a 10s wall-clock
        budget, and a starved ASGI thread occasionally blew that budget even
        though the server logic was correct. Testing _broadcast_event with
        fake subscribers removes the scheduling surface entirely while
        asserting the exact fan-out contract.
        """
        import asyncio
        from hermes_cli import web_server as ws_mod

        class _FakeSub:
            def __init__(self):
                self.sent: list[str] = []

            async def send_text(self, payload: str) -> None:
                self.sent.append(payload)

        app = ws_mod.app

        async def _run():
            sub_a1 = _FakeSub()
            sub_a2 = _FakeSub()
            sub_other = _FakeSub()
            frame = '{"type":"tool.start","payload":{"tool_id":"t1"}}'

            event_channels, event_lock = ws_mod._get_event_state(app)
            # Register two subscribers on the target channel and one on a
            # different channel, exactly as the /api/events handler does.
            async with event_lock:
                event_channels.setdefault("broadcast-test", set()).update(
                    {sub_a1, sub_a2}
                )
                event_channels.setdefault("other-channel", set()).add(sub_other)
            try:
                await ws_mod._broadcast_event(app, "broadcast-test", frame)
            finally:
                async with event_lock:
                    event_channels.pop("broadcast-test", None)
                    event_channels.pop("other-channel", None)

            return sub_a1, sub_a2, sub_other, frame

        sub_a1, sub_a2, sub_other, frame = asyncio.run(_run())

        # Every subscriber on the channel got the frame verbatim, exactly once.
        assert sub_a1.sent == [frame]
        assert sub_a2.sent == [frame]
        # A subscriber on a different channel got nothing.
        assert sub_other.sent == []


def test_resolve_chat_argv_injects_gateway_ws_url(monkeypatch):
    import hermes_cli.main as cli_main
    import hermes_cli.web_server as ws

    monkeypatch.setattr(
        cli_main,
        "_make_tui_argv",
        lambda *_args, **_kwargs: (["node", "fake-tui.js"], Path("/tmp")),
    )
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)

    _argv, _cwd, env = ws._resolve_chat_argv()

    assert env is not None
    gateway_url = env.get("HERMES_TUI_GATEWAY_URL", "")
    assert gateway_url.startswith("ws://127.0.0.1:9119/api/ws?")
    assert "token=" in gateway_url


class TestDashboardPluginStaticAssetAllowlist:
    """``/dashboard-plugins/<name>/<path>`` is unauthenticated by design —
    the SPA loads plugin JS via ``<script src>`` and CSS via
    ``<link href>``, neither of which can attach a custom auth header.
    Instead the route restricts file types to the browser-asset
    allowlist (JS/CSS/JSON/images/fonts) so that user-installed
    plugins shipping a ``plugin_api.py`` backend module don't leak
    their Python source to anyone reachable on the loopback port.

    Regression test for the dashboard pentest finding filed alongside
    the ``web-pentest`` skill (PR #32265 / issue #32267).
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home, _install_example_plugin):
        """Create a TestClient and install the example-dashboard fixture.

        The static-asset allowlist tests need a plugin to point at —
        they verify that ``/dashboard-plugins/example/manifest.json``
        is served while ``plugin_api.py`` and ``__pycache__/*.pyc``
        from the same directory are not. Since the example plugin is
        no longer bundled, ``_install_example_plugin`` lays it down in
        the per-test ``HERMES_HOME`` user-plugins dir.
        """
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app

        self.client = TestClient(app)

    def test_python_source_is_404(self):
        """The example plugin's ``plugin_api.py`` must NOT be served as
        a static asset, even though the file exists under the plugin's
        dashboard directory. Suffix not in the allowlist → 404."""
        resp = self.client.get("/dashboard-plugins/example/plugin_api.py")
        assert resp.status_code == 404


    def test_manifest_json_still_served(self):
        """JSON files remain browser-fetchable — manifests, localized
        data, source maps, etc. all sit in this bucket."""
        resp = self.client.get("/dashboard-plugins/example/manifest.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        # And the body is actually the manifest, not the SPA fallback.
        body = resp.json()
        assert body.get("name") == "example"


    def test_path_traversal_still_blocked(self):
        """The allowlist is on top of the existing ``.resolve()`` /
        ``is_relative_to()`` check — a ``.js`` named file at an
        out-of-base path is still rejected as traversal, not served."""
        resp = self.client.get(
            "/dashboard-plugins/example/..%2Fplugin_api.py"
        )
        # 403 traversal-blocked OR 404 (depending on URL decode order)
        # — never 200.
        assert resp.status_code in (403, 404)


def _fake_httpx_client(*, status: int | None = None, raise_exc: bool = False):
    """Build a drop-in for httpx.Client whose .get() returns a canned status
    (or raises a transport error). Patched in for the credential-validate probe
    so tests never touch the network."""
    class _Resp:
        def __init__(self, code):
            self.status_code = code

        @property
        def is_success(self):
            return 200 <= self.status_code < 300

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            if raise_exc:
                raise RuntimeError("connection refused")
            return _Resp(status)

    return _Client


class TestValidateProviderCredential:
    """Live-probe credential validation (/api/providers/validate)."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _post(self, key, value):
        return self.client.post("/api/providers/validate", json={"key": key, "value": value})



    def test_network_error_is_unreachable_not_blocking(self, monkeypatch):
        monkeypatch.setattr("httpx.Client", _fake_httpx_client(raise_exc=True))
        data = self._post("OPENROUTER_API_KEY", "sk-real").json()
        assert data["ok"] is False and data["reachable"] is False



    def test_local_endpoint_forwards_api_key_as_bearer(self, monkeypatch):
        """A custom endpoint that gates /v1/models behind auth must still
        enumerate models: the optional api_key is sent as a Bearer header so the
        probe doesn't come back empty (the desktop loop's root cause)."""
        captured = {}

        class _Resp:
            status_code = 200
            is_success = True

            def json(self):
                return {"data": [{"id": "gpt-oss-120b"}]}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, *a, headers=None, **k):
                captured["url"] = url
                captured["headers"] = headers
                return _Resp()

        monkeypatch.setattr("httpx.Client", _Client)

        resp = self.client.post(
            "/api/providers/validate",
            json={
                "key": "OPENAI_BASE_URL",
                "value": "https://text.example.com/v1",
                "api_key": "sk-secret",
            },
        )
        data = resp.json()
        assert data["ok"] is True and data["reachable"] is True
        assert data["models"] == ["gpt-oss-120b"]
        assert captured["url"] == "https://text.example.com/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-secret"}


class TestDesktopCronTicker:
    """The dashboard backend fires cron jobs itself only when desktop-spawned."""

    def _client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app

        return TestClient(app)

    def test_ticker_runs_when_desktop(self, monkeypatch, _isolate_hermes_home):
        import threading
        import cron.scheduler as sched

        called = threading.Event()
        monkeypatch.setattr(sched, "tick", lambda *a, **k: called.set())
        monkeypatch.setenv("HERMES_DESKTOP", "1")

        with self._client():
            assert called.wait(3.0), "expected cron tick under HERMES_DESKTOP=1"


class TestServeIndexMissingIndex:
    """_serve_index must not raise per-request when index.html vanishes
    (partial build, wiped dist) after mount_spa saw an existing dist dir.
    It should return the same JSON 404 payload mount_spa emits for a
    fully-missing dist."""

    @staticmethod
    def _client_with_dist(tmp_path, monkeypatch, *, write_index: bool):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import hermes_cli.web_server as ws

        dist = tmp_path / "web_dist"
        (dist / "assets").mkdir(parents=True)
        if write_index:
            (dist / "index.html").write_text(
                "<html><head></head><body>SPA</body></html>", encoding="utf-8"
            )
        monkeypatch.setattr(ws, "WEB_DIST", dist)
        monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
        spa_app = FastAPI()
        ws.mount_spa(spa_app)
        return TestClient(spa_app), dist

    def test_missing_index_inside_existing_dist_returns_json_404(
        self, tmp_path, monkeypatch
    ):
        client, _dist = self._client_with_dist(
            tmp_path, monkeypatch, write_index=False
        )
        for route in ("/", "/chat"):
            resp = client.get(route)
            assert resp.status_code == 404
            assert resp.json()["error"] == (
                "Frontend not built. Run: cd web && npm run build"
            )

    def test_index_deleted_after_mount_returns_json_404(self, tmp_path, monkeypatch):
        client, dist = self._client_with_dist(tmp_path, monkeypatch, write_index=True)
        assert client.get("/chat").status_code == 200  # healthy first
        (dist / "index.html").unlink()
        resp = client.get("/chat")
        assert resp.status_code == 404
        assert "Frontend not built" in resp.json()["error"]
        # And recovers once the index reappears (e.g. a rebuild finished).
        (dist / "index.html").write_text(
            "<html><head></head><body>SPA-rebuilt</body></html>", encoding="utf-8"
        )
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert "SPA-rebuilt" in resp.text


class TestDashboardComponentHealth:
    """Component-health rollup: error middleware, /api/status components, self-test."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        import hermes_cli.web_server as ws

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        # Fresh state holder per test so counters don't leak across tests.
        monkeypatch.setattr(ws, "DASHBOARD_HEALTH", ws.DashboardHealth())
        self.ws = ws
        self.client = TestClient(ws.app, raise_server_exceptions=False)
        self.client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN

    # -- middleware -------------------------------------------------------




    # -- /api/status components ------------------------------------------




    def test_public_component_payload_carries_no_secret_bearing_fields(self):
        """PUBLIC_API_PATHS contract: counts/enums only — no paths/messages."""
        self.ws.DASHBOARD_HEALTH.record_error("RuntimeError", "/api/secret-route?token=abc")
        resp = self.client.get("/api/status")
        payload = json.dumps(resp.json()["components"])
        assert "secret-route" not in payload
        assert "token=abc" not in payload
        assert "last_error_path" not in payload
        assert "last_error_type" not in payload
        assert "kaboom" not in payload

    # -- self-test ---------------------------------------------------------

    def test_selftest_records_failure_on_500(self, monkeypatch):
        httpx = pytest.importorskip("httpx")

        class _FakeResponse:
            status_code = 500

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, *args, **kwargs):
                return _FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
        asyncio.run(self.ws._dashboard_selftest_once())
        assert self.ws.DASHBOARD_HEALTH.selftest_status == "failing"
        assert self.ws.DASHBOARD_HEALTH.selftest_http_status == 500
        assert self.ws.DASHBOARD_HEALTH.snapshot()["status"] == "degraded"
