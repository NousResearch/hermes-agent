"""Tests for hermes_cli.web_server and related config utilities."""

import asyncio
import os
import json
import shutil
import sys
import threading
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

    def test_updates_changed_vars(self, tmp_path):
        """reload_env() updates vars whose value changed on disk."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_RELOAD_VAR=old_value\n")
        with patch.dict(reload_env.__globals__, {"get_env_path": lambda: env_file}):
            os.environ["TEST_RELOAD_VAR"] = "old_value"
            # Now change the file
            env_file.write_text("TEST_RELOAD_VAR=new_value\n")
            count = reload_env()
            assert count >= 1
            assert os.environ.get("TEST_RELOAD_VAR") == "new_value"
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

    def test_does_not_remove_unknown_vars(self, tmp_path):
        """reload_env() preserves non-Hermes env vars even when absent from .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("")
        with patch.dict(reload_env.__globals__, {"get_env_path": lambda: env_file}):
            os.environ["MY_CUSTOM_UNRELATED_VAR"] = "keep_me"
            reload_env()
            assert os.environ.get("MY_CUSTOM_UNRELATED_VAR") == "keep_me"
        os.environ.pop("MY_CUSTOM_UNRELATED_VAR", None)


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
        import importlib
        import hermes_cli.web_server as ws

        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-seeded-token")
        try:
            importlib.reload(ws)
            assert ws._SESSION_TOKEN == "desktop-seeded-token"
        finally:
            monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
            importlib.reload(ws)

    def test_falls_back_to_random_token(self, monkeypatch):
        import importlib
        import hermes_cli.web_server as ws

        monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
        importlib.reload(ws)

        assert ws._SESSION_TOKEN and len(ws._SESSION_TOKEN) >= 32


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

    def test_get_status(self):
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "hermes_home" in data
        assert "active_sessions" in data
        assert data["can_update_hermes"] is True

    def test_status_active_session_count_uses_read_only_db(self, monkeypatch, tmp_path):
        import hermes_cli.web_server as web_server
        import hermes_state

        # Satisfy the fresh-install guard: read_only opens require the DB
        # file to already exist.
        fake_db_path = tmp_path / "state.db"
        fake_db_path.touch()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", fake_db_path)

        captured = {}

        class _FakeDB:
            def __init__(self, *args, **kwargs):
                captured["read_only"] = kwargs.get("read_only")

            def list_sessions_rich(self, limit, compact_rows=False):
                captured["limit"] = limit
                captured["compact_rows"] = compact_rows
                return [
                    {"ended_at": None, "last_active": 95},
                    {"ended_at": 99, "last_active": 99},
                    {"ended_at": None, "last_active": -300},
                ]

            def close(self):
                captured["closed"] = True

        monkeypatch.setattr("hermes_state.SessionDB", _FakeDB)
        monkeypatch.setattr(web_server.time, "time", lambda: 100)

        assert web_server._count_status_active_sessions() == 1
        assert captured == {
            "read_only": True, "limit": 50, "compact_rows": True, "closed": True
        }

    def test_status_active_session_count_fresh_install_returns_zero(self, monkeypatch, tmp_path):
        """No state.db yet (fresh install): return 0 without attempting a
        read-only open, which would raise OperationalError on every poll."""
        import hermes_cli.web_server as web_server
        import hermes_state

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "absent.db")

        def _boom(*a, **k):
            raise AssertionError("SessionDB must not be constructed when db file is absent")

        monkeypatch.setattr("hermes_state.SessionDB", _boom)
        assert web_server._count_status_active_sessions() == 0

    def test_get_status_degrades_when_active_session_count_fails(self, monkeypatch):
        import hermes_cli.web_server as web_server

        def _locked_count():
            raise TimeoutError("database is locked")

        monkeypatch.setattr(web_server, "_count_status_active_sessions", _locked_count)

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["active_sessions"] == 0

    def test_get_status_uses_cached_gateway_pid_probe(self, monkeypatch):
        import hermes_cli.web_server as web_server

        calls = {"get_running_pid_cached": 0}

        def _cached_pid():
            calls["get_running_pid_cached"] += 1
            return None

        monkeypatch.setattr(web_server, "get_running_pid_cached", _cached_pid)

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        assert calls["get_running_pid_cached"] == 1

    def test_gateway_drain_begin_writes_marker(self):
        from gateway import drain_control

        resp = self.client.post("/api/gateway/drain", json={"action": "drain"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True and data["action"] == "drain"
        assert data["draining"] is True
        assert drain_control.drain_requested() is True
        # cleanup
        drain_control.clear_drain_request()

    def test_gateway_drain_defaults_to_begin(self):
        from gateway import drain_control

        resp = self.client.post("/api/gateway/drain", json={})
        assert resp.status_code == 200
        assert resp.json()["action"] == "drain"
        assert drain_control.drain_requested() is True
        drain_control.clear_drain_request()

    def test_gateway_drain_suppress_notification_passthrough(self):
        from gateway import drain_control

        resp = self.client.post(
            "/api/gateway/drain",
            json={"action": "drain", "suppress_notification": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["suppress_notification"] is True
        # The flag landed on the marker the gateway reads at shutdown.
        body = drain_control.read_drain_request()
        assert body is not None and body["suppress_notification"] is True
        assert drain_control.drain_notification_suppressed() is True
        drain_control.clear_drain_request()

    def test_gateway_drain_suppress_defaults_false(self):
        from gateway import drain_control

        resp = self.client.post("/api/gateway/drain", json={"action": "drain"})
        assert resp.status_code == 200
        assert resp.json()["suppress_notification"] is False
        assert drain_control.drain_notification_suppressed() is False
        drain_control.clear_drain_request()

    def test_gateway_drain_cancel_removes_marker(self):
        from gateway import drain_control

        drain_control.write_drain_request()
        resp = self.client.post("/api/gateway/drain", json={"action": "cancel"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True and data["action"] == "cancel"
        assert data["was_draining"] is True
        assert drain_control.drain_requested() is False

    def test_gateway_drain_cancel_idempotent(self):
        from gateway import drain_control

        resp = self.client.post("/api/gateway/drain", json={"action": "cancel"})
        assert resp.status_code == 200
        assert resp.json()["was_draining"] is False
        assert drain_control.drain_requested() is False

    def test_gateway_drain_bad_action_400(self):
        resp = self.client.post("/api/gateway/drain", json={"action": "explode"})
        assert resp.status_code == 400

    def test_get_status_hides_update_capability_in_managed_runtime(self, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server, "_dashboard_local_update_managed_externally", lambda: True)

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["can_update_hermes"] is False

    def test_dashboard_update_capability_detects_generic_container(self, monkeypatch):
        import hermes_constants
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        # A docker install inside a container should be managed externally.
        monkeypatch.setattr(web_server, "detect_install_method", lambda _root: "docker")

        assert web_server._dashboard_local_update_managed_externally() is True

    def test_dashboard_update_capability_allows_git_in_container(self, monkeypatch):
        """A git checkout inside a container (e.g. bind-mounted in hermes-webui)
        should still offer dashboard updates — the checkout is self-managed."""
        import hermes_constants
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        monkeypatch.setattr(web_server, "detect_install_method", lambda _root: "git")

        assert web_server._dashboard_local_update_managed_externally() is False

    def test_dashboard_update_capability_blocks_pip_in_container(self, monkeypatch):
        """A pip install inside a container is still managed externally."""
        import hermes_constants
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        monkeypatch.setattr(web_server, "detect_install_method", lambda _root: "pip")

        assert web_server._dashboard_local_update_managed_externally() is True

    @staticmethod
    def _provider_field_map(payload):
        return {field["key"]: field for field in payload["fields"]}

    def test_get_memory_provider_config_returns_safe_defaults(self):
        resp = self.client.get("/api/memory/providers/hindsight/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "hindsight"
        assert data["label"] == "Hindsight"

        fields = self._provider_field_map(data)
        assert fields["mode"]["kind"] == "select"
        assert fields["mode"]["value"] == "cloud"
        assert {opt["value"] for opt in fields["mode"]["options"]} >= {
            "cloud",
            "local_external",
        }
        assert fields["api_url"]["kind"] == "text"
        assert fields["api_url"]["value"]
        assert fields["bank_id"]["value"] == "hermes"
        assert fields["recall_budget"]["value"] == "mid"
        assert fields["api_key"]["kind"] == "secret"
        assert fields["api_key"]["is_set"] is False
        assert fields["api_key"]["required"] is False

    def test_get_memory_provider_config_loads_dynamic_plugin_schema(self):
        resp = self.client.get("/api/memory/providers/honcho/config")

        assert resp.status_code == 200
        data = resp.json()
        fields = self._provider_field_map(data)
        assert fields["api_key"]["kind"] == "secret"
        assert fields["api_key"]["url"] == "https://app.honcho.dev"
        assert fields["baseUrl"]["kind"] == "text"

    def test_instance_schema_serves_providers_without_declared_schema(self, monkeypatch):
        # The default surface serves the plugin instance's get_config_schema().
        from hermes_cli import web_server

        class _Stub:
            def get_config_schema(self):
                return [
                    {"key": "api_key", "description": "Stub API key", "secret": True, "url": "https://stub.example"},
                    {"key": "baseUrl", "description": "Stub base URL"},
                ]

        monkeypatch.setattr(web_server, "_load_memory_provider", lambda name: _Stub())

        resp = self.client.get("/api/memory/providers/mem0/config")

        assert resp.status_code == 200
        data = resp.json()
        fields = self._provider_field_map(data)
        assert fields["api_key"]["kind"] == "secret"
        assert fields["api_key"]["url"] == "https://stub.example"
        assert fields["baseUrl"]["kind"] == "text"

    def test_declared_surface_serves_curated_hindsight_schema(self):
        resp = self.client.get("/api/memory/providers/hindsight/config?surface=declared")

        assert resp.status_code == 200
        data = resp.json()
        fields = self._provider_field_map(data)
        assert set(fields) == {"mode", "api_key", "api_url", "bank_id", "recall_budget"}
        assert fields["mode"]["kind"] == "select"
        assert fields["api_key"]["kind"] == "secret"

    def test_declared_surface_hides_undeclared_providers(self):
        resp = self.client.get("/api/memory/providers/builtin/config?surface=declared")

        assert resp.status_code == 200
        assert resp.json()["fields"] == []

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

    def test_declared_surface_put_rejects_undeclared_provider(self):
        resp = self.client.put(
            "/api/memory/providers/builtin/config?surface=declared",
            json={"values": {"api_key": "x"}},
        )

        assert resp.status_code == 404

    def test_all_listed_memory_provider_configs_fetch(self):
        resp = self.client.get("/api/memory")

        assert resp.status_code == 200
        providers = resp.json()["providers"]
        assert providers

        failures = []
        for provider in providers:
            config_resp = self.client.get(
                f"/api/memory/providers/{provider['name']}/config"
            )
            if config_resp.status_code != 200:
                failures.append((provider["name"], config_resp.status_code, config_resp.text))

        assert failures == []

    def test_memory_provider_payloads_include_manifest_setup_hints(self):
        resp = self.client.get("/api/memory")

        assert resp.status_code == 200
        providers = {row["name"]: row for row in resp.json()["providers"]}

        byterover_setup = providers["byterover"]["setup"]
        assert byterover_setup["external_dependencies"] == [
            {
                "name": "brv",
                "install": "curl -fsSL https://byterover.dev/install.sh | sh",
                "check": "brv --version",
            }
        ]

        retaindb_setup = providers["retaindb"]["setup"]
        assert "requests" in retaindb_setup["pip_dependencies"]
        assert "RETAINDB_API_KEY" in retaindb_setup["required_env"]
        assert isinstance(byterover_setup["dependencies_installed"], bool)

        config_resp = self.client.get("/api/memory/providers/byterover/config")
        assert config_resp.status_code == 200
        assert config_resp.json()["setup"]["external_dependencies"] == byterover_setup["external_dependencies"]

    def test_memory_status_reports_honcho_needs_config_after_dependency_setup(self, monkeypatch, tmp_path):
        # Pin HOME so a developer's real ~/.honcho config can't flip the status.
        monkeypatch.setenv("HOME", str(tmp_path))
        import hermes_cli.web_server as web_server

        original_dependency_importable = web_server._dependency_importable
        monkeypatch.setattr(
            web_server,
            "_dependency_importable",
            lambda dep: True if dep == "honcho-ai" else original_dependency_importable(dep),
        )

        resp = self.client.get("/api/memory")

        assert resp.status_code == 200
        providers = {row["name"]: row for row in resp.json()["providers"]}
        assert providers["honcho"]["setup"]["dependencies_installed"] is True
        assert providers["honcho"]["status"] == "needs_config"

    def test_post_memory_provider_setup_runs_declared_external_install(self, monkeypatch):
        import subprocess

        import hermes_cli.web_server as web_server

        calls = []
        check_count = 0

        def fake_run(command, **kwargs):
            nonlocal check_count
            calls.append((command, kwargs))
            if command == ["brv", "--version"]:
                check_count += 1
                if check_count == 1:
                    raise FileNotFoundError("brv")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="brv 1.0.0",
                    stderr="",
                )
            if command == "curl -fsSL https://byterover.dev/install.sh | sh":
                assert kwargs["shell"] is True
                return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")
            raise AssertionError(f"Unexpected command: {command}")

        monkeypatch.setattr(web_server.subprocess, "run", fake_run)

        resp = self.client.post("/api/memory/providers/byterover/setup", json={"values": {}})

        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "byterover"
        assert data["ok"] is True
        assert [result["status"] for result in data["results"]] == [
            "missing",
            "installed",
            "verified",
        ]
        assert [call[0] for call in calls[:3]] == [
            ["brv", "--version"],
            "curl -fsSL https://byterover.dev/install.sh | sh",
            ["brv", "--version"],
        ]
        assert calls[-1][0] == ["brv", "--version"]

    def test_post_unknown_memory_provider_setup_returns_404(self):
        resp = self.client.post("/api/memory/providers/nope/setup", json={"values": {}})

        assert resp.status_code == 404

    def test_memory_provider_endpoints_reject_traversal_names(self):
        # Names with path separators / dots must never reach the filesystem
        # lookup or the setup command path. 404 = rejected by the name guard;
        # 405 = the router collapsed the dotted path onto a different route
        # (equally safe — the handler never ran).
        for bad in ("..", "..%2f..%2fetc", "a.b", "x/y", ".hidden", ""):
            resp = self.client.get(f"/api/memory/providers/{bad}/config")
            assert resp.status_code in (404, 405), (bad, resp.status_code)
            resp = self.client.post(
                f"/api/memory/providers/{bad}/setup", json={"values": {}}
            )
            assert resp.status_code in (404, 405), (bad, resp.status_code)
            resp = self.client.put(
                f"/api/memory/providers/{bad}/config", json={"values": {}}
            )
            assert resp.status_code in (404, 405), (bad, resp.status_code)

    def test_post_memory_provider_setup_persists_values_without_activation(self):
        from hermes_cli.config import load_config, load_env

        resp = self.client.post(
            "/api/memory/providers/retaindb/setup",
            json={"values": {"api_key": "retain-test-key", "project": "default"}},
        )

        assert resp.status_code == 200
        assert resp.json()["provider"] == "retaindb"
        assert load_env()["RETAINDB_API_KEY"] == "retain-test-key"
        assert load_config().get("memory", {}).get("provider") != "retaindb"

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

    def test_put_memory_provider_config_rejects_unsupported_select_value(self):
        resp = self.client.put(
            "/api/memory/providers/hindsight/config",
            json={
                "values": {
                    "mode": "spaceship",
                    "api_url": "http://localhost:8888",
                    "bank_id": "hermes",
                    "recall_budget": "mid",
                }
            },
        )

        assert resp.status_code == 400

    def test_put_unknown_memory_provider_returns_404(self):
        resp = self.client.put(
            "/api/memory/providers/nope/config", json={"values": {}}
        )

        assert resp.status_code == 404

    def test_get_unknown_memory_provider_returns_empty_schema(self):
        resp = self.client.get("/api/memory/providers/builtin/config")

        assert resp.status_code == 200
        assert resp.json()["fields"] == []

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

    def test_get_memory_status_reports_ready_and_missing_provider(self):
        from hermes_cli.config import load_config, save_config

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
        resp = self.client.get("/api/memory")
        assert resp.status_code == 200
        providers = {row["name"]: row for row in resp.json()["providers"]}
        assert providers["hindsight"]["configured"] is True
        assert providers["hindsight"]["status"] == "ready"
        assert "available" in providers["hindsight"]

        config = load_config()
        config.setdefault("memory", {})["provider"] = "not-installed"
        save_config(config)

        resp = self.client.get("/api/memory")
        assert resp.status_code == 200
        providers = {row["name"]: row for row in resp.json()["providers"]}
        assert providers["not-installed"]["status"] == "missing"
        assert providers["not-installed"]["available"] is False

        config = load_config()
        config.setdefault("memory", {})["provider"] = "builtin"
        save_config(config)

        resp = self.client.get("/api/memory")
        assert resp.status_code == 200
        assert resp.json()["active"] == ""
        assert "builtin" not in {row["name"] for row in resp.json()["providers"]}

    def test_set_memory_provider_rejects_unready_and_clears_builtin(self):
        from hermes_cli.config import load_config

        resp = self.client.put("/api/memory/provider", json={"provider": "supermemory"})
        assert resp.status_code == 400

        resp = self.client.put("/api/memory/provider", json={"provider": "built-in"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "active": ""}
        assert load_config()["memory"]["provider"] == ""

    def test_dashboard_plugin_providers_rejects_unready_memory_provider(self):
        resp = self.client.put(
            "/api/dashboard/plugin-providers",
            json={"memory_provider": "supermemory"},
        )

        assert resp.status_code == 400

    def test_dashboard_plugin_providers_accepts_builtin_alias(self):
        from hermes_cli.config import load_config

        resp = self.client.put(
            "/api/dashboard/plugin-providers",
            json={"memory_provider": "built-in"},
        )

        assert resp.status_code == 200
        assert load_config()["memory"]["provider"] == ""

    def test_get_moa_models_returns_provider_model_slots(self):
        resp = self.client.get("/api/model/moa")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reference_models"]
        assert all(set(slot) == {"provider", "model"} for slot in data["reference_models"])
        assert set(data["aggregator"]) == {"provider", "model"}

    def test_put_moa_models_persists_provider_model_slots(self):
        from hermes_cli.config import load_config

        payload = {
            "reference_models": [
                {"provider": "openai-codex", "model": "gpt-5.5"},
                {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
            ],
            "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
            "reference_temperature": 0.6,
            "aggregator_temperature": 0.4,
            "max_tokens": 4096,
            "enabled": True,
        }

        resp = self.client.put("/api/model/moa", json=payload)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        cfg = load_config()
        assert cfg["moa"]["reference_models"] == payload["reference_models"]
        assert cfg["moa"]["aggregator"] == payload["aggregator"]

    def test_put_moa_models_rejects_half_filled_slot_with_422(self):
        """#64156: a mid-edit autosave (provider picked, model empty) used to be
        silently normalized into the hardcoded default preset — the user's
        config was replaced without any error. The write path must reject it."""
        from hermes_cli.config import load_config

        original = load_config().get("moa")

        payload = {
            "presets": {
                "default": {
                    "reference_models": [{"provider": "kilo", "model": ""}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                }
            }
        }

        resp = self.client.put("/api/model/moa", json=payload)
        assert resp.status_code == 422
        assert "model is required" in resp.json()["detail"]
        # Config untouched — not swapped for defaults.
        assert load_config().get("moa") == original

    def test_put_moa_models_rejects_half_filled_aggregator_with_422(self):
        payload = {
            "presets": {
                "default": {
                    "reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
                    "aggregator": {"provider": "openrouter", "model": ""},
                }
            }
        }

        resp = self.client.put("/api/model/moa", json=payload)
        assert resp.status_code == 422
        assert "aggregator" in resp.json()["detail"]

    def test_put_moa_models_round_trips_fanout_and_reference_max_tokens(self):
        """GET → PUT round-trip must not erase newer per-preset knobs. The old
        Pydantic payload didn't declare fanout / reference_max_tokens, so any
        client save silently wiped hand-set values back to defaults."""
        from hermes_cli.config import load_config

        payload = {
            "presets": {
                "default": {
                    "reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                    "fanout": "user_turn",
                    "reference_max_tokens": 600,
                }
            }
        }

        resp = self.client.put("/api/model/moa", json=payload)
        assert resp.status_code == 200

        saved = load_config()["moa"]["presets"]["default"]
        assert saved["fanout"] == "user_turn"
        assert saved["reference_max_tokens"] == 600

        # And the GET view carries them back to the client.
        fetched = self.client.get("/api/model/moa").json()
        assert fetched["presets"]["default"]["fanout"] == "user_turn"
        assert fetched["presets"]["default"]["reference_max_tokens"] == 600
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

    def test_get_honcho_config_returns_safe_defaults(self, monkeypatch, tmp_path):
        # HOME isn't isolated by the suite; pin it so ~/.honcho can't leak in.
        monkeypatch.setenv("HOME", str(tmp_path))

        resp = self.client.get("/api/memory/providers/honcho/config?surface=declared")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "honcho"
        assert data["label"] == "Honcho"
        assert data["docs_url"] == "https://docs.honcho.dev/v3/guides/integrations/hermes"

        fields = self._provider_field_map(data)
        assert fields["environment"]["kind"] == "select"
        assert fields["environment"]["value"] == "production"
        assert {opt["value"] for opt in fields["environment"]["options"]} == {
            "production",
            "local",
        }
        assert fields["sessionStrategy"]["value"] == "per-directory"
        # Blank workspace/aiPeer surface the resolved host as the placeholder.
        assert fields["workspace"]["value"] == ""
        assert fields["workspace"]["placeholder"] == "hermes"
        assert fields["aiPeer"]["placeholder"] == "hermes"
        assert fields["apiKey"]["kind"] == "secret"
        assert fields["apiKey"]["is_set"] is False

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

    def test_put_honcho_blank_text_clears_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        self._seed_local_honcho()
        from hermes_constants import get_hermes_home

        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"workspace": "myws"}},
        )
        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"workspace": ""}},
        )

        cfg = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))
        assert "workspace" not in cfg.get("hosts", {}).get("hermes", {})

    def test_put_honcho_partial_save_preserves_other_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        self._seed_local_honcho()
        from hermes_constants import get_hermes_home

        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"workspace": "myws"}},
        )
        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"peerName": "eri"}},
        )

        host = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))["hosts"]["hermes"]
        assert host["workspace"] == "myws"
        assert host["peerName"] == "eri"

    def test_put_honcho_rejects_unsupported_select_value(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"environment": "bogus"}},
        )

        assert resp.status_code == 400

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

    def test_put_honcho_bool_stored_natively_and_false_survives(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        self._seed_local_honcho()
        from hermes_constants import get_hermes_home

        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"saveMessages": "false", "dialecticDynamic": "true"}},
        )

        host = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))["hosts"]["hermes"]
        # Native JSON bools, not the strings "false"/"true" (which read truthy).
        assert host["saveMessages"] is False
        assert host["dialecticDynamic"] is True

        fields = self._provider_field_map(self.client.get("/api/memory/providers/honcho/config?surface=declared").json())
        assert fields["saveMessages"]["value"] == "false"
        assert fields["dialecticDynamic"]["value"] == "true"

    def test_put_honcho_number_stored_as_native_number(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        self._seed_local_honcho()
        from hermes_constants import get_hermes_home

        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"dialecticMaxChars": "1200", "timeout": "2.5"}},
        )

        cfg = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))
        assert cfg["hosts"]["hermes"]["dialecticMaxChars"] == 1200
        assert isinstance(cfg["hosts"]["hermes"]["dialecticMaxChars"], int)
        # timeout is root-scoped and keeps its fractional part.
        assert cfg["timeout"] == 2.5

        fields = self._provider_field_map(self.client.get("/api/memory/providers/honcho/config?surface=declared").json())
        assert fields["dialecticMaxChars"]["value"] == "1200"

    def test_put_honcho_json_round_trips_object(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        self._seed_local_honcho()
        from hermes_constants import get_hermes_home

        self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"userPeerAliases": '{"telegram_1": "eri"}'}},
        )

        host = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))["hosts"]["hermes"]
        assert host["userPeerAliases"] == {"telegram_1": "eri"}

        fields = self._provider_field_map(self.client.get("/api/memory/providers/honcho/config?surface=declared").json())
        assert json.loads(fields["userPeerAliases"]["value"]) == {"telegram_1": "eri"}

    def test_put_honcho_first_save_merges_into_resolved_config(self, monkeypatch, tmp_path):
        # With no profile-local file, a save merges into the resolved global config.
        monkeypatch.setenv("HOME", str(tmp_path))
        from hermes_constants import get_hermes_home

        global_path = tmp_path / ".honcho" / "config.json"
        global_path.parent.mkdir(parents=True)
        global_path.write_text(
            json.dumps({"baseUrl": "https://kept.example", "hosts": {"hermes": {"workspace": "kept"}}}),
            encoding="utf-8",
        )

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"peerName": "eri"}},
        )

        assert resp.status_code == 200
        assert not (get_hermes_home() / "honcho.json").exists()
        cfg = json.loads(global_path.read_text(encoding="utf-8"))
        assert cfg["baseUrl"] == "https://kept.example"
        assert cfg["hosts"]["hermes"] == {"workspace": "kept", "peerName": "eri"}

    def test_put_honcho_updates_legacy_dot_form_host_block(self, monkeypatch, tmp_path):
        # The legacy dot-form block reads resolve is updated in place, not shadowed.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HONCHO_HOST", "hermes_work")

        path = self._seed_local_honcho({"hosts": {"hermes.work": {"workspace": "w", "peerName": "eri"}}})

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"sessionStrategy": "per-repo"}},
        )

        assert resp.status_code == 200
        hosts = json.loads(path.read_text(encoding="utf-8"))["hosts"]
        assert set(hosts) == {"hermes.work"}
        assert hosts["hermes.work"] == {"workspace": "w", "peerName": "eri", "sessionStrategy": "per-repo"}

    def test_put_honcho_api_key_never_overwrites_oauth_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HONCHO_API_KEY", "guard")
        monkeypatch.delenv("HONCHO_API_KEY")
        from hermes_cli.config import load_env

        path = self._seed_local_honcho({"hosts": {"hermes": {"apiKey": "hch-at-oauth-token"}}})

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"apiKey": "manual-key"}},
        )

        assert resp.status_code == 200
        cfg = json.loads(path.read_text(encoding="utf-8"))
        # The OAuth grant owns the JSON slot; the manual key lands in the env store.
        assert cfg["hosts"]["hermes"]["apiKey"] == "hch-at-oauth-token"
        assert load_env()["HONCHO_API_KEY"] == "manual-key"

    def test_put_honcho_tolerates_null_hosts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))

        path = self._seed_local_honcho({"hosts": None})

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"workspace": "myws"}},
        )

        assert resp.status_code == 200
        assert json.loads(path.read_text(encoding="utf-8"))["hosts"]["hermes"]["workspace"] == "myws"

    def test_memory_provider_config_honors_profile_param(self, monkeypatch, tmp_path):
        # A ?profile= save must land in that profile's config, not the serving
        # process's — same contract as the skills/toolsets endpoints.
        monkeypatch.setenv("HOME", str(tmp_path))
        # The suite pins HERMES_HONCHO_HOST=hermes; this test exercises
        # profile-driven host resolution, so drop the override explicitly.
        monkeypatch.delenv("HERMES_HONCHO_HOST", raising=False)
        from hermes_constants import get_hermes_home
        from hermes_cli.profiles import get_profile_dir

        self._seed_local_honcho()

        worker_home = get_profile_dir("worker")
        worker_home.mkdir(parents=True, exist_ok=True)
        worker_cfg = worker_home / "honcho.json"
        worker_cfg.write_text(json.dumps({"hosts": {"hermes_worker": {"workspace": "kept"}}}), encoding="utf-8")

        resp = self.client.put(
            "/api/memory/providers/honcho/config?surface=declared&profile=worker",
            json={"values": {"peerName": "eri"}},
        )

        assert resp.status_code == 200
        worker_hosts = json.loads(worker_cfg.read_text(encoding="utf-8"))["hosts"]
        host_block = next(iter(worker_hosts.values()))
        assert host_block["peerName"] == "eri"
        # The serving process's own config is untouched.
        own = json.loads((get_hermes_home() / "honcho.json").read_text(encoding="utf-8"))
        assert "peerName" not in json.dumps(own)

        fields = self._provider_field_map(
            self.client.get("/api/memory/providers/honcho/config?surface=declared&profile=worker").json()
        )
        assert fields["peerName"]["value"] == "eri"

    def test_put_honcho_rejects_malformed_number_and_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))

        assert self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"dialecticMaxChars": "lots"}},
        ).status_code == 400
        assert self.client.put(
            "/api/memory/providers/honcho/config?surface=declared",
            json={"values": {"userPeerAliases": "{not json"}},
        ).status_code == 400

    # ── GET /api/media (remote image display) ───────────────────────────

    def test_get_media_serves_image_in_root(self):
        """An image under the gateway's images dir is returned as a data URL."""
        from hermes_constants import get_hermes_home

        img_dir = get_hermes_home() / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img = img_dir / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

        resp = self.client.get("/api/media", params={"path": str(img)})
        assert resp.status_code == 200
        assert resp.json()["data_url"].startswith("data:image/png;base64,")

    def test_get_media_rejects_path_outside_roots(self, tmp_path):
        """An image-extension file outside the media roots is forbidden."""
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"\x89PNG\r\n\x1a\n")

        resp = self.client.get("/api/media", params={"path": str(outside)})
        assert resp.status_code == 403

    def test_get_media_rejects_non_image_extension(self):
        from hermes_constants import get_hermes_home

        img_dir = get_hermes_home() / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        env = img_dir / "leak.env"
        env.write_text("SECRET=1")

        resp = self.client.get("/api/media", params={"path": str(env)})
        assert resp.status_code == 415

    def test_get_media_404_for_missing_file(self):
        from hermes_constants import get_hermes_home

        missing = get_hermes_home() / "images" / "nope.png"
        resp = self.client.get("/api/media", params={"path": str(missing)})
        assert resp.status_code == 404

    def test_get_media_requires_auth(self):
        from hermes_cli.web_server import _SESSION_HEADER_NAME

        resp = self.client.get(
            "/api/media",
            params={"path": "/tmp/x.png"},
            headers={_SESSION_HEADER_NAME: "wrong-token"},
        )
        assert resp.status_code == 401

    # ── POST /api/chat/image-upload (browser clipboard/drop images) ─────

    def test_chat_image_upload_writes_to_default_profile_images(self):
        from hermes_constants import get_hermes_home

        data_url = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        resp = self.client.post(
            "/api/chat/image-upload",
            json={"data_url": data_url, "filename": "../../clip.png"},
        )

        assert resp.status_code == 200
        data = resp.json()
        target = Path(data["path"])
        assert data["ok"] is True
        assert data["mime_type"] == "image/png"
        assert target.parent == get_hermes_home() / "images"
        assert target.name.startswith("dashboard_")
        assert target.name.endswith("_clip.png")
        assert target.is_file()
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_chat_image_upload_writes_to_requested_profile_images(self):
        from hermes_cli import profiles as profiles_mod

        worker_home = profiles_mod.get_profile_dir("worker")
        worker_home.mkdir(parents=True)

        resp = self.client.post(
            "/api/chat/image-upload?profile=worker",
            json={
                "data_url": "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA=",
                "filename": "drop.gif",
            },
        )

        assert resp.status_code == 200
        target = Path(resp.json()["path"])
        assert target.parent == worker_home / "images"
        assert target.is_file()
        assert target.read_bytes().startswith(b"GIF89a")

    def test_chat_image_upload_rejects_non_image_payload(self):
        resp = self.client.post(
            "/api/chat/image-upload",
            json={"data_url": "data:text/plain;base64,aGVsbG8="},
        )

        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_chat_image_upload_rejects_spoofed_image_payload(self):
        resp = self.client.post(
            "/api/chat/image-upload",
            json={"data_url": "data:image/png;base64,aGVsbG8=", "filename": "fake.png"},
        )

        assert resp.status_code == 400
        assert "unsupported image type" in resp.json()["detail"].lower()

    def test_chat_image_upload_rejects_unknown_profile(self):
        resp = self.client.post(
            "/api/chat/image-upload?profile=missing-profile",
            json={"data_url": "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="},
        )

        assert resp.status_code == 404
        assert "does not exist" in resp.json()["detail"]

    def test_chat_image_upload_enforces_image_size_cap(self, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server, "_CHAT_IMAGE_UPLOAD_MAX_BYTES", 4)

        resp = self.client.post(
            "/api/chat/image-upload",
            json={
                "data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE=",
                "filename": "large.png",
            },
        )

        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_chat_image_upload_requires_auth(self):
        from hermes_cli.web_server import _SESSION_HEADER_NAME

        resp = self.client.post(
            "/api/chat/image-upload",
            json={"data_url": "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="},
            headers={_SESSION_HEADER_NAME: "wrong-token"},
        )

        assert resp.status_code == 401

    # ── Dashboard font override ─────────────────────────────────────────

    def test_get_dashboard_font_defaults_to_theme(self):
        """With no override persisted, the active font is the theme sentinel."""
        resp = self.client.get("/api/dashboard/font")
        assert resp.status_code == 200
        assert resp.json() == {"font": "theme"}

    def test_set_dashboard_font_persists_valid_choice(self):
        """A valid catalog id is accepted, persisted, and read back."""
        from hermes_cli.config import load_config

        resp = self.client.put("/api/dashboard/font", json={"font": "inter"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "font": "inter"}

        # Persisted to config.yaml under dashboard.font.
        config = load_config()
        assert config["dashboard"]["font"] == "inter"

        # And reflected by the GET endpoint.
        assert self.client.get("/api/dashboard/font").json() == {"font": "inter"}

    def test_set_dashboard_font_clears_with_theme_sentinel(self):
        """Setting 'theme' clears any prior override."""
        self.client.put("/api/dashboard/font", json={"font": "fraunces"})
        resp = self.client.put("/api/dashboard/font", json={"font": "theme"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "font": "theme"}
        assert self.client.get("/api/dashboard/font").json() == {"font": "theme"}

    def test_set_dashboard_font_rejects_unknown_id(self):
        """An id not in the curated catalog coerces to the theme sentinel,
        so a stale/hostile client can't inject an arbitrary font id."""
        resp = self.client.put(
            "/api/dashboard/font", json={"font": "../../etc/passwd"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "font": "theme"}

    def test_get_dashboard_font_coerces_stale_persisted_value(self):
        """A config value no longer in the catalog reads back as 'theme'."""
        from hermes_cli.config import load_config, save_config

        config = load_config()
        config.setdefault("dashboard", {})["font"] = "retired-font-id"
        save_config(config)

        assert self.client.get("/api/dashboard/font").json() == {"font": "theme"}

    def test_dashboard_font_override_independent_of_theme(self):
        """The font override and the theme are stored separately — setting
        one must not disturb the other."""
        from hermes_cli.config import load_config

        self.client.put("/api/dashboard/theme", json={"name": "ember"})
        self.client.put("/api/dashboard/font", json={"font": "jetbrains-mono"})

        config = load_config()
        assert config["dashboard"]["theme"] == "ember"
        assert config["dashboard"]["font"] == "jetbrains-mono"

    def test_get_sessions_uses_only_persisted_cwd(self, monkeypatch):
        """Session rows without persisted cwd must not inherit TERMINAL_CWD.

        /api/sessions should reflect per-session DB state, not process/global
        cwd settings, so workspace grouping stays stable and deterministic.
        """
        from hermes_state import SessionDB

        monkeypatch.setenv("TERMINAL_CWD", "/tmp/global-default")

        db = SessionDB()
        try:
            db.create_session(session_id="session-no-cwd", source="cli")
        finally:
            db.close()

        resp = self.client.get("/api/sessions?limit=20&offset=0")
        assert resp.status_code == 200

        rows = resp.json()["sessions"]
        row = next(s for s in rows if s["id"] == "session-no-cwd")
        assert row["cwd"] is None

    def test_get_sessions_forwards_min_messages(self, monkeypatch):
        """The ?min_messages= filter must reach SessionDB.

        The desktop session picker calls /api/sessions?...&min_messages=N to
        hide empty sessions. The param was silently dropped from the handler
        in a merge once (SessionDB still supported it); guard the wiring.
        """
        captured = {}

        class _FakeDB:
            def __init__(self, *args, **kwargs):
                pass

            def list_sessions_rich(self, limit, offset, min_message_count=0, **kwargs):
                captured["list"] = min_message_count
                return []

            def session_count(self, min_message_count=0, **kwargs):
                captured["count"] = min_message_count
                return 0

            def close(self):
                pass

        monkeypatch.setattr("hermes_state.SessionDB", _FakeDB)

        resp = self.client.get("/api/sessions?limit=5&offset=0&min_messages=3")
        assert resp.status_code == 200
        assert captured["list"] == 3
        assert captured["count"] == 3

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

    def test_get_sessions_strips_heavy_fields_by_default(self):
        """List rows must omit system_prompt/model_config.

        system_prompt is the fully rendered prompt (tens of KB per row) and
        dominated the sidebar payload (96% of a 528KB response); no list UI
        reads it. Detail reads (GET /api/sessions/{id}) stay complete.
        """
        self._create_session_with_heavy_fields("lean-list-row")

        resp = self.client.get("/api/sessions?limit=20&offset=0")
        assert resp.status_code == 200
        rows = [s for s in resp.json()["sessions"] if s["id"] == "lean-list-row"]
        assert rows, "created session missing from list"
        row = rows[0]
        assert "system_prompt" not in row
        assert "model_config" not in row
        # The light fields the sidebar actually renders must survive.
        for key in ("id", "source", "started_at", "message_count", "is_active"):
            assert key in row

    def test_get_sessions_full_param_keeps_heavy_fields(self):
        """?full=1 is the escape hatch for callers that need complete rows."""
        self._create_session_with_heavy_fields("full-list-row")

        resp = self.client.get("/api/sessions?limit=20&offset=0&full=1")
        assert resp.status_code == 200
        rows = [s for s in resp.json()["sessions"] if s["id"] == "full-list-row"]
        assert rows, "created session missing from list"
        row = rows[0]
        assert row["system_prompt"].startswith("# SOUL.md")
        assert "temperature" in (row["model_config"] or "")

    def test_profiles_sessions_strips_heavy_fields_by_default(self):
        """The cross-profile aggregate applies the same list projection."""
        self._create_session_with_heavy_fields("lean-profiles-row")

        resp = self.client.get("/api/profiles/sessions?limit=20&offset=0")
        assert resp.status_code == 200
        rows = [s for s in resp.json()["sessions"] if s["id"] == "lean-profiles-row"]
        assert rows, "created session missing from profiles list"
        row = rows[0]
        assert "system_prompt" not in row
        assert "model_config" not in row
        assert row["profile"] == "default"

        full = self.client.get("/api/profiles/sessions?limit=20&offset=0&full=1")
        assert full.status_code == 200
        full_rows = [s for s in full.json()["sessions"] if s["id"] == "lean-profiles-row"]
        assert full_rows and full_rows[0]["system_prompt"].startswith("# SOUL.md")

    def test_rename_session_updates_title(self):
        """PATCH /api/sessions/{id} renames a session (regression: the route
        was missing entirely, so the desktop rename dialog got a 405)."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="rename-me", source="cli")
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/rename-me", json={"title": "My Chat"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "title": "My Chat"}

        db = SessionDB()
        try:
            assert db.get_session_title("rename-me") == "My Chat"
        finally:
            db.close()

    def test_rename_session_clears_title_when_empty(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="clear-me", source="cli")
            db.set_session_title("clear-me", "Has A Title")
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/clear-me", json={"title": ""})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "title": ""}

        db = SessionDB()
        try:
            assert db.get_session_title("clear-me") is None
        finally:
            db.close()

    def test_rename_session_not_found(self):
        resp = self.client.patch("/api/sessions/does-not-exist", json={"title": "x"})
        assert resp.status_code == 404

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

    def test_import_sessions_endpoint_rejects_oversized_stream(self):
        import hermes_cli.web_server as web_server

        payload = b'{"sessions":[]}' + b" " * web_server._SESSION_IMPORT_MAX_BYTES
        response = self.client.post(
            "/api/sessions/import",
            content=payload,
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 413
        assert response.json() == {"detail": "Session import payload is too large"}

    def test_import_sessions_endpoint_rejects_metadata_that_would_break_session_list(self):
        invalid = self.client.post(
            "/api/sessions/import",
            json={
                "sessions": [
                    {
                        "id": "bad-model-config",
                        "source": "cli",
                        "model_config": "{not-json",
                        "messages": [],
                    }
                ]
            },
        )

        assert invalid.status_code == 400
        assert invalid.json()["detail"]["errors"] == [
            {
                "index": 0,
                "session_id": "bad-model-config",
                "error": "model_config must be valid JSON",
            }
        ]
        listed = self.client.get("/api/sessions")
        assert listed.status_code == 200

    @pytest.mark.parametrize(
        "message",
        [{"content": "missing role"}, {"role": None, "content": "null role"}],
    )
    def test_import_sessions_endpoint_rejects_missing_or_null_message_role(self, message):
        response = self.client.post(
            "/api/sessions/import",
            json={"sessions": [{"id": "bad-message-role", "messages": [message]}]},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["errors"] == [
            {
                "index": 0,
                "session_id": "bad-message-role",
                "error": "messages[0].role must be a non-empty string",
            }
        ]
        assert self.client.get("/api/sessions").status_code == 200

    def test_archive_session_via_patch(self):
        """PATCH archived=true soft-hides a session; archived=false restores it."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="arch-me", source="cli")
            db.append_message(session_id="arch-me", role="user", content="hi")
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/arch-me", json={"archived": True})
        assert resp.status_code == 200
        assert resp.json()["archived"] is True

        # Hidden from the default list, surfaced by archived=only.
        listed = self.client.get("/api/sessions").json()
        assert all(s["id"] != "arch-me" for s in listed["sessions"])
        only = self.client.get("/api/sessions?archived=only").json()
        assert any(s["id"] == "arch-me" for s in only["sessions"])

        resp = self.client.patch("/api/sessions/arch-me", json={"archived": False})
        assert resp.status_code == 200
        restored = self.client.get("/api/sessions").json()
        assert any(s["id"] == "arch-me" for s in restored["sessions"])

    def test_patch_session_without_fields_is_400(self):
        """An existing session + empty body is a bad request, not a 404."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="no-fields", source="cli")
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/no-fields", json={})
        assert resp.status_code == 400

    def test_profiles_sessions_tags_default_profile(self):
        """The cross-profile aggregator returns the default profile's rows
        tagged profile="default" (single-profile parity with /api/sessions)."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="agg-me", source="cli")
            db.append_message(session_id="agg-me", role="user", content="hi")
        finally:
            db.close()

        resp = self.client.get("/api/profiles/sessions?limit=20&min_messages=0")
        assert resp.status_code == 200
        data = resp.json()
        row = next(s for s in data["sessions"] if s["id"] == "agg-me")
        assert row["profile"] == "default"
        assert row["is_default_profile"] is True
        assert isinstance(data.get("errors"), list)

    def test_profiles_sessions_rejects_unknown_archived_value(self):
        resp = self.client.get("/api/profiles/sessions?archived=bogus")
        assert resp.status_code == 400

    def test_profiles_sessions_sidebar_batches_three_slices(self):
        """The batched sidebar endpoint returns recents/cron/messaging in one
        pass, each source-scoped by the caller-supplied excludes, so the desktop
        stops reopening every profile DB three times per refresh."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            for sid, src in (
                ("sb-desktop", "desktop"),
                ("sb-cron", "cron"),
                ("sb-telegram", "telegram"),
            ):
                db.create_session(session_id=sid, source=src)
                db.append_message(session_id=sid, role="user", content="hi")
        finally:
            db.close()

        resp = self.client.get(
            "/api/profiles/sessions/sidebar"
            "?recents_profile=all&recents_limit=20&recents_exclude=cron,telegram"
            "&cron_limit=50&messaging_limit=100"
            "&messaging_exclude=cron,cli,codex,desktop,gateway,local,tui"
        )
        assert resp.status_code == 200
        data = resp.json()

        recents_ids = {s["id"] for s in data["recents"]["sessions"]}
        cron_ids = {s["id"] for s in data["cron"]["sessions"]}
        messaging_ids = {s["id"] for s in data["messaging"]["sessions"]}

        # Each session lands only in its own slice.
        assert "sb-desktop" in recents_ids
        assert "sb-desktop" not in cron_ids and "sb-desktop" not in messaging_ids
        assert "sb-cron" in cron_ids
        assert "sb-cron" not in recents_ids and "sb-cron" not in messaging_ids
        assert "sb-telegram" in messaging_ids
        assert "sb-telegram" not in recents_ids and "sb-telegram" not in cron_ids

        # Rows carry profile tagging like /api/profiles/sessions.
        row = next(s for s in data["recents"]["sessions"] if s["id"] == "sb-desktop")
        assert row["profile"] == "default"
        assert row["is_default_profile"] is True
        assert isinstance(data.get("errors"), list)
        assert data["recents"]["total"] >= 1

    def test_sessions_endpoint_reads_requested_profile(self):
        """The machine dashboard's global profile switcher must retarget
        the Sessions page, not just config/skills/model pages."""
        from hermes_state import SessionDB
        from hermes_cli import profiles as profiles_mod

        worker_home = profiles_mod.get_profile_dir("worker")
        worker_home.mkdir(parents=True)

        default_db = SessionDB()
        try:
            default_db.create_session(session_id="default-only", source="cli")
            default_db.append_message("default-only", role="user", content="default")
        finally:
            default_db.close()

        worker_db = SessionDB(db_path=worker_home / "state.db")
        try:
            worker_db.create_session(session_id="worker-only", source="cli")
            worker_db.append_message("worker-only", role="user", content="worker")
        finally:
            worker_db.close()

        resp = self.client.get("/api/sessions?profile=worker&limit=20&min_messages=0")
        assert resp.status_code == 200
        data = resp.json()
        ids = {s["id"] for s in data["sessions"]}
        assert "worker-only" in ids
        assert "default-only" not in ids
        row = next(s for s in data["sessions"] if s["id"] == "worker-only")
        assert row["profile"] == "worker"
        assert row["is_default_profile"] is False

        stats = self.client.get("/api/sessions/stats?profile=worker").json()
        assert stats["total"] == 1
        assert stats["messages"] == 1

        messages = self.client.get("/api/sessions/worker-only/messages?profile=worker").json()
        assert [m["content"] for m in messages["messages"]] == ["worker"]

    def test_latest_descendant_reads_requested_profile(self):
        """Chat resume must resolve compression tips in the chat profile DB."""
        from hermes_state import SessionDB
        from hermes_cli import profiles as profiles_mod

        worker_home = profiles_mod.get_profile_dir("worker")
        worker_home.mkdir(parents=True)

        default_db = SessionDB()
        try:
            default_db.create_session(session_id="shared-root", source="cli")
        finally:
            default_db.close()

        worker_db = SessionDB(db_path=worker_home / "state.db")
        try:
            worker_db.create_session(session_id="shared-root", source="cli")
            worker_db.create_session(
                session_id="worker-tip",
                source="cli",
                parent_session_id="shared-root",
            )
        finally:
            worker_db.close()

        default_resp = self.client.get("/api/sessions/shared-root/latest-descendant")
        assert default_resp.status_code == 200
        assert default_resp.json()["session_id"] == "shared-root"

        worker_resp = self.client.get(
            "/api/sessions/shared-root/latest-descendant?profile=worker"
        )
        assert worker_resp.status_code == 200
        assert worker_resp.json()["session_id"] == "worker-tip"

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

    def test_analytics_endpoints_read_requested_profile(self):
        from hermes_state import SessionDB
        from hermes_cli import profiles as profiles_mod

        worker_home = profiles_mod.get_profile_dir("worker")
        worker_home.mkdir(parents=True)

        default_db = SessionDB()
        try:
            default_db.create_session(session_id="default-usage", source="cli", model="default/model")
            default_db.update_token_counts("default-usage", input_tokens=10, output_tokens=5)
        finally:
            default_db.close()

        worker_db = SessionDB(db_path=worker_home / "state.db")
        try:
            worker_db.create_session(session_id="worker-usage", source="cli", model="worker/model")
            worker_db.update_token_counts(
                "worker-usage",
                input_tokens=123,
                output_tokens=45,
                billing_provider="worker-provider",
            )
        finally:
            worker_db.close()

        usage = self.client.get("/api/analytics/usage?days=7&profile=worker").json()
        assert usage["totals"]["total_sessions"] == 1
        assert usage["totals"]["total_input"] == 123
        assert [m["model"] for m in usage["by_model"]] == ["worker/model"]

        models = self.client.get("/api/analytics/models?days=7&profile=worker").json()
        assert models["totals"]["distinct_models"] == 1
        assert models["totals"]["total_input"] == 123
        assert models["models"][0]["model"] == "worker/model"
        assert models["models"][0]["provider"] == "worker-provider"

        default_usage = self.client.get("/api/analytics/usage?days=7").json()
        assert default_usage["totals"]["total_input"] == 10
        assert default_usage["totals"]["total_output"] == 5

    def test_get_sessions_rejects_unknown_archived_value(self):
        resp = self.client.get("/api/sessions?archived=bogus")
        assert resp.status_code == 400

    def test_get_sessions_rejects_unknown_order_value(self):
        resp = self.client.get("/api/sessions?order=sideways")
        assert resp.status_code == 400

    def test_get_sessions_order_recent_surfaces_compression_tip(self):
        """A long-running conversation that auto-compresses must stay on the
        first page by recency, listed under its live continuation id."""
        import time as _time

        from hermes_state import SessionDB

        db = SessionDB()
        try:
            old = _time.time() - 86_400
            # Old conversation that later compresses into a fresh continuation.
            # The continuation must start at/after the parent's ended_at to be
            # recognised as a compression tip (not a sub-agent/branch).
            db.create_session(session_id="root-old", source="cli")
            db.append_message(session_id="root-old", role="user", content="kickoff")
            db.end_session("root-old", "compression")
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (old, old + 10, "root-old"),
            )
            db.create_session(session_id="tip-new", source="cli", parent_session_id="root-old")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (old + 10, "tip-new"))
            db.append_message(session_id="tip-new", role="user", content="continued just now")
            # A brand-new unrelated session started after the root but before now.
            db.create_session(session_id="mid", source="cli")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (_time.time() - 3600, "mid"))
            db.append_message(session_id="mid", role="user", content="hello")
            db._conn.commit()
        finally:
            db.close()

        rows = self.client.get("/api/sessions?order=recent&limit=5").json()["sessions"]
        ids = [r["id"] for r in rows]
        # The compressed conversation surfaces under its live tip id...
        assert "tip-new" in ids
        # ...carrying the durable lineage root so the desktop can match pins.
        tip = next(r for r in rows if r["id"] == "tip-new")
        assert tip.get("_lineage_root_id") == "root-old"

    def test_search_dedupes_compression_lineage_to_tip(self):
        """A conversation that auto-compresses leaves the matched term in both
        the root segment and the continuation. Search must collapse them to a
        single result keyed by the lineage root and pointing at the live tip,
        so the sidebar stops showing the same chat several times."""
        import time as _time

        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="search-root", source="cli")
            db.append_message(session_id="search-root", role="user", content="distinctneedle in the root")
            db.end_session("search-root", "compression")
            now = _time.time()
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (now - 100, now - 90, "search-root"),
            )
            db.create_session(session_id="search-tip", source="cli", parent_session_id="search-root")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 90, "search-tip"))
            db.append_message(session_id="search-tip", role="user", content="distinctneedle again in the tip")
            db._conn.commit()
        finally:
            db.close()

        resp = self.client.get("/api/sessions/search?q=distinctneedle")
        assert resp.status_code == 200
        results = resp.json()["results"]

        lineage_hits = [r for r in results if r.get("lineage_root") == "search-root"]
        # One conversation -> exactly one result despite two FTS hits.
        assert len(lineage_hits) == 1
        hit = lineage_hits[0]
        # Surfaced under the live tip so clicking resumes the current session.
        assert hit["session_id"] == "search-tip"
        assert hit["lineage_root"] == "search-root"

    def test_search_keeps_branch_specific_hits_on_branch(self):
        """Branch sessions share parent_session_id, but they are not compression
        continuations. A query that only exists in the branch must open the
        branch instead of being collapsed back to the parent/root."""
        import time as _time

        from hermes_state import SessionDB

        db = SessionDB()
        try:
            now = _time.time()
            db.create_session(session_id="branch-parent", source="cli")
            db.append_message(session_id="branch-parent", role="user", content="ancestor context")
            db.end_session("branch-parent", "branched")
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (now - 100, now - 90, "branch-parent"),
            )
            db.create_session(session_id="branch-child", source="cli", parent_session_id="branch-parent")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 80, "branch-child"))
            db.append_message(session_id="branch-child", role="user", content="branchspecificneedle only here")
            db._conn.commit()
        finally:
            db.close()

        resp = self.client.get("/api/sessions/search?q=branchspecificneedle")
        assert resp.status_code == 200
        results = resp.json()["results"]

        assert any(
            r["session_id"] == "branch-child" and r.get("lineage_root") == "branch-child"
            for r in results
        )

    def test_get_session_messages_follows_compression_tip(self):
        """Reading a compressed session by its old id should hydrate from the
        live continuation, matching /resume behavior."""
        import time as _time

        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="desktop-root", source="cli")
            db.append_message(session_id="desktop-root", role="user", content="before compression")
            db.end_session("desktop-root", "compression")
            now = _time.time()
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (now - 10, now - 5, "desktop-root"),
            )
            db.create_session(session_id="desktop-tip", source="cli", parent_session_id="desktop-root")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 4, "desktop-tip"))
            db.replace_messages("desktop-root", [])
            db.append_message(session_id="desktop-tip", role="user", content="after compression")
            db._conn.commit()
        finally:
            db.close()

        resp = self.client.get("/api/sessions/desktop-root/messages")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["session_id"] == "desktop-tip"
        assert [m["content"] for m in payload["messages"]] == ["after compression"]

    def test_get_sessions_archived_is_boolean(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="bool-arch", source="cli")
            db.append_message(session_id="bool-arch", role="user", content="hi")
        finally:
            db.close()

        row = next(s for s in self.client.get("/api/sessions").json()["sessions"] if s["id"] == "bool-arch")
        assert row["archived"] is False

    def test_rename_response_omits_archived_when_not_set(self):
        """Title-only PATCH keeps its legacy {ok, title} response shape."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(session_id="title-only", source="cli")
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/title-only", json={"title": "Hi"})
        assert resp.status_code == 200
        assert "archived" not in resp.json()

    def test_audio_transcription_endpoint(self, monkeypatch):
        import tools.transcription_tools as transcription_tools

        captured = {}

        def fake_transcribe_audio(path):
            captured["path"] = path
            return {
                "success": True,
                "transcript": "hello from voice mode",
                "provider": "test",
            }

        monkeypatch.setattr(transcription_tools, "transcribe_audio", fake_transcribe_audio)

        resp = self.client.post(
            "/api/audio/transcribe",
            json={
                "data_url": "data:audio/webm;base64,aGVsbG8=",
                "mime_type": "audio/webm",
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "ok": True,
            "transcript": "hello from voice mode",
            "provider": "test",
        }
        assert captured["path"].endswith(".webm")
        assert not Path(captured["path"]).exists()

    def test_audio_transcription_rejects_invalid_base64(self):
        resp = self.client.post(
            "/api/audio/transcribe",
            json={
                "data_url": "data:audio/webm;base64,not base64",
                "mime_type": "audio/webm",
            },
        )

        assert resp.status_code == 400
        assert "base64" in resp.json()["detail"]

    def test_desktop_audio_routes_registered(self):
        """All three desktop voice endpoints must exist.

        The renderer (apps/desktop) calls /api/audio/transcribe, /speak, and
        /elevenlabs/voices. /speak + /voices were silently dropped in a merge
        once; this guards the contract so a future merge can't lose them
        without failing CI.
        """
        from hermes_cli.web_server import app

        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/api/audio/transcribe" in paths
        assert "/api/audio/speak" in paths
        assert "/api/audio/elevenlabs/voices" in paths

    def test_elevenlabs_voices_unavailable_without_key(self, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server, "load_env", lambda: {})
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        resp = self.client.get("/api/audio/elevenlabs/voices")
        assert resp.status_code == 200
        assert resp.json() == {"available": False, "voices": []}

    def test_speak_text_returns_base64_data_url(self, monkeypatch, tmp_path):
        import tools.tts_tool as tts_tool

        audio_file = tmp_path / "speech.mp3"
        audio_file.write_bytes(b"ID3fake-audio-bytes")

        def fake_tts(text):
            return json.dumps({
                "success": True,
                "file_path": str(audio_file),
                "provider": "test",
            })

        monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

        resp = self.client.post("/api/audio/speak", json={"text": "hello there"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["mime_type"] == "audio/mpeg"
        assert body["data_url"].startswith("data:audio/mpeg;base64,")
        assert body["provider"] == "test"
        # The handler streams the bytes back and removes the temp file.
        assert not audio_file.exists()

    def test_speak_text_requires_nonempty_text(self):
        resp = self.client.post("/api/audio/speak", json={"text": "   "})
        assert resp.status_code == 400

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

    def test_update_hermes_returns_managed_runtime_guidance_without_spawning(self, monkeypatch):
        import hermes_cli.web_server as web_server

        spawned = False
        detected = False

        def fail_spawn(*_args, **_kwargs):
            nonlocal spawned
            spawned = True
            raise AssertionError("managed runtime update guard should not spawn hermes update")

        def fail_detect(*_args, **_kwargs):
            nonlocal detected
            detected = True
            raise AssertionError("managed runtime update guard should not detect install method")

        monkeypatch.setattr(web_server, "_dashboard_local_update_managed_externally", lambda: True)
        monkeypatch.setattr(web_server, "detect_install_method", fail_detect)
        monkeypatch.setattr(web_server, "_spawn_hermes_action", fail_spawn)
        web_server._ACTION_PROCS.pop("hermes-update", None)
        web_server._ACTION_RESULTS.pop("hermes-update", None)

        resp = self.client.post("/api/hermes/update")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["name"] == "hermes-update"
        assert data["pid"] is None
        assert data["error"] == "dashboard_update_managed_externally"
        assert "managed outside this dashboard" in data["message"]
        assert spawned is False
        assert detected is False

        status = self.client.get("/api/actions/hermes-update/status")
        assert status.status_code == 200
        status_data = status.json()
        assert status_data["running"] is False
        assert status_data["exit_code"] == 1
        assert status_data["pid"] is None
        assert any("managed outside this dashboard" in line for line in status_data["lines"])

    def test_update_hermes_spawns_on_non_docker_install(self, monkeypatch):
        import hermes_cli.web_server as web_server

        class Proc:
            pid = 12345

            def poll(self):
                return None

        calls = []

        def fake_spawn(subcommand, name):
            calls.append((subcommand, name))
            return Proc()

        monkeypatch.setattr(web_server, "detect_install_method", lambda _root: "git")
        monkeypatch.setattr(web_server, "_spawn_hermes_action", fake_spawn)
        web_server._ACTION_PROCS.pop("hermes-update", None)
        web_server._ACTION_RESULTS.pop("hermes-update", None)

        resp = self.client.post("/api/hermes/update")

        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "pid": 12345, "name": "hermes-update"}
        assert calls == [(["update"], "hermes-update")]

    def test_action_status_reaps_completed_process(self, monkeypatch):
        import hermes_cli.web_server as web_server

        waited = {"done": False}

        class _Proc:
            pid = 42424

            def poll(self):
                return 0

            def wait(self, timeout=None):
                waited["done"] = True

        proc = _Proc()
        web_server._ACTION_PROCS.pop("hermes-update", None)
        web_server._ACTION_RESULTS.pop("hermes-update", None)
        web_server._ACTION_PROCS["hermes-update"] = proc

        resp = self.client.get("/api/actions/hermes-update/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["exit_code"] == 0
        assert data["pid"] == 42424

        # Process should have been reaped and moved to results.
        assert waited["done"] is True
        assert "hermes-update" not in web_server._ACTION_PROCS
        assert web_server._ACTION_RESULTS["hermes-update"] == {
            "exit_code": 0,
            "pid": 42424,
        }

    def test_action_status_ignores_wait_failure(self, monkeypatch):
        import hermes_cli.web_server as web_server

        class _Proc:
            pid = 99

            def poll(self):
                return 1

            def wait(self, timeout=None):
                raise OSError("already reaped")

        proc = _Proc()
        web_server._ACTION_PROCS.pop("hermes-update", None)
        web_server._ACTION_RESULTS.pop("hermes-update", None)
        web_server._ACTION_PROCS["hermes-update"] = proc

        resp = self.client.get("/api/actions/hermes-update/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 1
        # Still reaped despite wait() raising.
        assert "hermes-update" not in web_server._ACTION_PROCS
        assert web_server._ACTION_RESULTS["hermes-update"] == {
            "exit_code": 1,
            "pid": 99,
        }

    def test_action_status_tails_large_log_without_read_text(self, tmp_path, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server, "_ACTION_LOG_DIR", tmp_path)
        web_server._ACTION_PROCS.pop("hermes-update", None)
        web_server._ACTION_RESULTS.pop("hermes-update", None)

        log_path = tmp_path / web_server._ACTION_LOG_FILES["hermes-update"]
        log_path.write_text(
            "stale-start\n"
            + ("x" * (web_server._ACTION_LOG_TAIL_MAX_BYTES + 1024))
            + "\ntail-one\ntail-two\n",
            encoding="utf-8",
        )
        assert log_path.stat().st_size > web_server._ACTION_LOG_TAIL_MAX_BYTES

        original_read_text = Path.read_text

        def fail_if_status_reads_whole_log(path, *args, **kwargs):
            if path == log_path:
                raise AssertionError("action status must not read the entire log")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_if_status_reads_whole_log)

        resp = self.client.get("/api/actions/hermes-update/status?lines=3")

        assert resp.status_code == 200
        assert resp.json()["lines"] == ["tail-one", "tail-two"]


    def test_get_status_filters_unconfigured_gateway_platforms(self, monkeypatch):
        import gateway.config as gateway_config
        import hermes_cli.web_server as web_server

        class _Platform:
            def __init__(self, value):
                self.value = value

        class _GatewayConfig:
            def get_connected_platforms(self):
                return [_Platform("telegram")]

        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda: {
                "gateway_state": "running",
                "updated_at": "2026-04-12T00:00:00+00:00",
                "platforms": {
                    "telegram": {"state": "connected", "updated_at": "2026-04-12T00:00:00+00:00"},
                    "whatsapp": {"state": "retrying", "updated_at": "2026-04-12T00:00:00+00:00"},
                    "feishu": {"state": "connected", "updated_at": "2026-04-12T00:00:00+00:00"},
                },
            },
        )
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: _GatewayConfig())

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["gateway_platforms"] == {
            "telegram": {"state": "connected", "updated_at": "2026-04-12T00:00:00+00:00"},
        }

    def test_get_status_hides_stale_platforms_when_gateway_not_running(self, monkeypatch):
        import gateway.config as gateway_config
        import hermes_cli.web_server as web_server

        class _GatewayConfig:
            def get_connected_platforms(self):
                return []

        monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda: {
                "gateway_state": "startup_failed",
                "updated_at": "2026-04-12T00:00:00+00:00",
                "platforms": {
                    "whatsapp": {"state": "retrying", "updated_at": "2026-04-12T00:00:00+00:00"},
                    "feishu": {"state": "connected", "updated_at": "2026-04-12T00:00:00+00:00"},
                },
            },
        )
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: _GatewayConfig())

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["gateway_state"] == "startup_failed"
        assert resp.json()["gateway_platforms"] == {}

    def test_cron_delivery_targets_lists_configured_platforms(self, monkeypatch):
        """The cron dropdown endpoint returns Local + configured platforms dynamically."""
        import gateway.config as gateway_config

        class _Platform:
            def __init__(self, value):
                self.value = value

        class _GatewayConfig:
            def get_connected_platforms(self):
                return [_Platform("matrix")]

        monkeypatch.setattr(
            gateway_config, "load_gateway_config", lambda: _GatewayConfig()
        )
        monkeypatch.setenv("MATRIX_HOME_ROOM", "!room:matrix.org")

        resp = self.client.get("/api/cron/delivery-targets")

        assert resp.status_code == 200
        targets = {t["id"]: t for t in resp.json()["targets"]}
        # Local is always offered; matrix appears because its gateway is configured.
        assert "local" in targets
        assert "matrix" in targets
        assert targets["matrix"]["home_target_set"] is True
        # No hardcoded telegram/discord/slack/email when they aren't configured.
        assert "telegram" not in targets

    def test_get_config_schema(self):
        resp = self.client.get("/api/config/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "fields" in data
        assert "category_order" in data
        schema = data["fields"]
        assert len(schema) > 100  # Should have 150+ fields
        assert "model" in schema
        # Verify category_order is a non-empty list
        assert isinstance(data["category_order"], list)
        assert len(data["category_order"]) > 0
        assert "general" in data["category_order"]

    def _schema_provider_options(self, key):
        resp = self.client.get("/api/config/schema")
        assert resp.status_code == 200
        return resp.json()["fields"][key]["options"]

    def test_config_schema_merges_custom_command_tts_provider(self):
        """A tts.providers.<name> command block appears in tts.provider options,
        appended AFTER the built-ins (original order preserved, no re-sort)."""
        from hermes_cli.config import load_config, save_config
        from hermes_cli.web_server import CONFIG_SCHEMA

        builtins = list(CONFIG_SCHEMA["tts.provider"]["options"])

        cfg = load_config()
        cfg.setdefault("tts", {}).setdefault("providers", {})["mycustomtts"] = {
            "type": "command",
            "command": "mytts --text {text} --out {output}",
        }
        save_config(cfg)

        options = self._schema_provider_options("tts.provider")
        assert options[: len(builtins)] == builtins  # built-in order kept
        assert "mycustomtts" in options
        assert options.count("mycustomtts") == 1
        # The module-level schema must NOT have been mutated.
        assert "mycustomtts" not in CONFIG_SCHEMA["tts.provider"]["options"]

    def test_config_schema_merges_custom_command_stt_provider(self):
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("stt", {}).setdefault("providers", {})["mywhisper"] = {
            "command": "whisper-cli {input}",  # type: omitted → command implied
        }
        save_config(cfg)

        options = self._schema_provider_options("stt.provider")
        assert "mywhisper" in options

    def test_config_schema_excludes_builtin_name_collisions(self):
        """A providers.EDGE command block must NOT be offered — the runtime
        rejects built-in names as command providers (case-insensitively)."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("tts", {}).setdefault("providers", {})["EDGE"] = {
            "type": "command",
            "command": "fake-edge {text}",
        }
        save_config(cfg)

        options = self._schema_provider_options("tts.provider")
        lowered = [o.lower() for o in options]
        assert lowered.count("edge") == 1  # only the built-in entry

    def test_config_schema_excludes_non_command_blocks(self):
        """Built-in-shaped blocks (voice/model, no command) and non-dicts are
        not offered as providers."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        tts = cfg.setdefault("tts", {})
        tts.setdefault("providers", {})["notacommand"] = {"voice": "en-US-Foo"}
        tts["stringy"] = "oops"
        save_config(cfg)

        options = self._schema_provider_options("tts.provider")
        assert "notacommand" not in options
        assert "stringy" not in options

    def test_config_schema_preserves_current_custom_provider_value(self):
        """A custom active tts.provider without a providers.<name> block stays
        selectable (current-value preservation, matching desktop behavior)."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("tts", {})["provider"] = "orphancustom"
        save_config(cfg)

        options = self._schema_provider_options("tts.provider")
        assert "orphancustom" in options

    def test_config_schema_reflects_config_changes_without_restart(self):
        """Options are computed per-request — adding a provider after the
        first schema fetch shows up on the next fetch."""
        from hermes_cli.config import load_config, save_config

        before = self._schema_provider_options("tts.provider")
        assert "latecomer" not in before

        cfg = load_config()
        cfg.setdefault("tts", {}).setdefault("providers", {})["latecomer"] = {
            "type": "command",
            "command": "late {text}",
        }
        save_config(cfg)

        after = self._schema_provider_options("tts.provider")
        assert "latecomer" in after

    def test_config_schema_legacy_toplevel_command_provider(self):
        """The legacy top-level ``tts.<name>`` command block (runtime
        back-compat fallback) is also offered."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("tts", {})["legacytts"] = {
            "type": "command",
            "command": "legacy {text}",
        }
        save_config(cfg)

        options = self._schema_provider_options("tts.provider")
        assert "legacytts" in options

    def test_get_config_defaults(self):
        resp = self.client.get("/api/config/defaults")
        assert resp.status_code == 200
        defaults = resp.json()
        assert "model" in defaults

    def test_get_env_vars(self):
        resp = self.client.get("/api/env")
        assert resp.status_code == 200
        data = resp.json()
        # Should contain known env var names
        assert any(k.endswith("_API_KEY") or k.endswith("_TOKEN") for k in data.keys())

    def test_get_env_vars_marks_channel_managed_keys(self):
        from hermes_cli.web_server import _channel_managed_env_keys

        data = self.client.get("/api/env").json()
        # Every entry carries the classification the Keys page relies on.
        assert all("channel_managed" in info for info in data.values())

        channel_keys = _channel_managed_env_keys()
        # Messaging-platform credentials owned by the Channels page are flagged;
        # everything else stays visible on the Keys page.
        for key, info in data.items():
            assert info["channel_managed"] is (key in channel_keys)

    def test_get_env_vars_surfaces_catalog_providers(self):
        """Every keys-tab provider in the unified catalog must appear in /api/env
        as a provider card, even when it has no hand entry in OPTIONAL_ENV_VARS.

        Regression for the GUI⇄CLI drift: openai-api, kilocode, novita,
        tencent-tokenhub, copilot were configurable via `hermes model` but
        invisible in the desktop Providers → API keys tab.
        """
        from hermes_cli.provider_catalog import provider_catalog

        data = self.client.get("/api/env").json()
        for d in provider_catalog():
            if d.tab != "keys" or not d.api_key_env_vars:
                continue
            # The PRIMARY credential var must surface as this provider's card.
            # (Shared aliases like GITHUB_TOKEN are intentionally left on their
            # existing tool category and not hijacked — see the copilot test.)
            primary = d.api_key_env_vars[0]
            assert primary in data, f"{primary} ({d.slug}) missing from /api/env"
            info = data[primary]
            assert info["category"] == "provider"
            assert info["provider"] == d.slug
            assert info["provider_label"] == d.label

    def test_get_env_vars_provider_rows_carry_grouping_hints(self):
        """Provider env rows expose the backend `provider`/`provider_label` the
        desktop Keys tab groups by (so it no longer relies on prefix guesses)."""
        data = self.client.get("/api/env").json()
        # OPENAI_API_KEY is a hand-listed protected var AND a catalog provider;
        # it must come back tagged to the openai-api provider.
        assert data["OPENAI_API_KEY"]["provider"] == "openai-api"
        assert data["OPENAI_API_KEY"]["category"] == "provider"

    def test_get_env_vars_copilot_uses_provider_token_not_shared_github_token(self):
        """Copilot surfaces as its own provider card via COPILOT_GITHUB_TOKEN;
        the shared GITHUB_TOKEN keeps its existing (tool) category."""
        data = self.client.get("/api/env").json()
        assert data["COPILOT_GITHUB_TOKEN"]["provider"] == "copilot"
        assert data["COPILOT_GITHUB_TOKEN"]["category"] == "provider"
        # Shared GITHUB_TOKEN must NOT be hijacked into the copilot provider card.
        assert data.get("GITHUB_TOKEN", {}).get("provider", "") != "copilot"

    def test_get_env_vars_bedrock_aws_vars_tagged_to_provider(self):
        """Bedrock (aws_sdk, no api-key) must still appear on the Keys tab: its
        AWS_REGION/AWS_PROFILE settings are tagged to the bedrock provider card.
        """
        data = self.client.get("/api/env").json()
        assert data["AWS_REGION"]["provider"] == "bedrock"
        assert data["AWS_REGION"]["category"] == "provider"
        assert data["AWS_PROFILE"]["provider"] == "bedrock"

    def test_platform_scoped_messaging_env_vars_are_channel_managed(self):
        from hermes_cli.web_server import (
            _MESSAGING_KEYS_PAGE_KEYS,
            _build_catalog_entry,
            _channel_managed_env_keys,
        )

        discord = _build_catalog_entry("discord")
        assert "DISCORD_HOME_CHANNEL" in discord["env_vars"]
        assert "DISCORD_ALLOW_ALL_USERS" in discord["env_vars"]

        managed = _channel_managed_env_keys()
        assert "DISCORD_HOME_CHANNEL" in managed
        assert "BLUEBUBBLES_ALLOW_ALL_USERS" in managed
        assert "MATTERMOST_ALLOW_ALL_USERS" in managed
        assert "GATEWAY_PROXY_URL" not in managed
        assert "GATEWAY_PROXY_URL" in _MESSAGING_KEYS_PAGE_KEYS

    def test_model_set_requires_confirmation_for_expensive_model(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: SimpleNamespace(message="EXPENSIVE MODEL WARNING"),
        )

        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "nous",
                "model": "openai/gpt-5.5-pro",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["confirm_required"] is True
        assert data["confirm_message"] == "EXPENSIVE MODEL WARNING"

        confirmed = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "nous",
                "model": "openai/gpt-5.5-pro",
                "confirm_expensive_model": True,
            },
        )

        assert confirmed.status_code == 200
        assert confirmed.json()["ok"] is True

    def test_model_set_normalizes_vendor_slug_for_native_provider(self, monkeypatch):
        """'Use as → Main' with an OpenRouter slug + native provider must not
        persist the vendor-prefixed slug verbatim (it 400s against the native
        API and reads as "changing models does nothing")."""
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: None,
        )
        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "anthropic",
                "model": "anthropic/claude-opus-4.6",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["provider"] == "anthropic"
        # Vendor prefix stripped + dots→hyphens for the native Anthropic API.
        assert data["model"] == "claude-opus-4-6"

        from hermes_cli.config import load_config
        cfg = load_config()
        assert cfg["model"]["provider"] == "anthropic"
        assert cfg["model"]["default"] == "claude-opus-4-6"

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

    def test_model_set_keeps_aggregator_slug_unchanged(self, monkeypatch):
        """The happy path (picker → openrouter + vendor/model) is untouched."""
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: None,
        )
        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["provider"] == "openrouter"
        assert data["model"] == "anthropic/claude-sonnet-4.6"

    def test_ops_import_passes_force_flag(self, tmp_path, monkeypatch):
        """force=True must append --force so the spawned non-interactive
        `hermes import` doesn't auto-abort at the overwrite prompt."""
        import hermes_cli.web_server as ws

        archive = tmp_path / "backup.zip"
        import zipfile
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("config.yaml", "model: {}\n")

        captured = {}

        def fake_spawn(subcommand, name):
            captured["args"] = subcommand
            captured["name"] = name
            from types import SimpleNamespace as NS
            return NS(pid=12345)

        monkeypatch.setattr(ws, "_spawn_hermes_action", fake_spawn)

        resp = self.client.post(
            "/api/ops/import", json={"archive": str(archive), "force": True},
        )
        assert resp.status_code == 200
        assert captured["args"] == ["import", str(archive), "--force"]

        resp = self.client.post(
            "/api/ops/import", json={"archive": str(archive)},
        )
        assert resp.status_code == 200
        assert captured["args"] == ["import", str(archive)]

    def test_ops_backup_defaults_to_dashboard_downloadable_archive(self, monkeypatch):
        from pathlib import Path

        import hermes_cli.web_server as ws
        from hermes_cli.config import get_hermes_home

        captured = {}

        def fake_spawn(subcommand, name):
            captured["args"] = subcommand
            captured["name"] = name
            from types import SimpleNamespace as NS
            return NS(pid=12345)

        monkeypatch.setattr(ws, "_spawn_hermes_action", fake_spawn)

        resp = self.client.post("/api/ops/backup", json={})
        assert resp.status_code == 200
        data = resp.json()
        archive = Path(data["archive"])

        assert data["name"] == "backup"
        assert captured["name"] == "backup"
        assert captured["args"] == ["backup", "-o", str(archive)]
        assert archive.parent == get_hermes_home() / "backups"
        assert archive.name.startswith("hermes-backup-")
        assert archive.suffix == ".zip"

    def test_ops_backup_uses_hosted_hermes_home(self, tmp_path, monkeypatch):
        from pathlib import Path

        import hermes_cli.web_server as ws

        hosted_home = tmp_path / "opt-data"
        monkeypatch.setenv("HERMES_HOME", str(hosted_home))
        captured = {}

        def fake_spawn(subcommand, name):
            captured["args"] = subcommand
            captured["name"] = name
            from types import SimpleNamespace as NS
            return NS(pid=12345)

        monkeypatch.setattr(ws, "_spawn_hermes_action", fake_spawn)

        resp = self.client.post("/api/ops/backup", json={})
        assert resp.status_code == 200
        archive = Path(resp.json()["archive"])

        assert archive.parent == hosted_home / "backups"
        assert captured["args"] == ["backup", "-o", str(archive)]
        assert archive.parent.is_dir()

    def test_ops_backup_download_streams_dashboard_backup(self, tmp_path):
        import hermes_cli.web_server as ws

        backup_dir = ws._dashboard_backup_dir()
        backup_dir.mkdir(parents=True, exist_ok=True)
        archive = backup_dir / "hermes-backup-test.zip"
        archive.write_bytes(b"zip bytes")

        resp = self.client.get(
            "/api/ops/backup/download",
            params={"archive": str(archive)},
        )
        assert resp.status_code == 200
        assert resp.content == b"zip bytes"
        assert "attachment" in resp.headers["content-disposition"]

        outside = tmp_path / "outside.zip"
        outside.write_bytes(b"nope")
        denied = self.client.get(
            "/api/ops/backup/download",
            params={"archive": str(outside)},
        )
        assert denied.status_code == 403

    def test_ops_import_upload_stages_archive_and_passes_force(self, tmp_path, monkeypatch):
        import zipfile
        from pathlib import Path

        import hermes_cli.web_server as ws

        archive = tmp_path / "backup.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("config.yaml", "model: {}\n")

        captured = {}

        def fake_spawn(subcommand, name):
            captured["args"] = subcommand
            captured["name"] = name
            from types import SimpleNamespace as NS
            return NS(pid=12345)

        monkeypatch.setattr(ws, "_spawn_hermes_action", fake_spawn)

        resp = self.client.post(
            "/api/ops/import-upload",
            data={"force": "true"},
            files={
                "file": (
                    "my backup.zip",
                    archive.read_bytes(),
                    "application/zip",
                ),
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "import"
        assert data["uploaded_bytes"] == archive.stat().st_size
        staged = Path(captured["args"][1])
        assert captured["name"] == "import"
        assert captured["args"] == ["import", str(staged), "--force"]
        assert staged.is_file()
        assert staged.name.startswith("dashboard-import-")
        assert staged.name.endswith("-my-backup.zip")
        assert zipfile.is_zipfile(staged)
        assert data["archive"] == str(staged)

    def test_ops_import_upload_rejects_invalid_zip(self, monkeypatch):
        import hermes_cli.web_server as ws

        def fail_spawn(*_args):
            raise AssertionError("invalid uploads must not spawn import")

        monkeypatch.setattr(ws, "_spawn_hermes_action", fail_spawn)

        resp = self.client.post(
            "/api/ops/import-upload",
            data={"force": "true"},
            files={"file": ("backup.zip", b"not a zip", "application/zip")},
        )

        assert resp.status_code == 400
        assert "valid zip" in resp.json()["detail"]


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

    def test_reveal_env_var_not_found(self):
        """POST /api/env/reveal should 404 for unknown keys."""
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "NONEXISTENT_KEY_XYZ"},
            headers={_SESSION_HEADER_NAME: _SESSION_TOKEN},
        )
        assert resp.status_code == 404

    def test_reveal_env_var_no_token(self, tmp_path):
        """POST /api/env/reveal without token should return 401."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app
        from hermes_cli.config import save_env_value
        save_env_value("TEST_REVEAL_NOAUTH", "secret-value")
        # Use a fresh client WITHOUT the dashboard session header
        unauth_client = TestClient(app)
        resp = unauth_client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_NOAUTH"},
        )
        assert resp.status_code == 401

    def test_reveal_env_var_bad_token(self, tmp_path):
        """POST /api/env/reveal with wrong token should return 401."""
        from hermes_cli.config import save_env_value
        from hermes_cli.web_server import _SESSION_HEADER_NAME
        save_env_value("TEST_REVEAL_BADAUTH", "secret-value")
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_BADAUTH"},
            headers={_SESSION_HEADER_NAME: "wrong-token-here"},
        )
        assert resp.status_code == 401

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

    def test_get_messaging_platforms(self):
        resp = self.client.get("/api/messaging/platforms")

        assert resp.status_code == 200
        platforms = resp.json()["platforms"]
        telegram = next(platform for platform in platforms if platform["id"] == "telegram")
        assert telegram["name"] == "Telegram"
        assert telegram["enabled"] is False
        fields = {field["key"]: field for field in telegram["env_vars"]}
        assert fields["TELEGRAM_BOT_TOKEN"]["required"] is True
        assert fields["TELEGRAM_BOT_TOKEN"]["url"] == "https://t.me/BotFather"
        assert "Complete Telegram bot token" in fields["TELEGRAM_BOT_TOKEN"]["description"]
        assert fields["TELEGRAM_ALLOWED_USERS"]["url"] == "https://t.me/userinfobot"
        assert "DM pairing" in fields["TELEGRAM_ALLOWED_USERS"]["description"]

    def test_slack_messaging_platform_exposes_user_allowlist(self):
        resp = self.client.get("/api/messaging/platforms")

        assert resp.status_code == 200
        platforms = resp.json()["platforms"]
        slack = next(platform for platform in platforms if platform["id"] == "slack")
        fields = {field["key"]: field for field in slack["env_vars"]}

        assert "allowed Slack member IDs" in slack["description"]
        assert set(fields) >= {
            "SLACK_BOT_TOKEN",
            "SLACK_APP_TOKEN",
            "SLACK_ALLOWED_USERS",
        }
        assert fields["SLACK_ALLOWED_USERS"]["prompt"] == "Allowed Slack member IDs"
        assert fields["SLACK_ALLOWED_USERS"]["is_password"] is False
        assert "member IDs" in fields["SLACK_ALLOWED_USERS"]["description"]
        assert "Bot User OAuth Token" in fields["SLACK_BOT_TOKEN"]["help"]
        assert "App-Level Tokens" in fields["SLACK_APP_TOKEN"]["help"]
        assert "Copy member ID" in fields["SLACK_ALLOWED_USERS"]["help"]

    def test_weixin_messaging_metadata_describes_personal_ilink_setup(self):
        resp = self.client.get("/api/messaging/platforms")

        assert resp.status_code == 200
        weixin = next(
            platform
            for platform in resp.json()["platforms"]
            if platform["id"] == "weixin"
        )
        assert weixin["name"] == "Weixin / WeChat (Personal)"
        assert "personal WeChat" in weixin["description"]
        assert "Official Account" not in f"{weixin['name']} {weixin['description']}"
        assert weixin["docs_url"] == (
            "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin/"
        )

        fields = {field["key"]: field for field in weixin["env_vars"]}
        for key in ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL"):
            assert "iLink" in fields[key]["description"]
            assert "QR login" in fields[key]["description"]
            assert "Official Account" not in fields[key]["description"]

    def test_teams_messaging_metadata_links_setup_guide(self):
        # Teams is a platform plugin, so the catalog entry is built from the
        # plugin registry. The override must still supply a docs link so the
        # Channels page renders a working "Open setup guide" button instead of
        # an empty href (which resolves to the packaged app's own index.html).
        from hermes_cli.web_server import _build_catalog_entry

        teams = _build_catalog_entry("teams")
        assert teams["docs_url"] == (
            "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/teams"
        )

    def test_google_chat_messaging_metadata_links_setup_guide(self):
        # Google Chat is a platform plugin, so the catalog entry is built from
        # the plugin registry. The override must supply a docs link so the
        # Channels page renders a working "Open setup guide" button instead of
        # an empty href (which resolves to the packaged app's own index.html).
        from hermes_cli.web_server import _build_catalog_entry

        google_chat = _build_catalog_entry("google_chat")
        assert google_chat["name"] == "Google Chat"
        assert google_chat["docs_url"] == (
            "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/google_chat"
        )

    def test_messaging_catalog_covers_gateway_platforms(self):
        """Catalog is derived from the Platform enum, so every built-in shows up."""
        from gateway.config import Platform

        resp = self.client.get("/api/messaging/platforms")
        platforms = {entry["id"] for entry in resp.json()["platforms"]}

        for member in Platform.__members__.values():
            if member.value == "local":
                continue
            assert member.value in platforms, f"Missing gateway platform {member.value} from /api/messaging/platforms"

    def test_messaging_catalog_includes_plugin_platforms(self, monkeypatch):
        """Plugin-registered adapters appear in the catalog without per-platform code."""
        from gateway.platform_registry import PlatformEntry, platform_registry

        entry = PlatformEntry(
            name="ircfake",
            label="IRC (test)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            required_env=["IRC_SERVER"],
            install_hint="Connect to IRC.",
            source="plugin",
        )
        platform_registry.register(entry)
        try:
            resp = self.client.get("/api/messaging/platforms")
            ids = {row["id"]: row for row in resp.json()["platforms"]}
            assert "ircfake" in ids
            assert ids["ircfake"]["name"] == "IRC (test)"
            assert any(field["key"] == "IRC_SERVER" and field["required"] for field in ids["ircfake"]["env_vars"])
        finally:
            platform_registry.unregister("ircfake")

    def test_update_messaging_platform_saves_env_and_enablement(self):
        from hermes_cli.config import load_config, load_env

        resp = self.client.put(
            "/api/messaging/platforms/telegram",
            json={
                "enabled": False,
                "env": {"TELEGRAM_BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234"},
            },
        )

        assert resp.status_code == 200
        assert load_env()["TELEGRAM_BOT_TOKEN"] == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234"
        assert load_config()["platforms"]["telegram"]["enabled"] is False

        status = self.client.get("/api/messaging/platforms").json()["platforms"]
        telegram = next(platform for platform in status if platform["id"] == "telegram")
        assert telegram["enabled"] is False

    def test_update_messaging_platform_rejects_invalid_telegram_bot_token(self):
        resp = self.client.put(
            "/api/messaging/platforms/telegram",
            json={"env": {"TELEGRAM_BOT_TOKEN": "not-a-botfather-token"}},
        )

        assert resp.status_code == 400
        assert "@BotFather" in resp.json()["detail"]

    def test_update_messaging_platform_rejects_invalid_telegram_allowed_users(self):
        resp = self.client.put(
            "/api/messaging/platforms/telegram",
            json={"env": {"TELEGRAM_ALLOWED_USERS": "123456,@username"}},
        )

        assert resp.status_code == 400
        assert "numeric user IDs" in resp.json()["detail"]

    def test_update_messaging_platform_saves_slack_allowed_users(self):
        from hermes_cli.config import load_env

        resp = self.client.put(
            "/api/messaging/platforms/slack",
            json={"env": {"SLACK_ALLOWED_USERS": "U01ABC2DEF3,U04XYZ5LMN6"}},
        )

        assert resp.status_code == 200
        assert load_env()["SLACK_ALLOWED_USERS"] == "U01ABC2DEF3,U04XYZ5LMN6"

    def test_update_messaging_platform_rejects_swapped_slack_bot_token(self):
        resp = self.client.put(
            "/api/messaging/platforms/slack",
            json={"env": {"SLACK_BOT_TOKEN": "xapp-wrong-token-type"}},
        )

        assert resp.status_code == 400
        assert "xoxb-" in resp.json()["detail"]

    def test_update_messaging_platform_rejects_swapped_slack_app_token(self):
        resp = self.client.put(
            "/api/messaging/platforms/slack",
            json={"env": {"SLACK_APP_TOKEN": "xoxb-wrong-token-type"}},
        )

        assert resp.status_code == 400
        assert "xapp-" in resp.json()["detail"]

    def test_update_messaging_platform_rejects_invalid_slack_allowed_users(self):
        resp = self.client.put(
            "/api/messaging/platforms/slack",
            json={"env": {"SLACK_ALLOWED_USERS": "U01ABC2DEF3,not-a-user"}},
        )

        assert resp.status_code == 400
        assert "member IDs" in resp.json()["detail"]

    def test_update_messaging_platform_accepts_slack_allowed_users_wildcard(self):
        # "*" is the gateway's allow-all wildcard (gateway/platforms/slack.py),
        # so the dashboard must accept it rather than rejecting it as malformed.
        from hermes_cli.config import load_env

        resp = self.client.put(
            "/api/messaging/platforms/slack",
            json={"env": {"SLACK_ALLOWED_USERS": "*"}},
        )

        assert resp.status_code == 200
        assert load_env()["SLACK_ALLOWED_USERS"] == "*"

    def test_update_messaging_platform_accepts_slack_allowed_users_trailing_comma(self):
        # The gateway drops empty entries (gateway/platforms/slack.py), so a
        # trailing/interior comma must not be rejected by the dashboard.
        from hermes_cli.config import load_env

        resp = self.client.put(
            "/api/messaging/platforms/slack",
            json={"env": {"SLACK_ALLOWED_USERS": "U01ABC2DEF3,,W04XYZ5LMN6,"}},
        )

        assert resp.status_code == 200
        assert load_env()["SLACK_ALLOWED_USERS"] == "U01ABC2DEF3,,W04XYZ5LMN6,"

    def test_messaging_platform_test_reports_missing_required_setup(self):
        resp = self.client.put("/api/messaging/platforms/discord", json={"enabled": True})
        assert resp.status_code == 200

        resp = self.client.post("/api/messaging/platforms/discord/test")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["state"] == "not_configured"
        assert "DISCORD_BOT_TOKEN" in data["message"]

    def test_telegram_onboarding_worker_request_uses_httpx(self, monkeypatch):
        import httpx
        import hermes_cli.web_server as ws

        calls = {}

        def fail_urlopen(*_args, **_kwargs):
            raise AssertionError("Telegram onboarding should not use urllib")

        class FakeHttpxClient:
            def __init__(self, *args, **kwargs):
                calls["client_kwargs"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info):
                return False

            def request(self, method, url, **kwargs):
                calls["request"] = (method, url, kwargs)
                return httpx.Response(
                    201,
                    json={"ok": True},
                    request=httpx.Request(method, url),
                )

        monkeypatch.setenv("TELEGRAM_ONBOARDING_URL", "https://worker.example")
        monkeypatch.setattr(ws.urllib.request, "urlopen", fail_urlopen)
        monkeypatch.setattr(httpx, "Client", FakeHttpxClient)

        payload = ws._telegram_onboarding_request_sync(
            "POST",
            "/v1/telegram/pairings",
            body={"bot_name": "Hermes Agent"},
            bearer_token="poll-secret",
        )

        assert payload == {"ok": True}
        method, url, kwargs = calls["request"]
        assert method == "POST"
        assert url == "https://worker.example/v1/telegram/pairings"
        assert kwargs["json"] == {"bot_name": "Hermes Agent"}
        assert kwargs["headers"]["Accept"] == "application/json"
        assert kwargs["headers"]["Authorization"] == "Bearer poll-secret"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["User-Agent"].startswith("HermesDashboard/")

    def test_telegram_onboarding_worker_request_maps_unexpected_errors(
        self, monkeypatch
    ):
        import hermes_cli.web_server as ws

        monkeypatch.setenv("TELEGRAM_ONBOARDING_URL", "not a valid url")

        with pytest.raises(ws.HTTPException) as exc:
            ws._telegram_onboarding_request_sync(
                "POST",
                "/v1/telegram/pairings",
                body={"bot_name": "Hermes Agent"},
            )

        assert exc.value.status_code == 502
        assert (
            exc.value.detail
            == "Telegram setup service is unavailable. Try again shortly."
        )

    def test_telegram_onboarding_start_strips_poll_token(self, monkeypatch):
        import hermes_cli.web_server as ws

        with ws._telegram_onboarding_lock:
            ws._telegram_onboarding_pairings.clear()

        calls = []

        def fake_request(method, path, *, body=None, bearer_token=None):
            calls.append((method, path, body, bearer_token))
            return {
                "pairing_id": "pair123",
                "poll_token": "poll-secret",
                "suggested_username": "hermes_pair123_bot",
                "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_pair123_bot",
                "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_pair123_bot",
                "expires_at": "2027-05-18T00:00:00.000Z",
            }

        monkeypatch.setattr(ws, "_telegram_onboarding_request_sync", fake_request)

        resp = self.client.post(
            "/api/messaging/telegram/onboarding/start",
            json={"bot_name": "Hosted Hermes"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["pairing_id"] == "pair123"
        assert "poll_token" not in data
        assert calls == [
            (
                "POST",
                "/v1/telegram/pairings",
                {"bot_name": "Hosted Hermes"},
                None,
            )
        ]

    def test_telegram_onboarding_ready_and_apply_never_returns_bot_token(self, monkeypatch):
        import hermes_cli.web_server as ws
        from hermes_cli.config import load_config, load_env

        with ws._telegram_onboarding_lock:
            ws._telegram_onboarding_pairings.clear()

        def fake_request(method, path, *, body=None, bearer_token=None):
            if method == "POST":
                return {
                    "pairing_id": "pair-ready",
                    "poll_token": "poll-secret",
                    "suggested_username": "hermes_pair_ready_bot",
                    "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_pair_ready_bot",
                    "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_pair_ready_bot",
                    "expires_at": "2027-05-18T00:00:00.000Z",
                }
            assert method == "GET"
            assert path == "/v1/telegram/pairings/pair-ready"
            assert bearer_token == "poll-secret"
            return {
                "status": "ready",
                "bot_username": "hermes_pair_ready_bot",
                "owner_user_id": 123456789,
                "token": "123456:SECRET",
            }

        monkeypatch.setattr(ws, "_telegram_onboarding_request_sync", fake_request)
        ws._ACTION_PROCS.pop("gateway-restart", None)
        restart_calls = []

        class FakeRestartProc:
            pid = 4242

        def fake_spawn_action(subcommand, name):
            restart_calls.append((subcommand, name))
            return FakeRestartProc()

        monkeypatch.setattr(ws, "_spawn_hermes_action", fake_spawn_action)

        start = self.client.post("/api/messaging/telegram/onboarding/start", json={})
        assert start.status_code == 200

        ready = self.client.get("/api/messaging/telegram/onboarding/pair-ready")
        assert ready.status_code == 200
        ready_data = ready.json()
        assert ready_data["status"] == "ready"
        assert ready_data["owner_user_id"] == "123456789"
        assert "token" not in ready_data

        applied = self.client.post(
            "/api/messaging/telegram/onboarding/pair-ready/apply",
            json={"allowed_user_ids": ["123456789", "123456789"]},
        )
        assert applied.status_code == 200
        applied_data = applied.json()
        assert applied_data == {
            "ok": True,
            "platform": "telegram",
            "bot_username": "hermes_pair_ready_bot",
            "needs_restart": False,
            "restart_started": True,
            "restart_action": "gateway-restart",
            "restart_pid": 4242,
        }
        assert restart_calls == [(["gateway", "restart"], "gateway-restart")]
        env = load_env()
        assert env["TELEGRAM_BOT_TOKEN"] == "123456:SECRET"
        assert env["TELEGRAM_ALLOWED_USERS"] == "123456789"
        assert load_config()["platforms"]["telegram"]["enabled"] is True

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

    def test_telegram_onboarding_apply_reuses_inflight_gateway_restart(
        self, monkeypatch
    ):
        """A live in-flight gateway restart is reused instead of spawning a
        second racing ``hermes gateway restart`` child (e.g. when a stale
        cached frontend also fires its own restart call)."""
        import hermes_cli.web_server as ws

        with ws._telegram_onboarding_lock:
            ws._telegram_onboarding_pairings.clear()

        def fake_request(method, path, *, body=None, bearer_token=None):
            if method == "POST":
                return {
                    "pairing_id": "pair-reuse",
                    "poll_token": "poll-secret",
                    "suggested_username": "hermes_pair_reuse_bot",
                    "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_pair_reuse_bot",
                    "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_pair_reuse_bot",
                    "expires_at": "2027-05-18T00:00:00.000Z",
                }
            return {
                "status": "ready",
                "bot_username": "hermes_pair_reuse_bot",
                "owner_user_id": 123456789,
                "token": "123456:SECRET",
            }

        monkeypatch.setattr(ws, "_telegram_onboarding_request_sync", fake_request)

        class FakeRunningProc:
            pid = 5151

            def poll(self):
                return None  # still running

        monkeypatch.setitem(ws._ACTION_PROCS, "gateway-restart", FakeRunningProc())

        def fail_spawn_action(subcommand, name):
            raise AssertionError("must not spawn a second concurrent restart")

        monkeypatch.setattr(ws, "_spawn_hermes_action", fail_spawn_action)

        start = self.client.post("/api/messaging/telegram/onboarding/start", json={})
        assert start.status_code == 200
        ready = self.client.get("/api/messaging/telegram/onboarding/pair-reuse")
        assert ready.status_code == 200

        applied = self.client.post(
            "/api/messaging/telegram/onboarding/pair-reuse/apply",
            json={"allowed_user_ids": ["123456789"]},
        )

        assert applied.status_code == 200
        applied_data = applied.json()
        assert applied_data["needs_restart"] is False
        assert applied_data["restart_started"] is True
        assert applied_data["restart_pid"] == 5151

    def test_telegram_onboarding_apply_requires_ready_pairing(self, monkeypatch):
        import hermes_cli.web_server as ws

        with ws._telegram_onboarding_lock:
            ws._telegram_onboarding_pairings.clear()

        def fake_request(method, path, *, body=None, bearer_token=None):
            return {
                "pairing_id": "pair-waiting",
                "poll_token": "poll-secret",
                "suggested_username": "hermes_pair_waiting_bot",
                "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_pair_waiting_bot",
                "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_pair_waiting_bot",
                "expires_at": "2027-05-18T00:00:00.000Z",
            }

        monkeypatch.setattr(ws, "_telegram_onboarding_request_sync", fake_request)

        start = self.client.post("/api/messaging/telegram/onboarding/start", json={})
        assert start.status_code == 200

        resp = self.client.post(
            "/api/messaging/telegram/onboarding/pair-waiting/apply",
            json={"allowed_user_ids": ["123456789"]},
        )

        assert resp.status_code == 409
        assert "not ready" in resp.json()["detail"]

    def test_telegram_onboarding_cancel_clears_local_session(self, monkeypatch):
        import hermes_cli.web_server as ws

        with ws._telegram_onboarding_lock:
            ws._telegram_onboarding_pairings.clear()

        def fake_request(method, path, *, body=None, bearer_token=None):
            return {
                "pairing_id": "pair-cancel",
                "poll_token": "poll-secret",
                "suggested_username": "hermes_pair_cancel_bot",
                "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_pair_cancel_bot",
                "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_pair_cancel_bot",
                "expires_at": "2027-05-18T00:00:00.000Z",
            }

        monkeypatch.setattr(ws, "_telegram_onboarding_request_sync", fake_request)

        start = self.client.post("/api/messaging/telegram/onboarding/start", json={})
        assert start.status_code == 200

        cancel = self.client.delete("/api/messaging/telegram/onboarding/pair-cancel")
        assert cancel.status_code == 200

        status = self.client.get("/api/messaging/telegram/onboarding/pair-cancel")
        assert status.status_code == 404

    def test_session_token_endpoint_removed(self):
        """GET /api/auth/session-token should no longer exist (token injected via HTML)."""
        resp = self.client.get("/api/auth/session-token")
        # The endpoint is gone — the catch-all SPA route serves index.html
        # or the middleware returns 401 for unauthenticated /api/ paths.
        assert resp.status_code in {200, 404}
        # Either way, it must NOT return the token as JSON
        try:
            data = resp.json()
            assert "token" not in data
        except Exception:
            pass  # Not JSON — that's fine (SPA HTML)

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

    def test_path_traversal_blocked(self):
        """Verify URL-encoded path traversal is blocked."""
        # %2e%2e = ..
        resp = self.client.get("/%2e%2e/%2e%2e/etc/passwd")
        # Should return 200 with index.html (SPA fallback), not the actual file
        assert resp.status_code in {200, 404}
        if resp.status_code == 200:
            # Should be the SPA fallback, not the system file
            assert "root:" not in resp.text

    def test_path_traversal_dotdot_blocked(self):
        """Direct .. path traversal via encoded sequences."""
        resp = self.client.get("/%2e%2e/hermes_cli/web_server.py")
        assert resp.status_code in {200, 404}
        if resp.status_code == 200:
            assert "FastAPI" not in resp.text  # Should not serve the actual source

    def test_spa_assets_are_read_as_utf8(self, monkeypatch, tmp_path):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import hermes_cli.web_server as ws

        dist = tmp_path / "web_dist"
        assets = dist / "assets"
        assets.mkdir(parents=True)
        index_path = dist / "index.html"
        css_path = assets / "app.css"
        index_path.write_text("<html><head></head><body>cafe cafe</body></html>", encoding="utf-8")
        css_path.write_text("body::before { content: 'cafe'; }", encoding="utf-8")

        original_read_text = Path.read_text
        seen_encodings = {}

        def tracking_read_text(path_self, *args, **kwargs):
            if path_self == index_path:
                seen_encodings["index"] = kwargs.get("encoding")
            elif path_self == css_path:
                seen_encodings["css"] = kwargs.get("encoding")
            return original_read_text(path_self, *args, **kwargs)

        monkeypatch.setattr(ws, "WEB_DIST", dist)
        monkeypatch.setattr(Path, "read_text", tracking_read_text)
        spa_app = FastAPI()
        ws.mount_spa(spa_app)
        spa_client = TestClient(spa_app)

        index_resp = spa_client.get("/chat")
        assert index_resp.status_code == 200
        assert "cafe cafe" in index_resp.text

        css_resp = spa_client.get("/assets/app.css", headers={"x-forwarded-prefix": "/hermes"})
        assert css_resp.status_code == 200
        assert "content: 'cafe';" in css_resp.text

        assert seen_encodings == {"index": "utf-8", "css": "utf-8"}

    def test_headless_serve_disables_spa_even_with_a_dist(self, monkeypatch, tmp_path):
        """`hermes serve` (HERMES_SERVE_HEADLESS) must NOT serve the SPA even
        when a built dist is present — only the API/WS surface is reachable."""
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import hermes_cli.web_server as ws

        dist = tmp_path / "web_dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<html><body>UI</body></html>", encoding="utf-8")

        monkeypatch.setattr(ws, "WEB_DIST", dist)
        monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
        app_ = FastAPI()
        ws.mount_spa(app_)

        for route in ("/", "/chat"):
            resp = TestClient(app_).get(route)
            assert resp.status_code == 404
            assert "web UI disabled" in resp.json()["error"]

    def test_set_model_main_nous_applies_gateway_defaults(self, monkeypatch):
        """Switching the main provider to Nous calls apply_nous_managed_defaults
        (mirroring the CLI's post-model-selection Tool Gateway routing) and
        surfaces the routed tools in the response."""
        import hermes_cli.nous_subscription as ns

        called = {}

        def fake_apply(config, *, enabled_toolsets=None, force_fresh=False):
            called["enabled"] = set(enabled_toolsets or ())
            called["force_fresh"] = force_fresh
            # Simulate routing the unconfigured web tool through the gateway.
            web = config.setdefault("web", {})
            web["backend"] = "firecrawl"
            return {"web"}

        monkeypatch.setattr(ns, "apply_nous_managed_defaults", fake_apply)

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "nous", "model": "hermes-4"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["provider"] == "nous"
        assert data["gateway_tools"] == ["web"]
        assert called["force_fresh"] is True

    def test_set_model_main_non_nous_skips_gateway_defaults(self, monkeypatch):
        """Non-Nous providers must NOT trigger Tool Gateway auto-routing."""
        import hermes_cli.nous_subscription as ns

        def boom(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("apply_nous_managed_defaults called for non-nous provider")

        monkeypatch.setattr(ns, "apply_nous_managed_defaults", boom)

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data.get("gateway_tools", []) == []

    def test_apply_main_model_assignment_base_url_and_context_reconcile(self):
        """The shared main-slot assignment helper must persist a supplied
        base_url, clear a stale base_url only when switching providers, preserve
        it on same-provider re-assignment, and always drop a hardcoded
        context_length override. Both POST /api/model/set and profile-model
        writes route through this, so the contract is pinned here."""
        from hermes_cli.web_server import _apply_main_model_assignment

        # Custom + base_url → persisted; stale context_length dropped.
        out = _apply_main_model_assignment(
            {"context_length": 8192}, "custom", "llama-3.1-8b", "http://127.0.0.1:8000/v1"
        )
        assert out["provider"] == "custom"
        assert out["default"] == "llama-3.1-8b"
        assert out["base_url"] == "http://127.0.0.1:8000/v1"
        assert "context_length" not in out

        # Switching providers (custom → openrouter) → stale base_url cleared.
        out = _apply_main_model_assignment(
            {"provider": "custom", "base_url": "http://127.0.0.1:8000/v1"},
            "openrouter",
            "anthropic/claude-opus-4.8",
        )
        assert out["provider"] == "openrouter"
        assert out["base_url"] == ""

        # Same provider, no new base_url → existing custom endpoint preserved.
        # Regression: picking a different MiMo model under xiaomi must NOT wipe a
        # Token Plan base_url (https://token-plan-*.xiaomimimo.com/v1).
        out = _apply_main_model_assignment(
            {"provider": "xiaomi", "base_url": "https://token-plan-ams.xiaomimimo.com/v1"},
            "xiaomi",
            "mimo-v2.5-pro",
        )
        assert out["provider"] == "xiaomi"
        assert out["default"] == "mimo-v2.5-pro"
        assert out["base_url"] == "https://token-plan-ams.xiaomimimo.com/v1"

        # A supplied base_url is honored for any provider, not just custom.
        out = _apply_main_model_assignment(
            {"provider": "xiaomi"},
            "xiaomi",
            "mimo-v2.5",
            "https://token-plan-cn.xiaomimimo.com/v1",
        )
        assert out["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"

        # Switching providers without a base_url → don't invent one, clear stale.
        out = _apply_main_model_assignment(
            {"provider": "openrouter", "base_url": "http://stale:1/v1"}, "custom", "m"
        )
        assert out["base_url"] == ""

        # Non-dict input is coerced to a fresh dict (never raises).
        out = _apply_main_model_assignment("not-a-dict", "custom", "m", "http://x/v1")
        assert out == {"provider": "custom", "default": "m", "base_url": "http://x/v1"}

        # api_key follows the same lifecycle as base_url:
        # supplied → persisted.
        out = _apply_main_model_assignment(
            {"api": "sk-legacy-old"}, "custom", "m", "http://x/v1", "sk-secret"
        )
        assert out["api_key"] == "sk-secret"
        assert "api" not in out

        # same provider, no new key → existing key preserved (re-picking a model
        # on the same custom endpoint must not wipe the saved key).
        out = _apply_main_model_assignment(
            {"provider": "custom", "base_url": "http://x/v1", "api_key": "sk-keep"},
            "custom",
            "m2",
        )
        assert out["api_key"] == "sk-keep"

        # switching providers without a new key → stale key cleared.
        out = _apply_main_model_assignment(
            {"provider": "custom", "api_key": "sk-old", "api_mode": "anthropic_messages"},
            "openrouter",
            "m",
        )
        assert "api_key" not in out
        assert "api_mode" not in out

        # switching providers when the stale secret lives under the legacy
        # ``api`` alias only (no api_key) → it must be cleared too. The resolver
        # reads ``model.api`` as a key, so leaving it behind keeps a secret in
        # config.yaml that contaminates the next custom resolution.
        out = _apply_main_model_assignment(
            {"provider": "custom", "api": "sk-legacy-stale", "base_url": "http://endpoint-a/v1"},
            "openrouter",
            "m",
        )
        assert "api" not in out
        assert "api_key" not in out

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

    def test_set_model_main_custom_persists_base_url(self):
        """Custom/local providers must persist model.base_url so the runtime
        resolver (which ignores OPENAI_BASE_URL) can route to a self-hosted
        endpoint without an API key. Regression for the desktop onboarding bug
        where 'Local / custom endpoint' could never be configured."""
        from hermes_cli.config import load_config

        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom",
                "model": "llama-3.1-8b",
                "base_url": "http://127.0.0.1:8000/v1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["provider"] == "custom"
        assert data["base_url"] == "http://127.0.0.1:8000/v1"

        model_cfg = load_config().get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["provider"] == "custom"
        assert model_cfg["default"] == "llama-3.1-8b"
        assert model_cfg["base_url"] == "http://127.0.0.1:8000/v1"

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

    def test_set_model_main_non_custom_clears_stale_base_url(self):
        """Switching to a hosted provider must clear a stale base_url so the
        resolver picks that provider's own default endpoint."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["model"] = {
            "provider": "custom",
            "default": "llama-3.1-8b",
            "base_url": "http://127.0.0.1:8000/v1",
        }
        save_config(cfg)

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        )
        assert resp.status_code == 200
        assert resp.json()["base_url"] == ""

    def test_set_model_main_same_provider_preserves_base_url(self):
        """Re-picking a model under the SAME provider must NOT wipe a configured
        base_url. Regression for the desktop bug where selecting a Xiaomi MiMo
        model reset a Token Plan endpoint back to the registry default, breaking
        Token Plan keys (https://token-plan-*.xiaomimimo.com/v1)."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["model"] = {
            "provider": "xiaomi",
            "default": "mimo-v2.5-pro",
            "base_url": "https://token-plan-ams.xiaomimimo.com/v1",
        }
        save_config(cfg)

        # Desktop model picker sends provider+model only (no base_url).
        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "xiaomi", "model": "mimo-v2.5"},
        )
        assert resp.status_code == 200
        assert resp.json()["base_url"] == "https://token-plan-ams.xiaomimimo.com/v1"

        model_cfg = load_config().get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["default"] == "mimo-v2.5"
        assert model_cfg["base_url"] == "https://token-plan-ams.xiaomimimo.com/v1"

    def test_set_model_main_reports_stale_auxiliary_pins(self):
        """Switching the main provider must report auxiliary slots still pinned
        to a *different* provider so the UI can warn the user their helper tasks
        aren't following the switch (the silent credit-burn path)."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["model"] = {"provider": "nous", "default": "hermes-4"}
        cfg["auxiliary"] = {
            # Pinned to nous — same as the OLD main, becomes stale after switch.
            "compression": {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
            # Auto — follows main, never stale.
            "vision": {"provider": "auto", "model": ""},
            # Pinned to a third provider — also stale vs the new main.
            "curator": {"provider": "deepseek", "model": "deepseek-chat"},
        }
        save_config(cfg)

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        )
        assert resp.status_code == 200
        stale = resp.json()["stale_aux"]
        stale_tasks = {entry["task"] for entry in stale}
        assert stale_tasks == {"compression", "curator"}
        # auto slot must never appear.
        assert "vision" not in stale_tasks
        # Provider/model echoed back for the UI label.
        comp = next(e for e in stale if e["task"] == "compression")
        assert comp["provider"] == "nous"
        assert comp["model"] == "anthropic/claude-sonnet-4.6"

    def test_set_model_main_no_stale_when_aux_matches_new_provider(self):
        """Aux slots pinned to the SAME provider as the new main are not stale."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["model"] = {"provider": "nous", "default": "hermes-4"}
        cfg["auxiliary"] = {
            "compression": {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
            "vision": {"provider": "auto", "model": ""},
        }
        save_config(cfg)

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        )
        assert resp.status_code == 200
        assert resp.json()["stale_aux"] == []

        model_cfg = load_config().get("model")
        assert model_cfg["provider"] == "openrouter"
        assert model_cfg.get("base_url", "") == ""

    def test_custom_endpoints_list_includes_direct_custom_config(self):
        """A bare model.provider=custom config should show up in Desktop even
        before the user has materialized it under providers.
        """
        from hermes_cli.config import save_config

        save_config({
            "model": {
                "provider": "custom",
                "default": "gpt-5.4",
                "base_url": "http://127.0.0.1:8081/v1",
                "api_key": "sk-local",
            },
            "providers": {},
        })

        resp = self.client.get("/api/providers/custom-endpoints")

        assert resp.status_code == 200
        data = resp.json()
        assert data["current"] == {
            "provider": "custom",
            "model": "gpt-5.4",
            "base_url": "http://127.0.0.1:8081/v1",
        }
        assert data["endpoints"][0]["id"] == "custom"
        assert data["endpoints"][0]["source"] == "direct-config"
        assert data["endpoints"][0]["has_api_key"] is True

    def test_custom_endpoint_upsert_persists_provider_and_sets_default(self):
        """Desktop can persist an OpenAI-compatible proxy in providers and make
        it the default for new chats.
        """
        from hermes_cli.config import load_config

        resp = self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "axet-proxy",
                "name": "Axet Proxy",
                "base_url": "http://127.0.0.1:8081/v1/",
                "model": "gpt-5.4",
                "api_key": "sk-local",
                "context_length": 262144,
                "make_default": True,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["id"] == "axet-proxy"
        endpoint = next(e for e in data["endpoints"] if e["id"] == "axet-proxy")
        assert endpoint["base_url"] == "http://127.0.0.1:8081/v1"
        assert endpoint["model"] == "gpt-5.4"
        assert endpoint["is_current"] is True

        cfg = load_config()
        assert cfg["providers"]["axet-proxy"]["base_url"] == "http://127.0.0.1:8081/v1"
        assert cfg["providers"]["axet-proxy"]["models"]["gpt-5.4"]["context_length"] == 262144
        assert cfg["model"]["provider"] == "axet-proxy"
        assert cfg["model"]["default"] == "gpt-5.4"
        assert cfg["model"]["base_url"] == "http://127.0.0.1:8081/v1"

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

    def test_set_model_main_honors_an_explicitly_supplied_api_key(self):
        """A key in the request must win over the provider entry's stored one.

        The entry-key fallback exists so switching to a configured provider
        picks up its credential. Applying it unconditionally discards a key the
        caller is rotating in — and ``model.api_key`` outranks the environment
        at client construction (#62269), so the stale key keeps authenticating
        while the UI reports the change saved.
        """
        from hermes_cli.config import load_config

        self._seed_custom_provider_with_key()

        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "acme",
                "model": "acme/m1",
                "api_key": "sk-new-rotated",
            },
        )
        assert resp.status_code == 200
        assert load_config()["model"]["api_key"] == "sk-new-rotated"

    def test_set_model_main_falls_back_to_the_provider_entry_key(self):
        """With no key in the request the stored one is still adopted."""
        from hermes_cli.config import load_config

        self._seed_custom_provider_with_key()

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "acme", "model": "acme/m1"},
        )
        assert resp.status_code == 200
        model_cfg = load_config()["model"]
        assert model_cfg["api_key"] == "sk-stored-old"
        # The sibling base_url fill is unaffected.
        assert model_cfg["base_url"] == "https://llm.acme.corp/v1"

    def test_custom_endpoint_edit_preserves_hand_written_provider_fields(self):
        """The panel edits a few fields; it does not own the whole entry.

        A ``providers.<name>`` block can carry keys the dashboard has no field
        for — ``api_mode``, ``key_env``, ``extra_headers`` (which may carry
        credentials), ``request_overrides``. Rebuilding the entry from scratch
        on an unrelated edit silently dropped all of them, leaving a provider
        that no longer authenticates or speaks the right protocol.
        """
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["providers"] = {
            "acme": {
                "name": "Acme",
                "base_url": "https://llm.acme.corp/v1",
                "model": "acme/model-1",
                "api_mode": "responses",
                "key_env": "ACME_API_KEY",
                "extra_headers": {"X-Org-Id": "org_123"},
                "request_overrides": {"reasoning_effort": "high"},
                "models": {
                    "acme/model-1": {"context_length": 200000},
                    "acme/model-2": {"context_length": 400000},
                },
            }
        }
        save_config(cfg)

        # The user opens the panel and only switches the default model.
        resp = self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "acme",
                "name": "Acme",
                "base_url": "https://llm.acme.corp/v1",
                "model": "acme/model-2",
            },
        )
        assert resp.status_code == 200

        entry = load_config()["providers"]["acme"]
        assert entry["api_mode"] == "responses"
        assert entry["key_env"] == "ACME_API_KEY"
        assert entry["extra_headers"] == {"X-Org-Id": "org_123"}
        assert entry["request_overrides"] == {"reasoning_effort": "high"}
        # The edit still applies.
        assert entry["model"] == "acme/model-2"

    def test_custom_endpoint_edit_keeps_the_other_models(self):
        """The panel names one default model; it doesn't enumerate the catalogue."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["providers"] = {
            "acme": {
                "name": "Acme",
                "base_url": "https://llm.acme.corp/v1",
                "model": "acme/model-1",
                "models": {
                    "acme/model-1": {"context_length": 200000},
                    "acme/model-2": {"context_length": 400000},
                },
            }
        }
        save_config(cfg)

        self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "acme",
                "name": "Acme",
                "base_url": "https://llm.acme.corp/v1",
                "model": "acme/model-2",
            },
        )

        models = load_config()["providers"]["acme"]["models"]
        assert sorted(models) == ["acme/model-1", "acme/model-2"]
        assert models["acme/model-1"]["context_length"] == 200000

    def test_deleting_the_active_custom_endpoint_clears_its_model_mirror(self):
        """Deleting an endpoint must not leave its key running the agent.

        ``activate`` copies the endpoint's base_url + api_key onto ``model``,
        and ``model.api_key`` outranks the environment at client construction
        (#62269). Without clearing that mirror the agent keeps authenticating
        to the deleted host with the deleted key, and the key the operator
        just removed through the dashboard stays in config.yaml.
        """
        from hermes_cli.config import load_config

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

        cfg = load_config()
        assert cfg["model"]["api_key"] == "sk-acme-secret"

        assert self.client.request(
            "DELETE", "/api/providers/custom-endpoints/acme"
        ).status_code == 200

        cfg = load_config()
        assert "acme" not in (cfg.get("providers") or {})
        model_cfg = cfg.get("model") or {}
        assert not model_cfg.get("api_key"), "deleted endpoint's key still in config.yaml"
        assert not model_cfg.get("base_url"), "deleted endpoint's host still routed to"
        assert not model_cfg.get("provider")

    def test_deleting_an_inactive_custom_endpoint_leaves_the_active_one_alone(self):
        """Only the mirror of the DELETED provider is scrubbed."""
        from hermes_cli.config import load_config

        for name, key in (("acme", "sk-acme"), ("other", "sk-other")):
            self.client.post(
                "/api/providers/custom-endpoints",
                json={
                    "id": name,
                    "name": name,
                    "base_url": f"https://llm.{name}.corp/v1",
                    "model": f"{name}/m",
                    "api_key": key,
                },
            )

        self.client.post("/api/providers/custom-endpoints/other/activate", json={})
        self.client.request("DELETE", "/api/providers/custom-endpoints/acme")

        model_cfg = load_config().get("model") or {}
        assert model_cfg.get("provider") == "other"
        assert model_cfg.get("api_key") == "sk-other"
        assert model_cfg.get("base_url") == "https://llm.other.corp/v1"

    def test_set_model_main_preserves_base_url_for_named_custom_provider(self):
        """Selecting a named custom endpoint from the Desktop model picker
        should keep its endpoint URL attached to model config.
        """
        from hermes_cli.config import load_config, save_config

        save_config({
            "model": {"provider": "nous", "default": "hermes-4"},
            "providers": {
                "axet-proxy": {
                    "name": "Axet Proxy",
                    "base_url": "http://127.0.0.1:8081/v1",
                    "api_key": "sk-local",
                    "model": "gpt-5.4",
                    "models": {"gpt-5.4": {}},
                }
            },
        })

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "axet-proxy", "model": "gpt-5.4"},
        )

        assert resp.status_code == 200
        model_cfg = load_config()["model"]
        assert model_cfg["provider"] == "axet-proxy"
        assert model_cfg["default"] == "gpt-5.4"
        assert model_cfg["base_url"] == "http://127.0.0.1:8081/v1"
        assert model_cfg["api_key"] == "sk-local"

    def test_set_model_main_gateway_failure_does_not_block_save(self, monkeypatch):
        """A Portal/gateway hiccup must never prevent saving the model."""
        import hermes_cli.nous_subscription as ns

        def boom(*args, **kwargs):
            raise RuntimeError("portal unreachable")

        monkeypatch.setattr(ns, "apply_nous_managed_defaults", boom)

        resp = self.client.post(
            "/api/model/set",
            json={"scope": "main", "provider": "nous", "model": "hermes-4"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data.get("gateway_tools", []) == []

    def test_recommended_default_nous_honors_free_tier(self, monkeypatch):
        """For a free-tier Nous user, the recommended default must be a free
        model (mirroring `hermes model`), not the first curated paid entry."""
        import hermes_cli.models as models_mod

        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda: ["paid/expensive", "free/cheap"])
        monkeypatch.setattr(
            models_mod, "get_pricing_for_provider",
            lambda provider: {"paid/expensive": {"input": "1"}, "free/cheap": {"input": "0"}},
        )
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda *, force_fresh=False: True)
        monkeypatch.setattr(
            models_mod, "union_with_portal_free_recommendations",
            lambda ids, pricing, url: (ids, pricing),
        )
        # Free partition keeps only the free model selectable.
        monkeypatch.setattr(
            models_mod, "partition_nous_models_by_tier",
            lambda ids, pricing, free_tier: (["free/cheap"], ["paid/expensive"]),
        )

        resp = self.client.get("/api/model/recommended-default?provider=nous")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "nous"
        assert data["model"] == "free/cheap"
        assert data["free_tier"] is True

    def test_recommended_default_nous_paid_uses_curated_default(self, monkeypatch):
        """A paid Nous user gets the first curated/paid-augmented model."""
        import hermes_cli.models as models_mod

        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda: ["top/model", "other/model"])
        monkeypatch.setattr(models_mod, "get_pricing_for_provider", lambda provider: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda *, force_fresh=False: False)
        monkeypatch.setattr(
            models_mod, "union_with_portal_paid_recommendations",
            lambda ids, pricing, url: (ids, pricing),
        )

        resp = self.client.get("/api/model/recommended-default?provider=nous")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "nous"
        assert data["model"] == "top/model"
        assert data["free_tier"] is False

    def test_recommended_default_handles_failure_gracefully(self, monkeypatch):
        """Endpoint never 500s — returns empty model on internal error."""
        import hermes_cli.models as models_mod

        def boom():
            raise RuntimeError("portal down")

        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", boom)

        resp = self.client.get("/api/model/recommended-default?provider=nous")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == ""
        assert data["free_tier"] is None


# ---------------------------------------------------------------------------
# _build_schema_from_config tests
# ---------------------------------------------------------------------------


class TestBuildSchemaFromConfig:
    def test_produces_expected_field_count(self):
        from hermes_cli.web_server import CONFIG_SCHEMA
        # DEFAULT_CONFIG has ~150+ leaf fields
        assert len(CONFIG_SCHEMA) > 100

    def test_schema_entries_have_required_fields(self):
        from hermes_cli.web_server import CONFIG_SCHEMA
        for key, entry in list(CONFIG_SCHEMA.items())[:10]:
            assert "type" in entry, f"Missing type for {key}"
            assert "category" in entry, f"Missing category for {key}"

    def test_overrides_applied(self):
        from hermes_cli.web_server import CONFIG_SCHEMA
        # terminal.backend should be a select with options
        if "terminal.backend" in CONFIG_SCHEMA:
            entry = CONFIG_SCHEMA["terminal.backend"]
            assert entry["type"] == "select"
            assert "options" in entry
            assert "local" in entry["options"]

    def test_memory_provider_field_present_as_select(self):
        """memory.provider must stay in the config schema.

        Desktop's settings page builds its field list from /api/config/schema —
        a key excluded here silently vanishes from Desktop's Memory section
        (regression: the dashboard's dedicated memory-provider UI excluded the
        key server-side, breaking Desktop's dropdown). The dashboard hides the
        field client-side instead.
        """
        from hermes_cli.web_server import CONFIG_SCHEMA
        entry = CONFIG_SCHEMA["memory.provider"]
        assert entry["type"] == "select"
        assert entry["category"] == "memory"
        options = entry["options"]
        # Built-in-only sentinel first, plus at least one discovered provider.
        # The literal "builtin" alias must NOT be offered — built-in memory is
        # not a provider plugin (#49513).
        assert options[0] == ""
        assert "builtin" not in options
        assert len(options) >= 2

    def test_memory_provider_options_cover_discovered_providers(self):
        """Every provider the /api/memory endpoint can activate is selectable."""
        from hermes_cli.web_server import CONFIG_SCHEMA
        from plugins.memory import list_memory_provider_names

        options = set(CONFIG_SCHEMA["memory.provider"]["options"])
        missing = set(list_memory_provider_names()) - options
        assert missing == set(), f"discovered providers missing from schema options: {missing}"

    def test_approvals_mode_options_match_config_values(self):
        """approvals.mode select options must match the values accepted by config.py.

        Previously the dashboard showed ['ask', 'yolo', 'deny'] which are stale
        names that don't correspond to any real config value. The correct values
        are 'manual', 'smart', and 'off' (see hermes_cli/config.py).
        'smart' was missing entirely, making it unreachable from the UI.
        """
        from hermes_cli.web_server import CONFIG_SCHEMA
        entry = CONFIG_SCHEMA["approvals.mode"]
        assert entry["type"] == "select"
        options = entry["options"]
        assert "manual" in options, "'manual' missing from approvals.mode options"
        assert "smart" in options, "'smart' missing from approvals.mode options"
        assert "off" in options, "'off' missing from approvals.mode options"
        # Stale names that were previously shown but don't match config values
        assert "ask" not in options, "stale option 'ask' should not appear"
        assert "yolo" not in options, "stale option 'yolo' should not appear"
        assert "deny" not in options, "stale option 'deny' should not appear"

    def test_empty_prefix_produces_correct_keys(self):
        from hermes_cli.web_server import _build_schema_from_config
        test_config = {"model": "test", "nested": {"key": "val"}}
        schema = _build_schema_from_config(test_config)
        assert "model" in schema
        assert "nested.key" in schema

    def test_top_level_scalars_get_general_category(self):
        """Top-level scalar fields should be in 'general' category."""
        from hermes_cli.web_server import CONFIG_SCHEMA
        assert CONFIG_SCHEMA["model"]["category"] == "general"

    def test_nested_keys_get_parent_category(self):
        """Nested fields should use the top-level parent as their category."""
        from hermes_cli.web_server import CONFIG_SCHEMA
        if "agent.max_turns" in CONFIG_SCHEMA:
            assert CONFIG_SCHEMA["agent.max_turns"]["category"] == "agent"

    def test_category_merge_applied(self):
        """Small categories should be merged into larger ones."""
        from hermes_cli.web_server import CONFIG_SCHEMA
        categories = {e["category"] for e in CONFIG_SCHEMA.values()}
        # These should be merged away
        assert "privacy" not in categories  # merged into security
        assert "context" not in categories  # merged into agent

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

    def test_get_config_no_internal_keys(self):
        """GET /api/config should not expose _config_version or _model_meta."""
        config = self.client.get("/api/config").json()
        internal = [k for k in config if k.startswith("_")]
        assert not internal, f"Internal keys leaked to frontend: {internal}"

    def test_get_config_model_is_string(self):
        """GET /api/config should normalize model dict to a string."""
        config = self.client.get("/api/config").json()
        assert isinstance(config.get("model"), str), \
            f"model should be string, got {type(config.get('model'))}"

    def test_round_trip_preserves_model_subkeys(self):
        """Save and reload should not lose model.provider, model.base_url, etc."""
        from hermes_cli.config import load_config, save_config

        # Set up a config with model as a dict (the common user config form)
        save_config({
            "model": {
                "default": "anthropic/claude-sonnet-4",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "openai",
            }
        })

        before = load_config()
        assert isinstance(before.get("model"), dict)
        original_keys = set(before["model"].keys())

        # GET → PUT unchanged
        web_config = self.client.get("/api/config").json()
        assert isinstance(web_config.get("model"), str), "GET should normalize model to string"

        self.client.put("/api/config", json={"config": web_config})

        after = load_config()
        assert isinstance(after.get("model"), dict), "model should still be a dict after save"
        assert set(after["model"].keys()) >= original_keys, \
            f"Lost model subkeys: {original_keys - set(after['model'].keys())}"

    def test_edit_model_name_preserved(self):
        """Changing the model string should update model.default on disk."""
        from hermes_cli.config import load_config

        web_config = self.client.get("/api/config").json()
        original_model = web_config["model"]

        # Change model
        web_config["model"] = "test/editing-model"
        self.client.put("/api/config", json={"config": web_config})

        after = load_config()
        if isinstance(after.get("model"), dict):
            assert after["model"]["default"] == "test/editing-model"
        else:
            assert after["model"] == "test/editing-model"

        # Restore
        web_config["model"] = original_model
        self.client.put("/api/config", json={"config": web_config})

    def test_edit_nested_value(self):
        """Editing a nested config value should persist correctly."""
        from hermes_cli.config import load_config

        web_config = self.client.get("/api/config").json()
        original_turns = web_config.get("agent", {}).get("max_turns")

        # Change max_turns
        if "agent" not in web_config:
            web_config["agent"] = {}
        web_config["agent"]["max_turns"] = 42

        self.client.put("/api/config", json={"config": web_config})

        after = load_config()
        assert after.get("agent", {}).get("max_turns") == 42

        # Restore
        web_config["agent"]["max_turns"] = original_turns
        self.client.put("/api/config", json={"config": web_config})

    def test_round_trip_preserves_custom_providers(self):
        """``custom_providers`` is not in the dashboard schema, so the
        frontend never sends it in PUT bodies. Saving must still preserve
        it on disk — otherwise every dashboard click that saves silently
        wipes the user's custom endpoints."""
        from hermes_cli.config import load_config, save_config

        save_config({
            "model": {"default": "test/model", "provider": "custom:myprov"},
            "custom_providers": [
                {
                    "name": "myprov",
                    "base_url": "https://example.invalid/v1",
                    "key_env": "MYPROV_API_KEY",
                    "api_mode": "chat_completions",
                    "model": "test/model",
                },
            ],
        })

        # Frontend behaviour: GET full config, then PUT without keys the
        # schema doesn't know about (custom_providers is the prime example).
        web_config = self.client.get("/api/config").json()
        web_config.pop("custom_providers", None)
        resp = self.client.put("/api/config", json={"config": web_config})
        assert resp.status_code == 200

        after = load_config()
        cps = after.get("custom_providers")
        assert isinstance(cps, list) and len(cps) == 1, \
            f"custom_providers wiped by lossy PUT: {cps!r}"
        assert cps[0].get("name") == "myprov"
        assert cps[0].get("base_url") == "https://example.invalid/v1"

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

    def test_get_logs_default(self):
        resp = self.client.get("/api/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert "file" in data
        assert "lines" in data
        assert isinstance(data["lines"], list)

    def test_get_logs_invalid_file(self):
        resp = self.client.get("/api/logs?file=nonexistent")
        assert resp.status_code == 400

    def test_cron_list(self):
        resp = self.client.get("/api/cron/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_cron_job_not_found(self):
        resp = self.client.get("/api/cron/jobs/nonexistent-id")
        assert resp.status_code == 404

    # --- Automation Blueprints ---

    def test_cron_blueprints_list(self):
        resp = self.client.get("/api/cron/blueprints")
        assert resp.status_code == 200
        blueprints = resp.json()["blueprints"]
        assert len(blueprints) >= 1
        first = blueprints[0]
        assert "fields" in first
        assert first["command"].startswith("/blueprint")
        assert first["appUrl"].startswith("hermes://")

    def test_blueprint_instantiate_creates_job(self):
        resp = self.client.post(
            "/api/cron/blueprints/instantiate",
            json={"blueprint": "morning-brief", "values": {"time": "07:30", "deliver": "local"}},
        )
        assert resp.status_code == 200
        job = resp.json()
        assert (job.get("schedule_display") or "").strip() == "30 7 * * *" or \
            (job.get("schedule", {}) or {}).get("expr") == "30 7 * * *"

    def test_blueprint_instantiate_unknown_404(self):
        resp = self.client.post(
            "/api/cron/blueprints/instantiate",
            json={"blueprint": "does-not-exist", "values": {}},
        )
        assert resp.status_code == 404

    def test_blueprint_instantiate_bad_value_422(self):
        resp = self.client.post(
            "/api/cron/blueprints/instantiate",
            json={"blueprint": "morning-brief", "values": {"time": "99:99"}},
        )
        assert resp.status_code == 422

    # --- Profiles ---

    def test_profiles_list_includes_default(self):
        from hermes_constants import get_hermes_home
        get_hermes_home().mkdir(parents=True, exist_ok=True)

        resp = self.client.get("/api/profiles")
        assert resp.status_code == 200
        names = [p["name"] for p in resp.json()["profiles"]]
        assert "default" in names

    def test_profiles_list_falls_back_when_profile_listing_fails(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod

        hermes_home = get_hermes_home()
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "model:\n  provider: openrouter\n  name: anthropic/claude-sonnet-4.6\n",
            encoding="utf-8",
        )
        named = hermes_home / "profiles" / "multi-agent"
        named.mkdir(parents=True)
        (named / ".env").write_text("EXAMPLE=1\n", encoding="utf-8")
        (named / "skills" / "demo").mkdir(parents=True)
        (named / "skills" / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")

        monkeypatch.setattr(
            profiles_mod,
            "list_profiles",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        resp = self.client.get("/api/profiles")

        assert resp.status_code == 200
        profiles = {p["name"]: p for p in resp.json()["profiles"]}
        assert profiles["default"]["is_default"] is True
        assert profiles["default"]["provider"] == "openrouter"
        assert profiles["multi-agent"]["has_env"] is True
        assert profiles["multi-agent"]["skill_count"] == 1

    def test_profiles_create_rename_delete_round_trip(self, monkeypatch):
        # Stub gateway service teardown so the test doesn't shell out to
        # launchctl/systemctl on the host.
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "_cleanup_gateway_service", lambda *a, **kw: None)

        created = self.client.post("/api/profiles", json={"name": "test-prof"})
        assert created.status_code == 200

        renamed = self.client.patch(
            "/api/profiles/test-prof",
            json={"new_name": "test-prof-2"},
        )
        assert renamed.status_code == 200

        names = [p["name"] for p in self.client.get("/api/profiles").json()["profiles"]]
        assert "test-prof" not in names
        assert "test-prof-2" in names

        deleted = self.client.delete("/api/profiles/test-prof-2")
        assert deleted.status_code == 200
        names = [p["name"] for p in self.client.get("/api/profiles").json()["profiles"]]
        assert "test-prof-2" not in names

    def test_profile_setup_command_uses_named_profile_wrapper(self):
        from hermes_constants import get_hermes_home

        (get_hermes_home() / "profiles" / "coder").mkdir(parents=True)

        resp = self.client.get("/api/profiles/coder/setup-command")

        assert resp.status_code == 200
        assert resp.json()["command"] == "coder setup"

    def test_profile_setup_command_uses_hermes_for_default_profile(self):
        from hermes_constants import get_hermes_home

        get_hermes_home().mkdir(parents=True, exist_ok=True)

        resp = self.client.get("/api/profiles/default/setup-command")

        assert resp.status_code == 200
        assert resp.json()["command"] == "hermes setup"

    def test_profiles_create_creates_wrapper_alias_when_safe(self, monkeypatch, tmp_path):
        import hermes_cli.profiles as profiles_mod

        wrapper_dir = tmp_path / "bin"
        wrapper_dir.mkdir()
        monkeypatch.setattr(profiles_mod, "_get_wrapper_dir", lambda: wrapper_dir)
        monkeypatch.setattr(profiles_mod.shutil, "which", lambda name: "/opt/hermes/bin/hermes")

        resp = self.client.post(
            "/api/profiles",
            json={"name": "writer", "clone_from": None},
        )

        assert resp.status_code == 200
        is_windows = sys.platform == "win32"
        wrapper_path = wrapper_dir / ("writer.bat" if is_windows else "writer")
        assert wrapper_path.exists()
        lines = [line.strip() for line in wrapper_path.read_text().splitlines() if line.strip()]
        if is_windows:
            assert lines == ["@echo off", "hermes -p writer %*"]
        else:
            assert lines == ["#!/bin/sh", 'exec /opt/hermes/bin/hermes -p writer "$@"']

    def test_profiles_create_with_clone_from_copies_source_skills(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)
        (get_hermes_home() / "config.yaml").write_text(
            "model:\n  provider: openrouter\n",
            encoding="utf-8",
        )
        default_skill = get_hermes_home() / "skills" / "custom" / "new-skill"
        default_skill.mkdir(parents=True)
        (default_skill / "SKILL.md").write_text("---\nname: new-skill\n---\n", encoding="utf-8")

        resp = self.client.post(
            "/api/profiles",
            json={"name": "cloned", "clone_from": "default"},
        )

        assert resp.status_code == 200
        cloned_root = get_hermes_home() / "profiles" / "cloned"
        cloned_skill = cloned_root / "skills" / "custom" / "new-skill" / "SKILL.md"
        assert cloned_skill.exists()
        cloned_config = yaml.safe_load((cloned_root / "config.yaml").read_text(encoding="utf-8"))
        assert cloned_config["_config_version"] == DEFAULT_CONFIG["_config_version"]
        profiles = {p["name"]: p for p in self.client.get("/api/profiles").json()["profiles"]}
        assert profiles["cloned"]["skill_count"] == 1

    def test_profiles_create_with_clone_from_duplicates_source(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        # Create a source profile and give it a distinctive skill.
        assert self.client.post("/api/profiles", json={"name": "source-prof"}).status_code == 200
        source_skill = get_hermes_home() / "profiles" / "source-prof" / "skills" / "custom" / "src-skill"
        source_skill.mkdir(parents=True)
        (source_skill / "SKILL.md").write_text("---\nname: src-skill\n---\n", encoding="utf-8")

        # Duplicate it via an explicit clone_from source (not "default").
        resp = self.client.post(
            "/api/profiles",
            json={"name": "source-prof-copy", "clone_from": "source-prof"},
        )

        assert resp.status_code == 200
        cloned_skill = (
            get_hermes_home() / "profiles" / "source-prof-copy" / "skills" / "custom" / "src-skill" / "SKILL.md"
        )
        assert cloned_skill.exists()

    def test_profiles_create_clone_all_from_named_source(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        assert self.client.post("/api/profiles", json={"name": "full-src"}).status_code == 200
        source_dir = get_hermes_home() / "profiles" / "full-src"
        (source_dir / "config.yaml").write_text("model:\n  provider: source-only\n", encoding="utf-8")
        (source_dir / "workspace" / "artifact.txt").parent.mkdir(parents=True, exist_ok=True)
        (source_dir / "workspace" / "artifact.txt").write_text("copied", encoding="utf-8")

        resp = self.client.post(
            "/api/profiles",
            json={"name": "full-copy", "clone_from": "full-src", "clone_all": True},
        )

        assert resp.status_code == 200
        target_dir = get_hermes_home() / "profiles" / "full-copy"
        assert (target_dir / "config.yaml").read_text(encoding="utf-8") == "model:\n  provider: source-only\n"
        assert (target_dir / "workspace" / "artifact.txt").read_text(encoding="utf-8") == "copied"

    def test_profiles_create_without_clone_seeds_bundled_skills(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        def fake_seed(profile_dir, quiet=False):
            skill_dir = profile_dir / "skills" / "software-development" / "plan"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: plan\n---\n", encoding="utf-8")
            return {"copied": ["plan"]}

        monkeypatch.setattr(profiles_mod, "seed_profile_skills", fake_seed)

        resp = self.client.post(
            "/api/profiles",
            json={"name": "fresh", "clone_from": None},
        )

        assert resp.status_code == 200
        seeded_skill = get_hermes_home() / "profiles" / "fresh" / "skills" / "software-development" / "plan" / "SKILL.md"
        assert seeded_skill.exists()
        profiles = {p["name"]: p for p in self.client.get("/api/profiles").json()["profiles"]}
        assert profiles["fresh"]["skill_count"] == 1

    def test_profiles_create_builder_fields_model_mcp_and_keep_skills(self, monkeypatch):
        """Profile-builder create: model + MCP servers + keep-skills selection
        all land in the NEW profile's config, and hub installs are spawned
        scoped to that profile via ``-p <name>``."""
        from hermes_constants import (
            get_hermes_home,
            set_hermes_home_override,
            reset_hermes_home_override,
        )
        from hermes_cli.config import load_config
        from hermes_cli.skills_config import get_disabled_skills
        import hermes_cli.profiles as profiles_mod
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        # Seed two known skills so keep-skills "replace" has something to act on.
        def fake_seed(profile_dir, quiet=False):
            for skill in ("keep-me", "drop-me"):
                d = profile_dir / "skills" / "custom" / skill
                d.mkdir(parents=True)
                (d / "SKILL.md").write_text(f"---\nname: {skill}\n---\n", encoding="utf-8")
            return {"copied": ["keep-me", "drop-me"]}

        monkeypatch.setattr(profiles_mod, "seed_profile_skills", fake_seed)

        # Capture hub-install spawns instead of launching real subprocesses.
        spawned = []

        class _FakeProc:
            pid = 4321

        def fake_spawn(subcommand, name):
            spawned.append((list(subcommand), name))
            return _FakeProc()

        monkeypatch.setattr(web_server, "_spawn_hermes_action", fake_spawn)

        resp = self.client.post(
            "/api/profiles",
            json={
                "name": "builder",
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
                "mcp_servers": [
                    {"name": "ctx7", "url": "https://mcp.context7.com/mcp"},
                    {"name": "bogus"},  # no url/command -> must be skipped, no 500
                ],
                "keep_skills": ["keep-me"],
                "hub_skills": ["someuser/some-skill"],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_set"] is True
        assert data["mcp_written"] == 1  # bogus skipped
        assert data["skills_disabled"] == 1  # drop-me disabled, keep-me kept
        assert data["hub_installs"] == [{"identifier": "someuser/some-skill", "pid": 4321}]

        # Hub install was scoped to the new profile.
        assert spawned == [
            (
                ["-p", "builder", "skills", "install", "someuser/some-skill", "--yes"],
                web_server._hub_action_name("install", "someuser/some-skill"),
            )
        ]

        # Verify the writes landed in the NEW profile's config, not the root.
        prof_dir = get_hermes_home() / "profiles" / "builder"
        token = set_hermes_home_override(str(prof_dir))
        try:
            cfg = load_config()
            assert cfg["model"]["default"] == "anthropic/claude-sonnet-4.6"
            assert cfg["model"]["provider"] == "openrouter"
            assert sorted((cfg.get("mcp_servers") or {}).keys()) == ["ctx7"]
            disabled = get_disabled_skills(cfg)
            assert "drop-me" in disabled
            assert "keep-me" not in disabled
        finally:
            reset_hermes_home_override(token)

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

    def test_profile_open_terminal_uses_macos_terminal(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.web_server as web_server

        (get_hermes_home() / "profiles" / "coder").mkdir(parents=True)
        calls = []
        monkeypatch.setattr(web_server.sys, "platform", "darwin")
        monkeypatch.setattr(web_server.subprocess, "Popen", lambda args, **kwargs: calls.append(args))

        resp = self.client.post("/api/profiles/coder/open-terminal")

        assert resp.status_code == 200
        assert calls
        assert calls[0][0] == "osascript"
        assert "coder setup" in " ".join(calls[0])

    def test_profile_open_terminal_uses_windows_cmd(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.web_server as web_server

        (get_hermes_home() / "profiles" / "coder").mkdir(parents=True)
        calls = []
        monkeypatch.setattr(web_server.sys, "platform", "win32")
        monkeypatch.setattr(web_server.subprocess, "Popen", lambda args, **kwargs: calls.append(args))

        resp = self.client.post("/api/profiles/coder/open-terminal")

        assert resp.status_code == 200
        assert calls
        assert calls[0][:4] == ["cmd.exe", "/c", "start", ""]
        assert calls[0][-1] == "coder setup"

    def test_profiles_create_rejects_invalid_name(self):
        resp = self.client.post("/api/profiles", json={"name": "Has Spaces"})
        assert resp.status_code == 400

    def test_profiles_delete_default_forbidden(self):
        resp = self.client.delete("/api/profiles/default")
        assert resp.status_code == 400

    def test_profiles_delete_not_found(self):
        resp = self.client.delete("/api/profiles/does-not-exist")
        assert resp.status_code == 404

    def test_profile_soul_round_trip(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "_cleanup_gateway_service", lambda *a, **kw: None)

        self.client.post("/api/profiles", json={"name": "soul-prof"})
        get1 = self.client.get("/api/profiles/soul-prof/soul")
        assert get1.status_code == 200
        assert get1.json()["exists"] is True

        put = self.client.put(
            "/api/profiles/soul-prof/soul",
            json={"content": "# Edited soul"},
        )
        assert put.status_code == 200

        got = self.client.get("/api/profiles/soul-prof/soul").json()
        assert got["content"] == "# Edited soul"

        self.client.delete("/api/profiles/soul-prof")

    def test_profile_soul_unknown_profile_404(self):
        resp = self.client.get("/api/profiles/nonexistent/soul")
        assert resp.status_code == 404

    # --- New profiles endpoints: active / description / model / describe-auto ---

    def test_profiles_active_defaults(self):
        from hermes_constants import get_hermes_home
        get_hermes_home().mkdir(parents=True, exist_ok=True)

        resp = self.client.get("/api/profiles/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] == "default"
        assert data["current"] == "default"

    def test_profiles_set_active_round_trip(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        self.client.post("/api/profiles", json={"name": "router"})

        resp = self.client.post("/api/profiles/active", json={"name": "router"})
        assert resp.status_code == 200
        assert resp.json()["active"] == "router"
        assert self.client.get("/api/profiles/active").json()["active"] == "router"

    def test_profiles_set_active_unknown_404(self):
        resp = self.client.post("/api/profiles/active", json={"name": "ghost"})
        assert resp.status_code == 404

    def test_profile_description_round_trip(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        self.client.post("/api/profiles", json={"name": "desc-prof"})

        put = self.client.put(
            "/api/profiles/desc-prof/description",
            json={"description": "Handles code review"},
        )
        assert put.status_code == 200
        body = put.json()
        assert body["description"] == "Handles code review"
        assert body["description_auto"] is False

        profiles = {p["name"]: p for p in self.client.get("/api/profiles").json()["profiles"]}
        assert profiles["desc-prof"]["description"] == "Handles code review"
        assert profiles["desc-prof"]["description_auto"] is False

    def test_profile_description_unknown_404(self):
        resp = self.client.put(
            "/api/profiles/nope/description", json={"description": "x"}
        )
        assert resp.status_code == 404

    def test_profile_model_round_trip(self, monkeypatch):
        from hermes_constants import get_hermes_home
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        self.client.post("/api/profiles", json={"name": "model-prof"})

        resp = self.client.put(
            "/api/profiles/model-prof/model",
            json={"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        )
        assert resp.status_code == 200
        assert resp.json()["provider"] == "openrouter"

        import yaml
        cfg_path = get_hermes_home() / "profiles" / "model-prof" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["model"]["provider"] == "openrouter"
        assert cfg["model"]["default"] == "anthropic/claude-sonnet-4.6"

    def test_profile_model_requires_provider_and_model(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        self.client.post("/api/profiles", json={"name": "model-prof2"})
        resp = self.client.put(
            "/api/profiles/model-prof2/model",
            json={"provider": "", "model": ""},
        )
        assert resp.status_code == 400

    def test_profile_describe_auto_success(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        self.client.post("/api/profiles", json={"name": "auto-prof"})

        from hermes_cli import profile_describer
        monkeypatch.setattr(
            profile_describer,
            "describe_profile",
            lambda name, overwrite=False: profile_describer.DescribeOutcome(
                name, True, "described", description="Generated blurb"
            ),
        )

        resp = self.client.post("/api/profiles/auto-prof/describe-auto", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["description"] == "Generated blurb"
        assert body["description_auto"] is True

    def test_profile_describe_auto_failure_is_not_auto(self, monkeypatch):
        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

        self.client.post("/api/profiles", json={"name": "auto-fail"})

        from hermes_cli import profile_describer
        monkeypatch.setattr(
            profile_describer,
            "describe_profile",
            lambda name, overwrite=False: profile_describer.DescribeOutcome(
                name, False, "no aux client", description=None
            ),
        )

        resp = self.client.post("/api/profiles/auto-fail/describe-auto", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["description_auto"] is False

    def test_skills_list(self):
        resp = self.client.get("/api/skills")
        assert resp.status_code == 200
        skills = resp.json()
        assert isinstance(skills, list)
        if skills:
            assert "name" in skills[0]
            assert "enabled" in skills[0]

    def test_skills_list_includes_disabled_skills(self, monkeypatch):
        import tools.skills_tool as skills_tool
        import hermes_cli.skills_config as skills_config
        import hermes_cli.web_server as web_server

        def _fake_find_all_skills(*, skip_disabled=False):
            if skip_disabled:
                return [
                    {"name": "active-skill", "description": "active", "category": "demo"},
                    {"name": "disabled-skill", "description": "disabled", "category": "demo"},
                ]
            return [
                {"name": "active-skill", "description": "active", "category": "demo"},
            ]

        monkeypatch.setattr(skills_tool, "_find_all_skills", _fake_find_all_skills)
        monkeypatch.setattr(skills_config, "get_disabled_skills", lambda config: {"disabled-skill"})
        monkeypatch.setattr(web_server, "load_config", lambda: {"skills": {"disabled": ["disabled-skill"]}})

        resp = self.client.get("/api/skills")

        assert resp.status_code == 200
        assert resp.json() == [
            {
                "name": "active-skill",
                "description": "active",
                "category": "demo",
                "enabled": True,
                "usage": 0,
                "provenance": "agent",
            },
            {
                "name": "disabled-skill",
                "description": "disabled",
                "category": "demo",
                "enabled": False,
                "usage": 0,
                "provenance": "agent",
            },
        ]

    def test_toolsets_list(self):
        resp = self.client.get("/api/tools/toolsets")
        assert resp.status_code == 200
        toolsets = resp.json()
        assert isinstance(toolsets, list)
        if toolsets:
            assert "name" in toolsets[0]
            assert "label" in toolsets[0]
            assert "enabled" in toolsets[0]

    def test_toolsets_list_matches_cli_enabled_state(self, monkeypatch):
        import hermes_cli.tools_config as tools_config
        import toolsets as toolsets_module
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            tools_config,
            "_get_effective_configurable_toolsets",
            lambda: [
                ("web", "🔍 Web Search & Scraping", "web_search, web_extract"),
                ("skills", "📚 Skills", "list, view, manage"),
                ("memory", "💾 Memory", "persistent memory across sessions"),
            ],
        )
        monkeypatch.setattr(
            tools_config,
            "_get_platform_tools",
            lambda config, platform, include_default_mcp_servers=False: {"web", "skills"},
        )
        monkeypatch.setattr(
            tools_config,
            "_toolset_has_keys",
            lambda ts_key, config=None: ts_key != "web",
        )
        monkeypatch.setattr(
            toolsets_module,
            "resolve_toolset",
            lambda name: {
                "web": ["web_search", "web_extract"],
                "skills": ["skills_list", "skill_view"],
                "memory": ["memory_read"],
            }[name],
        )
        monkeypatch.setattr(web_server, "load_config", lambda: {"platform_toolsets": {"cli": ["web", "skills"]}})

        resp = self.client.get("/api/tools/toolsets")

        assert resp.status_code == 200
        assert resp.json() == [
            {
                "name": "web",
                "label": "Web Search & Scraping",
                "description": "web_search, web_extract",
                "platform": "cli",
                "platform_label": "CLI",
                "enabled": True,
                "available": True,
                "configured": False,
                "tools": ["web_extract", "web_search"],
            },
            {
                "name": "skills",
                "label": "Skills",
                "description": "list, view, manage",
                "platform": "cli",
                "platform_label": "CLI",
                "enabled": True,
                "available": True,
                "configured": True,
                "tools": ["skill_view", "skills_list"],
            },
            {
                "name": "memory",
                "label": "Memory",
                "description": "persistent memory across sessions",
                "platform": "cli",
                "platform_label": "CLI",
                "enabled": False,
                "available": False,
                "configured": True,
                "tools": ["memory_read"],
            },
        ]

    def test_toggle_toolset_enable_disable(self):
        """PUT /api/tools/toolsets/{name} round-trips through config and the list view."""
        # Enable a toolset that is off-by-default so the state change is observable.
        resp = self.client.put("/api/tools/toolsets/x_search", json={"enabled": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["name"] == "x_search"
        assert body["enabled"] is True

        listing = {t["name"]: t for t in self.client.get("/api/tools/toolsets").json()}
        assert listing["x_search"]["enabled"] is True

        # Disable it again.
        resp = self.client.put("/api/tools/toolsets/x_search", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

        listing = {t["name"]: t for t in self.client.get("/api/tools/toolsets").json()}
        assert listing["x_search"]["enabled"] is False

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

    def test_toggle_toolset_unknown_returns_400(self):
        resp = self.client.put(
            "/api/tools/toolsets/not_a_real_toolset", json={"enabled": True}
        )
        assert resp.status_code == 400

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

    def test_get_toolset_config_status_ready_when_key_set(self, monkeypatch):
        """A keyed provider flips to status=ready once its env var is set."""
        monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")

        resp = self.client.get("/api/tools/toolsets/tts/config")
        assert resp.status_code == 200
        by_name = {p["name"]: p for p in resp.json()["providers"]}
        assert by_name["ElevenLabs"]["status"] == "ready"

    def test_get_toolset_config_tts_rows_carry_provider_key(self):
        """TTS provider rows surface their tts_provider config key.

        The desktop Capabilities panel renders the provider's voice/model
        config fields (tts.<key>.*) inline; without the key it can only show
        API keys. Every built-in TTS row declares one.
        """
        resp = self.client.get("/api/tools/toolsets/tts/config")
        assert resp.status_code == 200
        providers = resp.json()["providers"]
        assert providers
        for prov in providers:
            assert prov.get("tts_provider"), f"row {prov['name']!r} missing tts_provider"
        by_name = {p["name"]: p for p in providers}
        assert by_name["OpenAI TTS"]["tts_provider"] == "openai"
        assert by_name["Microsoft Edge TTS"]["tts_provider"] == "edge"
        # Non-TTS toolsets must not grow the field.
        web = self.client.get("/api/tools/toolsets/web/config").json()
        assert all("tts_provider" not in p for p in web["providers"])

    def test_get_toolset_config_reflects_selected_provider(self):
        """Selecting a provider is reflected in the next /config read.

        Regression: the GUI's provider panel highlighted the first keyless
        provider on relaunch because /config never reported which provider was
        actually active. After selecting one, is_active / active_provider must
        point at it.
        """
        sel = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "Firecrawl Self-Hosted"},
        )
        assert sel.status_code == 200

        resp = self.client.get("/api/tools/toolsets/web/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_provider"] == "Firecrawl Self-Hosted"
        active = [p["name"] for p in data["providers"] if p["is_active"]]
        # The first active row is what the GUI highlights; it must be the
        # selected provider.
        assert active, "expected at least one provider flagged active"
        assert active[0] == "Firecrawl Self-Hosted"

    def test_get_toolset_config_no_category_toolset(self):
        """A toolset without a TOOL_CATEGORIES entry returns has_category False."""
        resp = self.client.get("/api/tools/toolsets/todo/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "todo"
        assert data["has_category"] is False
        assert data["providers"] == []

    def test_get_toolset_config_unknown_returns_400(self):
        resp = self.client.get("/api/tools/toolsets/not_a_real_toolset/config")
        assert resp.status_code == 400

    def test_select_toolset_provider_persists_backend(self):
        """PUT .../provider writes the backend selection to config."""
        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "Firecrawl Self-Hosted"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["name"] == "web"
        assert body["provider"] == "Firecrawl Self-Hosted"

        from hermes_cli.config import load_config
        cfg = load_config()
        assert cfg["web"]["backend"] == "firecrawl"

    def test_select_toolset_provider_unknown_provider_returns_400(self):
        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "No Such Provider"},
        )
        assert resp.status_code == 400

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

    def test_select_managed_nous_provider_entitled_no_auth_flag(self, monkeypatch):
        """A signed-in, entitled subscriber gets no needs_nous_auth field."""
        from hermes_cli.nous_account import NousPortalAccountInfo

        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_portal_account_info",
            lambda *a, **k: NousPortalAccountInfo(
                logged_in=True, source="jwt", fresh=True, paid_service_access=True
            ),
        )

        resp = self.client.put(
            "/api/tools/toolsets/browser/provider",
            json={"provider": "Nous Subscription (Browser Use cloud)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "needs_nous_auth" not in data

    def test_select_unmanaged_provider_has_no_nous_auth_field(self):
        """Non-managed rows never carry the entitlement fields."""
        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "Firecrawl Self-Hosted"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "needs_nous_auth" not in data
        assert "feature" not in data

    def test_select_toolset_provider_unknown_toolset_returns_400(self):
        resp = self.client.put(
            "/api/tools/toolsets/not_a_real_toolset/provider",
            json={"provider": "whatever"},
        )
        assert resp.status_code == 400

    # -- Web capability split (search vs extract backends) ------------------

    def test_web_config_reports_per_capability_backends(self):
        """GET web/config carries the resolved search/extract backends.

        The runtime resolves web_search and web_extract independently
        (web.search_backend / web.extract_backend → web.backend → auto-detect);
        the config payload must surface both so the GUI can show which backend
        each capability actually hits.
        """
        resp = self.client.get("/api/tools/toolsets/web/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_search_backend" in data
        assert "active_extract_backend" in data
        # Provider rows carry their backend key + supported capabilities so
        # the GUI can hide "Use for Extract" on search-only rows.
        rows_with_backend = [p for p in data["providers"] if p.get("web_backend")]
        assert rows_with_backend, "expected at least one provider with a web backend key"
        for prov in rows_with_backend:
            assert isinstance(prov["capabilities"], list)
            assert set(prov["capabilities"]) <= {"search", "extract"}
            assert prov["capabilities"], "a web provider must support at least one capability"

    def test_web_capability_fields_only_on_web_toolset(self):
        resp = self.client.get("/api/tools/toolsets/tts/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_search_backend" not in data
        assert "active_extract_backend" not in data

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

    def test_select_web_extract_backend_writes_extract_key(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "Firecrawl Self-Hosted", "capability": "extract"},
        )
        assert resp.status_code == 200

        from hermes_cli.config import load_config
        cfg = load_config()
        assert cfg["web"]["extract_backend"] == "firecrawl"
        # Whole-provider/search keys untouched by a capability-scoped write
        # (the default config seeds them as empty strings).
        assert not cfg["web"].get("search_backend")

        from tools.web_tools import _get_extract_backend
        assert _get_extract_backend() == "firecrawl"

    def test_select_web_capability_rejects_unsupported_capability(self):
        """A search-only provider (ddgs) can't be set as the extract backend."""
        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "DuckDuckGo (ddgs)", "capability": "extract"},
        )
        assert resp.status_code == 400
        assert "does not support extract" in resp.json()["detail"]

    def test_select_web_capability_rejects_bad_values(self):
        resp = self.client.put(
            "/api/tools/toolsets/web/provider",
            json={"provider": "Firecrawl Self-Hosted", "capability": "browse"},
        )
        assert resp.status_code == 400

        # capability is a web-only concept.
        resp = self.client.put(
            "/api/tools/toolsets/tts/provider",
            json={"provider": "Microsoft Edge TTS", "capability": "search"},
        )
        assert resp.status_code == 400

    # -- Terminal execution backend picker ---------------------------------

    def test_get_terminal_backends_shape_and_local_ready(self, monkeypatch):
        """GET .../backends returns one row per backend; local is always ready."""
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server.shutil, "which", lambda name: None)

        resp = self.client.get("/api/tools/terminal/backends")
        assert resp.status_code == 200
        body = resp.json()
        names = [row["name"] for row in body["backends"]]
        assert names == ["local", "docker", "singularity", "modal", "daytona", "ssh"]
        assert body["active"] in set(names)
        for row in body["backends"]:
            assert row["status"] in {"ready", "needs_setup", "unavailable"}
            assert isinstance(row["label"], str) and row["label"]
            assert isinstance(row["description"], str)
            assert isinstance(row["detail"], str)
            assert isinstance(row["active"], bool)
        local = body["backends"][0]
        assert local["status"] == "ready"
        # Exactly one backend is flagged active, matching the summary field.
        active_rows = [r["name"] for r in body["backends"] if r["active"]]
        assert active_rows == [body["active"]]

    def test_terminal_docker_probe_missing_cli(self, monkeypatch):
        """No docker binary on PATH -> needs_setup with install guidance."""
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server.shutil, "which", lambda name: None)

        body = self.client.get("/api/tools/terminal/backends").json()
        docker = next(r for r in body["backends"] if r["name"] == "docker")
        assert docker["status"] == "needs_setup"
        assert "not found" in docker["detail"]

    def test_terminal_docker_probe_daemon_down(self, monkeypatch):
        """docker CLI present but daemon unreachable -> needs_setup."""
        import subprocess as subprocess_mod
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            web_server.shutil,
            "which",
            lambda name: "/usr/bin/docker" if name == "docker" else None,
        )
        monkeypatch.setattr(
            web_server.subprocess,
            "run",
            lambda cmd, **kw: subprocess_mod.CompletedProcess(cmd, 1, stdout="", stderr="daemon down"),
        )

        body = self.client.get("/api/tools/terminal/backends").json()
        docker = next(r for r in body["backends"] if r["name"] == "docker")
        assert docker["status"] == "needs_setup"
        assert "daemon" in docker["detail"].lower()

    def test_terminal_docker_probe_daemon_ready(self, monkeypatch):
        """docker CLI + reachable daemon -> ready."""
        import subprocess as subprocess_mod
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            web_server.shutil,
            "which",
            lambda name: "/usr/bin/docker" if name in {"docker", "singularity"} else None,
        )
        monkeypatch.setattr(
            web_server.subprocess,
            "run",
            lambda cmd, **kw: subprocess_mod.CompletedProcess(cmd, 0, stdout="27.0\n", stderr=""),
        )

        body = self.client.get("/api/tools/terminal/backends").json()
        rows = {r["name"]: r for r in body["backends"]}
        assert rows["docker"]["status"] == "ready"
        # singularity resolves via which() too
        assert rows["singularity"]["status"] == "ready"

    def test_terminal_probe_failure_is_a_status_not_a_500(self, monkeypatch):
        """A probe that raises must surface as a status row, never an error."""
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            web_server.shutil,
            "which",
            lambda name: "/usr/bin/docker" if name == "docker" else None,
        )

        def boom(cmd, **kw):
            raise OSError("exec format error")

        monkeypatch.setattr(web_server.subprocess, "run", boom)

        resp = self.client.get("/api/tools/terminal/backends")
        assert resp.status_code == 200
        docker = next(r for r in resp.json()["backends"] if r["name"] == "docker")
        assert docker["status"] == "unavailable"
        assert "probe failed" in docker["detail"].lower()

    def test_terminal_ssh_probe_reports_missing_keys(self, monkeypatch):
        """SSH without host/user config lists the missing terminal.* keys."""
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server.shutil, "which", lambda name: None)

        body = self.client.get("/api/tools/terminal/backends").json()
        ssh = next(r for r in body["backends"] if r["name"] == "ssh")
        assert ssh["status"] == "needs_setup"
        assert "terminal.ssh_host" in ssh["detail"]

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

    def test_select_terminal_backend_persists_config(self, monkeypatch):
        """PUT .../backend writes terminal.backend and the list reflects it."""
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server.shutil, "which", lambda name: None)

        resp = self.client.put(
            "/api/tools/terminal/backend", json={"backend": "docker"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "backend": "docker"}

        from hermes_cli.config import load_config
        assert load_config()["terminal"]["backend"] == "docker"

        body = self.client.get("/api/tools/terminal/backends").json()
        assert body["active"] == "docker"
        docker = next(r for r in body["backends"] if r["name"] == "docker")
        assert docker["active"] is True
        # Selecting a needs-setup backend is allowed; the row still carries
        # its guidance detail.
        assert docker["status"] == "needs_setup"

    def test_select_terminal_backend_unknown_returns_400(self):
        resp = self.client.put(
            "/api/tools/terminal/backend", json={"backend": "kubernetes"}
        )
        assert resp.status_code == 400
        assert "Unknown terminal backend" in resp.json()["detail"]

    def test_get_toolset_models_no_catalog_toolset(self):
        """Toolsets without a model catalog report has_models: false."""
        resp = self.client.get("/api/tools/toolsets/web/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_models"] is False
        assert body["models"] == []

    def test_get_toolset_models_fal_catalog(self):
        """image_gen with the FAL backend returns its model catalog."""
        resp = self.client.get(
            "/api/tools/toolsets/image_gen/models", params={"provider": "FAL.ai"}
        )
        assert resp.status_code == 200
        body = resp.json()
        # Behavior contract, not a snapshot: FAL always has >= 1 model and
        # each row carries the picker columns.
        assert body["has_models"] is True
        assert body["plugin"] == "fal"
        assert len(body["models"]) >= 1
        for row in body["models"]:
            assert "id" in row
            assert "speed" in row
            assert "strengths" in row
            assert "price" in row
        # current resolves to a real catalog entry (default when unset).
        ids = {row["id"] for row in body["models"]}
        assert body["current"] in ids
        assert body["default"] in ids

    def test_select_toolset_model_persists_and_validates(self):
        """PUT .../model writes image_gen.model; bad ids/toolsets are 400."""
        catalog = self.client.get(
            "/api/tools/toolsets/image_gen/models", params={"provider": "FAL.ai"}
        ).json()
        model_id = catalog["models"][0]["id"]

        resp = self.client.put(
            "/api/tools/toolsets/image_gen/model",
            json={"model": model_id, "provider": "FAL.ai"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        from hermes_cli.config import load_config
        cfg = load_config()
        assert cfg["image_gen"]["model"] == model_id

        # The next catalog read reflects the persisted choice.
        after = self.client.get(
            "/api/tools/toolsets/image_gen/models", params={"provider": "FAL.ai"}
        ).json()
        assert after["current"] == model_id

        # Unknown model id → 400.
        resp = self.client.put(
            "/api/tools/toolsets/image_gen/model",
            json={"model": "not-a-real-model", "provider": "FAL.ai"},
        )
        assert resp.status_code == 400

        # Toolset without a model catalog → 400.
        resp = self.client.put(
            "/api/tools/toolsets/web/model", json={"model": model_id}
        )
        assert resp.status_code == 400


    def test_config_raw_get(self):
        resp = self.client.get("/api/config/raw")
        assert resp.status_code == 200
        assert "yaml" in resp.json()

    def test_config_raw_put_valid(self):
        resp = self.client.put(
            "/api/config/raw",
            json={"yaml_text": "model: test\ntoolsets:\n  - all\n"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_config_raw_put_invalid(self):
        resp = self.client.put(
            "/api/config/raw",
            json={"yaml_text": "- this is a list not a dict"},
        )
        assert resp.status_code == 400

    def test_analytics_usage(self):
        resp = self.client.get("/api/analytics/usage?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "daily" in data
        assert "by_model" in data
        assert "totals" in data
        assert "skills" in data
        assert isinstance(data["daily"], list)
        assert "total_sessions" in data["totals"]
        assert "total_api_calls" in data["totals"]
        assert data["skills"] == {
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        }

    def test_models_analytics_merges_session_only_duplicate_into_accounted_provider(self):
        """Session-only model rows should not render as duplicate zero-token cards.

        Direct-provider-on-OpenRouter sessions can leave one row with only
        ``model`` populated and another row with token/API accounting plus
        ``billing_provider``. The Models dashboard should show one provider
        card, not a real card plus a misleading duplicate empty card.
        """
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session(
                session_id="deepseek-session-only",
                source="cli",
                model="deepseek/deepseek-v4-flash",
            )
            db.create_session(
                session_id="deepseek-accounted",
                source="cli",
                model="deepseek/deepseek-v4-flash",
            )
            db.update_token_counts(
                "deepseek-accounted",
                input_tokens=20_000,
                output_tokens=7_100,
                billing_provider="openrouter",
                api_call_count=9,
            )
        finally:
            db.close()

        resp = self.client.get("/api/analytics/models?days=7")
        assert resp.status_code == 200

        models = resp.json()["models"]
        deepseek_rows = [
            row for row in models
            if row["model"] == "deepseek/deepseek-v4-flash"
        ]

        assert len(deepseek_rows) == 1
        row = deepseek_rows[0]
        assert row["provider"] == "openrouter"
        assert row["sessions"] == 2
        assert row["input_tokens"] == 20_000
        assert row["output_tokens"] == 7_100
        assert row["api_calls"] == 9
        assert row["avg_tokens_per_session"] == 13_550

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

    def test_session_token_endpoint_removed(self):
        """GET /api/auth/session-token no longer exists."""
        resp = self.client.get("/api/auth/session-token")
        # Should not return a JSON token object
        assert resp.status_code in {200, 404}
        try:
            data = resp.json()
            assert "token" not in data
        except Exception:
            pass


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

    def test_normalize_dict_without_context_length_yields_zero(self):
        """normalize should default to 0 when model dict has no context_length."""
        from hermes_cli.web_server import _normalize_config_for_web

        cfg = {"model": {"default": "test/model", "provider": "openrouter"}}
        result = _normalize_config_for_web(cfg)
        assert result["model_context_length"] == 0

    def test_normalize_non_int_context_length_yields_zero(self):
        """normalize should coerce non-int context_length to 0."""
        from hermes_cli.web_server import _normalize_config_for_web

        cfg = {"model": {"default": "test/model", "context_length": "invalid"}}
        result = _normalize_config_for_web(cfg)
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

    def test_denormalize_zero_removes_context_length(self):
        """denormalize with model_context_length=0 should remove context_length key."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({
            "model": {
                "default": "anthropic/claude-opus-4.6",
                "provider": "openrouter",
                "context_length": 50000,
            }
        })

        result = _denormalize_config_from_web({
            "model": "anthropic/claude-opus-4.6",
            "model_context_length": 0,
        })
        assert isinstance(result["model"], dict)
        assert "context_length" not in result["model"]

    def test_denormalize_upgrades_bare_string_to_dict(self):
        """denormalize should upgrade bare string model to dict when context_length set."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        # Disk has model as bare string
        save_config({"model": "anthropic/claude-sonnet-4"})

        result = _denormalize_config_from_web({
            "model": "anthropic/claude-sonnet-4",
            "model_context_length": 65000,
        })
        assert isinstance(result["model"], dict)
        assert result["model"]["default"] == "anthropic/claude-sonnet-4"
        assert result["model"]["context_length"] == 65000

    def test_denormalize_bare_string_stays_string_when_zero(self):
        """denormalize should keep bare string model as string when context_length=0."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({"model": "anthropic/claude-sonnet-4"})

        result = _denormalize_config_from_web({
            "model": "anthropic/claude-sonnet-4",
            "model_context_length": 0,
        })
        assert result["model"] == "anthropic/claude-sonnet-4"

    def test_denormalize_coerces_string_context_length(self):
        """denormalize should handle string model_context_length from frontend."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({
            "model": {"default": "test/model", "provider": "openrouter"}
        })

        result = _denormalize_config_from_web({
            "model": "test/model",
            "model_context_length": "32000",
        })
        assert isinstance(result["model"], dict)
        assert result["model"]["context_length"] == 32000


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

    def test_unchanged_model_preserves_provider_and_base_url(self):
        """Saving with the model unchanged must never re-detect/overwrite the
        provider — protects unrelated config saves and custom endpoints."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({
            "model": {
                "default": "llama3.2",
                "provider": "ollama-local",
                "base_url": "http://localhost:11434/v1",
            }
        })

        result = _denormalize_config_from_web({"model": "llama3.2"})
        model = result["model"]
        assert model["provider"] == "ollama-local"
        assert model["base_url"] == "http://localhost:11434/v1"

    def test_bare_model_name_change_keeps_local_provider(self):
        """A bare (non-slug) model name gives no provider signal — leave the
        existing provider alone rather than guessing."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({
            "model": {
                "default": "llama3.2",
                "provider": "ollama-local",
                "base_url": "http://localhost:11434/v1",
            }
        })

        result = _denormalize_config_from_web({"model": "qwen2.5"})
        model = result["model"]
        assert model["provider"] == "ollama-local"
        assert model["default"] == "qwen2.5"

    def test_same_aggregator_model_swap_keeps_provider(self):
        """Swapping models within an aggregator must not change the provider."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({
            "model": {"default": "anthropic/claude-opus-4.6", "provider": "openrouter"}
        })

        result = _denormalize_config_from_web({"model": "google/gemini-2.5-flash"})
        model = result["model"]
        assert model["provider"] == "openrouter"
        assert model["default"] == "google/gemini-2.5-flash"

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

    def test_schema_has_model_context_length(self):
        from hermes_cli.web_server import CONFIG_SCHEMA
        assert "model_context_length" in CONFIG_SCHEMA

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

    def test_model_info_returns_200(self):
        resp = self.client.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "provider" in data
        assert "auto_context_length" in data
        assert "config_context_length" in data
        assert "effective_context_length" in data
        assert "capabilities" in data

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

    def test_model_info_auto_detect_when_no_override(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "load_config", lambda: {
            "model": {"default": "anthropic/claude-opus-4.6", "provider": "openrouter"}
        })

        with patch("agent.model_metadata.get_model_context_length", return_value=200000):
            resp = self.client.get("/api/model/info")

        data = resp.json()
        assert data["auto_context_length"] == 200000
        assert data["config_context_length"] == 0
        assert data["effective_context_length"] == 200000  # auto wins

    def test_model_info_empty_model(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "load_config", lambda: {"model": ""})

        resp = self.client.get("/api/model/info")
        data = resp.json()
        assert data["model"] == ""
        assert data["effective_context_length"] == 0

    def test_model_info_bare_string_model(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "load_config", lambda: {
            "model": "anthropic/claude-sonnet-4"
        })

        with patch("agent.model_metadata.get_model_context_length", return_value=200000):
            resp = self.client.get("/api/model/info")

        data = resp.json()
        assert data["model"] == "anthropic/claude-sonnet-4"
        assert data["provider"] == ""
        assert data["config_context_length"] == 0
        assert data["effective_context_length"] == 200000

    def test_model_info_capabilities(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "load_config", lambda: {
            "model": {"default": "anthropic/claude-opus-4.6", "provider": "openrouter"}
        })

        mock_caps = MagicMock()
        mock_caps.supports_tools = True
        mock_caps.supports_vision = True
        mock_caps.supports_reasoning = True
        mock_caps.context_window = 200000
        mock_caps.max_output_tokens = 32000
        mock_caps.model_family = "claude-opus"

        with patch("agent.model_metadata.get_model_context_length", return_value=200000), \
             patch("agent.models_dev.get_model_capabilities", return_value=mock_caps):
            resp = self.client.get("/api/model/info")

        caps = resp.json()["capabilities"]
        assert caps["supports_tools"] is True
        assert caps["supports_vision"] is True
        assert caps["supports_reasoning"] is True
        assert caps["max_output_tokens"] == 32000
        assert caps["model_family"] == "claude-opus"

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

    def test_returns_false_when_no_url_configured(self, monkeypatch):
        """When GATEWAY_HEALTH_URL is unset, the probe returns (False, None)."""
        import hermes_cli.web_server as ws
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", None)
        alive, body = ws._probe_gateway_health()
        assert alive is False
        assert body is None

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

    def test_normalizes_url_with_health_suffix(self, monkeypatch):
        """If the user sets the URL to include /health, it's stripped to base."""
        import hermes_cli.web_server as ws
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642/health")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        # Both paths should fail (no server), but we verify they were constructed
        # correctly by checking the URLs attempted.
        calls = []
        original_urlopen = ws.urllib.request.urlopen

        def mock_urlopen(req, **kwargs):
            calls.append(req.full_url)
            raise ConnectionError("mock")

        monkeypatch.setattr(ws.urllib.request, "urlopen", mock_urlopen)
        alive, body = ws._probe_gateway_health()
        assert alive is False
        assert "http://gw:8642/health/detailed" in calls
        assert "http://gw:8642/health" in calls

    def test_normalizes_url_with_health_detailed_suffix(self, monkeypatch):
        """If the user sets the URL to include /health/detailed, it's stripped to base."""
        import hermes_cli.web_server as ws
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642/health/detailed")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        calls = []

        def mock_urlopen(req, **kwargs):
            calls.append(req.full_url)
            raise ConnectionError("mock")

        monkeypatch.setattr(ws.urllib.request, "urlopen", mock_urlopen)
        ws._probe_gateway_health()
        assert "http://gw:8642/health/detailed" in calls
        assert "http://gw:8642/health" in calls

    def test_successful_detailed_probe(self, monkeypatch):
        """Successful /health/detailed probe returns (True, body_dict)."""
        import hermes_cli.web_server as ws
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)

        response_body = json.dumps({
            "status": "ok",
            "gateway_state": "running",
            "pid": 42,
        })

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = response_body.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(ws.urllib.request, "urlopen", lambda req, **kw: mock_resp)
        alive, body = ws._probe_gateway_health()
        assert alive is True
        assert body["status"] == "ok"
        assert body["pid"] == 42

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

    def test_status_bounds_the_complete_remote_probe(self, monkeypatch):
        """Two serial HTTP attempts cannot consume more than the route budget."""
        import hermes_cli.web_server as ws

        probe_started = threading.Event()

        def slow_probe():
            probe_started.set()
            threading.Event().wait(timeout=0.1)
            return True, {"status": "ok", "pid": 999}

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", 0.02)
        monkeypatch.setattr(ws, "_probe_gateway_health", slow_probe)

        resp = self.client.get("/api/status")

        assert probe_started.is_set()
        assert resp.status_code == 200
        assert resp.json()["gateway_running"] is False

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

    def test_status_remote_probe_not_attempted_when_no_url(self, monkeypatch):
        """When GATEWAY_HEALTH_URL is unset, no probe is attempted."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", None)

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is False
        assert data["gateway_health_url"] is None

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

    def test_busy_when_running_with_active_agents(self, monkeypatch):
        """gateway_busy is True iff running AND active_agents > 0."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 2,
            # A deliberately stale timestamp: busy must NOT depend on it.
            "updated_at": "2020-01-01T00:00:00+00:00",
        })

        data = self.client.get("/api/status").json()
        assert data["active_agents"] == 2
        assert data["gateway_busy"] is True
        assert data["gateway_drainable"] is True

    def test_idle_running_is_drainable_but_not_busy(self, monkeypatch):
        """A running gateway with zero in-flight turns is drainable, not busy."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 0,
        })

        data = self.client.get("/api/status").json()
        assert data["active_agents"] == 0
        assert data["gateway_busy"] is False
        assert data["gateway_drainable"] is True

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

    def test_down_gateway_degrades_to_safe_falsy(self, monkeypatch):
        """Gateway down (no PID, no remote probe): busy/drainable False,
        active_agents 0 — never a spurious busy that would wedge NAS."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", None)

        data = self.client.get("/api/status").json()
        assert data["gateway_running"] is False
        assert data["active_agents"] == 0
        assert data["gateway_busy"] is False
        assert data["gateway_drainable"] is False

    def test_down_gateway_with_stale_busy_file_still_not_busy(self, monkeypatch):
        """A leftover status file claiming running + active_agents>0 must NOT
        read as busy when the live PID probe says the gateway is down. Liveness
        wins over the file."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", None)
        # File says running with active turns, but get_running_pid_cached()==None and
        # get_runtime_status_running_pid finds no live PID → gateway_running False.
        monkeypatch.setattr(ws, "get_runtime_status_running_pid", lambda *_a, **_k: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 5,
        })

        data = self.client.get("/api/status").json()
        assert data["gateway_running"] is False
        assert data["gateway_busy"] is False
        assert data["gateway_drainable"] is False

    def test_restart_drain_timeout_surfaced_and_numeric(self, monkeypatch):
        """restart_drain_timeout is present and resolves to a non-negative
        float so NAS can size its poll deadline without out-of-band knowledge."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 0,
        })
        monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "90")

        data = self.client.get("/api/status").json()
        assert "restart_drain_timeout" in data
        assert isinstance(data["restart_drain_timeout"], (int, float))
        assert data["restart_drain_timeout"] == 90.0

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

    @pytest.mark.parametrize("updated_at", [
        1750000000.5,          # legacy epoch float
        1750000000,            # legacy epoch int
        "not-a-timestamp",     # garbage string
        None,                  # explicit null
        True,                  # bool (int subclass) garbage
        {"nested": "junk"},    # structured garbage
    ])
    def test_local_runtime_updated_at_normalized(self, monkeypatch, updated_at):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 0,
            "updated_at": updated_at,
        })

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        self._assert_contract(resp.json()["gateway_updated_at"])

    def test_local_runtime_updated_at_absent(self, monkeypatch):
        """Key missing entirely from the status file → null, not a crash."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 0,
        })

        data = self.client.get("/api/status").json()
        assert data["gateway_updated_at"] is None

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

    def test_local_runtime_valid_iso_passes_through_parseable(self, monkeypatch):
        """The canonical writer format survives normalization round-trip."""
        from datetime import datetime
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": 0,
            "updated_at": "2026-07-21T12:00:00+00:00",
        })

        value = self.client.get("/api/status").json()["gateway_updated_at"]
        assert isinstance(value, str)
        assert datetime.fromisoformat(value).tzinfo is not None

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

    def test_remote_health_garbage_updated_at_nulled(self, monkeypatch):
        """Remote body with unparseable updated_at → null, never verbatim."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid_cached", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_probe_gateway_health", lambda: (True, {
            "status": "ok",
            "gateway_state": "running",
            "platforms": {},
            "updated_at": "yesterday-ish",
        }))

        data = self.client.get("/api/status").json()
        assert data["gateway_updated_at"] is None


# ---------------------------------------------------------------------------
# Dashboard theme normaliser tests
# ---------------------------------------------------------------------------


class TestNormaliseThemeDefinition:
    """Tests for _normalise_theme_definition() — parses YAML theme files."""

    def test_rejects_missing_name(self):
        from hermes_cli.web_server import _normalise_theme_definition
        assert _normalise_theme_definition({}) is None
        assert _normalise_theme_definition({"name": ""}) is None
        assert _normalise_theme_definition({"name": "   "}) is None

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

    def test_full_palette_form(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({
            "name": "full",
            "palette": {
                "background": {"hex": "#0a1628", "alpha": 1.0},
                "midground": {"hex": "#a8d0ff", "alpha": 0.9},
                "warmGlow": "rgba(255, 0, 0, 0.5)",
                "noiseOpacity": 0.5,
            },
        })
        assert result["palette"]["background"]["hex"] == "#0a1628"
        assert result["palette"]["midground"]["alpha"] == 0.9
        assert result["palette"]["warmGlow"] == "rgba(255, 0, 0, 0.5)"
        assert result["palette"]["noiseOpacity"] == 0.5

    def test_default_typography_applied_when_missing(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({"name": "minimal"})
        typo = result["typography"]
        assert "fontSans" in typo
        assert "fontMono" in typo
        assert typo["baseSize"] == "15px"
        assert typo["lineHeight"] == "1.55"
        assert typo["letterSpacing"] == "0"

    def test_partial_typography_merges_with_defaults(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({
            "name": "partial",
            "typography": {
                "fontSans": "MyFont, sans-serif",
                "baseSize": "12px",
            },
        })
        assert result["typography"]["fontSans"] == "MyFont, sans-serif"
        assert result["typography"]["baseSize"] == "12px"
        # fontMono defaulted
        assert "monospace" in result["typography"]["fontMono"]

    def test_layout_defaults(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({"name": "minimal"})
        assert result["layout"]["radius"] == "0.5rem"
        assert result["layout"]["density"] == "comfortable"

    def test_invalid_density_falls_back(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({
            "name": "bad",
            "layout": {"density": "ultra-spacious"},
        })
        assert result["layout"]["density"] == "comfortable"

    def test_valid_densities_accepted(self):
        from hermes_cli.web_server import _normalise_theme_definition
        for d in ("compact", "comfortable", "spacious"):
            r = _normalise_theme_definition({"name": "x", "layout": {"density": d}})
            assert r["layout"]["density"] == d

    def test_color_overrides_filter_unknown_keys(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({
            "name": "o",
            "colorOverrides": {
                "card": "#123456",
                "fakeToken": "#abcdef",
                "primary": 42,  # non-string rejected
                "destructive": "#ff0000",
            },
        })
        assert result["colorOverrides"] == {
            "card": "#123456",
            "destructive": "#ff0000",
        }

    def test_color_overrides_omitted_when_empty(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({"name": "x"})
        assert "colorOverrides" not in result

    def test_alpha_clamped_to_unit_range(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "c",
            "palette": {"background": {"hex": "#000", "alpha": 99.5}},
        })
        assert r["palette"]["background"]["alpha"] == 1.0
        r2 = _normalise_theme_definition({
            "name": "c",
            "palette": {"background": {"hex": "#000", "alpha": -5}},
        })
        assert r2["palette"]["background"]["alpha"] == 0.0

    def test_invalid_alpha_uses_default(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "c",
            "palette": {"background": {"hex": "#000", "alpha": "not a number"}},
        })
        assert r["palette"]["background"]["alpha"] == 1.0


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

    def test_malformed_yaml_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "bad.yaml").write_text("::: not valid yaml :::\n\tindent wrong")
        (themes_dir / "nameless.yaml").write_text("label: No Name Here\n")
        (themes_dir / "ok.yaml").write_text("name: ok\n")
        from hermes_cli import web_server
        results = web_server._discover_user_themes()
        names = [r["name"] for r in results]
        assert "ok" in names
        assert "bad" not in names  # malformed YAML
        assert len(results) == 1  # only the valid one

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

    def test_builtin_theme_renders_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli import web_server
        for builtin in ("default", "midnight", "cyberpunk"):
            monkeypatch.setattr(
                web_server, "load_config",
                lambda b=builtin: {"dashboard": {"theme": b}},
            )
            assert web_server._render_active_theme_bootstrap_css() == ""

    def test_unknown_theme_renders_nothing(self, tmp_path, monkeypatch):
        """Configured theme has no YAML on disk → empty string, no crash."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli import web_server
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": "ghost"}}
        )
        assert web_server._render_active_theme_bootstrap_css() == ""

    def test_non_string_theme_renders_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli import web_server
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": 42}}
        )
        assert web_server._render_active_theme_bootstrap_css() == ""

    def test_malformed_theme_yaml_no_crash(self, tmp_path, monkeypatch):
        """A garbage YAML for the active theme name must not crash — the
        discover helper skips it, so no style block is emitted."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "broken.yaml").write_text(
            "::: not valid yaml :::\n\tindent wrong", encoding="utf-8"
        )
        from hermes_cli import web_server
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": "broken"}}
        )
        assert web_server._render_active_theme_bootstrap_css() == ""

    def test_load_config_exception_no_crash(self, monkeypatch):
        from hermes_cli import web_server

        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(web_server, "load_config", boom)
        assert web_server._render_active_theme_bootstrap_css() == ""

    def test_style_escape_defends_style_breakout(self, tmp_path, monkeypatch):
        """`</style>` in a theme value cannot break out of the block."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "sneaky.yaml").write_text(
            "name: sneaky\n"
            "typography:\n"
            "  fontSans: '</style><script>alert(1)</script>'\n",
            encoding="utf-8",
        )
        from hermes_cli import web_server
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": "sneaky"}}
        )
        css = web_server._render_active_theme_bootstrap_css()
        assert css.count("</style>") == 1  # only the legitimate closer
        assert "<\\/style>" in css  # payload was escaped, not emitted raw

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

    def test_serve_index_no_bootstrap_for_builtin_theme(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws
        monkeypatch.setattr(
            ws, "load_config", lambda: {"dashboard": {"theme": "default"}}
        )
        client = self._mount_spa_client(tmp_path, monkeypatch)
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert "hermes-theme-bootstrap" not in resp.text

    def test_serve_index_survives_render_failure(self, tmp_path, monkeypatch):
        """Even if theme rendering blows up internally, index serving
        must not crash (the helper swallows and returns '')."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(ws, "load_config", boom)
        client = self._mount_spa_client(tmp_path, monkeypatch)
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert "hermes-theme-bootstrap" not in resp.text
        assert "SPA" in resp.text


class TestNormaliseThemeExtensions:
    """Tests for the extended normaliser fields (assets, customCSS,
    componentStyles, layoutVariant) — the surfaces themes use to reskin
    the dashboard without shipping code."""

    def test_layout_variant_defaults_to_standard(self):
        from hermes_cli.web_server import _normalise_theme_definition
        result = _normalise_theme_definition({"name": "t"})
        assert result["layoutVariant"] == "standard"

    def test_layout_variant_accepts_known_values(self):
        from hermes_cli.web_server import _normalise_theme_definition
        for variant in ("standard", "cockpit", "tiled"):
            r = _normalise_theme_definition({"name": "t", "layoutVariant": variant})
            assert r["layoutVariant"] == variant

    def test_layout_variant_rejects_unknown(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({"name": "t", "layoutVariant": "warship"})
        assert r["layoutVariant"] == "standard"
        r2 = _normalise_theme_definition({"name": "t", "layoutVariant": 12})
        assert r2["layoutVariant"] == "standard"

    def test_assets_named_slots_passthrough(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "t",
            "assets": {
                "bg": "https://example.com/bg.jpg",
                "hero": "linear-gradient(180deg, red, blue)",
                "crest": "/ds-assets/crest.svg",
                "logo": "  ",  # whitespace-only — dropped
                "notAKnownKey": "ignored",
            },
        })
        assert r["assets"]["bg"] == "https://example.com/bg.jpg"
        assert r["assets"]["hero"].startswith("linear-gradient")
        assert r["assets"]["crest"] == "/ds-assets/crest.svg"
        assert "logo" not in r["assets"]  # whitespace-only rejected
        assert "notAKnownKey" not in r["assets"]  # unknown slot ignored

    def test_assets_custom_block(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "t",
            "assets": {
                "custom": {
                    "scan-lines": "/img/scan.png",
                    "my_overlay": "/img/ov.png",
                    "bad key!": "x",  # non-alnum key — rejected
                    "empty": "",        # empty value — rejected
                },
            },
        })
        assert r["assets"]["custom"] == {
            "scan-lines": "/img/scan.png",
            "my_overlay": "/img/ov.png",
        }

    def test_assets_absent_means_no_field(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({"name": "t"})
        assert "assets" not in r

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

    def test_custom_css_empty_dropped(self):
        from hermes_cli.web_server import _normalise_theme_definition
        for val in ("", "   \n\t", None):
            r = _normalise_theme_definition({"name": "t", "customCSS": val})
            assert "customCSS" not in r

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

    def test_component_styles_empty_buckets_dropped(self):
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "t",
            "componentStyles": {
                "card": {},        # empty — dropped entirely
                "header": {"bad prop!": "ignored"},  # all props rejected — bucket dropped
                "footer": {"background": "black"},
            },
        })
        assert "card" not in r.get("componentStyles", {})
        assert "header" not in r.get("componentStyles", {})
        assert r["componentStyles"]["footer"]["background"] == "black"

    def test_component_styles_accepts_numeric_values(self):
        """Numeric values (e.g. opacity: 0.8) are coerced to strings."""
        from hermes_cli.web_server import _normalise_theme_definition
        r = _normalise_theme_definition({
            "name": "t",
            "componentStyles": {"card": {"opacity": 0.8, "zIndex": 5}},
        })
        assert r["componentStyles"]["card"] == {"opacity": "0.8", "zIndex": "5"}


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

    def test_delete_existing_session(self):
        self._seed(["real_one"])
        resp = self.auth_client.delete("/api/sessions/real_one")
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        assert not self._exists("real_one")

    def test_delete_absent_session_is_idempotent(self):
        # PREMISE / regression: deleting a row that no longer exists must NOT
        # 404 — the desktop would resurrect the ghost row and show
        # "session not found". DELETE's contract is "ensure it's gone".
        resp = self.auth_client.delete("/api/sessions/never_existed")
        assert resp.status_code == 200
        assert resp.json().get("ok") is True

    def test_delete_resolves_unique_prefix(self):
        # Symmetry with the other session endpoints, which all resolve ids.
        self._seed(["20260618_abcdef_unique"])
        resp = self.auth_client.delete("/api/sessions/20260618_abcdef")
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        assert not self._exists("20260618_abcdef_unique")


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

    def test_requires_auth(self):
        resp = self.client.post("/api/sessions/bulk-delete", json={"ids": ["x"]})
        assert resp.status_code == 401

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

    def test_unknown_ids_silently_skipped(self):
        """The endpoint never 404s on a missing ID — it returns the
        real deleted count so a UI selection that raced against
        another tab still resolves cleanly."""
        self._seed(["real"])
        resp = self.auth_client.post(
            "/api/sessions/bulk-delete",
            json={"ids": ["real", "ghost1", "ghost2"]},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 1}

    def test_empty_list_is_noop(self):
        """``ids: []`` returns ``deleted: 0`` (200, not 400) — the UI
        treats an empty selection as a no-op rather than an error."""
        resp = self.auth_client.post(
            "/api/sessions/bulk-delete", json={"ids": []}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 0}

    def test_payload_cap_enforced(self):
        """501 IDs returns 400 — a hard cap stops a runaway selection
        from holding the SQLite writer for an extended window."""
        resp = self.auth_client.post(
            "/api/sessions/bulk-delete",
            json={"ids": [f"s{i}" for i in range(501)]},
        )
        assert resp.status_code == 400
        # 500 exactly still succeeds (no rows actually present, so
        # deleted=0 — but it's not the cap path).
        resp = self.auth_client.post(
            "/api/sessions/bulk-delete",
            json={"ids": [f"s{i}" for i in range(500)]},
        )
        assert resp.status_code == 200

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

    def test_count_endpoint_requires_auth(self):
        """GET /api/sessions/empty/count must 401 without the session token."""
        resp = self.client.get("/api/sessions/empty/count")
        assert resp.status_code == 401

    def test_delete_endpoint_requires_auth(self):
        """DELETE /api/sessions/empty must 401 without the session token.

        Regression guard for issue #19533 — the bulk-delete is a strictly
        destructive primitive, the middleware must gate it even if a
        future refactor introduces a non-auth path."""
        resp = self.client.delete("/api/sessions/empty")
        assert resp.status_code == 401

    def test_count_returns_only_empty_ended_unarchived(self):
        """With the standard corpus, the count is exactly 2 — only
        ``empty1`` and ``empty2`` qualify (``hasmsg`` has a message,
        ``live`` is active, ``archived`` is archived)."""
        self._seed()
        resp = self.auth_client.get("/api/sessions/empty/count")
        assert resp.status_code == 200
        assert resp.json() == {"count": 2}

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

    def test_delete_with_no_empties_returns_zero(self):
        """No empty sessions → endpoint returns ``deleted: 0`` (200,
        not 404). The dashboard relies on this no-op path to surface
        a "Nothing to clean up" toast instead of an error."""
        resp = self.auth_client.delete("/api/sessions/empty")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 0}

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

    def test_plugin_route_requires_auth(self):
        """Plugin API routes should return 401 without a valid session token."""
        # Use a known plugin route (kanban board)
        resp = self.client.get("/api/plugins/kanban/board")
        assert resp.status_code == 401

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

    def test_plugin_post_requires_auth(self):
        """Plugin POST routes should return 401 without a valid session token."""
        resp = self.client.post("/api/plugins/kanban/tasks", json={"title": "test"})
        assert resp.status_code == 401

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

    def test_plugin_delete_requires_auth(self):
        """Plugin DELETE routes should return 401 without a valid session token."""
        resp = self.client.delete("/api/plugins/kanban/tasks/t_fake")
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

    def test_override_requires_leading_slash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_plugin(tmp_path, "bad-override", {
            "name": "bad-override",
            "label": "Bad",
            "tab": {"path": "/bad", "override": "no-leading-slash"},
            "entry": "dist/index.js",
        })
        from hermes_cli import web_server
        web_server._dashboard_plugins_cache = None
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "bad-override")
        assert "override" not in entry["tab"]

    def test_slots_default_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_plugin(tmp_path, "no-slots", {
            "name": "no-slots",
            "label": "No Slots",
            "tab": {"path": "/no-slots"},
            "entry": "dist/index.js",
        })
        from hermes_cli import web_server
        web_server._dashboard_plugins_cache = None
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "no-slots")
        assert entry["slots"] == []
        assert "hidden" not in entry["tab"]
        assert "override" not in entry["tab"]

    def test_slots_filters_non_string_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_plugin(tmp_path, "mixed-slots", {
            "name": "mixed-slots",
            "label": "Mixed",
            "tab": {"path": "/mixed-slots"},
            "slots": ["sidebar", "", 42, None, "header-right"],
            "entry": "dist/index.js",
        })
        from hermes_cli import web_server
        web_server._dashboard_plugins_cache = None
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "mixed-slots")
        assert entry["slots"] == ["sidebar", "header-right"]

    def test_page_scoped_slots_preserved(self, tmp_path, monkeypatch):
        """Page-scoped slot names (e.g. ``sessions:top``) round-trip through
        the manifest loader untouched.  The backend has no allowlist — the
        frontend ``<PluginSlot name="...">`` placements decide what actually
        renders — but the loader must not mangle colons in slot names."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_plugin(tmp_path, "page-slots", {
            "name": "page-slots",
            "label": "Page Slots",
            "tab": {"path": "/page-slots", "hidden": True},
            "slots": [
                "sessions:top",
                "analytics:bottom",
                "logs:top",
                "skills:bottom",
                "config:top",
                "env:bottom",
                "docs:top",
                "cron:bottom",
                "chat:top",
            ],
            "entry": "dist/index.js",
        })
        from hermes_cli import web_server
        web_server._dashboard_plugins_cache = None
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "page-slots")
        assert entry["slots"] == [
            "sessions:top",
            "analytics:bottom",
            "logs:top",
            "skills:bottom",
            "config:top",
            "env:bottom",
            "docs:top",
            "cron:bottom",
            "chat:top",
        ]


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

    def test_resolve_chat_argv_uses_dashboard_scroll_env(self, monkeypatch):
        """Dashboard chat runs the TUI in browser-scrollback mode."""
        import hermes_cli.main as main_mod

        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env["HERMES_TUI_DASHBOARD"] == "1"
        assert env["HERMES_TUI_INLINE"] == "1"
        assert env["HERMES_TUI_DISABLE_MOUSE"] == "1"

    def test_resolve_chat_argv_backfills_colorterm_truecolor(self, monkeypatch):
        """Headless servers (cloud/systemd) have no COLORTERM, which made
        chalk in the TUI child degrade skin hex colors to the xterm 256
        palette (gold banner rendered salmon-red). xterm.js always supports
        24-bit color, so the PTY env must advertise truecolor."""
        import hermes_cli.main as main_mod

        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )
        monkeypatch.delenv("COLORTERM", raising=False)

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env["COLORTERM"] == "truecolor"

    def test_resolve_chat_argv_keeps_operator_colorterm(self, monkeypatch):
        """An explicit operator COLORTERM wins over the backfill."""
        import hermes_cli.main as main_mod

        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )
        monkeypatch.setenv("COLORTERM", "24bit")

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env["COLORTERM"] == "24bit"

    def test_resolve_chat_argv_sets_tui_python_environment(self, monkeypatch):
        """Dashboard chat gives the Node TUI the same Python env as CLI launches."""
        import hermes_cli.main as main_mod

        monkeypatch.delenv("HERMES_PYTHON_SRC_ROOT", raising=False)
        monkeypatch.delenv("HERMES_PYTHON", raising=False)
        monkeypatch.delenv("HERMES_CWD", raising=False)
        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env is not None
        assert env["HERMES_PYTHON_SRC_ROOT"] == str(main_mod.PROJECT_ROOT)
        assert env["HERMES_PYTHON"] == sys.executable
        assert env["HERMES_CWD"] == os.getcwd()

    def test_resolve_chat_argv_replaces_invalid_tui_python_environment(self, monkeypatch):
        """Dashboard chat does not preserve unusable inherited TUI Python env."""
        import hermes_cli.main as main_mod

        monkeypatch.setenv("HERMES_PYTHON_SRC_ROOT", "/definitely/missing/hermes-src")
        monkeypatch.setenv("HERMES_PYTHON", "/definitely/missing/python")
        monkeypatch.setenv("HERMES_CWD", "/definitely/missing/cwd")
        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env is not None
        assert env["HERMES_PYTHON_SRC_ROOT"] == str(main_mod.PROJECT_ROOT)
        assert env["HERMES_PYTHON"] == sys.executable
        assert env["HERMES_CWD"] == os.getcwd()

    def test_resolve_chat_argv_keeps_relative_python_under_tui_cwd(
        self, monkeypatch, tmp_path
    ):
        """Relative Python paths are resolved from the TUI child's cwd."""
        import hermes_cli.main as main_mod

        relative_python = Path(".review-venv") / "bin" / Path(sys.executable).name
        python_path = tmp_path / relative_python
        python_path.parent.mkdir(parents=True)
        # copy2, not os.link: tmp_path may sit on a different filesystem than
        # the venv (tmpfs /tmp vs disk home) where hard links raise EXDEV.
        shutil.copy2(sys.executable, python_path)
        monkeypatch.setenv("HERMES_CWD", str(tmp_path))
        monkeypatch.setenv("HERMES_PYTHON", str(relative_python))
        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env is not None
        assert env["HERMES_PYTHON"] == str(relative_python)

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

    def test_resolve_chat_argv_falls_back_when_getcwd_is_missing(self, monkeypatch, tmp_path):
        """Dashboard chat still starts if the service cwd was deleted."""
        import hermes_cli.main as main_mod

        monkeypatch.delenv("HERMES_CWD", raising=False)
        monkeypatch.setenv("PWD", str(tmp_path))
        monkeypatch.setattr(main_mod.os, "getcwd", lambda: (_ for _ in ()).throw(FileNotFoundError()))
        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env is not None
        assert env["HERMES_CWD"] == str(tmp_path)

    def test_resolve_chat_argv_applies_terminal_backend_config(
        self, monkeypatch, _isolate_hermes_home
    ):
        import hermes_cli.main as main_mod

        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "terminal:",
                    "  backend: docker",
                    "  docker_image: example/hermes-tools:latest",
                    "  docker_extra_args:",
                    "    - --network=host",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        monkeypatch.delenv("TERMINAL_DOCKER_IMAGE", raising=False)
        monkeypatch.delenv("TERMINAL_DOCKER_EXTRA_ARGS", raising=False)
        monkeypatch.setattr(
            main_mod,
            "_make_tui_argv",
            lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
        )

        _argv, _cwd, env = self.ws_module._resolve_chat_argv()

        assert env["TERMINAL_ENV"] == "docker"
        assert env["TERMINAL_DOCKER_IMAGE"] == "example/hermes-tools:latest"
        assert env["TERMINAL_DOCKER_EXTRA_ARGS"] == '["--network=host"]'

    def test_rejects_when_embedded_chat_disabled(self, monkeypatch):
        monkeypatch.setattr(self.ws_module, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", False)
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc:
            with self.client.websocket_connect(self._url()):
                pass
        assert exc.value.code == 4404

    def test_rejects_missing_token(self, monkeypatch):
        monkeypatch.setattr(
            self.ws_module,
            "_resolve_chat_argv",
            lambda resume=None, sidecar_url=None, profile=None: (["/bin/cat"], None, None),
        )
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc:
            with self.client.websocket_connect("/api/pty"):
                pass
        assert exc.value.code == 4401

    def test_rejects_bad_token(self, monkeypatch):
        monkeypatch.setattr(
            self.ws_module,
            "_resolve_chat_argv",
            lambda resume=None, sidecar_url=None, profile=None: (["/bin/cat"], None, None),
        )
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc:
            with self.client.websocket_connect(self._url(token="wrong")):
                pass
        assert exc.value.code == 4401

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

    def test_pty_ws_resolves_argv_through_async_wrapper(self, monkeypatch):
        captured: dict = {}

        async def fake_resolve_async(resume=None, sidecar_url=None, profile=None):
            captured["resume"] = resume
            captured["sidecar_url"] = sidecar_url
            captured["profile"] = profile
            return (["/bin/sh", "-c", "printf async-resolve-ok"], None, None)

        monkeypatch.setattr(self.ws_module, "_resolve_chat_argv_async", fake_resolve_async)

        with self.client.websocket_connect(self._url(resume="sess-99")) as conn:
            try:
                conn.receive_bytes()
            except Exception:
                pass

        assert captured["resume"] == "sess-99"

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

    def test_pty_ws_propagates_systemexit_through_async_wrapper(self, monkeypatch):
        """SystemExit from _make_tui_argv (node/npm missing) propagates through
        the async wrapper and is caught by pty_ws's ``except SystemExit``."""

        def boom(resume=None, sidecar_url=None, profile=None):
            raise SystemExit("node not found")

        self._assert_pty_propagates(monkeypatch, boom)

    def test_pty_ws_propagates_httpexception_through_async_wrapper(self, monkeypatch):
        """An invalid-profile HTTPException raised inside the threaded resolver
        propagates through the wrapper and hits pty_ws's ``except HTTPException``."""
        from fastapi import HTTPException

        def bad_profile(resume=None, sidecar_url=None, profile=None):
            raise HTTPException(status_code=404, detail="unknown profile")

        self._assert_pty_propagates(
            monkeypatch, bad_profile, profile="ghost", expect_detail="unknown profile"
        )

    def test_streams_child_stdout_to_client(self, monkeypatch):
        monkeypatch.setattr(
            self.ws_module,
            "_resolve_chat_argv",
            lambda resume=None, sidecar_url=None, profile=None: (
                ["/bin/sh", "-c", "printf hermes-ws-ok"],
                None,
                None,
            ),
        )
        with self.client.websocket_connect(self._url()) as conn:
            # Drain frames until we see the needle or time out.  TestClient's
            # recv_bytes blocks; loop until we have the signal byte string.
            buf = b""
            import time

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    frame = conn.receive_bytes()
                except Exception:
                    break
                if frame:
                    buf += frame
                if b"hermes-ws-ok" in buf:
                    break
            assert b"hermes-ws-ok" in buf

    def test_client_input_reaches_child_stdin(self, monkeypatch):
        # ``cat`` echoes stdin back, so a write → read round-trip proves
        # the full duplex path.
        monkeypatch.setattr(
            self.ws_module,
            "_resolve_chat_argv",
            lambda resume=None, sidecar_url=None, profile=None: (["/bin/cat"], None, None),
        )
        with self.client.websocket_connect(self._url()) as conn:
            conn.send_bytes(b"round-trip-payload\n")
            buf = b""
            import time

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                frame = conn.receive_bytes()
                if frame:
                    buf += frame
                if b"round-trip-payload" in buf:
                    break
            assert b"round-trip-payload" in buf

    def test_resize_escape_is_forwarded(self, monkeypatch):
        # Resize escape gets intercepted and applied via TIOCSWINSZ, then the
        # child reads the TTY ioctl directly. Avoid tput because CI may not set
        # TERM for non-interactive shells.
        import sys

        winsize_script = (
            "import fcntl, struct, termios, time; "
            "time.sleep(0.5); "
            "rows, cols, *_ = struct.unpack('HHHH', "
            "fcntl.ioctl(0, termios.TIOCGWINSZ, b'\\0' * 8)); "
            "print(cols); print(rows)"
        )
        monkeypatch.setattr(
            self.ws_module,
            "_resolve_chat_argv",
            # sleep gives the test time to push the resize before the child reads the ioctl.
            lambda resume=None, sidecar_url=None, profile=None: (
                [sys.executable, "-c", winsize_script],
                None,
                None,
            ),
        )
        with self.client.websocket_connect(self._url()) as conn:
            conn.send_text("\x1b[RESIZE:99;41]")
            buf = b""
            import time

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                # receive_bytes() blocks; once the child prints its winsize and
                # exits, the PTY closes and further reads raise. Without this
                # guard a missed-marker run blocks until a test timeout
                # (flaky failure) instead of failing fast on the assert below.
                try:
                    frame = conn.receive_bytes()
                except Exception:
                    break
                if frame:
                    buf += frame
                if b"99" in buf and b"41" in buf:
                    break
            assert b"99" in buf and b"41" in buf

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

    def test_resume_parameter_is_forwarded_to_argv(self, monkeypatch):
        captured: dict = {}

        def fake_resolve(resume=None, sidecar_url=None, profile=None):
            captured["resume"] = resume
            return (["/bin/sh", "-c", "printf resume-arg-ok"], None, None)

        monkeypatch.setattr(self.ws_module, "_resolve_chat_argv", fake_resolve)

        with self.client.websocket_connect(self._url(resume="sess-42")) as conn:
            # Drain briefly so the handler actually invokes the resolver.
            try:
                conn.receive_bytes()
            except Exception:
                pass
        assert captured.get("resume") == "sess-42"

    def test_channel_param_propagates_sidecar_url(self, monkeypatch):
        """When /api/pty is opened with ?channel=, the PTY child gets a
        HERMES_TUI_SIDECAR_URL env var pointing back at /api/pub on the
        same channel — which is how tool events reach the dashboard sidebar."""
        captured: dict = {}

        def fake_resolve(resume=None, sidecar_url=None, profile=None, active_session_file=None):
            captured["sidecar_url"] = sidecar_url
            captured["active_session_file"] = active_session_file
            return (["/bin/sh", "-c", "printf sidecar-ok"], None, None)

        monkeypatch.setattr(self.ws_module, "_resolve_chat_argv", fake_resolve)
        monkeypatch.setattr(
            self.ws_module.app.state, "bound_host", "127.0.0.1", raising=False
        )
        monkeypatch.setattr(
            self.ws_module.app.state, "bound_port", 9119, raising=False
        )

        headers = {"host": "127.0.0.1:9119", "origin": "http://127.0.0.1:9119"}
        with self.client.websocket_connect(
            self._url(channel="abc-123"), headers=headers
        ) as conn:
            try:
                conn.receive_bytes()
            except Exception:
                pass

        url = captured.get("sidecar_url") or ""
        assert url.startswith("ws://127.0.0.1:9119/api/pub?")
        assert "channel=abc-123" in url
        assert "token=" in url
        assert captured["active_session_file"]

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

    def test_events_rejects_missing_channel(self):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc:
            with self.client.websocket_connect(
                f"/api/events?token={self.token}"
            ):
                pass
        assert exc.value.code == 4400


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

    def test_pycache_is_404(self):
        """Same protection for compiled Python (``.pyc``) inside the
        plugin's ``__pycache__/``. Real plugins ship these as a
        side-effect of running tests / dashboard once."""
        # __pycache__ files are only generated after the api file has
        # been imported once. Use the path the example plugin actually
        # generates during the dashboard test boot.
        resp = self.client.get(
            "/dashboard-plugins/example/__pycache__/plugin_api.cpython-311.pyc"
        )
        # 404 either way (file may not exist on this CI Python version);
        # what matters is we never get a 200 with the bytes.
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

    def test_unknown_plugin_is_404(self):
        """Existing behaviour preserved: nonexistent plugin name → 404."""
        resp = self.client.get(
            "/dashboard-plugins/_definitely_not_a_plugin_/manifest.json"
        )
        assert resp.status_code == 404

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

    def test_rejected_key_blocks(self, monkeypatch):
        monkeypatch.setattr("httpx.Client", _fake_httpx_client(status=401))
        data = self._post("OPENROUTER_API_KEY", "sk-bogus").json()
        assert data["ok"] is False and data["reachable"] is True

    def test_valid_key_passes(self, monkeypatch):
        monkeypatch.setattr("httpx.Client", _fake_httpx_client(status=200))
        data = self._post("OPENAI_API_KEY", "sk-real").json()
        assert data["ok"] is True and data["reachable"] is True

    def test_rate_limited_counts_as_valid(self, monkeypatch):
        monkeypatch.setattr("httpx.Client", _fake_httpx_client(status=429))
        data = self._post("XAI_API_KEY", "xai-real").json()
        assert data["ok"] is True

    def test_network_error_is_unreachable_not_blocking(self, monkeypatch):
        monkeypatch.setattr("httpx.Client", _fake_httpx_client(raise_exc=True))
        data = self._post("OPENROUTER_API_KEY", "sk-real").json()
        assert data["ok"] is False and data["reachable"] is False

    def test_unknown_provider_is_not_validated(self):
        # No probe for this key → don't block (ok True, reachable False).
        data = self._post("SOME_OTHER_API_KEY", "whatever-value").json()
        assert data["ok"] is True and data["reachable"] is False

    def test_empty_value_rejected(self):
        data = self._post("OPENAI_API_KEY", "   ").json()
        assert data["ok"] is False

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

    def test_local_endpoint_without_key_sends_no_auth_header(self, monkeypatch):
        """No key → no Authorization header (keyless local servers unaffected)."""
        captured = {}

        class _Resp:
            status_code = 200
            is_success = True

            def json(self):
                return {"data": []}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, *a, headers=None, **k):
                captured["headers"] = headers
                return _Resp()

        monkeypatch.setattr("httpx.Client", _Client)

        self.client.post(
            "/api/providers/validate",
            json={"key": "OPENAI_BASE_URL", "value": "http://127.0.0.1:8000/v1"},
        )
        assert captured["headers"] is None


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

    def test_ticker_skipped_without_desktop(self, monkeypatch, _isolate_hermes_home):
        import threading
        import cron.scheduler as sched

        called = threading.Event()
        monkeypatch.setattr(sched, "tick", lambda *a, **k: called.set())
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)

        with self._client():
            assert not called.wait(0.5), "ticker must not run outside the desktop app"


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

    def test_middleware_counts_unhandled_exception(self):
        """An exception escaping a route must be recorded (and re-raised)."""
        route_path = "/api/_test_boom"

        async def _boom():
            raise RuntimeError("kaboom")

        from fastapi.routing import APIRoute

        self.ws.app.router.routes.insert(
            0, APIRoute(route_path, _boom, methods=["GET"])
        )
        try:
            resp = self.client.get(route_path)
            assert resp.status_code == 500
            assert self.ws.DASHBOARD_HEALTH.recent_error_count() == 1
            assert self.ws.DASHBOARD_HEALTH.last_error_type == "RuntimeError"
            # Path is retained internally only — snapshot must not export it.
            assert self.ws.DASHBOARD_HEALTH.last_error_path == route_path
        finally:
            self.ws.app.router.routes[:] = [
                r for r in self.ws.app.router.routes
                if getattr(r, "path", None) != route_path
            ]

    def test_middleware_counts_5xx_response(self):
        route_path = "/api/_test_teapot_fire"

        async def _fivehundred():
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=503, content={"detail": "down"})

        from fastapi.routing import APIRoute

        self.ws.app.router.routes.insert(
            0, APIRoute(route_path, _fivehundred, methods=["GET"])
        )
        try:
            resp = self.client.get(route_path)
            assert resp.status_code == 503
            assert self.ws.DASHBOARD_HEALTH.recent_error_count() == 1
            assert self.ws.DASHBOARD_HEALTH.last_error_type == "http_503"
        finally:
            self.ws.app.router.routes[:] = [
                r for r in self.ws.app.router.routes
                if getattr(r, "path", None) != route_path
            ]

    def test_error_window_expires_old_entries(self, monkeypatch):
        health = self.ws.DashboardHealth(window_seconds=300)
        now = {"t": 1000.0}
        monkeypatch.setattr(self.ws.time, "time", lambda: now["t"])
        health.record_error("RuntimeError", "/api/x")
        assert health.recent_error_count() == 1
        now["t"] = 1000.0 + 301
        assert health.recent_error_count() == 0

    # -- /api/status components ------------------------------------------

    def test_status_includes_components_and_overall(self):
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall"] in {"ok", "degraded"}
        components = data["components"]
        assert set(components) == {"gateway", "storage", "dashboard", "platforms"}
        for comp in components.values():
            assert comp["status"] in {"ok", "degraded"}
        dashboard = components["dashboard"]
        assert dashboard["recent_unhandled_errors"] == 0
        assert "last_error_at" in dashboard
        assert dashboard["selftest"] in {"unknown", "ok", "failing"}

    def test_storage_degraded_when_state_db_probe_fails(self, monkeypatch):
        import gateway.readiness as readiness

        monkeypatch.setattr(
            readiness, "_probe_state_db", lambda home: {"status": "degraded", "detail": "OperationalError"}
        )
        resp = self.client.get("/api/status")
        data = resp.json()
        assert data["components"]["storage"] == {"status": "degraded"}
        assert data["overall"] == "degraded"

    def test_dashboard_component_degraded_after_error(self):
        self.ws.DASHBOARD_HEALTH.record_error("RuntimeError", "/api/x")
        resp = self.client.get("/api/status")
        data = resp.json()
        dashboard = data["components"]["dashboard"]
        assert dashboard["status"] == "degraded"
        assert dashboard["recent_unhandled_errors"] == 1
        assert data["overall"] == "degraded"

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

    def test_selftest_records_pass_on_200(self, monkeypatch):
        httpx = pytest.importorskip("httpx")

        class _FakeResponse:
            status_code = 200

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None, **kwargs):
                # The probe must authenticate with the real session token.
                assert headers[self.ws_header] == self.ws_token
                return _FakeResponse()

        _FakeClient.ws_header = self.ws._SESSION_HEADER_NAME
        _FakeClient.ws_token = self.ws._SESSION_TOKEN
        monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
        asyncio.run(self.ws._dashboard_selftest_once())
        assert self.ws.DASHBOARD_HEALTH.selftest_status == "ok"
        assert self.ws.DASHBOARD_HEALTH.selftest_http_status == 200

    def test_selftest_real_asgi_roundtrip(self):
        """End-to-end: the in-process ASGI self-test hits the real route."""
        pytest.importorskip("httpx")
        asyncio.run(self.ws._dashboard_selftest_once())
        assert self.ws.DASHBOARD_HEALTH.selftest_status in {"ok", "failing"}
        assert self.ws.DASHBOARD_HEALTH.selftest_http_status is not None
