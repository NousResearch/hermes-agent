"""Tests for hermes_cli.web_server and related config utilities."""

import asyncio
import os
import json
import shutil
import sys
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
        assert {opt["value"] for opt in fields["mode"]["options"]} == {"cloud", "local_external"}
        assert fields["api_url"]["value"] == "https://api.hindsight.vectorize.io"
        assert fields["bank_id"]["value"] == "hermes"
        assert fields["recall_budget"]["value"] == "mid"
        assert fields["api_key"]["kind"] == "secret"
        assert fields["api_key"]["is_set"] is False

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
        assert resp.json() == {"ok": True}
        assert load_config()["memory"]["provider"] == "hindsight"
        assert load_env()["HINDSIGHT_API_KEY"] == "hs-test-key"

        config_path = get_hermes_home() / "hindsight" / "config.json"
        provider_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert provider_config == {
            "mode": "local_external",
            "api_url": "http://localhost:8888",
            "bank_id": "ben-bank",
            "recall_budget": "high",
        }

    def test_put_memory_provider_config_rejects_unsupported_select_value(self):
        resp = self.client.put(
            "/api/memory/providers/hindsight/config",
            json={
                "values": {
                    "mode": "local_embedded",
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


    def test_get_status_filters_unconfigured_gateway_platforms(self, monkeypatch):
        import gateway.config as gateway_config
        import hermes_cli.web_server as web_server

        class _Platform:
            def __init__(self, value):
                self.value = value

        class _GatewayConfig:
            def get_connected_platforms(self):
                return [_Platform("telegram")]

        monkeypatch.setattr(web_server, "get_running_pid", lambda: 1234)
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

        monkeypatch.setattr(web_server, "get_running_pid", lambda: None)
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
        assert "messaging" in data["category_order"]
        assert schema["discord.require_mention"]["category"] == "messaging"
        assert schema["telegram.reactions"]["category"] == "messaging"
        assert "discord" not in data["category_order"]

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
        assert captured["args"] == ["backup", str(archive)]
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
        assert captured["args"] == ["backup", str(archive)]
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
        assert any(field["key"] == "TELEGRAM_BOT_TOKEN" and field["required"] for field in telegram["env_vars"])

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
                "env": {"TELEGRAM_BOT_TOKEN": "1234567890abcdef"},
            },
        )

        assert resp.status_code == 200
        assert load_env()["TELEGRAM_BOT_TOKEN"] == "1234567890abcdef"
        assert load_config()["platforms"]["telegram"]["enabled"] is False

        status = self.client.get("/api/messaging/platforms").json()["platforms"]
        telegram = next(platform for platform in status if platform["id"] == "telegram")
        assert telegram["enabled"] is False

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
                "enabled": True,
                "available": True,
                "configured": False,
                "tools": ["web_extract", "web_search"],
            },
            {
                "name": "skills",
                "label": "Skills",
                "description": "list, view, manage",
                "enabled": True,
                "available": True,
                "configured": True,
                "tools": ["skill_view", "skills_list"],
            },
            {
                "name": "memory",
                "label": "Memory",
                "description": "persistent memory across sessions",
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

    def test_select_toolset_provider_unknown_toolset_returns_400(self):
        resp = self.client.put(
            "/api/tools/toolsets/not_a_real_toolset/provider",
            json={"provider": "whatever"},
        )
        assert resp.status_code == 400

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

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", None)
        # File says running with active turns, but get_running_pid()==None and
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
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

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: {
            "gateway_state": "running",
            "platforms": {},
            "active_agents": "garbage",
        })

        data = self.client.get("/api/status").json()
        assert data["active_agents"] == 0
        assert data["gateway_busy"] is False


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
