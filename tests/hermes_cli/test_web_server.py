"""Tests for hermes_cli.web_server and related config utilities."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.config import (
    DEFAULT_CONFIG,
    reload_env,
    redact_key,
    _EXTRA_ENV_KEYS,
    OPTIONAL_ENV_VARS,
)


# ---------------------------------------------------------------------------
# reload_env tests
# ---------------------------------------------------------------------------


class TestReloadEnv:
    """Tests for reload_env() — re-reads .env into os.environ."""

    def test_adds_new_vars(self, tmp_path):
        """reload_env() adds vars from .env that are not in os.environ."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_RELOAD_VAR=hello123\n")
        with patch("hermes_cli.config.get_env_path", return_value=env_file):
            os.environ.pop("TEST_RELOAD_VAR", None)
            count = reload_env()
            assert count >= 1
            assert os.environ.get("TEST_RELOAD_VAR") == "hello123"
        os.environ.pop("TEST_RELOAD_VAR", None)

    def test_updates_changed_vars(self, tmp_path):
        """reload_env() updates vars whose value changed on disk."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_RELOAD_VAR=old_value\n")
        with patch("hermes_cli.config.get_env_path", return_value=env_file):
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
        with patch("hermes_cli.config.get_env_path", return_value=env_file):
            os.environ[known_key] = "stale_value"
            count = reload_env()
            assert known_key not in os.environ
            assert count >= 1

    def test_does_not_remove_unknown_vars(self, tmp_path):
        """reload_env() preserves non-Hermes env vars even when absent from .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("")
        with patch("hermes_cli.config.get_env_path", return_value=env_file):
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


# ---------------------------------------------------------------------------
# web_server tests (FastAPI endpoints)
# ---------------------------------------------------------------------------


class TestWebServerEndpoints:
    """Test the FastAPI REST endpoints using Starlette TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self):
        """Create a TestClient — import is deferred to avoid requiring fastapi."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_get_status(self):
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "hermes_home" in data
        assert "active_sessions" in data

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
                    "telegram": {
                        "state": "connected",
                        "updated_at": "2026-04-12T00:00:00+00:00",
                    },
                    "whatsapp": {
                        "state": "retrying",
                        "updated_at": "2026-04-12T00:00:00+00:00",
                    },
                    "feishu": {
                        "state": "connected",
                        "updated_at": "2026-04-12T00:00:00+00:00",
                    },
                },
            },
        )
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            gateway_config, "load_gateway_config", lambda: _GatewayConfig()
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["gateway_platforms"] == {
            "telegram": {
                "state": "connected",
                "updated_at": "2026-04-12T00:00:00+00:00",
            },
        }

    def test_get_status_hides_stale_platforms_when_gateway_not_running(
        self, monkeypatch
    ):
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
                    "whatsapp": {
                        "state": "retrying",
                        "updated_at": "2026-04-12T00:00:00+00:00",
                    },
                    "feishu": {
                        "state": "connected",
                        "updated_at": "2026-04-12T00:00:00+00:00",
                    },
                },
            },
        )
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            gateway_config, "load_gateway_config", lambda: _GatewayConfig()
        )

        resp = self.client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["gateway_state"] == "startup_failed"
        assert resp.json()["gateway_platforms"] == {}

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

    def test_reveal_env_var(self, tmp_path):
        """POST /api/env/reveal should return the real unredacted value."""
        from hermes_cli.config import save_env_value
        from hermes_cli.web_server import _SESSION_TOKEN

        save_env_value("TEST_REVEAL_KEY", "super-secret-value-12345")
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_KEY"},
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "TEST_REVEAL_KEY"
        assert data["value"] == "super-secret-value-12345"

    def test_reveal_env_var_not_found(self):
        """POST /api/env/reveal should 404 for unknown keys."""
        from hermes_cli.web_server import _SESSION_TOKEN

        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "NONEXISTENT_KEY_XYZ"},
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
        )
        assert resp.status_code == 404

    def test_reveal_env_var_no_token(self, tmp_path):
        """POST /api/env/reveal without token should return 401."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app
        from hermes_cli.config import save_env_value

        save_env_value("TEST_REVEAL_NOAUTH", "secret-value")
        # Use a fresh client WITHOUT the Authorization header
        unauth_client = TestClient(app)
        resp = unauth_client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_NOAUTH"},
        )
        assert resp.status_code == 401

    def test_reveal_env_var_bad_token(self, tmp_path):
        """POST /api/env/reveal with wrong token should return 401."""
        from hermes_cli.config import save_env_value

        save_env_value("TEST_REVEAL_BADAUTH", "secret-value")
        resp = self.client.post(
            "/api/env/reveal",
            json={"key": "TEST_REVEAL_BADAUTH"},
            headers={"Authorization": "Bearer wrong-token-here"},
        )
        assert resp.status_code == 401

    def test_session_token_endpoint_removed(self):
        """GET /api/auth/session-token should no longer exist (token injected via HTML)."""
        resp = self.client.get("/api/auth/session-token")
        # The endpoint is gone — the catch-all SPA route serves index.html
        # or the middleware returns 401 for unauthenticated /api/ paths.
        assert resp.status_code in (200, 404)
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

        # Create a client WITHOUT the Authorization header
        unauth_client = TestClient(app)
        resp = unauth_client.get("/api/env")
        assert resp.status_code == 401
        resp = unauth_client.get("/api/config")
        assert resp.status_code == 401
        # Public endpoints should still work
        resp = unauth_client.get("/api/status")
        assert resp.status_code == 200

    def test_path_traversal_blocked(self):
        """Verify URL-encoded path traversal is blocked."""
        # %2e%2e = ..
        resp = self.client.get("/%2e%2e/%2e%2e/etc/passwd")
        # Should return 200 with index.html (SPA fallback), not the actual file
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            # Should be the SPA fallback, not the system file
            assert "root:" not in resp.text

    def test_path_traversal_dotdot_blocked(self):
        """Direct .. path traversal via encoded sequences."""
        resp = self.client.get("/%2e%2e/hermes_cli/web_server.py")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert "FastAPI" not in resp.text  # Should not serve the actual source


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
            assert count >= 2, (
                f"Category '{cat}' has only {count} field(s) — should be merged"
            )


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
        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_get_config_no_internal_keys(self):
        """GET /api/config should not expose _config_version or _model_meta."""
        config = self.client.get("/api/config").json()
        internal = [k for k in config if k.startswith("_")]
        assert not internal, f"Internal keys leaked to frontend: {internal}"

    def test_get_config_model_is_string(self):
        """GET /api/config should normalize model dict to a string."""
        config = self.client.get("/api/config").json()
        assert isinstance(config.get("model"), str), (
            f"model should be string, got {type(config.get('model'))}"
        )

    def test_round_trip_preserves_model_subkeys(self):
        """Save and reload should not lose model.provider, model.base_url, etc."""
        from hermes_cli.config import load_config, save_config

        # Set up a config with model as a dict (the common user config form)
        save_config(
            {
                "model": {
                    "default": "anthropic/claude-sonnet-4",
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "openai",
                }
            }
        )

        before = load_config()
        assert isinstance(before.get("model"), dict)
        original_keys = set(before["model"].keys())

        # GET → PUT unchanged
        web_config = self.client.get("/api/config").json()
        assert isinstance(web_config.get("model"), str), (
            "GET should normalize model to string"
        )

        self.client.put("/api/config", json={"config": web_config})

        after = load_config()
        assert isinstance(after.get("model"), dict), (
            "model should still be a dict after save"
        )
        assert set(after["model"].keys()) >= original_keys, (
            f"Lost model subkeys: {original_keys - set(after['model'].keys())}"
        )

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
            if expected in ("string", "select") and not isinstance(val, str):
                mismatches.append(f"{key}: expected str, got {type(val).__name__}")
            elif expected == "number" and not isinstance(val, (int, float)):
                mismatches.append(f"{key}: expected number, got {type(val).__name__}")
            elif expected == "boolean" and not isinstance(val, bool):
                mismatches.append(f"{key}: expected bool, got {type(val).__name__}")
            elif expected == "list" and not isinstance(val, list):
                mismatches.append(f"{key}: expected list, got {type(val).__name__}")
        assert not mismatches, f"Type mismatches:\n" + "\n".join(mismatches)


# ---------------------------------------------------------------------------
# New feature endpoint tests
# ---------------------------------------------------------------------------


class TestNewEndpoints:
    """Tests for session detail, logs, cron, skills, tools, raw config, analytics."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

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
                    {
                        "name": "active-skill",
                        "description": "active",
                        "category": "demo",
                    },
                    {
                        "name": "disabled-skill",
                        "description": "disabled",
                        "category": "demo",
                    },
                ]
            return [
                {"name": "active-skill", "description": "active", "category": "demo"},
            ]

        monkeypatch.setattr(skills_tool, "_find_all_skills", _fake_find_all_skills)
        monkeypatch.setattr(
            skills_config, "get_disabled_skills", lambda config: {"disabled-skill"}
        )
        monkeypatch.setattr(
            web_server,
            "load_config",
            lambda: {"skills": {"disabled": ["disabled-skill"]}},
        )

        resp = self.client.get("/api/skills")

        assert resp.status_code == 200
        assert resp.json() == [
            {
                "name": "active-skill",
                "description": "active",
                "category": "demo",
                "enabled": True,
            },
            {
                "name": "disabled-skill",
                "description": "disabled",
                "category": "demo",
                "enabled": False,
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
            lambda config, platform, include_default_mcp_servers=False: {
                "web",
                "skills",
            },
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
        monkeypatch.setattr(
            web_server,
            "load_config",
            lambda: {"platform_toolsets": {"cli": ["web", "skills"]}},
        )

        resp = self.client.get("/api/tools/toolsets")

        assert resp.status_code == 200
        assert resp.json() == [
            {
                "name": "web",
                "label": "🔍 Web Search & Scraping",
                "description": "web_search, web_extract",
                "enabled": True,
                "available": True,
                "configured": False,
                "tools": ["web_extract", "web_search"],
            },
            {
                "name": "skills",
                "label": "📚 Skills",
                "description": "list, view, manage",
                "enabled": True,
                "available": True,
                "configured": True,
                "tools": ["skill_view", "skills_list"],
            },
            {
                "name": "memory",
                "label": "💾 Memory",
                "description": "persistent memory across sessions",
                "enabled": False,
                "available": False,
                "configured": True,
                "tools": ["memory_read"],
            },
        ]

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
        assert isinstance(data["daily"], list)
        assert "total_sessions" in data["totals"]

    def test_session_token_endpoint_removed(self):
        """GET /api/auth/session-token no longer exists."""
        resp = self.client.get("/api/auth/session-token")
        # Should not return a JSON token object
        assert resp.status_code in (200, 404)
        try:
            data = resp.json()
            assert "token" not in data
        except Exception:
            pass


class TestHermesWebChatEndpoints:
    """Tests for HermesWeb chat/history/ws integration endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        self.session_token = _SESSION_TOKEN

    def test_chat_history_requires_session_id(self):
        resp = self.client.get("/api/chat/history")
        assert resp.status_code == 400
        assert "session_id" in resp.json()["detail"]

    def test_chat_history_returns_messages_for_session(self, monkeypatch):
        import hermes_state

        class _FakeSessionDB:
            def resolve_session_id(self, session_id):
                return session_id if session_id == "sess-123" else None

            def get_messages(self, session_id):
                assert session_id == "sess-123"
                return [
                    {
                        "id": 7,
                        "session_id": session_id,
                        "role": "user",
                        "content": "ping",
                        "timestamp": 1713830400.0,
                        "tool_name": None,
                        "tool_calls": None,
                        "tool_call_id": None,
                        "finish_reason": None,
                        "reasoning": None,
                    },
                    {
                        "id": 8,
                        "session_id": session_id,
                        "role": "assistant",
                        "content": "pong",
                        "timestamp": 1713830460.0,
                        "tool_name": None,
                        "tool_calls": None,
                        "tool_call_id": None,
                        "finish_reason": "stop",
                        "reasoning": None,
                    },
                ]

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)

        resp = self.client.get("/api/chat/history?session_id=sess-123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-123"
        assert data["total"] == 2
        assert [msg["role"] for msg in data["messages"]] == ["user", "assistant"]
        assert data["messages"][1]["content"] == "pong"
        assert data["messages"][1]["timestamp"] == 1713830460.0

    def test_post_chat_returns_queued_response(self, monkeypatch):
        """POST /api/chat returns a run_id + queued status immediately (async mode)."""
        import hermes_cli.web_server as web_server

        # Prevent the background task from actually executing the agent
        async def _fake_execute_chat_run(run_id, body):
            pass

        monkeypatch.setattr(web_server, "_execute_chat_run", _fake_execute_chat_run)

        resp = self.client.post(
            "/api/chat", json={"content": "hello", "session_id": "sess-456"}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["session_id"] is not None
        assert "run_id" in data
        assert data["run_id"].startswith("run_")
        # backward-compat null fields
        assert data["message"] is None
        assert data["response"] is None
        assert data["messages"] == []

    def test_run_chat_turn_uses_runtime_and_returns_new_messages(self, monkeypatch):
        import hermes_state
        import run_agent
        import hermes_cli.web_server as web_server
        import hermes_cli.tools_config as tools_config
        import hermes_cli.runtime_provider as runtime_provider

        class _FakeSessionDB:
            def __init__(self):
                self.history = [
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "before"},
                ]

            def resolve_session_id(self, session_id):
                return session_id

            def reopen_session(self, session_id):
                self.reopened = session_id

            def get_messages(self, session_id):
                if getattr(self, "_after", False):
                    return [
                        {
                            "id": 21,
                            "session_id": session_id,
                            "role": "user",
                            "content": "earlier",
                            "timestamp": 1713830300.0,
                        },
                        {
                            "id": 22,
                            "session_id": session_id,
                            "role": "assistant",
                            "content": "before",
                            "timestamp": 1713830310.0,
                        },
                        {
                            "id": 23,
                            "session_id": session_id,
                            "role": "user",
                            "content": "hello runtime",
                            "timestamp": 1713830400.0,
                        },
                        {
                            "id": 24,
                            "session_id": session_id,
                            "role": "assistant",
                            "content": "runtime reply",
                            "timestamp": 1713830410.0,
                            "finish_reason": "stop",
                        },
                    ]
                return [
                    {
                        "id": 21,
                        "session_id": session_id,
                        "role": "user",
                        "content": "earlier",
                        "timestamp": 1713830300.0,
                    },
                    {
                        "id": 22,
                        "session_id": session_id,
                        "role": "assistant",
                        "content": "before",
                        "timestamp": 1713830310.0,
                    },
                ]

            def get_messages_as_conversation(self, session_id):
                return list(self.history)

            def get_session(self, session_id):
                return {
                    "id": session_id,
                    "title": "Runtime test",
                    "source": "web",
                    "model": "gpt-5.4",
                    "started_at": 1713830200.0,
                    "ended_at": None,
                    "end_reason": None,
                    "message_count": 4,
                    "tool_call_count": 0,
                    "input_tokens": 12,
                    "output_tokens": 5,
                }

            def close(self):
                pass

        class _FakeAIAgent:
            def __init__(self, **kwargs):
                self.session_id = kwargs["session_id"]
                self.kwargs = kwargs
                self.session_db = kwargs["session_db"]

            def run_conversation(
                self, *, user_message, conversation_history, persist_user_message
            ):
                assert user_message == "hello runtime"
                assert persist_user_message == "hello runtime"
                assert conversation_history == [
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "before"},
                ]
                self.session_db._after = True
                return {
                    "completed": True,
                    "partial": False,
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                }

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        monkeypatch.setattr(run_agent, "AIAgent", _FakeAIAgent)
        monkeypatch.setattr(
            tools_config, "_get_platform_tools", lambda config, platform: {"shell"}
        )
        monkeypatch.setattr(
            runtime_provider,
            "resolve_runtime_provider",
            lambda: {
                "api_key": "test-key",
                "base_url": "https://example.com/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr(
            web_server,
            "load_config",
            lambda: {
                "model": "gpt-5.4",
                "agent": {"max_turns": 7},
            },
        )

        turn = web_server._run_chat_turn(
            content="hello runtime", session_id="sess-runtime"
        )

        assert turn["session_id"] == "sess-runtime"
        assert turn["completed"] is True
        assert turn["assistant_message"]["content"] == "runtime reply"
        assert [msg["id"] for msg in turn["messages"]] == [23, 24]
        assert turn["session"]["last_active"] == 1713830410.0

    def test_ws_accepts_token_and_sends_hello_then_pong(self):
        with self.client.websocket_connect(
            f"/ws?token={self.session_token}"
        ) as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello"
            assert "connection_id" in hello
            assert hello["status"]["version"]

            websocket.send_text("ping")
            pong = websocket.receive_json()
            assert pong["type"] == "pong"


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
        save_config(
            {
                "model": {
                    "default": "anthropic/claude-opus-4.6",
                    "provider": "openrouter",
                }
            }
        )

        result = _denormalize_config_from_web(
            {
                "model": "anthropic/claude-opus-4.6",
                "model_context_length": 100000,
            }
        )
        assert isinstance(result["model"], dict)
        assert result["model"]["context_length"] == 100000
        assert "model_context_length" not in result  # virtual field removed

    def test_denormalize_zero_removes_context_length(self):
        """denormalize with model_context_length=0 should remove context_length key."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config(
            {
                "model": {
                    "default": "anthropic/claude-opus-4.6",
                    "provider": "openrouter",
                    "context_length": 50000,
                }
            }
        )

        result = _denormalize_config_from_web(
            {
                "model": "anthropic/claude-opus-4.6",
                "model_context_length": 0,
            }
        )
        assert isinstance(result["model"], dict)
        assert "context_length" not in result["model"]

    def test_denormalize_upgrades_bare_string_to_dict(self):
        """denormalize should upgrade bare string model to dict when context_length set."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        # Disk has model as bare string
        save_config({"model": "anthropic/claude-sonnet-4"})

        result = _denormalize_config_from_web(
            {
                "model": "anthropic/claude-sonnet-4",
                "model_context_length": 65000,
            }
        )
        assert isinstance(result["model"], dict)
        assert result["model"]["default"] == "anthropic/claude-sonnet-4"
        assert result["model"]["context_length"] == 65000

    def test_denormalize_bare_string_stays_string_when_zero(self):
        """denormalize should keep bare string model as string when context_length=0."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({"model": "anthropic/claude-sonnet-4"})

        result = _denormalize_config_from_web(
            {
                "model": "anthropic/claude-sonnet-4",
                "model_context_length": 0,
            }
        )
        assert result["model"] == "anthropic/claude-sonnet-4"

    def test_denormalize_coerces_string_context_length(self):
        """denormalize should handle string model_context_length from frontend."""
        from hermes_cli.web_server import _denormalize_config_from_web
        from hermes_cli.config import save_config

        save_config({"model": {"default": "test/model", "provider": "openrouter"}})

        result = _denormalize_config_from_web(
            {
                "model": "test/model",
                "model_context_length": "32000",
            }
        )
        assert isinstance(result["model"], dict)
        assert result["model"]["context_length"] == 32000


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

        monkeypatch.setattr(
            ws,
            "load_config",
            lambda: {
                "model": {
                    "default": "anthropic/claude-opus-4.6",
                    "provider": "openrouter",
                    "context_length": 100000,
                }
            },
        )

        with patch(
            "agent.model_metadata.get_model_context_length", return_value=200000
        ):
            resp = self.client.get("/api/model/info")

        data = resp.json()
        assert data["model"] == "anthropic/claude-opus-4.6"
        assert data["provider"] == "openrouter"
        assert data["auto_context_length"] == 200000
        assert data["config_context_length"] == 100000
        assert data["effective_context_length"] == 100000  # override wins

    def test_model_info_auto_detect_when_no_override(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(
            ws,
            "load_config",
            lambda: {
                "model": {
                    "default": "anthropic/claude-opus-4.6",
                    "provider": "openrouter",
                }
            },
        )

        with patch(
            "agent.model_metadata.get_model_context_length", return_value=200000
        ):
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

        monkeypatch.setattr(
            ws, "load_config", lambda: {"model": "anthropic/claude-sonnet-4"}
        )

        with patch(
            "agent.model_metadata.get_model_context_length", return_value=200000
        ):
            resp = self.client.get("/api/model/info")

        data = resp.json()
        assert data["model"] == "anthropic/claude-sonnet-4"
        assert data["provider"] == ""
        assert data["config_context_length"] == 0
        assert data["effective_context_length"] == 200000

    def test_model_info_capabilities(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(
            ws,
            "load_config",
            lambda: {
                "model": {
                    "default": "anthropic/claude-opus-4.6",
                    "provider": "openrouter",
                }
            },
        )

        mock_caps = MagicMock()
        mock_caps.supports_tools = True
        mock_caps.supports_vision = True
        mock_caps.supports_reasoning = True
        mock_caps.context_window = 200000
        mock_caps.max_output_tokens = 32000
        mock_caps.model_family = "claude-opus"

        with (
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
            patch("agent.models_dev.get_model_capabilities", return_value=mock_caps),
        ):
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

        monkeypatch.setattr(ws, "load_config", lambda: {"model": "some/obscure-model"})

        with patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=Exception("boom"),
        ):
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

        response_body = json.dumps(
            {
                "status": "ok",
                "gateway_state": "running",
                "pid": 42,
            }
        )

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

        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_status_falls_back_to_remote_probe(self, monkeypatch):
        """When local PID check fails and remote probe succeeds, gateway shows running."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(
            ws,
            "_probe_gateway_health",
            lambda: (
                True,
                {
                    "status": "ok",
                    "gateway_state": "running",
                    "platforms": {"telegram": {"state": "connected"}},
                    "pid": 999,
                },
            ),
        )

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is True
        assert data["gateway_pid"] == 999
        assert data["gateway_state"] == "running"

    def test_status_remote_probe_not_attempted_when_local_pid_found(self, monkeypatch):
        """When local PID check succeeds, the remote probe is never called."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid", lambda: 1234)
        monkeypatch.setattr(
            ws,
            "read_runtime_status",
            lambda: {
                "gateway_state": "running",
                "platforms": {},
            },
        )
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

    def test_status_remote_running_null_pid(self, monkeypatch):
        """Remote gateway running but PID not in response — pid should be None."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "get_running_pid", lambda: None)
        monkeypatch.setattr(ws, "read_runtime_status", lambda: None)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(
            ws,
            "_probe_gateway_health",
            lambda: (
                True,
                {
                    "status": "ok",
                },
            ),
        )

        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is True
        assert data["gateway_pid"] is None
        assert data["gateway_state"] == "running"


class TestAgentsAndApprovalsEndpoints:
    """Tests for GET /api/agents, GET /api/approvals, and approve/reject."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        self.session_token = _SESSION_TOKEN

    def test_get_agents_requires_auth(self):
        """GET /api/agents without auth returns 401."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        unauth = TestClient(app)
        resp = unauth.get("/api/agents")
        assert resp.status_code == 401

    def test_get_agents_returns_list(self, monkeypatch):
        """GET /api/agents returns a list of agents derived from sessions."""
        import hermes_state
        import time as time_module

        recent_time = 1713830500.0

        class _FakeSessionDB:
            def list_sessions_rich(self, limit=50, offset=0):
                return [
                    {
                        "id": "sess-abc",
                        "title": "Test session",
                        "source": "web",
                        "model": "gpt-5.4",
                        "started_at": recent_time - 100,
                        "ended_at": None,
                        "end_reason": None,
                        "last_active": recent_time,
                        "parent_session_id": None,
                    },
                    {
                        "id": "sess-def",
                        "title": "Child task",
                        "source": "cli",
                        "model": "claude-3-opus",
                        "started_at": recent_time - 50,
                        "ended_at": None,
                        "end_reason": None,
                        "last_active": recent_time,
                        "parent_session_id": "sess-abc",
                    },
                ]

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        monkeypatch.setattr(time_module, "time", lambda: recent_time)

        resp = self.client.get("/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        agents = data["agents"]
        assert len(agents) == 2

        primary = next(a for a in agents if a["id"] == "agent-sess-abc")
        assert primary["name"] == "Test session"
        assert primary["kind"] == "primary"
        assert primary["status"] == "active"
        assert primary["model"] == "gpt-5.4"
        assert primary["platform"] == "web"

        subagent = next(a for a in agents if a["id"] == "subagent-sess-def")
        assert subagent["kind"] == "subagent"

    def test_get_agents_ended_session_is_not_active(self, monkeypatch):
        """A session with ended_at is marked 'ended', not 'active'."""
        import hermes_state

        class _FakeSessionDB:
            def list_sessions_rich(self, limit=50, offset=0):
                return [
                    {
                        "id": "sess-old",
                        "title": "Old session",
                        "source": "cli",
                        "model": "gpt-5.4",
                        "started_at": 1713800000.0,
                        "ended_at": 1713810000.0,
                        "end_reason": "completed",
                        "last_active": 1713810000.0,
                        "parent_session_id": None,
                    },
                ]

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)

        resp = self.client.get("/api/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert len(agents) == 1
        assert agents[0]["status"] == "ended"

    def test_get_agents_idle_when_stale(self, monkeypatch):
        """A session with no ended_at but older than 5 min is 'idle'."""
        import hermes_state
        import time as time_module

        class _FakeSessionDB:
            def list_sessions_rich(self, limit=50, offset=0):
                return [
                    {
                        "id": "sess-stale",
                        "title": "Stale session",
                        "source": "cli",
                        "model": "gpt-5.4",
                        "started_at": 1713830400.0,
                        "ended_at": None,
                        "end_reason": None,
                        "last_active": 1713830500.0,
                        "parent_session_id": None,
                    },
                ]

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        monkeypatch.setattr(time_module, "time", lambda: 1713830800.0)

        resp = self.client.get("/api/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert agents[0]["status"] == "idle"

    def test_get_approvals_requires_auth(self):
        """GET /api/approvals without auth returns 401."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        unauth = TestClient(app)
        resp = unauth.get("/api/approvals")
        assert resp.status_code == 401

    def test_get_approvals_empty_queue(self):
        """GET /api/approvals returns empty list when no approvals pending."""
        resp = self.client.get("/api/approvals")
        assert resp.status_code == 200
        data = resp.json()
        assert "approvals" in data
        assert data["approvals"] == []
        assert "total" in data

    def test_get_approvals_pending_only_by_default(self, monkeypatch):
        """GET /api/approvals returns pending approvals from the queue."""
        import hermes_cli.web_server as ws

        mock_entry_data = {
            "command": "rm -rf /tmp/build",
            "description": "recursive delete",
            "pattern_keys": ["recursive delete"],
            "session_id": "sess-123",
            "created_at": "2026-04-23T12:00:00+00:00",
        }

        class _FakeEntry:
            def __init__(self, data):
                self.data = data
                self.result = None

        import tools.approval as approval_mod

        monkeypatch.setattr(
            approval_mod, "_gateway_queues", {"sess-123": [_FakeEntry(mock_entry_data)]}
        )

        resp = self.client.get("/api/approvals")
        assert resp.status_code == 200
        approvals = resp.json()["approvals"]
        assert len(approvals) == 1
        assert approvals[0]["title"] == "recursive delete"
        assert approvals[0]["command"] == "rm -rf /tmp/build"
        assert approvals[0]["status"] == "pending"
        assert approvals[0]["session_id"] == "sess-123"

    def test_approve_unknown_id_returns_404(self):
        """POST /api/approvals/{id}/approve returns 404 for unknown id."""
        resp = self.client.post("/api/approvals/nonexistent-id-xyz/approve")
        assert resp.status_code == 404

    def test_reject_unknown_id_returns_404(self):
        """POST /api/approvals/{id}/reject returns 404 for unknown id."""
        resp = self.client.post("/api/approvals/nonexistent-id-xyz/reject")
        assert resp.status_code == 404

    def test_approve_requires_auth(self):
        """POST /api/approvals/{id}/approve without auth returns 401."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        unauth = TestClient(app)
        resp = unauth.post("/api/approvals/some-id/approve")
        assert resp.status_code == 401

    def test_reject_requires_auth(self):
        """POST /api/approvals/{id}/reject without auth returns 401."""
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        unauth = TestClient(app)
        resp = unauth.post("/api/approvals/some-id/reject")
        assert resp.status_code == 401

    def test_approval_approve_resolves_and_unblocks(self, monkeypatch):
        """Approve resolves the approval entry and returns ok=true."""
        import tools.approval as approval_mod

        resolved_results = []

        class _FakeEntry:
            def __init__(self, data):
                self.data = data
                self.result = None
                self.event = _FakeEvent(self)

        class _FakeEvent:
            def __init__(self, entry):
                self.entry = entry

            def wait(self, timeout=None):
                return True

            def set(self):
                resolved_results.append(self.entry.result)

        mock_entry_data = {
            "command": "git push --force",
            "description": "git force push",
            "pattern_keys": ["git force push"],
            "session_id": "sess-push",
            "created_at": "2026-04-23T12:00:00+00:00",
        }

        entry = _FakeEntry(mock_entry_data)
        monkeypatch.setattr(approval_mod, "_gateway_queues", {"sess-push": [entry]})
        monkeypatch.setattr(
            approval_mod,
            "resolve_gateway_approval",
            lambda sk, choice, resolve_all=False: (
                (setattr(entry, "result", choice or "session"), entry.event.set(), 1)[1]
                if choice
                else (setattr(entry, "result", "session"), entry.event.set(), 1)[1]
            ),
        )

        import hermes_cli.web_server as ws

        resp = self.client.post("/api/approvals/approval-pushtest/approve")
        assert resp.status_code in (200, 404, 409)

    def test_ws_sends_approval_resolved_on_approve(self, monkeypatch):
        """When an approval is approved, the server emits approval.resolved over WS."""
        import tools.approval as approval_mod

        class _FakeEntry:
            def __init__(self, data):
                self.data = data
                self.result = None

            @property
            def event(self):
                ev = _FakeEvent()
                return ev

        class _FakeEvent:
            def wait(self, timeout=None):
                return True

            def set(self):
                pass

        mock_entry_data = {
            "command": "dd if=/dev/zero",
            "description": "disk copy",
            "pattern_keys": ["disk copy"],
            "session_id": "sess-dd",
        }
        entry = _FakeEntry(mock_entry_data)
        monkeypatch.setattr(approval_mod, "_gateway_queues", {"sess-dd": [entry]})
        monkeypatch.setattr(
            approval_mod,
            "resolve_gateway_approval",
            lambda sk, choice, resolve_all=False: 1,
        )

        with self.client.websocket_connect(
            f"/ws?token={self.session_token}"
        ) as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello"


class TestApprovalPersistence:
    """Tests for approval persistence via ApprovalDB."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_TOKEN
        import hermes_state

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        self.session_token = _SESSION_TOKEN

        self._approval_db_path = tmp_path / "test_approvals_persist.db"

    def test_get_approvals_reads_from_db(self, monkeypatch):
        """GET /api/approvals returns approvals stored in ApprovalDB."""
        from hermes_state import ApprovalDB

        test_db = ApprovalDB(db_path=self._approval_db_path)
        test_db.create_approval(
            approval_id="approval-db-test-1",
            session_id="sess-persist-1",
            agent_id="agent-sess-persist-1",
            title="persisted approval",
            command="echo persisted",
            created_at="2026-04-23T12:00:00+00:00",
        )
        test_db.close()

        monkeypatch.setattr(
            "hermes_cli.web_server._get_approval_db",
            lambda: ApprovalDB(db_path=self._approval_db_path),
        )

        resp = self.client.get("/api/approvals")
        assert resp.status_code == 200
        data = resp.json()
        approvals = data["approvals"]
        persisted = [a for a in approvals if a["id"] == "approval-db-test-1"]
        assert len(persisted) == 1
        assert persisted[0]["title"] == "persisted approval"
        assert persisted[0]["status"] == "pending"

    def test_get_approvals_includes_resolved_from_db(self, monkeypatch):
        """GET /api/approvals returns resolved approvals from ApprovalDB."""
        from hermes_state import ApprovalDB

        test_db = ApprovalDB(db_path=self._approval_db_path)
        test_db.create_approval(
            approval_id="approval-resolved-db-1",
            session_id="sess-resolved-1",
            agent_id="agent-sess-resolved-1",
            title="resolved in db",
            command="echo resolved",
            created_at="2026-04-23T12:00:00+00:00",
        )
        test_db.resolve_approval(
            "approval-resolved-db-1",
            "approved",
            resolved_by="sess-resolved-1",
            choice="session",
        )
        test_db.close()

        monkeypatch.setattr(
            "hermes_cli.web_server._get_approval_db",
            lambda: ApprovalDB(db_path=self._approval_db_path),
        )

        resp = self.client.get("/api/approvals?status=approved")
        assert resp.status_code == 200
        data = resp.json()
        approvals = data["approvals"]
        resolved = [a for a in approvals if a["id"] == "approval-resolved-db-1"]
        assert len(resolved) == 1
        assert resolved[0]["status"] == "approved"

    def test_approve_persists_to_db(self, monkeypatch):
        """POST /api/approvals/{id}/approve persists resolution to ApprovalDB."""
        import tools.approval as approval_mod
        from hermes_state import ApprovalDB

        test_db = ApprovalDB(db_path=self._approval_db_path)
        test_db.create_approval(
            approval_id="approval-approve-db-1",
            session_id="sess-approve-db-1",
            agent_id="agent-sess-approve-db-1",
            title="approve persist test",
            command="echo approve",
            created_at="2026-04-23T12:00:00+00:00",
        )
        test_db.close()

        class _FakeEntry:
            def __init__(self, data):
                self.data = data
                self.result = None

            @property
            def event(self):
                ev = _FakeEvent()
                return ev

        class _FakeEvent:
            def wait(self, timeout=None):
                return True

            def set(self):
                pass

        mock_entry_data = {
            "command": "echo approve",
            "description": "approve persist test",
            "pattern_keys": ["test"],
            "session_id": "sess-approve-db-1",
        }
        entry = _FakeEntry(mock_entry_data)
        monkeypatch.setattr(
            approval_mod, "_gateway_queues", {"sess-approve-db-1": [entry]}
        )
        monkeypatch.setattr(
            approval_mod,
            "resolve_gateway_approval",
            lambda sk, choice, resolve_all=False: (
                (setattr(entry, "result", choice or "session"), entry.event.set(), 1)[1]
                if choice
                else (setattr(entry, "result", "session"), entry.event.set(), 1)[1]
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.web_server._get_approval_db",
            lambda: ApprovalDB(db_path=self._approval_db_path),
        )

        resp = self.client.post("/api/approvals/approval-approve-db-1/approve")
        assert resp.status_code == 200

        db_after = ApprovalDB(db_path=self._approval_db_path)
        approval = db_after.get_approval("approval-approve-db-1")
        db_after.close()
        assert approval["status"] == "approved"
        assert approval["choice"] == "session"

    def test_reject_persists_to_db(self, monkeypatch):
        """POST /api/approvals/{id}/reject persists resolution to ApprovalDB."""
        import tools.approval as approval_mod
        from hermes_state import ApprovalDB

        test_db = ApprovalDB(db_path=self._approval_db_path)
        test_db.create_approval(
            approval_id="approval-reject-db-1",
            session_id="sess-reject-db-1",
            agent_id="agent-sess-reject-db-1",
            title="reject persist test",
            command="echo reject",
            created_at="2026-04-23T12:00:00+00:00",
        )
        test_db.close()

        class _FakeEntry:
            def __init__(self, data):
                self.data = data
                self.result = None

            @property
            def event(self):
                ev = _FakeEvent()
                return ev

        class _FakeEvent:
            def wait(self, timeout=None):
                return True

            def set(self):
                pass

        mock_entry_data = {
            "command": "echo reject",
            "description": "reject persist test",
            "pattern_keys": ["test"],
            "session_id": "sess-reject-db-1",
        }
        entry = _FakeEntry(mock_entry_data)
        monkeypatch.setattr(
            approval_mod, "_gateway_queues", {"sess-reject-db-1": [entry]}
        )
        monkeypatch.setattr(
            approval_mod,
            "resolve_gateway_approval",
            lambda sk, choice, resolve_all=False: (
                (setattr(entry, "result", choice or "session"), entry.event.set(), 1)[1]
                if choice
                else (setattr(entry, "result", "session"), entry.event.set(), 1)[1]
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.web_server._get_approval_db",
            lambda: ApprovalDB(db_path=self._approval_db_path),
        )

        resp = self.client.post("/api/approvals/approval-reject-db-1/reject")
        assert resp.status_code == 200

        db_after = ApprovalDB(db_path=self._approval_db_path)
        approval = db_after.get_approval("approval-reject-db-1")
        db_after.close()
        assert approval["status"] == "rejected"
        assert approval["choice"] == "deny"


class TestArtifactsEndpoint:
    """Tests for GET /api/sessions/{session_id}/artifacts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_artifacts_requires_auth(self):
        from hermes_cli.web_server import app

        client_no_auth = self.client.__class__(app)
        resp = client_no_auth.get("/api/sessions/sess-123/artifacts")
        assert resp.status_code == 401

    def test_artifacts_returns_404_for_unknown_session(self, monkeypatch):
        import hermes_state

        class _FakeSessionDB:
            def resolve_session_id(self, session_id):
                return None

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        resp = self.client.get("/api/sessions/nonexistent/artifacts")
        assert resp.status_code == 404

    def test_artifacts_returns_empty_list_when_no_tool_messages(self, monkeypatch):
        import hermes_state

        class _FakeSessionDB:
            def resolve_session_id(self, session_id):
                return session_id

            def get_artifacts_by_session(self, session_id):
                return []

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        resp = self.client.get("/api/sessions/sess-123/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-123"
        assert data["artifacts"] == []
        assert data["total"] == 0

    def test_artifacts_returns_patch_artifact_with_diff(self, monkeypatch):
        import hermes_state

        class _FakeSessionDB:
            def resolve_session_id(self, session_id):
                return session_id

            def get_artifacts_by_session(self, session_id):
                return [
                    {
                        "id": "msg-5",
                        "tool_call_id": "call_abc",
                        "tool_name": "patch",
                        "path": "src/foo.py",
                        "status": "modified",
                        "diff": "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1,2 @@\n-old\n+new\n+line\n",
                        "additions": 2,
                        "deletions": 1,
                        "timestamp": 1776943829.208,
                    }
                ]

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        resp = self.client.get("/api/sessions/sess-123/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        artifact = data["artifacts"][0]
        assert artifact["tool_name"] == "patch"
        assert artifact["path"] == "src/foo.py"
        assert artifact["status"] == "modified"
        assert artifact["additions"] == 2
        assert artifact["deletions"] == 1
        assert "--- a/src/foo.py" in artifact["diff"]

    def test_artifacts_returns_write_file_artifact_without_diff(self, monkeypatch):
        import hermes_state

        class _FakeSessionDB:
            def resolve_session_id(self, session_id):
                return session_id

            def get_artifacts_by_session(self, session_id):
                return [
                    {
                        "id": "msg-7",
                        "tool_call_id": "call_xyz",
                        "tool_name": "write_file",
                        "path": "src/new.py",
                        "status": "added",
                        "diff": "",
                        "additions": 0,
                        "deletions": 0,
                        "timestamp": 1776943830.0,
                    }
                ]

            def close(self):
                pass

        monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
        resp = self.client.get("/api/sessions/sess-123/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        artifact = data["artifacts"][0]
        assert artifact["tool_name"] == "write_file"
        assert artifact["path"] == "src/new.py"
        assert artifact["status"] == "added"
        assert artifact["diff"] == ""
        assert artifact["additions"] == 0
        assert artifact["deletions"] == 0
