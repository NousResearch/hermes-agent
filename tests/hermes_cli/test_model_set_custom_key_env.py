"""POST /api/model/set for custom endpoints must persist key_env, never the secret (#57547).

Kept in its own module so the key_env regression coverage lives in a bounded
owner instead of growing tests/hermes_cli/test_web_server.py.
"""

import pytest


class TestModelSetCustomKeyEnv:
    """Direct custom-model route: secret goes to .env, config.yaml holds only key_env."""

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

    def test_set_model_main_custom_persists_key_env_and_registers_provider(self):
        """The direct custom-model route must keep its key out of config.yaml."""
        from hermes_cli.config import (
            custom_endpoint_key_env,
            get_env_value,
            load_config,
        )

        key_env = custom_endpoint_key_env("https://text.example.com/v1")

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
        assert model_cfg["key_env"] == key_env
        assert "api_key" not in model_cfg
        assert get_env_value(key_env) == "sk-secret"

        # Registered in custom_providers (dedup by base_url) so the picker shows
        # a proper ready row instead of the "needs setup" dead-end.
        custom = cfg.get("custom_providers") or []
        assert any(
            isinstance(e, dict)
            and e.get("base_url") == "https://text.example.com/v1"
            and e.get("key_env") == key_env
            and not e.get("api_key")
            and e.get("model") == "gpt-oss-120b"
            for e in custom
        )

    def test_set_model_main_named_custom_keeps_secret_out_of_config(self):
        """The durable custom:<name> namespace has the same storage boundary."""
        from hermes_cli.config import (
            custom_endpoint_key_env,
            get_config_path,
            get_env_path,
            get_env_value,
            read_raw_config,
        )

        base_url = "https://litellm.example.com/v1"
        secret = "sk-named-custom-secret"
        response = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom:litellm",
                "model": "ollama/glm-5.2",
                "base_url": base_url,
                "api_key": secret,
                "confirm_expensive_model": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert secret not in get_config_path().read_text(encoding="utf-8")

        model_cfg = read_raw_config()["model"]
        key_env = custom_endpoint_key_env(base_url)
        assert model_cfg["provider"] == "custom:litellm"
        assert model_cfg["base_url"] == base_url
        assert model_cfg["key_env"] == key_env
        assert "api_key" not in model_cfg
        assert get_env_value(key_env) == secret
        assert secret in get_env_path().read_text(encoding="utf-8")

    def test_set_model_main_custom_endpoint_change_drops_stale_key_env(self):
        """Changing hosts without a new key must not reuse the old host's key."""
        from hermes_cli.config import load_config

        first = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom",
                "model": "model-a",
                "base_url": "https://a.example.com/v1",
                "api_key": "sk-a-secret",
            },
        )
        assert first.status_code == 200

        second = self.client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom",
                "model": "model-b",
                "base_url": "https://b.example.com/v1",
            },
        )
        assert second.status_code == 200

        model_cfg = load_config()["model"]
        assert model_cfg["base_url"] == "https://b.example.com/v1"
        assert "key_env" not in model_cfg
        assert "api_key" not in model_cfg
