"""Custom Endpoints credential references across the endpoint lifecycle.

``api_key_env`` is the documented alias of ``key_env`` and resolves identically at runtime, so
every dashboard path that writes, clears, mirrors or displays an endpoint credential must treat
the two spellings alike. Moved out of ``test_web_server.py``.
"""

import pytest


class TestCustomEndpointCredentialReferences:
    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
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

    def _listed(self, endpoint_id):
        resp = self.client.get("/api/providers/custom-endpoints")
        assert resp.status_code == 200, resp.text
        return next(e for e in resp.json()["endpoints"] if e["id"] == endpoint_id)

    @pytest.mark.parametrize("clear_key", [False, True])
    @pytest.mark.parametrize("credential_field", ["key_env", "api_key_env"])
    @pytest.mark.parametrize("shared", [False, True])
    def test_custom_endpoint_edit_preserves_or_clears_api_key_env(self, clear_key, credential_field, shared):
        """Real edits preserve omitted references; blank removes pointers, not shared secrets."""
        from hermes_cli.config import (
            custom_endpoint_key_env, get_compatible_custom_providers, load_config,
            load_env, read_raw_config, save_config, save_env_value,
        )
        from hermes_cli.runtime_provider_custom import _get_named_custom_provider

        env_var = "SHARED_ENDPOINT_KEY" if shared else custom_endpoint_key_env("local")
        save_env_value(env_var, "shared-placeholder")
        cfg = load_config()
        cfg["providers"] = {
            "local": {
                "name": "Local", "base_url": "http://127.0.0.1:8000/v1",
                "model": "local-model", credential_field: env_var,
            },
        }
        if clear_key:
            # All credential spellings may coexist in a hand-written record.
            cfg["providers"]["local"].update(api_key="inline-placeholder", key_env=env_var, api_key_env=env_var)
        save_config(cfg)
        assert _get_named_custom_provider("custom:local")["api_key"] == "shared-placeholder"

        payload = {
            "id": "local", "name": "Local edited",
            "base_url": "http://127.0.0.1:8000/v1", "model": "local-model",
        }
        if clear_key:
            payload["api_key"] = ""
        response = self.client.post("/api/providers/custom-endpoints", json=payload)
        assert response.status_code == 200, response.text

        raw_entry = read_raw_config()["providers"]["local"]
        reloaded = load_config()
        entry = reloaded["providers"]["local"]
        normalized = next(e for e in get_compatible_custom_providers(reloaded) if e.get("provider_key") == "local")
        runtime = _get_named_custom_provider("custom:local")
        assert raw_entry["name"] == entry["name"] == runtime["name"] == "Local edited"
        assert "local-model" in raw_entry["models"]
        if clear_key:
            for persisted in (raw_entry, entry, normalized):
                assert not {"api_key", "key_env", "api_key_env"}.intersection(persisted)
            assert runtime["api_key"] == ""
            assert self._listed("local")["has_api_key"] is False
        else:
            assert raw_entry[credential_field] == env_var
            assert normalized["key_env"] == env_var
            assert runtime["key_env"] == env_var
            assert runtime["api_key"] == "shared-placeholder"
        if clear_key and not shared:
            assert env_var not in load_env()
        else:
            assert load_env()[env_var] == "shared-placeholder"

    @pytest.mark.parametrize("credential_field", ["key_env", "api_key_env"])
    def test_submitting_a_key_replaces_a_hand_written_credential_reference(self, credential_field):
        """A pasted key supersedes every earlier credential spelling.

        The entry must point at the dedicated ``.env`` slot alone: an alias left behind
        would name a second credential, and the shared variable it named stays for its
        other consumers.
        """
        from hermes_cli.config import custom_endpoint_key_env, load_config, load_env, read_raw_config, save_config, save_env_value
        from hermes_cli.runtime_provider_custom import _get_named_custom_provider

        save_env_value("SHARED_ENDPOINT_KEY", "shared-placeholder")
        cfg = load_config()
        cfg["providers"] = {"local": {
            "name": "Local", "base_url": "http://127.0.0.1:8000/v1", "model": "local-model",
            credential_field: "SHARED_ENDPOINT_KEY",
        }}
        save_config(cfg)

        response = self.client.post("/api/providers/custom-endpoints", json={
            "id": "local", "name": "Local", "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model", "api_key": "pasted-key-placeholder",
        })
        assert response.status_code == 200, response.text

        env_var = custom_endpoint_key_env("local")
        raw_entry = read_raw_config()["providers"]["local"]
        assert raw_entry["key_env"] == env_var
        assert not {"api_key", "api_key_env"}.intersection(raw_entry)
        assert _get_named_custom_provider("custom:local")["api_key"] == "pasted-key-placeholder"
        assert self._listed("local")["api_key_preview"] == f"${{{env_var}}}"
        env = load_env()
        assert env[env_var] == "pasted-key-placeholder"
        assert env["SHARED_ENDPOINT_KEY"] == "shared-placeholder"

    @pytest.mark.parametrize("credential_field", ["key_env", "api_key_env"])
    def test_delete_custom_endpoint_clears_hand_written_model_credential(self, credential_field):
        """Delete drops the main-slot mirror under either spelling (#62269 for the alias)."""
        from hermes_cli.auth import _model_level_key_env
        from hermes_cli.config import load_config, load_env, read_raw_config, save_config, save_env_value
        from hermes_cli.runtime_provider_custom import _get_named_custom_provider

        save_env_value("MY_RELAY_KEY", "shared-placeholder")
        cfg = load_config()
        cfg["providers"] = {"relay": {
            "name": "Relay", "base_url": "https://relay.example/v1", "model": "m",
            credential_field: "MY_RELAY_KEY",
        }}
        cfg["model"] = {
            "provider": "relay", "default": "m", "base_url": "https://relay.example/v1",
            credential_field: "MY_RELAY_KEY",
        }
        save_config(cfg)
        assert _model_level_key_env("relay") == "MY_RELAY_KEY"
        assert _get_named_custom_provider("custom:relay")["api_key"] == "shared-placeholder"

        response = self.client.delete("/api/providers/custom-endpoints/relay")
        assert response.status_code == 200, response.text
        for persisted in (read_raw_config(), load_config()):
            assert "relay" not in persisted.get("providers", {})
            assert not {"api_key", "key_env", "api_key_env"}.intersection(persisted["model"])
        assert _model_level_key_env("relay") == ""
        assert _get_named_custom_provider("custom:relay") is None
        assert load_env()["MY_RELAY_KEY"] == "shared-placeholder"

    @pytest.mark.parametrize("credential_field", ["key_env", "api_key_env"])
    def test_custom_endpoint_response_reports_a_key_held_in_env(self, credential_field):
        """has_api_key must follow key_env or its alias, not just a plaintext api_key.

        Reading only ``api_key`` made the panel report "no API key" for every
        endpoint whose credential had been moved to .env; reading only ``key_env``
        did the same for an alias-only entry the runtime authenticates with.
        """
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["providers"] = {"proxy": {
            "name": "Proxy", "base_url": "https://proxy.example/v1", "model": "m",
            credential_field: "SHARED_ENDPOINT_KEY",
        }}
        save_config(cfg)

        endpoint = self._listed("proxy")
        assert endpoint["has_api_key"] is True
        assert endpoint["api_key_preview"] == "${SHARED_ENDPOINT_KEY}"

    @pytest.mark.parametrize("credential_field", ["key_env", "api_key_env"])
    @pytest.mark.parametrize("make_default", [False, True])
    def test_activating_an_endpoint_carries_its_credential_either_way(self, credential_field, make_default):
        """Activate and make-default carry the credential POINTER under either spelling;
        a pre-#69449 plaintext entry still activates."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["providers"] = {
            "legacy": {
                "name": "Legacy",
                "base_url": "https://legacy.example/v1",
                "model": "m",
                "api_key": "legacy-key-placeholder",
                "models": {"m": {}},
            },
            "modern": {
                "name": "Modern",
                "base_url": "https://modern.example/v1",
                "model": "m",
                credential_field: "MODERN_API_KEY",
                "models": {"m": {}},
            },
        }
        save_config(cfg)

        if make_default:
            response = self.client.post("/api/providers/custom-endpoints", json={
                "id": "modern", "name": "Modern", "base_url": "https://modern.example/v1",
                "model": "m", "make_default": True,
            })
        else:
            response = self.client.post("/api/providers/custom-endpoints/modern/activate", json={})
        assert response.status_code == 200, response.text
        model_cfg = load_config()["model"]
        assert model_cfg["key_env"] == "MODERN_API_KEY"
        assert "api_key" not in model_cfg

        self.client.post("/api/providers/custom-endpoints/legacy/activate", json={})
        model_cfg = load_config()["model"]
        assert model_cfg["api_key"] == "legacy-key-placeholder"
