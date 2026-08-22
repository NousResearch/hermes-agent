"""Regression tests for profile-scoped dashboard model settings."""

from copy import deepcopy

import pytest


class TestProfileModelReasoningEndpoints:
    @pytest.fixture(autouse=True)
    def _setup_client(self, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    @staticmethod
    def _profile_home(name="worker"):
        from hermes_cli import profiles

        home = profiles.get_profile_dir(name)
        home.mkdir(parents=True, exist_ok=True)
        return home

    @staticmethod
    def _read_profile_config(profile_home):
        from hermes_cli.config import load_config
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(str(profile_home))
        try:
            return load_config()
        finally:
            reset_hermes_home_override(token)

    def test_schema_exposes_main_reasoning_effort(self):
        response = self.client.get("/api/config/schema")

        assert response.status_code == 200
        entry = response.json()["fields"]["agent.reasoning_effort"]
        assert entry["type"] == "select"
        assert entry["category"] == "agent"
        assert entry["emptyLabel"] == "Inherit provider default"
        assert entry["options"] == [
            "",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "ultra",
        ]

    @pytest.mark.parametrize("invalid", ["turbo", 42, True, {"effort": "high"}])
    def test_global_reasoning_effort_rejects_invalid_values(self, invalid):
        web_config = self.client.get("/api/config").json()
        web_config.setdefault("agent", {})["reasoning_effort"] = invalid

        response = self.client.put("/api/config", json={"config": web_config})

        assert response.status_code == 400
        assert response.json()["detail"] == "invalid agent.reasoning_effort"

    def test_model_info_keeps_profile_scope_for_metadata_resolution(self, monkeypatch):
        from pathlib import Path
        from hermes_cli.config import load_config, save_config
        from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

        profile_home = self._profile_home()
        home_token = set_hermes_home_override(str(profile_home))
        try:
            config = load_config()
            config["model"] = {
                "provider": "provider-worker",
                "default": "model-worker",
            }
            save_config(config)
        finally:
            reset_hermes_home_override(home_token)

        def context_length(**_kwargs):
            return 111 if Path(get_hermes_home()).resolve() == profile_home.resolve() else 222

        monkeypatch.setattr("agent.model_metadata.get_model_context_length", context_length)
        response = self.client.get("/api/model/info?profile=worker")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "model-worker"
        assert data["provider"] == "provider-worker"
        assert data["auto_context_length"] == 111

    def test_profile_settings_are_scoped_and_saved_once(self, monkeypatch):
        import hermes_cli.web_server as web_server
        from hermes_cli.config import load_config, save_config
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_home = self._profile_home()
        token = set_hermes_home_override(str(profile_home))
        try:
            profile_config = load_config()
            profile_config["model"] = {
                "provider": "provider-a",
                "default": "model-a",
                "context_length": 12345,
                "base_url": "https://profile.example/v1",
                "api_key": "profile-secret",
            }
            save_config(profile_config)
        finally:
            reset_hermes_home_override(token)

        default_config_before = deepcopy(load_config())
        saves = []
        original_save_config = web_server.save_config

        def tracked_save_config(config):
            saves.append(deepcopy(config))
            return original_save_config(config)

        monkeypatch.setattr(web_server, "save_config", tracked_save_config)

        response = self.client.put(
            "/api/profiles/worker/settings",
            json={"provider": "", "model": "", "effort": "high"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "provider": None,
            "model": None,
            "reasoning_effort": "high",
        }
        assert len(saves) == 1
        profile_config = self._read_profile_config(profile_home)
        assert profile_config["model"]["context_length"] == 12345
        assert profile_config["model"]["base_url"] == "https://profile.example/v1"
        assert profile_config["model"]["api_key"] == "profile-secret"
        assert profile_config["agent"]["reasoning_effort"] == "high"
        assert load_config() == default_config_before

        response = self.client.put(
            "/api/profiles/worker/settings",
            json={"provider": "provider-b", "model": "model-b", "effort": "low"},
        )

        assert response.status_code == 200
        assert len(saves) == 2
        profile_config = self._read_profile_config(profile_home)
        assert profile_config["model"]["provider"] == "provider-b"
        assert profile_config["model"]["default"] == "model-b"
        assert "context_length" not in profile_config["model"]
        assert "api_key" not in profile_config["model"]
        assert profile_config["agent"]["reasoning_effort"] == "low"
        assert load_config() == default_config_before

    def test_profile_settings_require_a_complete_model_pair(self):
        self._profile_home()

        response = self.client.put(
            "/api/profiles/worker/settings",
            json={"provider": "provider-a", "model": "", "effort": "high"},
        )

        assert response.status_code == 400
        assert "provided together" in response.json()["detail"]

    def test_fallback_routes_hide_credentials_and_reject_duplicate_routes(self):
        profile_home = self._profile_home()
        token_config = self._read_profile_config(profile_home)
        token_config["fallback_providers"] = [
            {
                "provider": "provider-a",
                "model": "model-a",
                "api_key": "fallback-secret-a",
                "key_env": "FALLBACK_A",
                "base_url": "https://a.example/v1",
                "reasoning_effort": "high",
            },
            {
                "provider": "provider-b",
                "model": "model-b",
                "api_key": "fallback-secret-b",
                "key_env": "FALLBACK_B",
                "base_url": "https://b.example/v1",
            },
        ]
        from hermes_cli.config import save_config
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(str(profile_home))
        try:
            save_config(token_config)
        finally:
            reset_hermes_home_override(token)

        response = self.client.get("/api/profiles/worker/fallbacks")
        assert response.status_code == 200
        entries = response.json()["fallbacks"]
        assert [entry["model"] for entry in entries] == ["model-a", "model-b"]
        assert entries[0]["reasoning_effort"] == "high"
        for entry in entries:
            assert "api_key" not in entry
            assert "key_env" not in entry

        duplicate = {
            "fallbacks": [
                {
                    "source_index": 0,
                    "source_provider": "provider-a",
                    "source_model": "model-a",
                    "source_base_url": "https://a.example/v1",
                    "provider": "provider-a",
                    "model": "model-b",
                    "reasoning_effort": "",
                },
                {
                    "source_index": 0,
                    "source_provider": "provider-a",
                    "source_model": "model-a",
                    "source_base_url": "https://a.example/v1",
                    "provider": "provider-a",
                    "model": "model-b",
                    "reasoning_effort": "",
                },
            ]
        }
        response = self.client.put("/api/profiles/worker/fallbacks", json=duplicate)
        assert response.status_code == 400
        assert "unique" in response.json()["detail"]

        changed_provider = {
            "fallbacks": [
                {
                    "source_index": 0,
                    "source_provider": "provider-a",
                    "source_model": "model-a",
                    "source_base_url": "https://a.example/v1",
                    "provider": "provider-c",
                    "model": "model-c",
                    "reasoning_effort": "low",
                }
            ]
        }
        response = self.client.put("/api/profiles/worker/fallbacks", json=changed_provider)
        assert response.status_code == 200
        saved = self._read_profile_config(profile_home)["fallback_providers"][0]
        assert saved["provider"] == "provider-c"
        assert saved["model"] == "model-c"
        assert saved["reasoning_effort"] == "low"
        assert "api_key" not in saved
        assert "key_env" not in saved

    def test_fallback_duplicate_route_rejects_api_mode_only_difference(self, monkeypatch):
        self._profile_home()
        existing = [
            {
                "provider": "provider-a",
                "model": "model-a",
                "base_url": "https://a.example/v1",
                "api_mode": "chat_completions",
            },
            {
                "provider": "provider-a",
                "model": "model-a",
                "base_url": "https://a.example/v1",
                "api_mode": "responses",
            },
        ]
        monkeypatch.setattr(
            "hermes_cli.fallback_config.get_fallback_chain",
            lambda _config: [dict(entry) for entry in existing],
        )

        response = self.client.put(
            "/api/profiles/worker/fallbacks",
            json={
                "fallbacks": [
                    {
                        "source_index": 0,
                        "source_provider": "provider-a",
                        "source_model": "model-a",
                        "source_base_url": "https://a.example/v1",
                        "source_api_mode": "chat_completions",
                        "provider": "provider-a",
                        "model": "model-a",
                        "reasoning_effort": "",
                    },
                    {
                        "source_index": 1,
                        "source_provider": "provider-a",
                        "source_model": "model-a",
                        "source_base_url": "https://a.example/v1",
                        "source_api_mode": "responses",
                        "provider": "provider-a",
                        "model": "model-a",
                        "reasoning_effort": "",
                    },
                ]
            },
        )

        assert response.status_code == 400
        assert "unique" in response.json()["detail"]

    def test_profile_reasoning_endpoint_round_trips_and_clears(self):
        profile_home = self._profile_home()

        response = self.client.put(
            "/api/profiles/worker/reasoning",
            json={"effort": "HIGH"},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True, "reasoning_effort": "high"}
        assert self._read_profile_config(profile_home)["agent"]["reasoning_effort"] == "high"

        response = self.client.put(
            "/api/profiles/worker/reasoning",
            json={"effort": ""},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True, "reasoning_effort": ""}
        assert "reasoning_effort" not in self._read_profile_config(profile_home).get("agent", {})

    def test_profile_reasoning_endpoint_rejects_unknown_value(self):
        self._profile_home()

        response = self.client.put(
            "/api/profiles/worker/reasoning",
            json={"effort": "turbo"},
        )

        assert response.status_code == 400
        assert "effort must be one of" in response.json()["detail"]
