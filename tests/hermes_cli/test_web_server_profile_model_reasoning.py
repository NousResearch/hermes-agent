"""Regression tests for profile-scoped model/reasoning dashboard settings."""

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

    def test_global_reasoning_effort_normalizes_and_clears(self):
        from hermes_cli.config import load_config, save_config

        save_config({"agent": {"reasoning_effort": "high"}})
        web_config = self.client.get("/api/config").json()
        assert web_config["agent"]["reasoning_effort"] == "high"

        web_config["agent"]["reasoning_effort"] = ""
        response = self.client.put("/api/config", json={"config": web_config})

        assert response.status_code == 200
        assert "reasoning_effort" not in load_config().get("agent", {})

    @pytest.mark.parametrize("invalid", ["turbo", 42, True, {"effort": "high"}])
    def test_global_reasoning_effort_rejects_invalid_values(self, invalid):
        web_config = self.client.get("/api/config").json()
        web_config.setdefault("agent", {})["reasoning_effort"] = invalid

        response = self.client.put("/api/config", json={"config": web_config})

        assert response.status_code == 400
        assert response.json()["detail"] == "invalid agent.reasoning_effort"

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

        listed = self.client.get("/api/profiles")
        assert listed.status_code == 200
        worker = next(
            profile
            for profile in listed.json()["profiles"]
            if profile["name"] == "worker"
        )
        assert worker["reasoning_effort"] == ""

    def test_profile_reasoning_endpoint_rejects_unknown_value(self):
        self._profile_home()

        response = self.client.put(
            "/api/profiles/worker/reasoning",
            json={"effort": "turbo"},
        )

        assert response.status_code == 400
        assert "effort must be one of" in response.json()["detail"]
