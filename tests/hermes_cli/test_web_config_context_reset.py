"""Dashboard config PUT: a context-window reset persists as auto-detect (no legacy root alias can
restore the old override), and a model change keeps the config's other contracts."""

from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_yaml as yaml


class TestConfigContextReset:
    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    @pytest.mark.parametrize("profile_source", ["current", "query", "body"])
    @pytest.mark.parametrize(
        "layout,full_form,context_value",
        [("nested", False, 0), ("nested", True, 0), ("nested", False, "0"),
         ("legacy", False, 0), ("legacy", True, 0),
         ("nested", False, 200000), ("legacy", False, 200000),
         ("nested", False, None), ("legacy", False, None)],
        ids=["reset", "full-reset", "string-reset", "legacy-reset", "legacy-full-reset",
             "positive", "legacy-positive", "omitted", "legacy-omitted"],
    )
    def test_context_length_round_trip(self, tmp_path, monkeypatch, profile_source, layout, full_form, context_value):
        from hermes_cli.profiles import get_profile_dir

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        default_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        homes = {"current": default_home, "worker": get_profile_dir("worker"),
                 "other": get_profile_dir("other")}
        for home in homes.values():
            home.mkdir(parents=True, exist_ok=True)
            context_override = {"context_length": 123456}
            seed = {
                "model": {"default": "context-test-model", "x_context_sibling": "keep",
                          **(context_override if layout == "nested" else {})},
                "agent": {"x_context_sibling": "keep"},
                "x_context_sibling": "keep",
                **(context_override if layout == "legacy" else {}),
            }
            (home / "config.yaml").write_text(yaml.safe_dump(seed), encoding="utf-8")
        target = "current" if profile_source == "current" else "worker"
        config_path = homes[target] / "config.yaml"
        untouched = {home / "config.yaml": (home / "config.yaml").read_bytes()
                     for name, home in homes.items() if name != target}
        read_params = {} if target == "current" else {"profile": target}
        payload = {}
        if full_form:
            initial = self.client.get("/api/config", params=read_params)
            assert initial.status_code == 200
            payload = initial.json()
            assert isinstance(payload["model"], str)
            assert payload["model_context_length"] == 123456
        payload.pop("x_context_sibling", None)
        if "agent" in payload:
            payload["agent"].pop("x_context_sibling", None)
        payload.setdefault("display", {})["skin"] = "mono"
        if context_value is None:
            payload.pop("model_context_length", None)
        else:
            payload["model_context_length"] = context_value
        body = {"config": payload}
        write_params = read_params
        if profile_source == "body":
            body["profile"] = target
            write_params = {"profile": "other"}

        response = self.client.put("/api/config", params=write_params, json=body)

        assert response.status_code == 200, response.text
        assert response.json()["ok"] is True
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = 123456 if context_value is None else int(context_value)
        assert "model_context_length" not in raw
        assert "context_length" not in raw
        if expected == 0:
            assert "context_length" not in raw["model"]
        else:
            assert raw["model"]["context_length"] == expected
        assert raw["model"]["default"] == "context-test-model"
        assert raw["model"]["x_context_sibling"] == "keep"
        assert raw["agent"]["x_context_sibling"] == "keep"
        assert raw["x_context_sibling"] == "keep"
        assert raw["display"]["skin"] == "mono"
        reloaded = self.client.get("/api/config", params=read_params)
        assert reloaded.status_code == 200
        assert reloaded.json()["model_context_length"] == expected
        for path, original in untouched.items():
            assert path.read_bytes() == original

    @pytest.mark.parametrize(
        "payload,expected_model,expected_provider,expected_context",
        [({"model": "local-model-next"}, "local-model-next", "ollama-local", 123456),
         ({"model": {"default": "local-model-next"}}, "local-model-next", "ollama-local", 123456),
         ({"model": "google/gemini-2.5-flash"}, "google/gemini-2.5-flash", "openrouter", 0),
         ({"model": "google/gemini-2.5-flash", "model_context_length": 0},
          "google/gemini-2.5-flash", "openrouter", 0),
         ({"model": "google/gemini-2.5-flash", "model_context_length": 200000},
          "google/gemini-2.5-flash", "openrouter", 200000)],
        ids=["same-provider", "nested-model", "provider-switch", "provider-switch-reset", "provider-switch-positive"],
    )
    def test_model_update_preserves_config_contracts(self, payload, expected_model, expected_provider, expected_context):
        from hermes_cli.config import get_config_path

        seed = {"model": {"default": "llama3.2", "provider": "ollama-local",
                          "base_url": "http://localhost:11434/v1", "api_mode": "chat_completions",
                          "api_key": "test-placeholder", "api": "legacy-test-placeholder",
                          "context_length": 123456, "x_context_sibling": "keep"},
                "agent": {"x_context_sibling": "keep"}}
        config_path = get_config_path()
        config_path.write_text(yaml.safe_dump(seed), encoding="utf-8")

        with patch("hermes_cli.models_detect.provider_has_credentials", lambda p: p == "openrouter"):
            response = self.client.put("/api/config", json={"config": payload})

        assert response.status_code == 200, response.text
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["model"]["default"] == expected_model
        assert raw["model"]["provider"] == expected_provider
        if expected_context:
            assert raw["model"]["context_length"] == expected_context
        else:
            assert "context_length" not in raw["model"]
        assert raw["model"]["x_context_sibling"] == "keep"
        assert raw["agent"]["x_context_sibling"] == "keep"
        if expected_provider == "ollama-local":
            assert raw["model"]["base_url"] == seed["model"]["base_url"]
            assert raw["model"]["api_mode"] == seed["model"]["api_mode"]
            assert raw["model"]["api_key"] == seed["model"]["api_key"]
            assert raw["model"]["api"] == seed["model"]["api"]
        else:
            assert raw["model"]["base_url"] != seed["model"]["base_url"]
            assert "api_key" not in raw["model"]
            assert "api" not in raw["model"]
        reloaded = self.client.get("/api/config").json()
        assert reloaded["model"] == expected_model
        assert reloaded["model_context_length"] == expected_context
