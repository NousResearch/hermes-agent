from contextlib import nullcontext

import pytest

from hermes_cli.web_models import ReasoningEffortUpdate
from hermes_cli.web_routers import models


def test_reasoning_raw_persistence_reset_and_preservation(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: main\n  provider: p\nagent:\n  reasoning_effort: medium\ndelegation:\n  reasoning_effort: low\n  model: worker\n", encoding="utf-8")
    import hermes_cli.config as config
    monkeypatch.setattr(config, "get_config_path", lambda: path)
    monkeypatch.setattr(models, "_profile_scope", lambda profile: nullcontext())
    monkeypatch.setattr(models, "read_user_config_raw", config.read_user_config_raw)
    monkeypatch.setattr(models, "save_config", config.save_config)

    assert models.get_reasoning_effort() == {"main_raw": "medium", "delegation_raw": "low"}
    result = models.set_reasoning_effort(ReasoningEffortUpdate(scope="main", effort=""))
    assert result["raw"] == ""
    raw = config.read_user_config_raw(path)
    assert raw["model"]["default"] == "main"
    assert raw["delegation"]["model"] == "worker"
    assert "reasoning_effort" not in raw["agent"]


def test_reasoning_invalid_value_does_not_write(monkeypatch):
    with pytest.raises(Exception):
        models.set_reasoning_effort(ReasoningEffortUpdate(scope="main", effort="bogus"))


def test_delegation_assignment_uses_raw_and_endpoint(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: main\n  provider: p\nagent:\n  max_turns: 90\n", encoding="utf-8")
    import hermes_cli.config as config
    monkeypatch.setattr(config, "get_config_path", lambda: path)
    monkeypatch.setattr(models, "_profile_scope", lambda profile: nullcontext())
    monkeypatch.setattr(models, "read_user_config_raw", config.read_user_config_raw)
    monkeypatch.setattr(models, "save_config", config.save_config)
    monkeypatch.setattr(models, "_dashboard_code_skew_guard", lambda: None)
    monkeypatch.setattr(models, "combined_selection_warning", lambda *a, **k: None, raising=False)
    from hermes_cli.web_models import ModelAssignment
    import asyncio
    result = asyncio.run(models.set_model_assignment(ModelAssignment(scope="delegation", provider="local", model="worker", base_url="http://lan/v1", api_key="key"), None))
    assert result["ok"]
    raw = config.read_user_config_raw(path)
    assert raw["delegation"] == {"provider": "local", "model": "worker", "base_url": "http://lan/v1", "api_key": "key"}


def test_http_effort_roundtrip_and_model_reset(monkeypatch, tmp_path):
    import hermes_cli.config as config
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: parent\n  provider: custom\ndelegation:\n  model: worker\n  max_iterations: 40\nagent:\n  max_turns: 77\n", encoding="utf-8")
    monkeypatch.setattr(config, "get_config_path", lambda: path)
    monkeypatch.setattr(models, "_profile_scope", lambda profile: nullcontext())
    monkeypatch.setattr(models, "_dashboard_code_skew_guard", lambda: None)
    app = FastAPI()
    app.include_router(models.router)
    with TestClient(app) as client:
        for scope in ("main", "delegation"):
            for effort in ("low", "medium", "none", "high", ""):
                response = client.put("/api/model/reasoning-effort", json={"scope": scope, "effort": effort})
                assert response.status_code == 200, response.text
                assert client.get("/api/model/reasoning-effort").json()[scope + "_raw"] == effort
                raw = config.read_user_config_raw(path)
                assert raw["model"] == {"default": "parent", "provider": "custom"}
                assert raw["delegation"]["max_iterations"] == 40
                assert raw["agent"]["max_turns"] == 77
        before = path.read_bytes()
        assert client.put("/api/model/reasoning-effort", json={"scope": "delegation", "effort": "bogus"}).status_code == 422
        assert path.read_bytes() == before
        result = client.post("/api/model/set", json={"scope": "delegation", "provider": "", "model": ""})
        assert result.status_code == 200, result.text
        assert config.read_user_config_raw(path)["delegation"] == {"max_iterations": 40}
