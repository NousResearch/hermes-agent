"""HTTP contracts backed by independent, real profile config files."""

import json
from ruamel.yaml import YAML
from pathlib import Path

import pytest
from hermes_constants import resolve_reasoning_config
from fastapi.testclient import TestClient

from hermes_cli.web_server import app, _SESSION_TOKEN, _SESSION_HEADER_NAME
from hermes_cli.web_routers import models as routes


@pytest.fixture
def profile_client(monkeypatch, tmp_path):
    """Use the production profile scope and YAML config IO, redirecting only profile lookup."""
    from hermes_cli import web_server_profiles as profiles

    homes = {name: tmp_path / name for name in ("A", "B")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setattr(profiles, "_resolve_profile_dir", lambda name: homes[name])
    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    return client, homes


def write_config(home: Path, config: dict) -> Path:
    path = home / "config.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def read_config(path: Path) -> dict:
    return YAML(typ="safe").load(path.read_text(encoding="utf-8"))


def test_reasoning_http_real_disk_profile_isolation_and_reverse_alias_clear(profile_client):
    client, homes = profile_client
    path_a = write_config(homes["A"], {
        "model": {"default": "qwen3.6:27b", "provider": "ollama-local"},
        "agent": {"reasoning_effort": "low", "reasoning_overrides": {
            "ollama-local/qwen3.6:27b": "high",
            "openai/gpt-5": "max", "unrelated": "high",
        }},
        "delegation": {"provider": "openai", "model": "d0", "base_url": "https://custom",
                       "api_key": "secret", "reasoning_effort": False},
    })
    path_b = write_config(homes["B"], {
        "model": {"default": "anthropic/claude", "provider": "anthropic"},
        "agent": {"reasoning_effort": "medium"},
        "delegation": {"provider": "anthropic"},
    })

    def get_effort(profile):
        response = client.get("/api/model/reasoning-effort", params={"profile": profile})
        assert response.status_code == 200, response.text
        return response.json()

    a = get_effort("A")
    assert (a["main_raw"], a["main_effective"], a["main_source"]) == ("low", "high", "model_override")
    assert a["main_model"] == "qwen3.6:27b"
    assert a["delegation_raw"] == "none" and "secret" not in str(a)

    put = client.put("/api/model/reasoning-effort", params={"profile": "A"}, json={
        "scope": "main", "target": "model", "model": "qwen3.6:27b", "effort": "medium",
    })
    assert put.status_code == 200 and put.json()["main_effective"] == "medium"
    stored = read_config(path_a)
    assert stored["agent"]["reasoning_overrides"]["qwen3.6:27b"] == "medium"
    assert stored["model"]["default"] == "qwen3.6:27b"

    b = get_effort("B")
    assert (b["main_raw"], b["main_source"], b["main_model"]) == ("medium", "global", "anthropic/claude")
    stale = client.put("/api/model/reasoning-effort", params={"profile": "A"}, json={
        "scope": "main", "target": "model", "model": "old-model", "effort": "high",
    })
    assert stale.status_code == 409

    # Remove the selected prefixed model's override and its reverse-prefix alias only.
    clear = client.put("/api/model/reasoning-effort", params={"profile": "A"}, json={
        "scope": "main", "target": "model", "model": "qwen3.6:27b", "effort": "",
    })
    assert clear.status_code == 200 and clear.json()["main_source"] == "global"
    stored = read_config(path_a)
    assert stored["agent"]["reasoning_overrides"] == {"openai/gpt-5": "max", "unrelated": "high"}
    assert get_effort("A")["main_effective"] == "low"
    assert resolve_reasoning_config(stored) == {"enabled": True, "effort": "low"}
    assert get_effort("B")["main_raw"] == "medium"
    assert "reasoning_overrides" not in read_config(path_b)["agent"]

    # A -> B -> A reads prove persisted state, not an in-memory shared fixture.
    assert get_effort("A")["main_source"] == "global"
    assert read_config(path_a)["agent"]["reasoning_overrides"] == {"openai/gpt-5": "max", "unrelated": "high"}


@pytest.mark.parametrize("scope", ["main", "delegation"])
@pytest.mark.parametrize("value", [False, "false", "disabled", "none"])
def test_disabled_reasoning_form_is_a_read_only_noop(profile_client, scope, value):
    client, homes = profile_client
    section = "agent" if scope == "main" else "delegation"
    path = write_config(homes["A"], {"model": {"default": "openai/gpt-5"}, section: {"reasoning_effort": value}})
    before = path.read_bytes()
    response = client.get("/api/model/reasoning-effort", params={"profile": "A"})
    assert response.status_code == 200
    assert response.json()[f"{scope}_raw"] == "none"
    assert path.read_bytes() == before
    assert not (homes["B"] / "config.yaml").exists()


def test_custom_reasoning_values_persist_and_same_route_edit_preserves_routing(profile_client):
    client, homes = profile_client
    path = write_config(homes["A"], {
        "model": {"default": "openai/gpt-5", "provider": "openai"},
        "agent": {"reasoning_effort": "medium"},
        "delegation": {"provider": "openai", "model": "worker", "base_url": "https://custom/v1",
                       "api_key": "secret", "api_mode": "responses", "reasoning_effort": "high"},
        "budgets": {"daily": 12},
    })
    for scope, effort in (("main", "xhigh"), ("delegation", "ultra")):
        response = client.put("/api/model/reasoning-effort", params={"profile": "A"}, json={
            "scope": scope, "effort": effort,
        })
        assert response.status_code == 200, response.text
        assert response.json()["raw"] == effort
        assert client.get("/api/model/reasoning-effort", params={"profile": "A"}).json()[f"{scope}_raw"] == effort
    response = client.post("/api/model/set", json={
        "scope": "delegation", "provider": "openai", "model": "worker-v2", "profile": "A",
    })
    assert response.status_code == 200 and response.json()["ok"]
    stored = read_config(path)
    assert stored["delegation"]["model"] == "worker-v2"
    assert {key: stored["delegation"][key] for key in ("provider", "base_url", "api_key", "api_mode")} == {
        "provider": "openai", "base_url": "https://custom/v1", "api_key": "secret", "api_mode": "responses",
    }
    assert stored["budgets"] == {"daily": 12}
    assert not (homes["B"] / "config.yaml").exists()


def test_provider_switch_confirmation_reject_and_confirmed_reset_preserves_effort_budgets(profile_client):
    client, homes = profile_client
    path = write_config(homes["A"], {
        "model": {"default": "openai/gpt-5", "provider": "openai"},
        "agent": {"reasoning_effort": "medium", "max_turns": 77},
        "delegation": {"provider": "openai", "model": "worker", "base_url": "https://custom/v1",
                       "api_key": "secret", "api_mode": "responses", "reasoning_effort": "high",
                       "max_iterations": 40},
        "budgets": {"daily": 12},
    })
    before = path.read_bytes()
    rejected = client.post("/api/model/set", json={
        "scope": "delegation", "provider": "anthropic", "model": "claude-worker",
        "api_key": "must-not-store", "profile": "A",
    })
    assert rejected.status_code == 400
    assert read_config(path) == json.loads(before)

    needs_confirm = client.post("/api/model/set", json={
        "scope": "delegation", "provider": "anthropic", "model": "worker", "profile": "A",
    })
    assert needs_confirm.status_code == 200 and needs_confirm.json()["routing_confirmation_required"]
    assert path.read_bytes() == before
    changed = client.post("/api/model/set", json={
        "scope": "delegation", "provider": "anthropic", "model": "claude-worker",
        "confirm_clear_routing": True, "profile": "A",
    })
    assert changed.status_code == 200 and changed.json()["ok"]
    saved = read_config(path)["delegation"]
    assert saved["provider"] == "anthropic" and saved["model"] == "claude-worker"
    assert "base_url" not in saved and "api_key" not in saved


    confirmed = client.post("/api/model/set", json={
        "scope": "delegation", "provider": "", "model": "", "reset_routing": True,
        "confirm_clear_routing": True, "profile": "A",
    })
    assert confirmed.status_code == 200 and confirmed.json()["ok"]
    stored = read_config(path)
    assert stored["delegation"] == {"reasoning_effort": "high", "max_iterations": 40}
    assert stored["agent"] == {"reasoning_effort": "medium", "max_turns": 77}
    assert stored["budgets"] == {"daily": 12}


@pytest.mark.parametrize("target", ["global", "model"])
def test_disabled_save_and_custom_display(profile_client, target):
    client, homes = profile_client
    path = write_config(homes["A"], {"model": {"default": "demo"}, "agent": {"reasoning_effort": {"enabled": True, "effort": "bespoke"}}})
    before = path.read_bytes()
    data = client.get("/api/model/reasoning-effort", params={"profile": "A"}).json()
    assert data["main_raw"] == "__custom__" and data["main_effective"] == "bespoke"
    assert path.read_bytes() == before
    response = client.put("/api/model/reasoning-effort", json={"profile": "A", "scope": "main", "target": target, "model": "demo", "effort": "none"})
    assert response.status_code == 200 and response.json()["ok"]
    assert response.json()["raw"] == "none"
    assert resolve_reasoning_config(read_config(path)) == {"enabled": False}


def test_absent_provider_route_uses_model_without_creating_or_clearing_routing(profile_client):
    client, homes = profile_client
    path = write_config(homes["A"], {
        "model": {"default": "openai/gpt-5", "provider": "openai"},
        "agent": {"reasoning_effort": "low"},
        "delegation": {"model": "worker", "base_url": "https://custom/v1", "api_key": "secret",
                       "reasoning_effort": "high"},
    })
    response = client.post("/api/model/set", json={
        "scope": "delegation", "provider": "", "model": "worker-v2", "profile": "A",
    })
    assert response.status_code == 200 and response.json()["ok"]
    stored = read_config(path)
    assert "provider" not in stored["delegation"]
    assert stored["delegation"]["model"] == "worker-v2"
    assert stored["delegation"]["base_url"] == "https://custom/v1"
    assert stored["delegation"]["api_key"] == "secret"
    assert stored["delegation"]["reasoning_effort"] == "high"
    assert not (homes["B"] / "config.yaml").exists()
