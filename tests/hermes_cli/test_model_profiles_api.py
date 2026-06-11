"""Tests for dashboard model profile presets."""

from starlette.testclient import TestClient


def _test_client():
    from hermes_cli import web_server

    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    client.headers.update({web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN})
    return client


def test_model_profile_save_apply_and_active_sync():
    from hermes_cli.config import load_config, save_config

    save_config(
        {
            "model": {"provider": "openai-codex", "default": "gpt-5.4"},
            "auxiliary": {
                "compression": {"provider": "minimax-oneai-openai", "model": "MiniMax-M3"},
            },
        }
    )

    client = _test_client()

    created = client.post("/api/model/profiles", json={"name": "Stable Codex"})
    assert created.status_code == 200
    assert created.json()["profile"]["id"] == "stable-codex"
    assert {
        "task": "compression",
        "provider": "minimax-oneai-openai",
        "model": "MiniMax-M3",
        "base_url": "",
    } in created.json()["profile"]["auxiliary"]

    duplicate = client.post("/api/model/profiles", json={"name": " stable codex "})
    assert duplicate.status_code == 409

    changed = client.post(
        "/api/model/set",
        json={
            "scope": "main",
            "provider": "deepseek",
            "model": "deepseek-chat",
        },
    )
    assert changed.status_code == 200

    applied = client.post("/api/model/profiles/stable-codex/apply")
    assert applied.status_code == 200

    cfg = load_config()
    assert cfg["active_model_profile"] == "stable-codex"
    assert cfg["model"]["provider"] == "openai-codex"
    assert cfg["model"]["default"] == "gpt-5.4"
    assert cfg["auxiliary"]["compression"]["provider"] == "minimax-oneai-openai"

    updated_aux = client.post(
        "/api/model/set",
        json={
            "scope": "auxiliary",
            "task": "compression",
            "provider": "openai-codex",
            "model": "gpt-5.4",
        },
    )
    assert updated_aux.status_code == 200

    cfg = load_config()
    saved_profile = cfg["model_profiles"]["stable-codex"]
    assert cfg["active_model_profile"] == ""
    assert saved_profile["auxiliary"]["compression"]["provider"] == "minimax-oneai-openai"
    assert saved_profile["auxiliary"]["compression"]["model"] == "MiniMax-M3"

    saved = client.put("/api/model/profiles/stable-codex", json={"from_current": True})
    assert saved.status_code == 200

    cfg = load_config()
    saved_profile = cfg["model_profiles"]["stable-codex"]
    assert saved_profile["auxiliary"]["compression"]["provider"] == "openai-codex"
    assert saved_profile["auxiliary"]["compression"]["model"] == "gpt-5.4"


def test_model_profile_create_and_update_from_routing_payload():
    from hermes_cli.config import load_config, save_config

    save_config(
        {
            "model": {"provider": "openai-codex", "default": "gpt-5.4"},
            "auxiliary": {},
        }
    )

    client = _test_client()

    routing = {
        "main": {"provider": "deepseek", "model": "deepseek-chat"},
        "auxiliary": [
            {"task": "vision", "provider": "openai-codex", "model": "gpt-5.5"},
            {"task": "compression", "provider": "auto", "model": ""},
        ],
    }
    created = client.post(
        "/api/model/profiles",
        json={"name": "Draft Routing", "routing": routing},
    )
    assert created.status_code == 200

    cfg = load_config()
    saved_profile = cfg["model_profiles"]["draft-routing"]
    assert saved_profile["model"]["provider"] == "deepseek"
    assert saved_profile["model"]["default"] == "deepseek-chat"
    assert saved_profile["auxiliary"]["vision"]["provider"] == "openai-codex"
    assert saved_profile["auxiliary"]["vision"]["model"] == "gpt-5.5"

    next_routing = {
        "main": {"provider": "minimax-oneai-openai", "model": "MiniMax-M3"},
        "auxiliary": [
            {"task": "web_extract", "provider": "deepseek", "model": "deepseek-chat"},
        ],
    }
    updated = client.put(
        "/api/model/profiles/draft-routing",
        json={"routing": next_routing},
    )
    assert updated.status_code == 200
    assert updated.json()["profile"]["main"] == {
        "provider": "minimax-oneai-openai",
        "model": "MiniMax-M3",
    }
    assert {
        "task": "web_extract",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "",
    } in updated.json()["profile"]["auxiliary"]
