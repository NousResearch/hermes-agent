"""Refusal preflights preserve credentials and submitted-value validation."""
from pathlib import Path

import pytest

from hermes_cli.config_backend import ConfigWriteError


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    (root / "config.yaml").write_text("unknown_sibling: keep\n")
    return root


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    from hermes_cli import web_server
    result = TestClient(web_server.app, raise_server_exceptions=False)
    result.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return result


@pytest.mark.parametrize("ignored", [False, True])
def test_profile_clone_preserves_token_on_refused_mcp_save(home, client, monkeypatch, ignored):
    if not ignored:
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG")
    source = home / "profiles" / "source"
    source.mkdir(parents=True)
    (source / "config.yaml").write_text(
        "mcp_servers:\n  probe:\n    url: https://old.example.invalid/mcp\n"
        "    headers:\n      Authorization: Bearer ${MCP_PROBE_API_KEY}\n")
    (source / ".env").write_text("MCP_PROBE_API_KEY=original-value\n")
    response = client.post("/api/profiles", json={
        "name": "destination", "clone_from": "source", "mirror_credentials": False, "share_auth": False,
        "mcp_servers": [{"name": "probe", "url": "https://new.example.invalid/mcp",
                         "auth": "header", "bearer_token": "replacement-value"}],
    })
    assert response.status_code == 200, response.text
    target = home / "profiles" / "destination"
    assert (source / ".env").read_text() == "MCP_PROBE_API_KEY=original-value\n"
    if ignored:
        assert response.json()["mcp_written"] == 0
        assert any("--ignore-user-config" in warning for warning in response.json()["warnings"])
        assert (target / ".env").read_bytes() == (source / ".env").read_bytes()
        assert "https://old.example.invalid/mcp" in (target / "config.yaml").read_text()
    else:
        assert response.json()["mcp_written"] == 1
        assert "replacement-value" in (target / ".env").read_text()
        assert "https://new.example.invalid/mcp" in (target / "config.yaml").read_text()


@pytest.mark.parametrize("model", ["picked/model", ""])
@pytest.mark.parametrize("key", ["replacement-value", ""])
def test_custom_cli_refuses_before_key_or_auth_mutation(home, monkeypatch, model, key):
    from hermes_cli import auth, config, main_provider_setup, model_setup_flows_custom as flow, secret_prompt
    env = home / ".env"
    env.write_text(f"{config.custom_endpoint_key_env('probe.example.invalid')}=original-value\n")
    before = env.read_bytes(), (home / "config.yaml").read_bytes()
    answers = iter(["https://probe.example.invalid/v1", "4096", "Probe"])
    monkeypatch.setattr(flow, "line_input", lambda *args: next(answers))
    monkeypatch.setattr(secret_prompt, "masked_secret_prompt", lambda *args: key)
    monkeypatch.setattr(flow, "_probe_custom_endpoint", lambda token, url: ({"models": []}, url))
    monkeypatch.setattr(flow, "_pick_detected_model", lambda models: model)
    monkeypatch.setattr(main_provider_setup, "_prompt_custom_api_mode_selection", lambda *args, **kwargs: "")
    monkeypatch.setattr(config, "get_env_value", lambda name: "")
    deactivations = []
    monkeypatch.setattr(auth, "deactivate_provider", lambda: deactivations.append(True))
    with pytest.raises(ConfigWriteError):
        flow._model_flow_custom({})
    assert deactivations == []
    assert (env.read_bytes(), (home / "config.yaml").read_bytes()) == before


@pytest.mark.parametrize("surface", ["declared", "native"])
@pytest.mark.parametrize("ignored", [False, True])
def test_invalid_memory_values_still_return_400_without_writes(home, client, monkeypatch, surface, ignored):
    from hermes_cli.web_routers import memory_providers as memory
    from plugins.memory.config_schema import ProviderConfigSchema, ProviderField
    if not ignored:
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG")
    env = home / ".env"
    env.write_text("PROBE_TOKEN=original-value\n")
    before = env.read_bytes(), (home / "config.yaml").read_bytes()
    writes = []
    if surface == "declared":
        schema = ProviderConfigSchema(name="probe", label="Probe", fields=(
            ProviderField(key="token", label="Token", kind="secret", env_key="PROBE_TOKEN"),
            ProviderField(key="amount", label="Amount", kind="number"),
        ))
        monkeypatch.setattr(memory, "get_provider_config_schema", lambda name: schema)
    else:
        monkeypatch.setattr(memory, "_load_memory_provider", lambda name: object())
        monkeypatch.setattr(memory, "_read_memory_provider_existing_values", lambda name: {})
        monkeypatch.setattr(memory, "_normalize_memory_provider_schema", lambda *args: [
            {"key": "token", "kind": "secret", "_env_key": "PROBE_TOKEN"},
            {"key": "amount", "kind": "number", "default": 0},
        ])
        monkeypatch.setattr(memory, "_save_memory_provider_native_config", lambda *args: writes.append(args))
    response = client.put("/api/memory/providers/probe/config", params={"surface": surface}, json={
        "values": {"amount": "not-a-number", "token": "replacement-value"},
    })
    assert response.status_code == 400, response.text
    assert "amount" in response.json()["detail"]
    assert writes == []
    assert (env.read_bytes(), (home / "config.yaml").read_bytes()) == before
