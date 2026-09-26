"""Picker availability follows credential removal, not leftover catalog metadata."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def picker_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()
    # Only catalog/network boundaries are replaced; auth, pools, removal and inventory are real.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {
        "openrouter": {"env": ["OPENROUTER_API_KEY"], "models": {}}})
    def offline(*args, **kwargs):
        raise OSError("network disabled for picker regression")
    monkeypatch.setattr("socket.socket.connect", offline)
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *a, **k: ["test-model"])
    monkeypatch.setattr("hermes_cli.models.get_curated_nous_model_ids", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.fetch_ollama_cloud_models", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])
    return tmp_path


@pytest.mark.parametrize("provider,env_var,models_only,lingering_env", [
    ("zai", "ZAI_API_KEY", False, False),
    ("opencode-go", "OPENCODE_GO_API_KEY", True, False),
    ("opencode-go", "OPENCODE_GO_API_KEY", False, True),
    ("openrouter", "OPENROUTER_API_KEY", False, True),
])
def test_picker_removal_and_readd(picker_home, monkeypatch, provider, env_var, models_only, lingering_env):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _save_auth_store
    from hermes_cli.credential_lifecycle import remove_provider_env_credential, save_provider_env_credential
    from hermes_cli.inventory import build_models_payload, load_picker_context

    config = {"model": {"provider": provider, "default": "test-model"}}
    if models_only:
        config["providers"] = {provider: {"models": {"test-model": {}}, "stale_timeout_seconds": 60}}
    picker_home.joinpath("config.yaml").write_text(json.dumps(config), encoding="utf-8")
    metadata = {provider: {"detected_endpoint": "https://example.invalid/v1"}} if provider == "zai" else {}
    _save_auth_store({"version": 1, "providers": metadata})
    # A base override keeps Z.ai region detection offline.
    monkeypatch.setenv("ZAI_BASE_URL", "https://example.invalid/v1")
    key = "test-" + "a" * 24

    def row():
        payload = build_models_payload(load_picker_context(), explicit_only=True, picker_hints=True,
                                       probe_custom_providers=False, for_picker=True)
        return next(p for p in payload["providers"] if p["slug"] == provider)

    save_provider_env_credential(env_var, key)
    assert load_pool(provider).has_credentials()
    assert row()["authenticated"] is True
    remove_provider_env_credential(env_var)
    if lingering_env:
        monkeypatch.setenv(env_var, key)  # another process still inherited the removed key
    assert not load_pool(provider).has_credentials()
    removed = row()
    assert removed["authenticated"] is False
    assert removed["source"] == "configured-current"
    assert removed["warning"]
    unavailable = build_models_payload(
        load_picker_context().with_overrides(current_provider="opencode-free"),
        probe_custom_providers=False, for_picker=True)["providers"]
    assert provider not in {p["slug"] for p in unavailable}
    assert removed["models"] == ["test-model"]  # saved selection, never the old catalog
    save_provider_env_credential(env_var, key)
    assert load_pool(provider).has_credentials()
    assert row()["authenticated"] is True


@pytest.mark.parametrize("provider", ["opencode-go", "openai-codex"])
def test_picker_keeps_remaining_credentials_and_keyless_endpoints(picker_home, provider):
    from agent.credential_pool import load_pool, PooledCredential
    from hermes_cli.inventory import build_models_payload, ConfigContext

    pool = load_pool(provider)
    for ident in ("first", "second"):
        pool.add_entry(PooledCredential.from_dict(provider, {
            "id": ident, "source": "manual", "auth_type": "oauth" if provider == "openai-codex" else "api_key",
            "access_token": "test-" + ident * 8,
        }))
    local = {"local-test": {"base_url": "http://127.0.0.1:9999/v1", "models": ["local-model"]}}
    ctx = ConfigContext("", "", "", local, [])

    def rows():
        return {r["slug"]: r for r in build_models_payload(
            ctx, for_picker=True, picker_hints=True, probe_custom_providers=False)["providers"]}

    pool.remove_index(1)
    assert load_pool(provider).has_credentials()
    assert rows()[provider]["authenticated"] is True
    pool.remove_index(1)
    remaining = rows()
    assert provider not in remaining
    # The explicit keyless local endpoint survives credential churn.
    assert remaining["local-test"]["models"] == ["local-model"]
    assert remaining["local-test"]["authenticated"] is True
