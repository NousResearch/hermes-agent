"""OpenCode wire normalization must retain upstream's custom-entry pool ownership."""

import json

import pytest

from hermes_cli.config import atomic_config_replace, atomic_config_write

RAW_URL = "https://opencode.ai/zen/go"
WIRE_URL = f"{RAW_URL}/v1"
ENTRY_KEY = "fixture-entry-key-0123456789"
BUILTIN_KEY = "fixture-builtin-key-0123456789"
MANUAL_KEY = "fixture-manual-key-0123456789"


def _configure_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", BUILTIN_KEY)
    monkeypatch.setenv("ENTRY_KEY_VAR", ENTRY_KEY)
    config = {"providers": {"deepseek": {
        "base_url": RAW_URL, "api_mode": "chat_completions", "key_env": "ENTRY_KEY_VAR",
    }}}
    atomic_config_write(tmp_path / "config.yaml", config)
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"deepseek": [{
            "id": "manual", "label": "entry's manual account", "auth_type": "api_key",
            "priority": 0, "source": "manual", "access_token": MANUAL_KEY, "base_url": RAW_URL,
        }]},
    }), encoding="utf-8")
    return config


@pytest.mark.parametrize("requested", ["custom:deepseek", "custom"])
@pytest.mark.parametrize("edit", ["endpoint", "declaration"])
def test_wire_route_restores_only_its_recorded_owners_accounts(
    tmp_path, monkeypatch, requested, edit,
):
    from agent.credential_pool import load_pool
    from agent.custom_entry_pool import SNAPSHOT_OWNER_KEY, load_runtime_pool, runtime_entry_owner
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = _configure_entry(tmp_path, monkeypatch)
    runtime = resolve_runtime_provider(requested=requested, explicit_base_url=WIRE_URL)
    assert (runtime["base_url"], runtime["api_key"]) == (WIRE_URL, MANUAL_KEY)
    assert {row.access_token for row in runtime["credential_pool"].entries()} == {MANUAL_KEY}
    owner = runtime_entry_owner(runtime["provider"], requested, runtime["base_url"])
    assert owner is not None
    assert owner["name"] == "deepseek"
    assert owner["declares_credential"] is True
    assert MANUAL_KEY not in json.dumps(owner)
    if edit == "endpoint":
        config["providers"]["deepseek"]["base_url"] = "https://another.example.test/v1"
    else:
        del config["providers"]["deepseek"]["key_env"]
    atomic_config_replace(tmp_path / "config.yaml", config)

    restored = load_runtime_pool(
        runtime["provider"], WIRE_URL, requested, snapshot={SNAPSHOT_OWNER_KEY: owner},
    )
    assert {row.access_token for row in restored.entries()} == {MANUAL_KEY}
    # A scoped view must neither select nor delete the built-in's account.
    assert BUILTIN_KEY in {row.access_token for row in load_pool("deepseek").entries()}


@pytest.mark.parametrize("requested", ["custom:deepseek", "custom"])
def test_frozen_wire_route_keeps_its_declared_key_after_a_live_owner_edit(
    tmp_path, monkeypatch, requested,
):
    from agent.custom_entry_pool import runtime_entry_owner
    from hermes_cli.runtime_provider import frozen_runtime_config, load_config, resolve_runtime_provider

    config = _configure_entry(tmp_path, monkeypatch)
    (tmp_path / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}), encoding="utf-8",
    )
    frozen = load_config()
    config["providers"] = {"other": {"base_url": WIRE_URL, "api_key": "fixture-other-key"}}
    atomic_config_replace(tmp_path / "config.yaml", config)
    with frozen_runtime_config(frozen):
        runtime = resolve_runtime_provider(requested=requested, explicit_base_url=WIRE_URL)
        owner = runtime_entry_owner(runtime["provider"], requested, runtime["base_url"])
    assert (runtime["base_url"], runtime["api_key"]) == (WIRE_URL, ENTRY_KEY)
    assert owner is not None
    assert owner["name"] == "deepseek"
    assert owner["declares_credential"] is True
