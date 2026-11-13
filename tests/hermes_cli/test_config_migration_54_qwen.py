"""Migration 53 -> 54: a saved ``model.provider: qwen`` keeps a Qwen-CLI-only install on its login.

Bare ``qwen`` became an Alibaba DashScope alias. A home whose only Qwen credential is the Qwen CLI
login is rewritten to ``qwen-oauth`` once; a home with a DashScope key keeps ``qwen`` (now alibaba).
"""

import json
import time

import pytest

import hermes_yaml as yaml


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    creds = tmp_path / ".qwen" / "oauth_creds.json"
    creds.parent.mkdir()
    creds.write_text(json.dumps({
        "access_token": "qwen-cli-token", "refresh_token": "r", "token_type": "Bearer",
        "expiry_date": int((time.time() + 3600) * 1000), "resource_url": "portal.qwen.ai"}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.auth._qwen_cli_auth_path", lambda: creds)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(
        {"_config_version": 53, "model": {"provider": "qwen", "default": "qwen3-coder-plus"}}), encoding="utf-8")
    return hermes_home


def _migrate(home):
    from hermes_cli.config import migrate_config
    migrate_config(interactive=False, quiet=True)
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def _runtime_provider():
    from hermes_cli.runtime_provider import resolve_runtime_provider
    return resolve_runtime_provider()["provider"]


def test_qwen_cli_only_home_moves_to_qwen_oauth_once(home):
    raw = _migrate(home)
    assert raw["model"]["provider"] == "qwen-oauth"
    assert raw["_config_version"] >= 54
    assert _runtime_provider() == "qwen-oauth"

    # Stamped past 54: a bare ``qwen`` the user picks afterwards means alibaba and is left alone.
    raw["model"]["provider"] = "qwen"
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert _migrate(home)["model"]["provider"] == "qwen"


def test_home_with_a_dashscope_key_keeps_qwen_and_routes_to_alibaba(home):
    (home / ".env").write_text("DASHSCOPE_API_KEY=sk-dashscope-test-1234567890\n", encoding="utf-8")
    assert _migrate(home)["model"]["provider"] == "qwen"
    assert _runtime_provider() == "alibaba"


def test_configured_provider_named_qwen_is_not_rewritten(home):
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    raw["providers"] = {"qwen": {"name": "qwen", "base_url": "http://127.0.0.1:9999/v1"}}
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert _migrate(home)["model"]["provider"] == "qwen"


@pytest.mark.parametrize("field", ["key_env", "api_key_env"])
def test_explicit_model_key_environment_keeps_alibaba(home, monkeypatch, field):
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    raw["model"][field] = "MY_DASHSCOPE_KEY"
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("MY_DASHSCOPE_KEY", "sk-custom-dashscope")
    assert _migrate(home)["model"]["provider"] == "qwen"


def test_oauth_pool_without_qwen_cli_file_is_migrated(home, tmp_path):
    (tmp_path / ".qwen" / "oauth_creds.json").unlink()
    (home / "auth.json").write_text(json.dumps({"credential_pool": {
        "qwen-oauth": [{"id": "qwen-pool", "access_token": "oauth-token"}]
    }}), encoding="utf-8")
    assert _migrate(home)["model"]["provider"] == "qwen-oauth"


def test_provider_selectors_migrate_without_touching_models_or_payloads(home):
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    raw.update({
        "fallback_model": {"provider": "qwen", "model": "qwen"},
        "fallback_providers": [{"provider": " QWEN ", "model": "qwen"}, None],
        "delegation": {"provider": "qwen", "model": "qwen", "fallback_providers": [
            {"provider": "qwen", "model": "qwen"}]},
        "auxiliary": {"vision": {"provider": "qwen", "model": "qwen", "extra_body": {
            "provider": "qwen"}, "fallback_chain": [{"provider": "qwen", "model": "qwen"}]}},
        "moa": {"aggregator": {"provider": "qwen", "model": "qwen"},
                "reference_models": [{"provider": "qwen", "model": "qwen"}],
                "presets": {"sample": {"aggregator": {"provider": "qwen", "model": "qwen"},
            "reference_models": [{"provider": "qwen", "model": "qwen"}]}}},
        "tts": {"provider": "qwen"},
    })
    raw["model"]["default"] = "qwen"
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = _migrate(home)
    entries = [out["model"], out["fallback_model"], out["fallback_providers"][0],
               out["delegation"], out["delegation"]["fallback_providers"][0],
               out["auxiliary"]["vision"], out["auxiliary"]["vision"]["fallback_chain"][0],
               out["moa"]["aggregator"], out["moa"]["reference_models"][0],
               out["moa"]["presets"]["sample"]["aggregator"],
               out["moa"]["presets"]["sample"]["reference_models"][0]]
    assert all(entry["provider"] == "qwen-oauth" for entry in entries)
    assert out["model"]["default"] == "qwen"
    assert all(entry["model"] == "qwen" for entry in entries[1:])
    assert out["auxiliary"]["vision"]["extra_body"]["provider"] == "qwen"
    assert out["tts"]["provider"] == "qwen"


def test_guard_applies_to_each_provider_selector(home, monkeypatch):
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    raw["fallback_providers"] = [{"provider": "qwen", "key_env": "OTHER_KEY"}, {"provider": "qwen"}]
    monkeypatch.setenv("OTHER_KEY", "sk-custom-secret")
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = _migrate(home)
    assert out["fallback_providers"][0]["provider"] == "qwen"
    assert out["fallback_providers"][1]["provider"] == "qwen-oauth"


def test_cron_and_gateway_channel_overrides_are_not_migrated(home):
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    raw["cron"] = {"model_provider": "qwen"}
    raw["platforms"] = {"discord": {"channel_overrides": {"123": {"provider": "qwen"}}}}
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = _migrate(home)
    assert out["cron"]["model_provider"] == "qwen"
    assert out["platforms"]["discord"]["channel_overrides"]["123"]["provider"] == "qwen"


@pytest.mark.parametrize("shape", ["root", "nested-default", "single-fallback-provider"])
def test_other_provider_shapes_runtime_accepts_are_migrated(home, shape):
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    if shape == "root":
        raw.pop("model", None)
        raw.update({"provider": "qwen", "model": "qwen3-coder-plus"})
    elif shape == "nested-default":
        raw["model"] = {"default": {"provider": "qwen", "model": "qwen3-coder-plus"}}
    else:
        raw["fallback_providers"] = {"provider": "qwen", "model": "qwen3-coder-plus"}
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = _migrate(home)
    # The save after migrating folds a root ``provider:`` and a nested ``model.default`` mapping
    # into ``model``, so those two shapes read back flat.
    slot = out["fallback_providers"] if shape == "single-fallback-provider" else out["model"]
    assert slot["provider"] == "qwen-oauth"
    if shape == "single-fallback-provider":
        assert slot["model"] == "qwen3-coder-plus"
    else:
        assert slot["default"] == "qwen3-coder-plus"


def test_empty_alibaba_pool_entry_does_not_block_oauth_migration(home):
    (home / "auth.json").write_text(json.dumps({"credential_pool": {
        "alibaba": [{"id": "empty", "access_token": ""}]
    }}), encoding="utf-8")
    assert _migrate(home)["model"]["provider"] == "qwen-oauth"
