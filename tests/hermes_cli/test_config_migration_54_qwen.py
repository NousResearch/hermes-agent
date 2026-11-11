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
