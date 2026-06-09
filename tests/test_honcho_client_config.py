"""Tests for Honcho client configuration."""

import json
import os
import stat
from pathlib import Path

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho import HonchoMemoryProvider


def test_host_block_cadence_overrides_root_defaults(tmp_path):
    """Cadence knobs written by setup under hosts.hermes must be honored."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "apiKey": "test-api-key-12345",
        "contextCadence": 9,
        "dialecticCadence": 8,
        "hosts": {
            "hermes": {
                "contextCadence": 3,
                "dialecticCadence": 4,
            }
        },
    }))

    cfg = HonchoClientConfig.from_global_config(config_path=config_path, host="hermes")

    assert cfg.context_cadence == 3
    assert cfg.dialectic_cadence == 4


def test_gateway_session_key_wins_over_session_title():
    """Gateway title changes must not fragment one chat into many Honcho sessions."""
    cfg = HonchoClientConfig(session_strategy="per-directory")

    resolved = cfg.resolve_session_name(
        session_title="Auto Generated Conversation Title",
        gateway_session_key="agent:main:telegram:dm:7724476685",
        session_id="20260608_test",
    )

    assert resolved == "agent-main-telegram-dm-7724476685"


def test_profile_local_config_uses_global_credentials_fallback(tmp_path, monkeypatch):
    """Profile-local Honcho config may omit shared API credentials."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HONCHO_API_KEY", raising=False)
    monkeypatch.delenv("HONCHO_BASE_URL", raising=False)

    global_dir = tmp_path / ".honcho"
    global_dir.mkdir()
    (global_dir / "config.json").write_text(json.dumps({
        "apiKey": "global-api-key",
        "baseUrl": "https://honcho.example",
        "workspace": "global-workspace",
        "environment": "staging",
        "timeout": 12.5,
    }))

    profile_config = tmp_path / ".hermes" / "honcho.json"
    profile_config.parent.mkdir()
    profile_config.write_text(json.dumps({
        "enabled": True,
        "workspace": "wildeboer-fleet",
        "hosts": {
            "hermes": {
                "aiPeer": "hermes",
                "peerName": "Arta",
                "recallMode": "hybrid",
            }
        },
    }))

    cfg = HonchoClientConfig.from_global_config(config_path=profile_config, host="hermes")

    assert cfg.enabled is True
    assert cfg.api_key == "global-api-key"
    assert cfg.base_url == "https://honcho.example"
    assert cfg.workspace_id == "wildeboer-fleet"
    assert cfg.environment == "staging"
    assert cfg.timeout == 12.5
    assert cfg.peer_name == "Arta"


def test_windows_localappdata_honcho_config_uses_global_credentials_fallback(tmp_path, monkeypatch):
    """Windows HERMES_HOME ends in AppData/Local/hermes, not ~/.hermes."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HONCHO_API_KEY", raising=False)
    monkeypatch.delenv("HONCHO_BASE_URL", raising=False)

    global_dir = tmp_path / ".honcho"
    global_dir.mkdir()
    (global_dir / "config.json").write_text(json.dumps({
        "apiKey": "global-api-key",
        "baseUrl": "https://honcho.example",
    }))

    profile_config = tmp_path / "AppData" / "Local" / "hermes" / "honcho.json"
    profile_config.parent.mkdir(parents=True)
    profile_config.write_text(json.dumps({
        "enabled": True,
        "workspace": "wildeboer-fleet",
    }))

    cfg = HonchoClientConfig.from_global_config(config_path=profile_config, host="hermes")

    assert cfg.api_key == "global-api-key"
    assert cfg.base_url == "https://honcho.example"
    assert cfg.workspace_id == "wildeboer-fleet"


class TestHonchoClientConfigAutoEnable:
    """Test auto-enable behavior when API key is present."""

    def test_auto_enables_when_api_key_present_no_explicit_enabled(self, tmp_path):
        """When API key exists and enabled is not set, should auto-enable."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            # Note: no "enabled" field
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is True  # Auto-enabled because API key exists

    def test_respects_explicit_enabled_false(self, tmp_path):
        """When enabled is explicitly False, should stay disabled even with API key."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            "enabled": False,  # Explicitly disabled
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is False  # Respects explicit setting

    def test_respects_explicit_enabled_true(self, tmp_path):
        """When enabled is explicitly True, should be enabled."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            "enabled": True,
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is True

    def test_disabled_when_no_api_key_or_base_url_and_no_explicit_enabled(self, tmp_path):
        """When no API key/baseUrl and enabled not set, should be disabled."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "workspace": "test",
            # No apiKey, no baseUrl, no enabled
        }))

        # Clear env vars if set so this test does not inherit host-level status mode.
        env_key = os.environ.pop("HONCHO_API_KEY", None)
        env_base_url = os.environ.pop("HONCHO_BASE_URL", None)
        try:
            cfg = HonchoClientConfig.from_global_config(config_path=config_path)
            assert cfg.api_key is None
            assert cfg.base_url is None
            assert cfg.enabled is False  # No API key/baseUrl = not enabled
        finally:
            if env_key:
                os.environ["HONCHO_API_KEY"] = env_key
            if env_base_url:
                os.environ["HONCHO_BASE_URL"] = env_base_url

    def test_auto_enables_with_env_var_api_key(self, tmp_path, monkeypatch):
        """When API key is in env var (not config), should auto-enable."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "workspace": "test",
            # No apiKey in config
        }))

        monkeypatch.setenv("HONCHO_API_KEY", "env-api-key-67890")

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "env-api-key-67890"
        assert cfg.enabled is True  # Auto-enabled from env var API key

    def test_from_env_always_enabled(self, monkeypatch):
        """from_env() should always set enabled=True."""
        monkeypatch.setenv("HONCHO_API_KEY", "env-test-key")

        cfg = HonchoClientConfig.from_env()

        assert cfg.api_key == "env-test-key"
        assert cfg.enabled is True

    def test_falls_back_to_env_when_no_config_file(self, tmp_path, monkeypatch):
        """When config file doesn't exist, should fall back to from_env()."""
        nonexistent = tmp_path / "nonexistent.json"
        monkeypatch.setenv("HONCHO_API_KEY", "fallback-key")

        cfg = HonchoClientConfig.from_global_config(config_path=nonexistent)

        assert cfg.api_key == "fallback-key"
        assert cfg.enabled is True  # from_env() sets enabled=True


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path, monkeypatch):
    """honcho.json is created atomically with 0o600, not chmod-after-write."""
    import utils
    calls = []
    real_atomic = utils.atomic_json_write

    def spy(path, data, **kwargs):
        calls.append(kwargs.get("mode"))
        return real_atomic(path, data, **kwargs)

    monkeypatch.setattr(utils, "atomic_json_write", spy)
    provider = HonchoMemoryProvider()
    provider.save_config({"api_key": "hc-test-key"}, str(tmp_path))
    assert calls == [0o600]
    config_file = tmp_path / "honcho.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"
