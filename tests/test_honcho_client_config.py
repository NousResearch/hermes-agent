"""Tests for Honcho client configuration."""

import json
import os
import stat
from pathlib import Path

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho import HonchoMemoryProvider


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

    def test_disabled_when_no_api_key_and_no_explicit_enabled(self, tmp_path):
        """When no API key and enabled not set, should be disabled."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "workspace": "test",
            # No apiKey, no enabled
        }))

        # Clear env var if set
        env_key = os.environ.pop("HONCHO_API_KEY", None)
        try:
            cfg = HonchoClientConfig.from_global_config(config_path=config_path)
            assert cfg.api_key is None
            assert cfg.enabled is False  # No API key = not enabled
        finally:
            if env_key:
                os.environ["HONCHO_API_KEY"] = env_key

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


class TestHonchoCadenceHostScoped:
    """Test that cadence fields resolve with host-first precedence (#35359)."""

    def test_host_scoped_dialectic_cadence(self, tmp_path):
        """dialecticCadence in host block takes precedence over root."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "dialecticCadence": 1,
            "hosts": {
                "hermes": {
                    "dialecticCadence": 5,
                },
            },
        }))

        cfg = HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path,
        )

        assert cfg.dialectic_cadence == 5

    def test_root_dialectic_cadence_fallback(self, tmp_path):
        """dialecticCadence falls back to root when not in host block."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "dialecticCadence": 3,
        }))

        cfg = HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path,
        )

        assert cfg.dialectic_cadence == 3

    def test_default_dialectic_cadence(self, tmp_path):
        """dialecticCadence defaults to 1 when not set anywhere."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
        }))

        cfg = HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path,
        )

        assert cfg.dialectic_cadence == 1

    def test_host_scoped_context_cadence(self, tmp_path):
        """contextCadence in host block takes precedence over root."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "contextCadence": 1,
            "hosts": {
                "hermes": {
                    "contextCadence": 4,
                },
            },
        }))

        cfg = HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path,
        )

        assert cfg.context_cadence == 4

    def test_host_scoped_injection_frequency(self, tmp_path):
        """injectionFrequency in host block takes precedence over root."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "injectionFrequency": "every-turn",
            "hosts": {
                "hermes": {
                    "injectionFrequency": "session",
                },
            },
        }))

        cfg = HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path,
        )

        assert cfg.injection_frequency == "session"

    def test_root_injection_frequency_fallback(self, tmp_path):
        """injectionFrequency falls back to root when not in host block."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "injectionFrequency": "async",
        }))

        cfg = HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path,
        )

        assert cfg.injection_frequency == "async"
