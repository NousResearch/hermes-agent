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


    def test_from_env_always_enabled(self, monkeypatch):
        """from_env() should always set enabled=True."""
        monkeypatch.setenv("HONCHO_API_KEY", "env-test-key")

        cfg = HonchoClientConfig.from_env()

        assert cfg.api_key == "env-test-key"
        assert cfg.enabled is True



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


class TestLatencyFlagResolution:

    def test_host_block_wins(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_BASE_URL', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'queryRewrite': False,
            'firstTurnBaseWait': 3,
            'hosts': {'hermes': {
                'queryRewrite': True,
                'firstTurnBaseWait': 0,
                'firstTurnDialecticWait': 0.5,
            }},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.query_rewrite is True
        assert cfg.first_turn_base_wait == 0.0
        assert cfg.first_turn_dialectic_wait == 0.5

    def test_per_host_timeout_wins_over_global(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_TIMEOUT', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'timeout': 30,
            'hosts': {'hermes': {'timeout': 5}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.timeout == 5.0


class TestHonchoBaseUrlSanitize:
    def test_clean_base_url_accepted(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_BASE_URL', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'baseUrl': 'https://honcho.example.com',
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.base_url == 'https://honcho.example.com'

    def test_nonprintable_base_url_dropped(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_BASE_URL', raising=False)
        config_path = tmp_path / 'config.json'
        bad = 'https://honcho.example.com\x1b'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'baseUrl': bad,
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.base_url is None

    def test_env_nonprintable_dropped(self, monkeypatch):
        monkeypatch.setenv('HONCHO_BASE_URL', 'https://x.example\x1b')
        monkeypatch.delenv('HONCHO_API_KEY', raising=False)
        cfg = HonchoClientConfig.from_env()
        assert cfg.base_url is None


class TestProfileKeyIsolationWarning:
    """#36098 / #66125: a named-profile host block without apiKey does NOT
    inherit the default host's key (isolation by design), but the failure
    must be loud, not silent."""

    def test_keyless_profile_block_warns_when_default_has_key(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.delenv('HONCHO_API_KEY', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'hosts': {
                'hermes': {'apiKey': 'shared-key'},
                'hermes_coder': {'baseUrl': 'http://192.168.1.50:8000'},
            },
        }))
        with caplog.at_level(logging.WARNING, logger='plugins.memory.honcho.client'):
            cfg = HonchoClientConfig.from_global_config(
                host='hermes_coder', config_path=config_path,
            )
        assert cfg.api_key is None  # isolation preserved — no silent inheritance
        assert any('NOT inherited' in r.message for r in caplog.records)

    def test_no_warning_when_profile_block_has_key(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.delenv('HONCHO_API_KEY', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'hosts': {
                'hermes': {'apiKey': 'shared-key'},
                'hermes_coder': {'apiKey': 'coder-key'},
            },
        }))
        with caplog.at_level(logging.WARNING, logger='plugins.memory.honcho.client'):
            cfg = HonchoClientConfig.from_global_config(
                host='hermes_coder', config_path=config_path,
            )
        assert cfg.api_key == 'coder-key'
        assert not any('NOT inherited' in r.message for r in caplog.records)

    def test_no_warning_for_default_host(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.delenv('HONCHO_API_KEY', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'hosts': {'hermes': {'baseUrl': 'http://localhost:8000'}},
        }))
        with caplog.at_level(logging.WARNING, logger='plugins.memory.honcho.client'):
            HonchoClientConfig.from_global_config(
                host='hermes', config_path=config_path,
            )
        assert not any('NOT inherited' in r.message for r in caplog.records)
