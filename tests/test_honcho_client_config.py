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


class TestLatencyFlagResolution:
    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_BASE_URL', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({'apiKey': 'k'}))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.query_rewrite is False
        assert cfg.first_turn_base_wait == 3.0
        assert cfg.first_turn_dialectic_wait == 2.0

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

    def test_timeout_falls_back_to_global(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_TIMEOUT', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({'apiKey': 'k', 'timeout': 30}))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.timeout == 30.0


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
        bad = 'https://honcho.example.com'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'baseUrl': bad,
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.base_url is None

    def test_env_nonprintable_dropped(self, monkeypatch):
        monkeypatch.setenv('HONCHO_BASE_URL', 'https://x.example')
        monkeypatch.delenv('HONCHO_API_KEY', raising=False)
        cfg = HonchoClientConfig.from_env()
        assert cfg.base_url is None

    def test_config_yaml_override_uses_sanitize_helper(self, monkeypatch):
        """Regression: config.yaml path must call `_sanitize_url`, not undefined `sanitize_url`."""
        import plugins.memory.honcho.client as client_mod

        calls = []
        real = client_mod._sanitize_url

        def spy(url):
            calls.append(url)
            return real(url)

        monkeypatch.setattr(client_mod, '_sanitize_url', spy)

        # Force override path: no base_url on config, provide control-char in config.yaml
        cfg = client_mod.HonchoClientConfig(api_key='k', base_url=None, enabled=True)
        bad = 'https://yaml.example.com\x1b'

        import sys, types
        config_mod = types.ModuleType('hermes_cli.config')
        config_mod.load_config = lambda: {'honcho': {'base_url': bad}}
        hermes_cli = types.ModuleType('hermes_cli')
        monkeypatch.setitem(sys.modules, 'hermes_cli', hermes_cli)
        monkeypatch.setitem(sys.modules, 'hermes_cli.config', config_mod)

        class FakeHoncho:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Inject FakeHoncho via import path used inside _build
        honcho_mod = types.ModuleType('honcho')
        honcho_mod.Honcho = FakeHoncho
        monkeypatch.setitem(sys.modules, 'honcho', honcho_mod)

        # reset singleton if any
        if hasattr(client_mod, '_client'):
            try:
                client_mod._client = None
            except Exception:
                pass

        client = client_mod.get_honcho_client(cfg)
        assert calls, 'expected _sanitize_url to be invoked for config.yaml base_url'
        assert any(c == bad for c in calls)
        # non-printable dropped => base_url key not set or None when passed
        if hasattr(client, 'kwargs'):
            assert client.kwargs.get('base_url') in (None, '')
