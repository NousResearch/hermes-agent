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

    def test_api_key_file_resolves_relative_to_config_file(self, tmp_path, monkeypatch):
        """apiKeyFile keeps scoped JWTs out of reviewable honcho.json."""
        key_file = tmp_path / "secrets" / "ops.jwt"
        key_file.parent.mkdir()
        key_file.write_text(" file-backed-key \n")
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "baseUrl": "https://honcho.example.test",
            "hosts": {
                "hermes": {
                    "workspace": "ops-prod",
                    "apiKeyFile": "secrets/ops.jwt",
                }
            },
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "file-backed-key"
        assert cfg.workspace_id == "ops-prod"
        assert cfg.enabled is True

    def test_named_compartments_parse_to_separate_workspace_key_pairs(self, tmp_path, monkeypatch):
        """Milo/Miloh need ops and personal compartments at key/workspace level."""
        ops_key = tmp_path / "ops.jwt"
        personal_key = tmp_path / "personal.jwt"
        ops_key.write_text("ops-key\n")
        personal_key.write_text("personal-key\n")
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "baseUrl": "https://honcho.example.test",
            "hosts": {
                "hermes": {
                    "workspace": "personal-prod",
                    "apiKeyFile": "personal.jwt",
                    "compartments": {
                        "ops": {
                            "workspace": "ops-prod",
                            "apiKeyFile": "ops.jwt",
                            "write": True,
                        },
                        "personal": {
                            "workspace": "personal-prod",
                            "apiKeyFile": "personal.jwt",
                            "write": True,
                        },
                    },
                    "agents": {
                        "miloh": ["ops", "personal"],
                        "echo": ["ops"],
                    },
                }
            },
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert sorted(cfg.compartments) == ["ops", "personal"]
        assert cfg.agent_compartments == {"miloh": ["ops", "personal"], "echo": ["ops"]}

        ops_cfg = cfg.for_compartment("ops")
        personal_cfg = cfg.for_compartment("personal")

        assert ops_cfg.workspace_id == "ops-prod"
        assert ops_cfg.api_key == "ops-key"
        assert ops_cfg.base_url == "https://honcho.example.test"
        assert ops_cfg.host == "hermes:ops"

        assert personal_cfg.workspace_id == "personal-prod"
        assert personal_cfg.api_key == "personal-key"
        assert personal_cfg.host == "hermes:personal"

    def test_compartment_without_key_inherits_active_host_key(self, tmp_path, monkeypatch):
        """CLI-listed inherited/default auth should inherit the active host key."""
        host_key = tmp_path / "host.jwt"
        host_key.write_text("host-key\n")
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "baseUrl": "https://honcho.example.test",
            "apiKey": "root-key",
            "hosts": {
                "hermes": {
                    "workspace": "personal-prod",
                    "apiKeyFile": "host.jwt",
                    "compartments": {
                        "ops": {"workspace": "ops-prod"},
                    },
                }
            },
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        ops_cfg = cfg.for_compartment("ops")

        assert cfg.api_key == "host-key"
        assert ops_cfg.workspace_id == "ops-prod"
        assert ops_cfg.api_key == "host-key"
        assert ops_cfg.api_key_explicit is False


def test_honcho_config_schema_documents_file_backed_compartments():
    provider = HonchoMemoryProvider()

    keys = {item["key"] for item in provider.get_config_schema()}

    assert "api_key" in keys
    assert "apiKeyFile" in keys
    assert "baseUrl" in keys
    assert "compartments" in keys


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
