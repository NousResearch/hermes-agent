"""Tests for Honcho CLI helpers."""

import json
from unittest.mock import patch

from honcho_integration.cli import _resolve_api_key, clone_honcho_for_profile


class TestResolveApiKey:
    def test_prefers_host_scoped_key(self):
        cfg = {
            "apiKey": "root-key",
            "hosts": {
                "hermes": {
                    "apiKey": "host-key",
                }
            },
        }
        assert _resolve_api_key(cfg) == "host-key"

    def test_falls_back_to_root_key(self):
        cfg = {
            "apiKey": "root-key",
            "hosts": {"hermes": {}},
        }
        assert _resolve_api_key(cfg) == "root-key"

    def test_falls_back_to_env_key(self, monkeypatch):
        monkeypatch.setenv("HONCHO_API_KEY", "env-key")
        assert _resolve_api_key({}) == "env-key"
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)


class TestCloneHonchoForProfile:
    def test_clones_default_settings_to_new_profile(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "test-key",
            "hosts": {
                "hermes": {
                    "peerName": "alice",
                    "memoryMode": "honcho",
                    "recallMode": "tools",
                    "writeFrequency": "turn",
                    "dialecticReasoningLevel": "medium",
                    "enabled": True,
                },
            },
        }))

        with patch("honcho_integration.cli._config_path", return_value=config_file):
            result = clone_honcho_for_profile("coder")

        assert result is True

        cfg = json.loads(config_file.read_text())
        new_block = cfg["hosts"]["hermes.coder"]
        assert new_block["peerName"] == "alice"
        assert new_block["memoryMode"] == "honcho"
        assert new_block["recallMode"] == "tools"
        assert new_block["writeFrequency"] == "turn"
        assert new_block["aiPeer"] == "hermes.coder"
        assert new_block["workspace"] == "hermes"  # shared, not profile-derived
        assert new_block["enabled"] is True

    def test_skips_when_no_honcho_configured(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        with patch("honcho_integration.cli._config_path", return_value=config_file):
            result = clone_honcho_for_profile("coder")

        assert result is False

    def test_skips_when_host_block_already_exists(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "hosts": {
                "hermes": {"peerName": "alice"},
                "hermes.coder": {"peerName": "existing"},
            },
        }))

        with patch("honcho_integration.cli._config_path", return_value=config_file):
            result = clone_honcho_for_profile("coder")

        assert result is False
        cfg = json.loads(config_file.read_text())
        assert cfg["hosts"]["hermes.coder"]["peerName"] == "existing"

    def test_inherits_peer_name_from_root_when_not_in_host(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "peerName": "root-alice",
            "hosts": {"hermes": {}},
        }))

        with patch("honcho_integration.cli._config_path", return_value=config_file):
            clone_honcho_for_profile("dreamer")

        cfg = json.loads(config_file.read_text())
        assert cfg["hosts"]["hermes.dreamer"]["peerName"] == "root-alice"

    def test_works_with_api_key_only_no_host_block(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "key"}))

        with patch("honcho_integration.cli._config_path", return_value=config_file):
            result = clone_honcho_for_profile("coder")

        assert result is True
        cfg = json.loads(config_file.read_text())
        assert cfg["hosts"]["hermes.coder"]["aiPeer"] == "hermes.coder"
        assert cfg["hosts"]["hermes.coder"]["workspace"] == "hermes"  # shared

