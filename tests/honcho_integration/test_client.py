"""Tests for honcho_integration/client.py — Honcho client configuration."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from honcho_integration.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
    GLOBAL_CONFIG_PATH,
    HOST,
)


class TestHonchoClientConfigDefaults:
    def test_default_values(self):
        config = HonchoClientConfig()
        assert config.host == "hermes"
        assert config.workspace_id == "hermes"
        assert config.api_key is None
        assert config.environment == "production"
        assert config.enabled is False
        assert config.save_messages is True
        assert config.session_strategy == "per-directory"
        assert config.session_peer_prefix is False
        assert config.linked_hosts == []
        assert config.sessions == {}


class TestFromEnv:
    def test_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"HONCHO_API_KEY": "test-key-123"}):
            config = HonchoClientConfig.from_env()
        assert config.api_key == "test-key-123"
        assert config.enabled is True

    def test_reads_environment_from_env(self):
        with patch.dict(os.environ, {
            "HONCHO_API_KEY": "key",
            "HONCHO_ENVIRONMENT": "staging",
        }):
            config = HonchoClientConfig.from_env()
        assert config.environment == "staging"

    def test_defaults_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove HONCHO_API_KEY if it exists
            os.environ.pop("HONCHO_API_KEY", None)
            os.environ.pop("HONCHO_ENVIRONMENT", None)
            config = HonchoClientConfig.from_env()
        assert config.api_key is None
        assert config.environment == "production"

    def test_custom_workspace(self):
        config = HonchoClientConfig.from_env(workspace_id="custom")
        assert config.workspace_id == "custom"


class TestFromGlobalConfig:
    def test_missing_config_falls_back_to_env(self, tmp_path):
        config = HonchoClientConfig.from_global_config(
            config_path=tmp_path / "nonexistent.json"
        )
        # Should fall back to from_env
        assert config.enabled is True or config.api_key is None  # depends on env

    def test_reads_full_config(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "my-honcho-key",
            "workspace": "my-workspace",
            "environment": "staging",
            "peerName": "alice",
            "aiPeer": "hermes-custom",
            "enabled": True,
            "saveMessages": False,
            "contextTokens": 2000,
            "sessionStrategy": "per-project",
            "sessionPeerPrefix": True,
            "sessions": {"/home/user/proj": "my-session"},
            "hosts": {
                "hermes": {
                    "workspace": "override-ws",
                    "aiPeer": "override-ai",
                    "linkedHosts": ["cursor"],
                }
            }
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.api_key == "my-honcho-key"
        # Host block workspace overrides root workspace
        assert config.workspace_id == "override-ws"
        assert config.ai_peer == "override-ai"
        assert config.linked_hosts == ["cursor"]
        assert config.environment == "staging"
        assert config.peer_name == "alice"
        assert config.enabled is True
        assert config.save_messages is False
        assert config.session_strategy == "per-project"
        assert config.session_peer_prefix is True

    def test_host_block_overrides_root(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "workspace": "root-ws",
            "aiPeer": "root-ai",
            "hosts": {
                "hermes": {
                    "workspace": "host-ws",
                    "aiPeer": "host-ai",
                }
            }
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.workspace_id == "host-ws"
        assert config.ai_peer == "host-ai"

    def test_root_fields_used_when_no_host_block(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "workspace": "root-ws",
            "aiPeer": "root-ai",
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.workspace_id == "root-ws"
        assert config.ai_peer == "root-ai"

    def test_corrupt_config_falls_back_to_env(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json{{{")

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        # Should fall back to from_env without crashing
        assert isinstance(config, HonchoClientConfig)

    def test_api_key_env_fallback(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"enabled": True}))

        with patch.dict(os.environ, {"HONCHO_API_KEY": "env-key"}):
            config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.api_key == "env-key"


class TestResolveSessionName:
    def test_manual_override(self):
        config = HonchoClientConfig(sessions={"/home/user/proj": "custom-session"})
        assert config.resolve_session_name("/home/user/proj") == "custom-session"

    def test_derive_from_dirname(self):
        config = HonchoClientConfig()
        result = config.resolve_session_name("/home/user/my-project")
        assert result == "my-project"

    def test_peer_prefix(self):
        config = HonchoClientConfig(peer_name="alice", session_peer_prefix=True)
        result = config.resolve_session_name("/home/user/proj")
        assert result == "alice-proj"

    def test_no_peer_prefix_when_no_peer_name(self):
        config = HonchoClientConfig(session_peer_prefix=True)
        result = config.resolve_session_name("/home/user/proj")
        assert result == "proj"

    def test_default_cwd(self):
        config = HonchoClientConfig()
        result = config.resolve_session_name()
        # Should use os.getcwd() basename
        assert result == Path.cwd().name


class TestGetLinkedWorkspaces:
    def test_resolves_linked_hosts(self):
        config = HonchoClientConfig(
            workspace_id="hermes-ws",
            linked_hosts=["cursor", "windsurf"],
            raw={
                "hosts": {
                    "cursor": {"workspace": "cursor-ws"},
                    "windsurf": {"workspace": "windsurf-ws"},
                }
            },
        )
        workspaces = config.get_linked_workspaces()
        assert "cursor-ws" in workspaces
        assert "windsurf-ws" in workspaces

    def test_excludes_own_workspace(self):
        config = HonchoClientConfig(
            workspace_id="hermes-ws",
            linked_hosts=["other"],
            raw={"hosts": {"other": {"workspace": "hermes-ws"}}},
        )
        workspaces = config.get_linked_workspaces()
        assert workspaces == []

    def test_uses_host_key_as_fallback(self):
        config = HonchoClientConfig(
            workspace_id="hermes-ws",
            linked_hosts=["cursor"],
            raw={"hosts": {"cursor": {}}},  # no workspace field
        )
        workspaces = config.get_linked_workspaces()
        assert "cursor" in workspaces


class TestResetHonchoClient:
    def test_reset_clears_singleton(self):
        import honcho_integration.client as mod
        mod._honcho_client = MagicMock()
        assert mod._honcho_client is not None
        reset_honcho_client()
        assert mod._honcho_client is None
