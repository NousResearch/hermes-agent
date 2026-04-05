"""Tests for plugins/memory/honcho/client.py — Honcho client configuration."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
    resolve_active_host,
    resolve_config_path,
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
        assert config.recall_mode == "hybrid"
        assert config.session_peer_prefix is False
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

    def test_reads_base_url_from_env(self):
        with patch.dict(os.environ, {"HONCHO_BASE_URL": "http://localhost:8000"}, clear=False):
            config = HonchoClientConfig.from_env()
        assert config.base_url == "http://localhost:8000"
        assert config.enabled is True

    def test_enabled_without_api_key_when_base_url_set(self):
        """base_url alone (no API key) is sufficient to enable a local instance."""
        with patch.dict(os.environ, {"HONCHO_BASE_URL": "http://localhost:8000"}, clear=False):
            os.environ.pop("HONCHO_API_KEY", None)
            config = HonchoClientConfig.from_env()
        assert config.api_key is None
        assert config.base_url == "http://localhost:8000"
        assert config.enabled is True


class TestFromGlobalConfig:
    def test_missing_config_falls_back_to_env(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            config = HonchoClientConfig.from_global_config(
                config_path=tmp_path / "nonexistent.json"
            )
        # Should fall back to from_env
        assert config.enabled is False
        assert config.api_key is None

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
                }
            }
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.api_key == "my-honcho-key"
        # Host block workspace overrides root workspace
        assert config.workspace_id == "override-ws"
        assert config.ai_peer == "override-ai"
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

    def test_session_strategy_default_from_global_config(self, tmp_path):
        """from_global_config with no sessionStrategy should match dataclass default."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "key"}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.session_strategy == "per-directory"

    def test_context_tokens_host_block_wins(self, tmp_path):
        """Host block contextTokens should override root."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "contextTokens": 1000,
            "hosts": {"hermes": {"contextTokens": 2000}},
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.context_tokens == 2000

    def test_recall_mode_from_config(self, tmp_path):
        """recallMode is read from config, host block wins."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "recallMode": "tools",
            "hosts": {"hermes": {"recallMode": "context"}},
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.recall_mode == "context"

    def test_recall_mode_default(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "key"}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.recall_mode == "hybrid"

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

    def test_base_url_env_fallback(self, tmp_path):
        """HONCHO_BASE_URL env var is used when no baseUrl in config JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"workspace": "local"}))

        with patch.dict(os.environ, {"HONCHO_BASE_URL": "http://localhost:8000"}, clear=False):
            config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.base_url == "http://localhost:8000"
        assert config.enabled is True

    def test_base_url_from_config_root(self, tmp_path):
        """baseUrl in config root is read and takes precedence over env var."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"baseUrl": "http://config-host:9000"}))

        with patch.dict(os.environ, {"HONCHO_BASE_URL": "http://localhost:8000"}, clear=False):
            config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.base_url == "http://config-host:9000"

    def test_base_url_not_read_from_host_block(self, tmp_path):
        """baseUrl is a root-level connection setting, not overridable per-host (consistent with apiKey)."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "baseUrl": "http://root:9000",
            "hosts": {"hermes": {"baseUrl": "http://host-block:9001"}},
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.base_url == "http://root:9000"


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

    def test_per_repo_uses_git_root(self):
        config = HonchoClientConfig(session_strategy="per-repo")
        with patch.object(
            HonchoClientConfig, "_git_repo_name", return_value="hermes-agent"
        ):
            result = config.resolve_session_name("/home/user/hermes-agent/subdir")
        assert result == "hermes-agent"

    def test_per_repo_with_peer_prefix(self):
        config = HonchoClientConfig(
            session_strategy="per-repo", peer_name="eri", session_peer_prefix=True
        )
        with patch.object(
            HonchoClientConfig, "_git_repo_name", return_value="groudon"
        ):
            result = config.resolve_session_name("/home/user/groudon/src")
        assert result == "eri-groudon"

    def test_per_repo_falls_back_to_dirname_outside_git(self):
        config = HonchoClientConfig(session_strategy="per-repo")
        with patch.object(
            HonchoClientConfig, "_git_repo_name", return_value=None
        ):
            result = config.resolve_session_name("/home/user/not-a-repo")
        assert result == "not-a-repo"

    def test_per_repo_manual_override_still_wins(self):
        config = HonchoClientConfig(
            session_strategy="per-repo",
            sessions={"/home/user/proj": "custom-session"},
        )
        result = config.resolve_session_name("/home/user/proj")
        assert result == "custom-session"


class TestResolveConfigPath:
    def test_prefers_hermes_home_when_exists(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        local_cfg = hermes_home / "honcho.json"
        local_cfg.write_text('{"apiKey": "local"}')

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            result = resolve_config_path()
        assert result == local_cfg

    def test_falls_back_to_global_when_no_local(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        # No honcho.json in HERMES_HOME — also isolate ~/.hermes so
        # the default-profile fallback doesn't hit the real filesystem.
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}), \
             patch.object(Path, "home", return_value=fake_home):
            result = resolve_config_path()
        assert result == GLOBAL_CONFIG_PATH

    def test_falls_back_to_global_without_hermes_home_env(self, tmp_path):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(Path, "home", return_value=fake_home):
            os.environ.pop("HERMES_HOME", None)
            result = resolve_config_path()
        assert result == GLOBAL_CONFIG_PATH

    def test_from_global_config_uses_local_path(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        local_cfg = hermes_home / "honcho.json"
        local_cfg.write_text(json.dumps({
            "apiKey": "local-key",
            "workspace": "local-ws",
        }))

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            config = HonchoClientConfig.from_global_config()
        assert config.api_key == "local-key"
        assert config.workspace_id == "local-ws"


class TestResolveActiveHost:
    def test_default_returns_hermes(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            os.environ.pop("HERMES_HOME", None)
            assert resolve_active_host() == "hermes"

    def test_explicit_env_var_wins(self):
        with patch.dict(os.environ, {"HERMES_HONCHO_HOST": "hermes.coder"}):
            assert resolve_active_host() == "hermes.coder"

    def test_profile_name_derives_host(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="coder"):
                assert resolve_active_host() == "hermes.coder"

    def test_default_profile_returns_hermes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
                assert resolve_active_host() == "hermes"

    def test_custom_profile_returns_hermes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="custom"):
                assert resolve_active_host() == "hermes"

    def test_profiles_import_failure_falls_back(self):
        import sys
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            # Temporarily remove hermes_cli.profiles to simulate import failure
            saved = sys.modules.get("hermes_cli.profiles")
            sys.modules["hermes_cli.profiles"] = None  # type: ignore
            try:
                assert resolve_active_host() == "hermes"
            finally:
                if saved is not None:
                    sys.modules["hermes_cli.profiles"] = saved
                else:
                    sys.modules.pop("hermes_cli.profiles", None)


class TestProfileScopedConfig:
    def test_from_env_uses_profile_host(self):
        with patch.dict(os.environ, {"HONCHO_API_KEY": "key"}):
            config = HonchoClientConfig.from_env(host="hermes.coder")
        assert config.host == "hermes.coder"
        assert config.workspace_id == "hermes"  # shared workspace
        assert config.ai_peer == "hermes.coder"

    def test_from_env_default_workspace_preserved_for_default_host(self):
        with patch.dict(os.environ, {"HONCHO_API_KEY": "key"}):
            config = HonchoClientConfig.from_env(host="hermes")
        assert config.host == "hermes"
        assert config.workspace_id == "hermes"

    def test_from_global_config_reads_profile_host_block(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "shared-key",
            "hosts": {
                "hermes": {"aiPeer": "hermes", "peerName": "alice"},
                "hermes.coder": {
                    "aiPeer": "hermes.coder",
                    "peerName": "alice-coder",
                    "workspace": "coder-ws",
                },
            },
        }))
        config = HonchoClientConfig.from_global_config(
            host="hermes.coder", config_path=config_file,
        )
        assert config.host == "hermes.coder"
        assert config.workspace_id == "coder-ws"
        assert config.ai_peer == "hermes.coder"
        assert config.peer_name == "alice-coder"

    def test_from_global_config_auto_resolves_host(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "hosts": {
                "hermes.dreamer": {"peerName": "dreamer-user"},
            },
        }))
        with patch("plugins.memory.honcho.client.resolve_active_host", return_value="hermes.dreamer"):
            config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.host == "hermes.dreamer"
        assert config.peer_name == "dreamer-user"


class TestResetHonchoClient:
    def test_reset_clears_singleton(self):
        import plugins.memory.honcho.client as mod
        mod._honcho_client = MagicMock()
        assert mod._honcho_client is not None
        reset_honcho_client()
        assert mod._honcho_client is None
