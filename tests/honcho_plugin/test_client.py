"""Tests for plugins/memory/honcho/client.py — Honcho client configuration."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

from hermes_cli.profiles import _get_default_hermes_home

import pytest

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    profile_host_key,
    reset_honcho_client,
    resolve_active_host,
    resolve_config_path,
    resolve_global_config_path,
)


class TestHonchoClientConfigDefaults:
    def test_default_values(self):
        config = HonchoClientConfig()
        assert config.host == "hermes"
        assert config.workspace_id == "hermes"
        assert config.api_key is None
        assert config.environment == "production"
        assert config.timeout is None
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

    def test_reads_timeout_from_env(self):
        with patch.dict(os.environ, {"HONCHO_TIMEOUT": "90"}, clear=True):
            config = HonchoClientConfig.from_env()
        assert config.timeout == 90.0


class TestFromGlobalConfig:
    def test_missing_config_falls_back_to_env(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            config = HonchoClientConfig.from_global_config(
                config_path=tmp_path / "nonexistent.json"
            )
        # Should fall back to from_env
        assert config.enabled is False
        assert config.api_key is None

    def test_reads_full_config(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
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
        # Isolate from real ~/.hermes/honcho.json
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.api_key == "***"
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
        config_file.write_text(json.dumps({"apiKey": "***"}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.session_strategy == "per-directory"

    def test_context_tokens_default_is_none(self, tmp_path):
        """Default context_tokens should be None (uncapped) unless explicitly set."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***"}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.context_tokens is None

    def test_context_tokens_explicit_sets_cap(self, tmp_path):
        """Explicit contextTokens in config sets the cap."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "contextTokens": 1200}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.context_tokens == 1200

    def test_context_tokens_explicit_overrides_default(self, tmp_path):
        """Explicit contextTokens in config should override the default."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "contextTokens": 2000}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.context_tokens == 2000

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

    def test_timeout_from_config_root(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"timeout": 75}))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.timeout == 75.0

    def test_request_timeout_alias_from_config_root(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"requestTimeout": "82.5"}))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.timeout == 82.5


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

    def test_falls_back_to_default_profile_when_no_local(self, tmp_path, monkeypatch):
        # Profile mode: HERMES_HOME points at ~/.hermes/profiles/<name>, so
        # _get_default_hermes_home() must resolve back to ~/.hermes — that's
        # the bug the HOME-anchored helper fixes (vs. blindly using Path.home()).
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        default_home = fake_home / ".hermes"
        profile_home = default_home / "profiles" / "work"
        profile_home.mkdir(parents=True)
        default_cfg = default_home / "honcho.json"
        default_cfg.write_text('{"apiKey": "default-key"}')

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        result = resolve_config_path()

        assert _get_default_hermes_home() == default_home
        assert result == default_cfg

    def test_falls_back_to_global_without_hermes_home_env(self, tmp_path):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(Path, "home", return_value=fake_home):
            os.environ.pop("HERMES_HOME", None)
            result = resolve_config_path()
        assert result == fake_home / ".honcho" / "config.json"

    def test_global_fallback_uses_home_at_call_time(self, tmp_path):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}), \
             patch.object(Path, "home", return_value=fake_home):
            assert resolve_global_config_path() == fake_home / ".honcho" / "config.json"
            assert resolve_config_path() == fake_home / ".honcho" / "config.json"

    def test_from_global_config_uses_default_profile_fallback(self, tmp_path, monkeypatch):
        # Profile mode: from_global_config() reads the default-profile honcho.json
        # via the HOME-anchored helper, not Path.home() / ".hermes".
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        default_home = fake_home / ".hermes"
        profile_home = default_home / "profiles" / "work"
        profile_home.mkdir(parents=True)
        default_cfg = default_home / "honcho.json"
        default_cfg.write_text(json.dumps({
            "apiKey": "default-key",
            "workspace": "default-ws",
        }))

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        config = HonchoClientConfig.from_global_config()

        assert config.api_key == "default-key"
        assert config.workspace_id == "default-ws"

    def test_from_global_config_uses_local_path(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        local_cfg = hermes_home / "honcho.json"
        local_cfg.write_text(json.dumps({
            "apiKey": "***",
            "workspace": "local-ws",
        }))

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}), \
             patch.object(Path, "home", return_value=tmp_path):
            config = HonchoClientConfig.from_global_config()
        assert config.api_key == "***"
        assert config.workspace_id == "local-ws"


class TestResolveActiveHost:
    def test_profile_host_key_uses_honcho_safe_separator(self):
        assert profile_host_key("coder") == "hermes_coder"
        assert profile_host_key("default") == "hermes"

    def test_default_returns_hermes(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            os.environ.pop("HERMES_HOME", None)
            with patch(
                "plugins.memory.honcho.client.resolve_config_path",
                return_value=Path("/nonexistent/honcho.json"),
            ):
                assert resolve_active_host() == "hermes"

    def test_explicit_env_var_wins(self):
        with patch.dict(os.environ, {"HERMES_HONCHO_HOST": "hermes.coder"}):
            assert resolve_active_host() == "hermes.coder"

    def test_profile_name_derives_host(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="coder"):
                assert resolve_active_host() == "hermes_coder"

    def test_default_host_does_not_override_named_profile(self, tmp_path):
        """defaultHost is not applied before active-profile resolution."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "defaultHost": "local",
            "hosts": {"local": {"workspace": "local-ws"}},
        }))

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="coder"), \
                 patch("plugins.memory.honcho.client.resolve_config_path", return_value=config_file):
                assert resolve_active_host() == "hermes_coder"

    def test_default_host_applies_to_default_profile_only(self, tmp_path):
        """default profile can use setup-generated defaultHost without leaking to other profiles."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "defaultHost": "local",
            "hosts": {"local": {"workspace": "local-ws"}},
        }))

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
                 patch("plugins.memory.honcho.client.resolve_config_path", return_value=config_file):
                assert resolve_active_host() == "local"

    def test_default_profile_returns_hermes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
                 patch(
                     "plugins.memory.honcho.client.resolve_config_path",
                     return_value=Path("/nonexistent/honcho.json"),
                 ):
                assert resolve_active_host() == "hermes"

    def test_custom_profile_returns_hermes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HONCHO_HOST", None)
            with patch("hermes_cli.profiles.get_active_profile_name", return_value="custom"), \
                 patch(
                     "plugins.memory.honcho.client.resolve_config_path",
                     return_value=Path("/nonexistent/honcho.json"),
                 ):
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
            config = HonchoClientConfig.from_env(host="hermes_coder")
        assert config.host == "hermes_coder"
        assert config.workspace_id == "hermes"  # shared workspace
        assert config.ai_peer == "hermes_coder"

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
                "hermes_coder": {
                    "aiPeer": "hermes_coder",
                    "peerName": "alice-coder",
                    "workspace": "coder-ws",
                },
            },
        }))
        config = HonchoClientConfig.from_global_config(
            host="hermes_coder", config_path=config_file,
        )
        assert config.host == "hermes_coder"
        assert config.workspace_id == "coder-ws"
        assert config.ai_peer == "hermes_coder"
        assert config.peer_name == "alice-coder"

    def test_from_global_config_auto_resolves_host(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "hosts": {
                "hermes_dreamer": {"peerName": "dreamer-user"},
            },
        }))
        with patch("plugins.memory.honcho.client.resolve_active_host", return_value="hermes_dreamer"):
            config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.host == "hermes_dreamer"
        assert config.peer_name == "dreamer-user"

    def test_from_global_config_reads_legacy_dot_profile_host_block(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "key",
            "hosts": {
                "hermes.dreamer": {"peerName": "dreamer-user"},
            },
        }))
        config = HonchoClientConfig.from_global_config(
            host="hermes_dreamer",
            config_path=config_file,
        )
        assert config.host == "hermes_dreamer"
        assert config.peer_name == "dreamer-user"
        assert config.workspace_id == "hermes_dreamer"


class TestObservationModeMigration:
    """Existing configs without explicit observationMode keep 'unified' default."""

    def test_existing_config_defaults_to_unified(self, tmp_path):
        """Config with host block but no observationMode → 'unified' (old default)."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "hosts": {"hermes": {"enabled": True, "aiPeer": "hermes"}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.observation_mode == "unified"

    def test_new_config_defaults_to_directional(self, tmp_path):
        """Config with no host block and no credentials → 'directional' (new default)."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.observation_mode == "directional"

    def test_explicit_directional_respected(self, tmp_path):
        """Existing config with explicit observationMode → uses what's set."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "hosts": {"hermes": {"enabled": True, "observationMode": "directional"}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.observation_mode == "directional"

    def test_explicit_unified_respected(self, tmp_path):
        """Existing config with explicit observationMode unified → stays unified."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "observationMode": "unified",
            "hosts": {"hermes": {"enabled": True}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.observation_mode == "unified"

    def test_granular_observation_overrides_preset(self, tmp_path):
        """Explicit observation object overrides both preset and migration default."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "hosts": {"hermes": {
                "enabled": True,
                "observation": {
                    "user": {"observeMe": True, "observeOthers": False},
                    "ai": {"observeMe": False, "observeOthers": True},
                },
            }},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        # observation_mode falls back to "unified" (migration), but
        # granular booleans from the observation object win
        assert cfg.user_observe_me is True
        assert cfg.user_observe_others is False
        assert cfg.ai_observe_me is False
        assert cfg.ai_observe_others is True


class TestGetHonchoClient:
    def teardown_method(self):
        reset_honcho_client()

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_passes_timeout_from_config(self):
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="test-key",
            timeout=91.0,
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho:
            client = get_honcho_client(cfg)

        assert client is fake_honcho
        mock_honcho.assert_called_once()
        assert mock_honcho.call_args.kwargs["timeout"] == 91.0

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_hermes_config_timeout_override_used_when_config_timeout_missing(self):
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="test-key",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={"honcho": {"timeout": 88}}):
            client = get_honcho_client(cfg)

        assert client is fake_honcho
        mock_honcho.assert_called_once()
        assert mock_honcho.call_args.kwargs["timeout"] == 88.0

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_defaults_to_30s_when_no_timeout_configured(self):
        from plugins.memory.honcho.client import _DEFAULT_HTTP_TIMEOUT

        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="test-key",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            client = get_honcho_client(cfg)

        assert client is fake_honcho
        mock_honcho.assert_called_once()
        assert mock_honcho.call_args.kwargs["timeout"] == _DEFAULT_HTTP_TIMEOUT

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_hermes_request_timeout_alias_used(self):
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="test-key",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={"honcho": {"request_timeout": "77.5"}}):
            client = get_honcho_client(cfg)

        assert client is fake_honcho
        mock_honcho.assert_called_once()
        assert mock_honcho.call_args.kwargs["timeout"] == 77.5

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_timeout_change_triggers_client_rebuild(self):
        """Changing timeout config must rebuild the cached client."""
        fake_honcho_1 = MagicMock(name="Honcho_v1")
        fake_honcho_2 = MagicMock(name="Honcho_v2")
        cfg = HonchoClientConfig(
            api_key="test-key",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho_1) as mock_h1, \
             patch("hermes_cli.config.load_config", return_value={"honcho": {"timeout": 30}}):
            client1 = get_honcho_client(cfg)

        assert client1 is fake_honcho_1
        assert mock_h1.call_args.kwargs["timeout"] == 30.0

        # Same config — should return cached client (no rebuild)
        with patch("honcho.Honcho", return_value=fake_honcho_2) as mock_h2, \
             patch("hermes_cli.config.load_config", return_value={"honcho": {"timeout": 30}}):
            client2 = get_honcho_client(cfg)

        assert client2 is fake_honcho_1  # still cached
        mock_h2.assert_not_called()

        # Changed timeout — must rebuild
        with patch("honcho.Honcho", return_value=fake_honcho_2) as mock_h3, \
             patch("hermes_cli.config.load_config", return_value={"honcho": {"timeout": 300}}):
            client3 = get_honcho_client(cfg)

        assert client3 is fake_honcho_2  # rebuilt
        mock_h3.assert_called_once()
        assert mock_h3.call_args.kwargs["timeout"] == 300.0


class TestResolveSessionNameGatewayKey:
    """Regression tests for gateway_session_key priority in resolve_session_name.

    Ensures gateway platforms get stable per-chat Honcho sessions even when
    sessionStrategy=per-session would otherwise create ephemeral sessions.
    Regression: plugin refactor 924bc67e dropped gateway key plumbing.
    """

    def test_gateway_key_overrides_per_session_strategy(self):
        """gateway_session_key must win over per-session session_id."""
        config = HonchoClientConfig(session_strategy="per-session")
        result = config.resolve_session_name(
            session_id="20260412_171002_69bb38",
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563"

    def test_gateway_key_not_remapped_by_title(self):
        """A title never remaps a stable identifier — the gateway per-chat key
        wins over the title so a generated title can't split a live conversation
        onto a new Honcho session."""
        config = HonchoClientConfig(session_strategy="per-session")
        result = config.resolve_session_name(
            session_title="my-custom-title",
            session_id="20260412_171002_69bb38",
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563"

    def test_per_session_fallback_without_gateway_key(self):
        """Without gateway_session_key, per-session returns session_id (CLI path)."""
        config = HonchoClientConfig(session_strategy="per-session")
        result = config.resolve_session_name(
            session_id="20260412_171002_69bb38",
            gateway_session_key=None,
        )
        assert result == "20260412_171002_69bb38"

    def test_gateway_key_sanitizes_special_chars(self):
        """Colons and other non-alphanumeric chars are replaced with hyphens."""
        config = HonchoClientConfig()
        result = config.resolve_session_name(
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563"
        assert ":" not in result


class TestResolveSessionNameLengthLimit:
    """Regression tests for Honcho's 100-char session ID limit (issue #13868).

    Long gateway session keys (Matrix room+event IDs, Telegram supergroup
    reply chains, Slack thread IDs with long workspace prefixes) can overflow
    Honcho's 100-char session_id limit after sanitization. Before this fix,
    every Honcho API call for those sessions 400'd with "session_id too long".
    """

    HONCHO_MAX = 100

    def test_short_gateway_key_unchanged(self):
        """Short keys must not get a hash suffix appended."""
        config = HonchoClientConfig()
        result = config.resolve_session_name(
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        # Unchanged fast-path: sanitize only, no truncation, no hash suffix.
        assert result == "agent-main-telegram-dm-8439114563"
        assert len(result) <= self.HONCHO_MAX

    def test_key_at_exact_limit_unchanged(self):
        """A sanitized key that is exactly 100 chars must be returned as-is."""
        key = "a" * self.HONCHO_MAX
        config = HonchoClientConfig()
        result = config.resolve_session_name(gateway_session_key=key)
        assert result == key
        assert len(result) == self.HONCHO_MAX

    def test_long_gateway_key_truncated_to_limit(self):
        """An over-limit sanitized key must truncate to exactly 100 chars."""
        key = "!roomid:matrix.example.org|" + "$event_" + ("a" * 300)
        config = HonchoClientConfig()
        result = config.resolve_session_name(gateway_session_key=key)
        assert result is not None
        assert len(result) == self.HONCHO_MAX

    def test_truncation_is_deterministic(self):
        """Same long key must always produce the same truncated session ID."""
        key = "matrix-" + ("a" * 300)
        config = HonchoClientConfig()
        first = config.resolve_session_name(gateway_session_key=key)
        second = config.resolve_session_name(gateway_session_key=key)
        assert first == second

    def test_truncated_result_respects_char_allowlist(self):
        """Truncated result must still match Honcho's [a-zA-Z0-9_-] allowlist."""
        import re
        key = "slack:T12345:thread-reply:" + ("x" * 300) + ":with:colons:and:slashes/here"
        config = HonchoClientConfig()
        result = config.resolve_session_name(gateway_session_key=key)
        assert result is not None
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", result)

    def test_distinct_long_keys_do_not_collide(self):
        """Two long keys sharing a prefix must produce different truncated IDs."""
        prefix = "matrix:!room:example.org|" + "a" * 200
        key_a = prefix + "-suffix-alpha"
        key_b = prefix + "-suffix-beta"
        config = HonchoClientConfig()
        result_a = config.resolve_session_name(gateway_session_key=key_a)
        result_b = config.resolve_session_name(gateway_session_key=key_b)
        assert result_a != result_b
        assert len(result_a) == self.HONCHO_MAX
        assert len(result_b) == self.HONCHO_MAX

    def test_truncated_result_has_hash_suffix(self):
        """Truncated IDs must end with '-<8 hex chars>' for collision resistance."""
        import re
        key = "matrix-" + ("a" * 300)
        config = HonchoClientConfig()
        result = config.resolve_session_name(gateway_session_key=key)
        # Last 9 chars: '-' + 8 hex chars.
        assert re.search(r"-[0-9a-f]{8}$", result)


class TestResetHonchoClient:
    def test_reset_clears_singleton(self):
        import plugins.memory.honcho.client as mod

        # Seed the cached client through the slot's public surface, then
        # verify reset_honcho_client() clears it. (The client is cached in
        # mod._honcho_client_slot, a thread-safe SingletonSlot, not a bare
        # module global anymore — see #24759.)
        mod._honcho_client_slot.get(lambda: MagicMock())
        assert mod._honcho_client_slot.peek() is not None
        reset_honcho_client()
        assert mod._honcho_client_slot.peek() is None


class TestDialecticDepthParsing:
    """Tests for _parse_dialectic_depth and _parse_dialectic_depth_levels."""

    def test_default_depth_is_1(self, tmp_path):
        """Default dialecticDepth should be 1."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***"}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth == 1

    def test_depth_from_root(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "dialecticDepth": 2}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth == 2

    def test_depth_host_block_wins(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "dialecticDepth": 1,
            "hosts": {"hermes": {"dialecticDepth": 3}},
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth == 3

    def test_depth_clamped_high(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "dialecticDepth": 10}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth == 3

    def test_depth_clamped_low(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "dialecticDepth": -1}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth == 1

    def test_depth_levels_default_none(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***"}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth_levels is None

    def test_depth_levels_from_config(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "dialecticDepth": 2,
            "dialecticDepthLevels": ["minimal", "high"],
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth_levels == ["minimal", "high"]

    def test_depth_levels_padded_if_short(self, tmp_path):
        """Array shorter than depth gets padded with 'low'."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "dialecticDepth": 3,
            "dialecticDepthLevels": ["high"],
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth_levels == ["high", "low", "low"]

    def test_depth_levels_truncated_if_long(self, tmp_path):
        """Array longer than depth gets truncated."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "dialecticDepth": 1,
            "dialecticDepthLevels": ["high", "max", "medium"],
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth_levels == ["high"]

    def test_depth_levels_invalid_values_default_to_low(self, tmp_path):
        """Invalid reasoning levels in the array fall back to 'low'."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "dialecticDepth": 2,
            "dialecticDepthLevels": ["invalid", "high"],
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth_levels == ["low", "high"]


class TestGetHonchoClientBaseUrlDoublePrefixFix:
    """Regression tests for #20688 — Honcho SDK double-prefixing of /v3 for
    self-hosted instances where base_url already contains a version path."""

    def teardown_method(self):
        reset_honcho_client()

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_local_base_url_with_v3_suffix_stripped(self):
        """base_url 'http://localhost:38000/v3' must become 'http://localhost:38000'
        before passing to the Honcho SDK to avoid double '/v3/v3' prefixing."""
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key=None,
            base_url="http://localhost:38000/v3",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        passed_base_url = mock_honcho.call_args.kwargs.get("base_url")
        assert passed_base_url == "http://localhost:38000", (
            f"Expected 'http://localhost:38000', got {passed_base_url!r}"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_local_base_url_without_version_unchanged(self):
        """base_url 'http://localhost:38000' (no version) must be passed unchanged."""
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key=None,
            base_url="http://localhost:38000",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        passed_base_url = mock_honcho.call_args.kwargs.get("base_url")
        assert passed_base_url == "http://localhost:38000", (
            f"Expected 'http://localhost:38000', got {passed_base_url!r}"
        )

    def test_lan_default_host_empty_key_uses_local_placeholder(self, tmp_path):
        """Regression for #61661: setup-style root baseUrl + defaultHost + LAN IP
        must not pass an empty/None api_key to the SDK for a no-auth local server."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "defaultHost": "local",
            "baseUrl": "http://192.168.2.112:8000",
            "hosts": {
                "local": {
                    "workspace": "local-ws",
                    "aiPeer": "local-ai",
                    "apiKey": "",
                },
            },
        }))

        with patch.dict(os.environ, {}, clear=True), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("plugins.memory.honcho.client.resolve_config_path", return_value=config_file):
            cfg = HonchoClientConfig.from_global_config(config_path=config_file)

        assert cfg.host == "local"
        assert cfg.workspace_id == "local-ws"
        assert cfg.ai_peer == "local-ai"
        assert cfg.api_key is None
        assert cfg.base_url == "http://192.168.2.112:8000"

        fake_honcho = MagicMock(name="Honcho")
        mock_honcho = MagicMock(return_value=fake_honcho)
        fake_honcho_module = types.SimpleNamespace(Honcho=mock_honcho)
        with patch.dict(sys.modules, {"honcho": fake_honcho_module}), \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        assert mock_honcho.call_args.kwargs["api_key"] == "local"
        assert mock_honcho.call_args.kwargs["base_url"] == "http://192.168.2.112:8000"

    def test_lan_default_host_explicit_host_key_preserved(self, tmp_path):
        """A host-block local JWT still wins for LAN/VPN local URLs."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "defaultHost": "local",
            "baseUrl": "http://192.168.2.112:8000",
            "hosts": {
                "local": {
                    "workspace": "local-ws",
                    "aiPeer": "local-ai",
                    "apiKey": "local-jwt",
                },
            },
        }))

        with patch.dict(os.environ, {}, clear=True), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("plugins.memory.honcho.client.resolve_config_path", return_value=config_file):
            cfg = HonchoClientConfig.from_global_config(config_path=config_file)

        fake_honcho = MagicMock(name="Honcho")
        mock_honcho = MagicMock(return_value=fake_honcho)
        fake_honcho_module = types.SimpleNamespace(Honcho=mock_honcho)
        with patch.dict(sys.modules, {"honcho": fake_honcho_module}), \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        assert mock_honcho.call_args.kwargs["api_key"] == "local-jwt"

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_cloud_base_url_without_version_unchanged(self):
        """A cloud base_url with no version segment must pass through untouched."""
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="cloud-key",
            base_url="https://api.honcho.dev",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        passed_base_url = mock_honcho.call_args.kwargs.get("base_url")
        assert passed_base_url == "https://api.honcho.dev", (
            f"Expected 'https://api.honcho.dev', got {passed_base_url!r}"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_cloud_base_url_with_version_stripped(self):
        """A version segment double-prefixes regardless of host, so a cloud
        base_url that ends in '/v3' must also be stripped (the SDK re-adds it)."""
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="cloud-key",
            base_url="https://api.honcho.dev/v3",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        passed_base_url = mock_honcho.call_args.kwargs.get("base_url")
        assert passed_base_url == "https://api.honcho.dev", (
            f"Expected 'https://api.honcho.dev', got {passed_base_url!r}"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    @pytest.mark.parametrize(
        "raw_url, expected",
        [
            # LAN IP self-host
            ("http://10.0.0.5:8000/v3", "http://10.0.0.5:8000"),
            ("http://192.168.1.20:38000/v3/", "http://192.168.1.20:38000"),
            # Tailscale / custom-domain self-host
            ("https://honcho.my.ts.net/v3", "https://honcho.my.ts.net"),
            ("https://honcho.lab.internal/v3", "https://honcho.lab.internal"),
            ("https://honcho.fly.dev/v3", "https://honcho.fly.dev"),
            # higher version segments are also stripped
            ("https://honcho.lab.internal/v12", "https://honcho.lab.internal"),
            # self-host without a version segment is left unchanged
            ("https://honcho.my.ts.net", "https://honcho.my.ts.net"),
            ("http://10.0.0.5:8000", "http://10.0.0.5:8000"),
        ],
    )
    def test_self_hosted_base_url_version_stripped(self, raw_url, expected):
        """Non-loopback self-hosted instances (LAN IPs, Tailscale, custom
        domains) must get the same version-segment stripping as localhost.
        Regression for #20688 recurring on any non-loopback self-host."""
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key="self-host-key",
            base_url=raw_url,
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        passed_base_url = mock_honcho.call_args.kwargs.get("base_url")
        assert passed_base_url == expected, (
            f"Expected {expected!r}, got {passed_base_url!r}"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_local_base_url_with_trailing_slash_stripped(self):
        """base_url 'http://127.0.0.1:38000/v3/' must also be cleaned up."""
        fake_honcho = MagicMock(name="Honcho")
        cfg = HonchoClientConfig(
            api_key=None,
            base_url="http://127.0.0.1:38000/v3/",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        passed_base_url = mock_honcho.call_args.kwargs.get("base_url")
        assert passed_base_url == "http://127.0.0.1:38000", (
            f"Expected 'http://127.0.0.1:38000', got {passed_base_url!r}"
        )
