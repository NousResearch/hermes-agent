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


    def test_defaults_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove HONCHO_API_KEY if it exists
            os.environ.pop("HONCHO_API_KEY", None)
            os.environ.pop("HONCHO_ENVIRONMENT", None)
            config = HonchoClientConfig.from_env()
        assert config.api_key is None
        assert config.environment == "production"


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


    def test_context_tokens_explicit_sets_cap(self, tmp_path):
        """Explicit contextTokens in config sets the cap."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "contextTokens": 1200}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.context_tokens == 1200


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


    def test_corrupt_config_falls_back_to_env(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json{{{")

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        # Should fall back to from_env without crashing
        assert isinstance(config, HonchoClientConfig)


class TestResolveSessionName:
    def test_manual_override(self):
        config = HonchoClientConfig(sessions={"/home/user/proj": "custom-session"})
        assert config.resolve_session_name("/home/user/proj") == "custom-session"

    def test_derive_from_dirname(self):
        config = HonchoClientConfig()
        result = config.resolve_session_name("/home/user/my-project")
        assert result == "my-project"


    def test_per_repo_uses_git_root(self):
        config = HonchoClientConfig(session_strategy="per-repo")
        with patch.object(
            HonchoClientConfig, "_git_repo_name", return_value="hermes-agent"
        ):
            result = config.resolve_session_name("/home/user/hermes-agent/subdir")
        assert result == "hermes-agent"


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
             patch.object(Path, "home", return_value=fake_home), \
             patch(
                 "plugins.memory.honcho.client.get_hermes_home",
                 return_value=fake_home / ".hermes",
             ), \
             patch(
                 "plugins.memory.honcho.client._get_default_hermes_home",
                 return_value=fake_home / ".hermes",
             ):
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


    def test_explicit_env_var_wins(self):
        with patch.dict(os.environ, {"HERMES_HONCHO_HOST": "hermes.coder"}):
            assert resolve_active_host() == "hermes.coder"


    def test_profiles_import_failure_falls_back(self):
        import sys
        with patch.dict(os.environ, {}, clear=False), patch(
            "plugins.memory.honcho.client.resolve_config_path",
            return_value=Path("/nonexistent/test-honcho-config.json"),
        ):
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
    def test_timeout_change_triggers_client_rebuild(self):
        """Changing timeout config must rebuild the cached client."""
        from hermes_constants import get_hermes_home

        cfg_yaml = get_hermes_home() / "config.yaml"
        cfg_yaml.write_text("honcho:\n  timeout: 30\n")

        fake_honcho_1 = MagicMock(name="Honcho_v1")
        fake_honcho_2 = MagicMock(name="Honcho_v2")
        cfg = HonchoClientConfig(
            api_key="test-key",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho_1) as mock_h1:
            client1 = get_honcho_client(cfg)

        assert client1 is fake_honcho_1
        assert mock_h1.call_args.kwargs["timeout"] == 30.0

        # Same config — should return cached client (no rebuild)
        with patch("honcho.Honcho", return_value=fake_honcho_2) as mock_h2:
            client2 = get_honcho_client(cfg)

        assert client2 is fake_honcho_1  # still cached
        mock_h2.assert_not_called()

        # Changed timeout — must rebuild
        cfg_yaml.write_text("honcho:\n  timeout: 300\n")
        st = cfg_yaml.stat()
        os.utime(cfg_yaml, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        with patch("honcho.Honcho", return_value=fake_honcho_2) as mock_h3:
            client3 = get_honcho_client(cfg)

        assert client3 is fake_honcho_2  # rebuilt
        mock_h3.assert_called_once()
        assert mock_h3.call_args.kwargs["timeout"] == 300.0

    @pytest.mark.skipif(
        not importlib.util.find_spec("honcho"),
        reason="honcho SDK not installed"
    )
    def test_managed_config_timeout_does_not_thrash_singleton(self, tmp_path, monkeypatch):
        """A managed-scope honcho.timeout with no user config.yaml must be seen
        by the staleness check (stable reuse), and a managed edit must trigger
        a rebuild. Regression for a memo that keyed only on the user file."""
        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        managed_cfg = managed_dir / "config.yaml"
        managed_cfg.write_text("honcho:\n  timeout: 88\n")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))

        fake_honcho_1 = MagicMock(name="Honcho_v1")
        fake_honcho_2 = MagicMock(name="Honcho_v2")
        cfg = HonchoClientConfig(
            api_key="test-key",
            workspace_id="hermes",
            environment="production",
        )

        with patch("honcho.Honcho", return_value=fake_honcho_1) as mock_h1:
            client1 = get_honcho_client(cfg)
            client2 = get_honcho_client(cfg)

        assert client1 is fake_honcho_1
        assert client2 is fake_honcho_1
        assert mock_h1.call_count == 1
        assert mock_h1.call_args.kwargs["timeout"] == 88.0

        # A managed-timeout edit is detected (same-size write, so bump mtime).
        managed_cfg.write_text("honcho:\n  timeout: 99\n")
        st = managed_cfg.stat()
        os.utime(managed_cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        with patch("honcho.Honcho", return_value=fake_honcho_2) as mock_h2:
            client3 = get_honcho_client(cfg)

        assert client3 is fake_honcho_2
        mock_h2.assert_called_once()
        assert mock_h2.call_args.kwargs["timeout"] == 99.0


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


    def test_long_gateway_key_truncated_to_limit(self):
        """An over-limit sanitized key must truncate to exactly 100 chars."""
        key = "!roomid:matrix.example.org|" + "$event_" + ("a" * 300)
        config = HonchoClientConfig()
        result = config.resolve_session_name(gateway_session_key=key)
        assert result is not None
        assert len(result) == self.HONCHO_MAX


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


    def test_depth_clamped_high(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"apiKey": "***", "dialecticDepth": 10}))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.dialectic_depth == 3


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

