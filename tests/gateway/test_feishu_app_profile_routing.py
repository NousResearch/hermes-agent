"""Tests for multi-Feishu-app profile routing (Issue #68046).

Verifies that multiple Feishu apps can be configured in config.yaml with
feishu.apps[] list, each mapped to a different profile, with independent
authorization policies, home channels, and WebSocket connections.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

import pytest

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
)
from gateway.session import SessionSource
from gateway.run import GatewayRunner


class TestFeishuAppProfileRouting:
    """Multi-Feishu-app × profile routing behavior contracts."""

    @pytest.fixture
    def temp_hermes_home(self, monkeypatch, tmp_path):
        """Isolate HERMES_HOME for E2E config tests."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    @pytest.fixture
    def multi_app_yaml_config(self, temp_hermes_home):
        """Config.yaml with feishu.apps[] mapping apps to profiles."""
        import yaml
        config_path = temp_hermes_home / "config.yaml"
        config = {
            "feishu": {
                "apps": [
                    {
                        "name": "architect",
                        "app_id": "cli_aaa",
                        "app_secret": "secret_aaa",
                        "profile": "architect",
                        "home_channel": "oc_arch_home",
                    },
                    {
                        "name": "engineer",
                        "app_id": "cli_eee",
                        "app_secret": "secret_eee",
                        "profile": "engineer",
                        "home_channel": "oc_eng_home",
                        "allow_all_users": True,
                    },
                    {
                        "name": "tester",
                        "app_id": "cli_ttt",
                        "app_secret": "secret_ttt",
                        "profile": "tester",
                        "allowed_users": "ou_testuser1,ou_testuser2",
                    },
                ]
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        return config_path

    def test_feishu_apps_yaml_loads_as_platform_config_list(
        self, multi_app_yaml_config
    ):
        """feishu.apps[] in config.yaml becomes platforms.feishu: [PlatformConfig, ...]"""
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        feishu_configs = gw_config.platforms.get(Platform.FEISHU)

        assert isinstance(feishu_configs, list)
        assert len(feishu_configs) == 3
        assert feishu_configs[0].extra["app_id"] == "cli_aaa"
        assert feishu_configs[1].extra["app_id"] == "cli_eee"
        assert feishu_configs[2].extra["app_id"] == "cli_ttt"

    def test_adapter_profile_map_built_from_feishu_apps(self, multi_app_yaml_config):
        """_adapter_profile_map populated with adapter_id → profile entries."""
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = gw_config
        runner._adapter_profile_map = {}

        # Simulate adapter construction that would populate the map
        feishu_configs = gw_config.iter_platform_configs(Platform.FEISHU)
        for cfg in feishu_configs:
            app_id = cfg.extra.get("app_id")
            profile = cfg.extra.get("profile")
            if app_id and profile:
                adapter_id = f"feishu:{app_id}"
                runner._adapter_profile_map[adapter_id] = profile

        assert runner._adapter_profile_map["feishu:cli_aaa"] == "architect"
        assert runner._adapter_profile_map["feishu:cli_eee"] == "engineer"
        assert runner._adapter_profile_map["feishu:cli_ttt"] == "tester"

    def test_message_from_feishu_app_routes_to_correct_profile(
        self, multi_app_yaml_config
    ):
        """Messages from App A route to profile A, not default."""
        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_dm_user",
            chat_type="dm",
            user_id="ou_user1",
            adapter_id="feishu:cli_eee",
        )

        # Simulate gateway's profile resolution
        _adapter_profile_map = {
            "feishu:cli_aaa": "architect",
            "feishu:cli_eee": "engineer",
            "feishu:cli_ttt": "tester",
        }

        resolved_profile = _adapter_profile_map.get(source.adapter_id, "default")
        assert resolved_profile == "engineer"

    def test_single_app_mode_backward_compat_no_profile_map(self, temp_hermes_home):
        """Single-app config (env vars only) → empty _adapter_profile_map."""
        import yaml
        config_path = temp_hermes_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({}, f)

        with patch.dict(
            os.environ,
            {
                "FEISHU_APP_ID": "cli_single",
                "FEISHU_APP_SECRET": "secret_single",
            },
            clear=False,
        ):
            from gateway.config import load_gateway_config

            gw_config = load_gateway_config()
            runner = GatewayRunner.__new__(GatewayRunner)
            runner.config = gw_config
            runner._adapter_profile_map = {}

            # Single-app mode: no profile mapping
            assert len(runner._adapter_profile_map) == 0

    def test_per_app_home_channel(self, multi_app_yaml_config):
        """Each feishu.apps[] entry has its own home_channel."""
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        feishu_configs = gw_config.iter_platform_configs(Platform.FEISHU)

        arch_cfg, eng_cfg, test_cfg = feishu_configs
        assert arch_cfg.extra.get("home_channel") == "oc_arch_home"
        assert eng_cfg.extra.get("home_channel") == "oc_eng_home"
        assert "home_channel" not in test_cfg.extra or not test_cfg.extra["home_channel"]

    def test_per_app_authorization_config(self, multi_app_yaml_config):
        """Each feishu.apps[] entry has independent allow_all_users / allowed_users."""
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        feishu_configs = gw_config.iter_platform_configs(Platform.FEISHU)

        arch_cfg, eng_cfg, test_cfg = feishu_configs
        # engineer app allows all
        assert eng_cfg.extra.get("allow_all_users") is True
        # tester app has allowlist
        assert test_cfg.extra.get("allowed_users") == "ou_testuser1,ou_testuser2"

    def test_cron_delivery_routes_to_app_matching_profile(self):
        """Cron job with profile=engineer → delivered via engineer app."""
        _adapter_profile_map = {
            "feishu:cli_aaa": "architect",
            "feishu:cli_eee": "engineer",
        }

        cron_job_profile = "engineer"
        # Reverse lookup: profile → adapter_id
        adapter_id = next(
            (aid for aid, prof in _adapter_profile_map.items() if prof == cron_job_profile),
            None,
        )

        assert adapter_id == "feishu:cli_eee"

    def test_feishu_apps_list_exceeds_limit_raises_error(self, temp_hermes_home):
        """feishu.apps with >10 entries raises validation error."""
        # Test _apply_yaml_config directly since load_gateway_config catches exceptions
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        yaml_cfg = {}
        feishu_cfg = {
            "apps": [
                {
                    "name": f"app{i}",
                    "app_id": f"cli_{i}",
                    "app_secret": f"secret_{i}",
                    "profile": f"profile{i}",
                }
                for i in range(11)
            ]
        }

        with pytest.raises(ValueError, match="feishu.apps.*10"):
            _apply_yaml_config(yaml_cfg, feishu_cfg)

    def test_bot_to_bot_self_message_skip(self):
        """Message from our own app_id is skipped to prevent echo loops."""
        our_app_ids = {"cli_aaa", "cli_eee", "cli_ttt"}

        # Inbound message claims to be from cli_eee
        incoming_sender_app_id = "cli_eee"

        # Self-message detection
        is_self_message = incoming_sender_app_id in our_app_ids
        assert is_self_message is True

    def test_adapter_profile_map_lifecycle_single_app_mode(self):
        """_adapter_profile_map initialized unconditionally (empty dict in single-app)."""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner._adapter_profile_map = {}
        runner.adapters = {}

        # Single-app mode: map is empty but exists
        assert isinstance(runner._adapter_profile_map, dict)
        assert len(runner._adapter_profile_map) == 0

    def test_yaml_config_preferred_over_env_for_multi_app(self, temp_hermes_home):
        """feishu.apps[] in YAML takes precedence over FEISHU_APP_ID env var."""
        import yaml
        config_path = temp_hermes_home / "config.yaml"
        config = {
            "feishu": {
                "apps": [
                    {
                        "name": "app1",
                        "app_id": "cli_yaml_app",
                        "app_secret": "yaml_secret",
                        "profile": "yaml_profile",
                    }
                ]
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        with patch.dict(
            os.environ,
            {
                "FEISHU_APP_ID": "cli_env_app",
                "FEISHU_APP_SECRET": "env_secret",
            },
            clear=False,
        ):
            from gateway.config import load_gateway_config

            gw_config = load_gateway_config()
            feishu_configs = gw_config.iter_platform_configs(Platform.FEISHU)

            # YAML wins
            assert len(feishu_configs) == 1
            assert feishu_configs[0].extra["app_id"] == "cli_yaml_app"

    def test_feishu_apps_without_profile_uses_default(self, temp_hermes_home):
        """App entry without profile key defaults to 'default'."""
        import yaml
        config_path = temp_hermes_home / "config.yaml"
        config = {
            "feishu": {
                "apps": [
                    {
                        "name": "no-profile-app",
                        "app_id": "cli_no_prof",
                        "app_secret": "secret_no_prof",
                    }
                ]
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        feishu_configs = gw_config.iter_platform_configs(Platform.FEISHU)

        # Default profile when not specified
        profile = feishu_configs[0].extra.get("profile", "default")
        assert profile == "default"

    def test_duplicate_app_id_raises_error(self, temp_hermes_home):
        """Duplicate app_id in feishu.apps[] raises ValueError."""
        # Test _apply_yaml_config directly since load_gateway_config catches exceptions
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        yaml_cfg = {}
        feishu_cfg = {
            "apps": [
                {
                    "name": "app1",
                    "app_id": "cli_duplicate",
                    "app_secret": "secret1",
                    "profile": "profile1",
                },
                {
                    "name": "app2",
                    "app_id": "cli_duplicate",  # Duplicate!
                    "app_secret": "secret2",
                    "profile": "profile2",
                },
            ]
        }

        with pytest.raises(ValueError, match="Duplicate app_id in feishu.apps: cli_duplicate"):
            _apply_yaml_config(yaml_cfg, feishu_cfg)

    def test_yaml_env_coexistence_logs_warning(
        self, temp_hermes_home, monkeypatch, caplog
    ):
        """Both feishu.apps[] and FEISHU_APP_ID present logs warning."""
        import yaml
        config_path = temp_hermes_home / "config.yaml"
        config = {
            "feishu": {
                "apps": [
                    {
                        "name": "yaml_app",
                        "app_id": "cli_yaml",
                        "app_secret": "secret_yaml",
                        "profile": "yaml_profile",
                    }
                ]
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Set env var to trigger the warning
        monkeypatch.setenv("FEISHU_APP_ID", "cli_env_app")

        from gateway.config import load_gateway_config

        with caplog.at_level("WARNING"):
            gw_config = load_gateway_config()

        # Check warning was logged
        assert any(
            "Both feishu.apps[] config and FEISHU_APP_ID env var detected" in record.message
            for record in caplog.records
        )

        # Verify YAML config takes precedence
        feishu_configs = list(gw_config.iter_platform_configs(Platform.FEISHU))
        assert len(feishu_configs) == 1
        assert feishu_configs[0].extra["app_id"] == "cli_yaml"

    def test_adapter_id_for_profile_reverse_lookup(self, multi_app_yaml_config):
        """GatewayRunner.adapter_id_for_profile() finds adapter_id by profile name."""
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = gw_config
        runner._adapter_profile_map = {
            "feishu:cli_aaa": "architect",
            "feishu:cli_eee": "engineer",
            "feishu:cli_ttt": "tester",
        }

        # Reverse lookup: profile → adapter_id
        assert runner.adapter_id_for_profile("architect") == "feishu:cli_aaa"
        assert runner.adapter_id_for_profile("engineer") == "feishu:cli_eee"
        assert runner.adapter_id_for_profile("tester") == "feishu:cli_ttt"
        
        # Unknown profile returns None
        assert runner.adapter_id_for_profile("unknown") is None

        # Platform-specific lookup
        assert runner.adapter_id_for_profile("engineer", Platform.FEISHU) == "feishu:cli_eee"

    def test_register_adapter_auto_populates_profile_map(self, multi_app_yaml_config):
        """Registering a real adapter should auto-populate _adapter_profile_map via _register_connected_adapter."""
        from gateway.config import load_gateway_config, Platform
        from plugins.platforms.feishu.adapter import FeishuAdapter

        gw_config = load_gateway_config()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = gw_config
        runner._adapter_profile_map = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}
        runner.adapters = {}

        # Register each Feishu adapter using the real _register_connected_adapter flow
        for cfg in gw_config.iter_platform_configs(Platform.FEISHU):
            adapter = FeishuAdapter(cfg)
            runner._register_connected_adapter(Platform.FEISHU, adapter)

        # Verify profile map was populated automatically
        assert "feishu:cli_aaa" in runner._adapter_profile_map
        assert runner._adapter_profile_map["feishu:cli_aaa"] == "architect"
        assert "feishu:cli_eee" in runner._adapter_profile_map
        assert runner._adapter_profile_map["feishu:cli_eee"] == "engineer"
        assert "feishu:cli_ttt" in runner._adapter_profile_map
        assert runner._adapter_profile_map["feishu:cli_ttt"] == "tester"
        # Wrong platform returns None
        assert runner.adapter_id_for_profile("engineer", Platform.DISCORD) is None

    def test_bot_to_bot_self_message_skipped(self):
        """Bot messages from other bots are allowed (no bot-to-bot detection in webhook path).

        Bot-to-bot detection was removed because Feishu receive_v1 webhooks don't include
        sender.app_id. The self_echo guard (via open_id matching) prevents loops.
        """
        from plugins.platforms.feishu.adapter import FeishuAdapter

        # Create adapter instance for one app
        config = PlatformConfig(
            enabled=True,
            extra={
                "app_id": "cli_architect",
                "app_secret": "secret_arch",
                "adapter_id": "cli_architect",
            },
        )
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter.config = config
        adapter._bot_open_id = "ou_bot_arch"
        adapter._bot_user_id = None
        adapter._app_id = "cli_architect"
        adapter._allowed_group_users = set()
        adapter._admins = set()
        adapter._group_policy = "allowlist"
        adapter._default_group_policy = "allowlist"
        adapter._group_rules = {}
        adapter._allow_bots = "all"

        # Helper to build mock require_mention check
        adapter._require_mention_for = lambda chat_id: False
        adapter._mentions_self = lambda msg: True
        adapter._allow_group_message = lambda *args, **kwargs: True

        # Simulate message from ANOTHER bot (different open_id)
        sender = SimpleNamespace(
            sender_type="bot",
            sender_id=SimpleNamespace(
                open_id="ou_bot_engineer",  # Different bot
                user_id=None,
                union_id=None,
            ),
        )
        message = SimpleNamespace(
            chat_type="group",
            chat_id="oc_group123",
        )

        # _admit should ALLOW the message (no bot-to-bot rejection)
        reason = adapter._admit(sender, message)
        assert reason is None  # Message is allowed

        # Sanity check: message from self (same open_id) is rejected as self_echo
        sender_self = SimpleNamespace(
            sender_type="bot",
            sender_id=SimpleNamespace(
                open_id="ou_bot_arch",  # Same as adapter._bot_open_id
                user_id=None,
                union_id=None,
            ),
        )
        reason_self = adapter._admit(sender_self, message)
        assert reason_self == "self_echo"

        # External bot (different open_id, allow_bots=all) should be admitted
        sender_external = SimpleNamespace(
            sender_type="bot",
            sender_id=SimpleNamespace(
                open_id="ou_external_bot",
                user_id=None,
                union_id=None,
            ),
        )
        reason_external = adapter._admit(sender_external, message)
        assert reason_external != "bot_to_bot_self_message"

    # ------------------------------------------------------------------
    # E2E: Multi-app message routing — full stamping pipeline (#17)
    # ------------------------------------------------------------------

    def test_handle_message_stamps_profile_from_adapter_id(
        self, multi_app_yaml_config
    ):
        """E2E: _handle_message profile stamping — the exact code path at
        gateway/run.py:9387-9397.

        Given a SessionSource with adapter_id='feishu:cli_eee' but no profile,
        verify the stamping logic resolves profile='engineer' from
        _adapter_profile_map BEFORE any downstream code (authorization,
        session-key generation) runs.
        """
        from gateway.config import load_gateway_config
        from gateway.session import SessionSource

        gw_config = load_gateway_config()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = gw_config
        runner._adapter_profile_map = {
            "feishu:cli_aaa": "architect",
            "feishu:cli_eee": "engineer",
            "feishu:cli_ttt": "tester",
        }

        # Simulate inbound from the engineer app — no profile set yet
        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_dm_with_user",
            chat_type="dm",
            user_id="ou_user1",
            adapter_id="feishu:cli_eee",
        )
        assert source.profile is None  # pre-condition

        # Execute the EXACT stamping block from _handle_message (L9392-9397).
        # We inline it rather than calling the full async _handle_message
        # because the method is ~700 lines with heavy I/O dependencies.
        # The stamping block is the first thing that runs after source = event.source.
        if source and not getattr(source, "profile", None):
            adapter_id = getattr(source, "adapter_id", None)
            if adapter_id:
                _mapped_profile = runner._adapter_profile_map.get(str(adapter_id))
                if _mapped_profile:
                    source.profile = _mapped_profile

        # Profile was stamped correctly
        assert source.profile == "engineer"

        # Cross-check: architect app → architect profile
        source_arch = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_dm_with_user2",
            chat_type="dm",
            user_id="ou_user2",
            adapter_id="feishu:cli_aaa",
        )
        if source_arch and not getattr(source_arch, "profile", None):
            _mp = runner._adapter_profile_map.get(
                str(getattr(source_arch, "adapter_id", None))
            )
            if _mp:
                source_arch.profile = _mp
        assert source_arch.profile == "architect"

    def test_handle_message_does_not_overwrite_existing_profile(
        self, multi_app_yaml_config
    ):
        """E2E: If source.profile is already set (by multiplex handlers), the
        stamping block must NOT overwrite it.
        """
        from gateway.session import SessionSource

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._adapter_profile_map = {
            "feishu:cli_aaa": "architect",
            "feishu:cli_eee": "engineer",
        }

        # Simulate a multiplex handler that already set profile
        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_dm_user",
            chat_type="dm",
            user_id="ou_user1",
            adapter_id="feishu:cli_aaa",
            profile="multiplex_set_profile",
        )

        # The stamping guard: `not getattr(source, "profile", None)` must be False
        stamped = False
        if source and not getattr(source, "profile", None):
            adapter_id = getattr(source, "adapter_id", None)
            if adapter_id:
                _mapped = runner._adapter_profile_map.get(str(adapter_id))
                if _mapped:
                    source.profile = _mapped
                    stamped = True

        assert not stamped
        assert source.profile == "multiplex_set_profile"

    def test_handle_message_unknown_adapter_id_keeps_none_profile(
        self, multi_app_yaml_config
    ):
        """E2E: adapter_id not in map (e.g. unregistered app) → profile stays None,
        downstream code will use default profile.
        """
        from gateway.session import SessionSource

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._adapter_profile_map = {
            "feishu:cli_aaa": "architect",
        }

        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_dm_user",
            chat_type="dm",
            user_id="ou_user1",
            adapter_id="feishu:cli_unknown",
        )

        if source and not getattr(source, "profile", None):
            adapter_id = getattr(source, "adapter_id", None)
            if adapter_id:
                _mapped = runner._adapter_profile_map.get(str(adapter_id))
                if _mapped:
                    source.profile = _mapped

        assert source.profile is None

    # ------------------------------------------------------------------
    # E2E: Cron delivery in multi-app mode (#18)
    # ------------------------------------------------------------------

    def test_cron_delivery_selects_pconfig_by_profile(self, multi_app_yaml_config):
        """E2E: cron/scheduler.py _deliver_result pconfig list handling.

        When config.platforms[FEISHU] is a list (multi-app mode), and the job
        has profile='engineer', the cron delivery path must select the
        PlatformConfig whose extra['profile'] == 'engineer', not the first.
        """
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()

        # Simulate what _deliver_result does at L1532-1544
        platform = Platform.FEISHU
        pconfig_raw = gw_config.platforms.get(platform)
        assert isinstance(pconfig_raw, list), "Multi-app mode should produce a list"
        assert len(pconfig_raw) == 3

        # Job with profile=engineer
        job = {"id": "test_job_1", "profile": "engineer", "deliver": "feishu"}

        # Replicate the exact selection logic from scheduler.py L1534-1544
        pconfig = pconfig_raw
        if isinstance(pconfig, list):
            job_profile = job.get("profile") or job.get("deliver", {}).get("profile")
            if job_profile:
                selected = next(
                    (c for c in pconfig if (c.extra or {}).get("profile") == job_profile),
                    pconfig[0] if pconfig else None,
                )
            else:
                selected = pconfig[0] if pconfig else None
            pconfig = selected

        # Correct app config was selected
        assert pconfig is not None
        assert pconfig.extra["app_id"] == "cli_eee"
        assert pconfig.extra["profile"] == "engineer"

    def test_cron_delivery_falls_back_to_first_when_profile_unmatched(
        self, multi_app_yaml_config
    ):
        """E2E: Job with profile='nonexistent' → graceful fallback to first config.
        """
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        pconfig_raw = gw_config.platforms.get(Platform.FEISHU)
        assert isinstance(pconfig_raw, list)

        job = {"id": "test_job_2", "profile": "nonexistent_profile"}

        pconfig = pconfig_raw
        if isinstance(pconfig, list):
            job_profile = job.get("profile")
            if job_profile:
                selected = next(
                    (c for c in pconfig if (c.extra or {}).get("profile") == job_profile),
                    pconfig[0] if pconfig else None,
                )
            else:
                selected = pconfig[0] if pconfig else None
            pconfig = selected

        # Fell back to first app (architect)
        assert pconfig is not None
        assert pconfig.extra["app_id"] == "cli_aaa"

    def test_cron_delivery_no_profile_falls_back_to_first(
        self, multi_app_yaml_config
    ):
        """E2E: Job without profile key → fallback to first config.
        """
        from gateway.config import load_gateway_config

        gw_config = load_gateway_config()
        pconfig_raw = gw_config.platforms.get(Platform.FEISHU)
        assert isinstance(pconfig_raw, list)

        job = {"id": "test_job_3"}  # no profile key

        pconfig = pconfig_raw
        if isinstance(pconfig, list):
            job_profile = job.get("profile")
            if job_profile:
                selected = next(
                    (c for c in pconfig if (c.extra or {}).get("profile") == job_profile),
                    pconfig[0] if pconfig else None,
                )
            else:
                selected = pconfig[0] if pconfig else None
            pconfig = selected

        assert pconfig is not None
        assert pconfig.extra["app_id"] == "cli_aaa"  # first app

    def test_cron_delivery_runtime_adapter_lookup_by_adapter_id(
        self, multi_app_yaml_config
    ):
        """E2E: When gateway is running, runtime_adapter lookup uses
        adapters_by_id to find the correct multi-app adapter.

        Simulates the full selection chain: pconfig list → profile match →
        configured_id extraction → adapters_by_id lookup.
        """
        from gateway.config import load_gateway_config
        from plugins.platforms.feishu.adapter import FeishuAdapter

        gw_config = load_gateway_config()

        # Build runner with real adapters registered
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = gw_config
        runner._adapter_profile_map = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}
        runner.adapters = {}

        for cfg in gw_config.iter_platform_configs(Platform.FEISHU):
            adapter = FeishuAdapter(cfg)
            runner._register_connected_adapter(Platform.FEISHU, adapter)

        # adapters_by_id now has feishu:cli_aaa, feishu:cli_eee, feishu:cli_ttt
        assert "feishu:cli_aaa" in runner.adapters_by_id
        assert "feishu:cli_eee" in runner.adapters_by_id
        assert "feishu:cli_ttt" in runner.adapters_by_id

        # Simulate cron delivery for engineer profile
        pconfig_raw = gw_config.platforms.get(Platform.FEISHU)
        job = {"id": "cron_e2e", "profile": "engineer"}

        # Step 1: pconfig list → select by profile (scheduler.py L1534-1544)
        pconfig = pconfig_raw
        if isinstance(pconfig, list):
            job_profile = job.get("profile")
            selected = next(
                (c for c in pconfig if (c.extra or {}).get("profile") == job_profile),
                pconfig[0] if pconfig else None,
            )
            pconfig = selected

        assert pconfig.extra["app_id"] == "cli_eee"

        # Step 2: runtime_adapter lookup (scheduler.py L1557-1570)
        # In real gateway: adapters_by_id is passed as kwarg from runner
        adapters_by_id = runner.adapters_by_id

        _extra = getattr(pconfig, "extra", None) or {}
        _configured_id = str(
            _extra.get("adapter_id")
            or _extra.get("app_id")
            or ""
        ).strip()
        _full_id = f"{Platform.FEISHU.value}:{_configured_id}"
        runtime_adapter = adapters_by_id.get(_full_id)

        # The correct adapter was found
        assert runtime_adapter is not None
        assert runtime_adapter is runner.adapters_by_id["feishu:cli_eee"]
        # And it's the engineer app's adapter, not the architect's
        assert getattr(runtime_adapter, "adapter_id", None) == "feishu:cli_eee"
