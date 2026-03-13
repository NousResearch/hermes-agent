"""Tests for gateway configuration management."""

import ast
import json
from pathlib import Path

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
)


class TestHomeChannelRoundtrip:
    def test_to_dict_from_dict(self):
        hc = HomeChannel(platform=Platform.DISCORD, chat_id="999", name="general")
        d = hc.to_dict()
        restored = HomeChannel.from_dict(d)

        assert restored.platform == Platform.DISCORD
        assert restored.chat_id == "999"
        assert restored.name == "general"


class TestPlatformConfigRoundtrip:
    def test_to_dict_from_dict(self):
        pc = PlatformConfig(
            enabled=True,
            token="tok_123",
            home_channel=HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="555",
                name="Home",
            ),
            extra={"foo": "bar"},
        )
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)

        assert restored.enabled is True
        assert restored.token == "tok_123"
        assert restored.home_channel.chat_id == "555"
        assert restored.extra == {"foo": "bar"}

    def test_disabled_no_token(self):
        pc = PlatformConfig()
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)
        assert restored.enabled is False
        assert restored.token is None


class TestGetConnectedPlatforms:
    def test_returns_enabled_with_token(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
                Platform.DISCORD: PlatformConfig(enabled=False, token="d"),
                Platform.SLACK: PlatformConfig(enabled=True),  # no token
            },
        )
        connected = config.get_connected_platforms()
        assert Platform.TELEGRAM in connected
        assert Platform.DISCORD not in connected
        assert Platform.SLACK not in connected

    def test_empty_platforms(self):
        config = GatewayConfig()
        assert config.get_connected_platforms() == []


class TestSessionResetPolicy:
    def test_roundtrip(self):
        policy = SessionResetPolicy(mode="idle", at_hour=6, idle_minutes=120)
        d = policy.to_dict()
        restored = SessionResetPolicy.from_dict(d)
        assert restored.mode == "idle"
        assert restored.at_hour == 6
        assert restored.idle_minutes == 120

    def test_defaults(self):
        policy = SessionResetPolicy()
        assert policy.mode == "both"
        assert policy.at_hour == 4
        assert policy.idle_minutes == 1440


class TestGatewayConfigRoundtrip:
    def test_full_roundtrip(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="tok",
                    home_channel=HomeChannel(Platform.TELEGRAM, "123", "Home"),
                ),
            },
            reset_triggers=["/new"],
        )
        d = config.to_dict()
        restored = GatewayConfig.from_dict(d)

        assert Platform.TELEGRAM in restored.platforms
        assert restored.platforms[Platform.TELEGRAM].token == "tok"
        assert restored.reset_triggers == ["/new"]


# =========================================================================
# Config bridging (gateway/run.py module-level yaml->env bridge)
# =========================================================================


class TestConfigBridgeJsonImport:
    """json.dumps is called at module level in gateway/run.py for list config
    values (e.g. docker_volumes). json must be imported at module scope."""

    def test_json_imported_at_module_level(self):
        run_py = Path(__file__).parent.parent.parent / "gateway" / "run.py"
        with open(run_py) as f:
            tree = ast.parse(f.read())

        json_imported = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "json":
                        json_imported = True
            elif isinstance(node, ast.ImportFrom) and node.module == "json":
                json_imported = True

        assert json_imported, (
            "json must be imported at module level in gateway/run.py — "
            "line 85 calls json.dumps() during config bridging"
        )

    def test_list_config_value_serialized_to_json(self):
        terminal_cfg = {"docker_volumes": ["/home:/workspace", "/data:/data"]}
        env_map = {"docker_volumes": "TERMINAL_DOCKER_VOLUMES"}

        result_env = {}
        for cfg_key, env_var in env_map.items():
            if cfg_key in terminal_cfg:
                val = terminal_cfg[cfg_key]
                if isinstance(val, list):
                    result_env[env_var] = json.dumps(val)
                else:
                    result_env[env_var] = str(val)

        assert result_env["TERMINAL_DOCKER_VOLUMES"] == json.dumps(
            ["/home:/workspace", "/data:/data"]
        )
        assert json.loads(result_env["TERMINAL_DOCKER_VOLUMES"]) == [
            "/home:/workspace",
            "/data:/data",
        ]
