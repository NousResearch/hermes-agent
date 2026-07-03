"""Non-runtime plugin/platform env config parity tests for gateway loading.

These tests use temporary HERMES_HOME directories and dummy env values only.
They do not start the gateway, discover real plugins, load real profile config,
or touch credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry


_PLATFORM_SHAPES = pytest.mark.parametrize(
    ("shape_name", "yaml_config"),
    [
        (
            "top_level_platform",
            {
                "discord": {
                    "enabled": True,
                    "require_mention": True,
                    "free_response_channels": ["chan-top"],
                    "reply_to_mode": "off",
                    "extra": {"reply_prefix": "[top]"},
                }
            },
        ),
        (
            "platforms_platform",
            {
                "platforms": {
                    "discord": {
                        "enabled": True,
                        "require_mention": True,
                        "free_response_channels": ["chan-platforms"],
                        "reply_to_mode": "off",
                        "extra": {"reply_prefix": "[platforms]"},
                    }
                }
            },
        ),
        (
            "gateway_platforms_platform",
            {
                "gateway": {
                    "platforms": {
                        "discord": {
                            "enabled": True,
                            "require_mention": True,
                            "free_response_channels": ["chan-gateway"],
                            "reply_to_mode": "off",
                            "extra": {"reply_prefix": "[gateway]"},
                        }
                    }
                }
            },
        ),
    ],
)


def _load_config_with_fake_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_config: Mapping[str, Any],
    *,
    dummy_env: Mapping[str, str] | None = None,
    hook_seen: list[dict[str, Any]] | None = None,
):
    """Load gateway config with a fake Discord plugin entry and dummy env only."""
    from gateway import config as gateway_config
    from hermes_cli import managed_scope, plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(managed_scope, "apply_managed_overlay", lambda cfg: cfg)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)

    env_values = dict(dummy_env or {})

    def fake_getenv(name: str, default: str | None = None) -> str | None:
        return env_values.get(name, default)

    monkeypatch.setattr(gateway_config.os, "getenv", fake_getenv)

    seen = hook_seen if hook_seen is not None else []

    def fake_apply_yaml_config(_yaml_cfg: dict, platform_cfg: dict) -> dict[str, Any]:
        seen.append(dict(platform_cfg))
        return {
            "hook_seen_require_mention": platform_cfg.get("require_mention"),
            "hook_seen_free_response_channels": platform_cfg.get("free_response_channels"),
            "hook_seen_reply_to_mode": platform_cfg.get("reply_to_mode"),
        }

    fake_entry = PlatformEntry(
        name="discord",
        label="Discord Test Plugin",
        adapter_factory=lambda cfg: object(),
        check_fn=lambda: True,
        is_connected=lambda cfg: bool(getattr(cfg, "token", None)),
        required_env=["DISCORD_BOT_TOKEN"],
        source="plugin",
        plugin_name="discord-test-plugin",
        apply_yaml_config_fn=fake_apply_yaml_config,
        cron_deliver_env_var="DISCORD_HOME_CHANNEL",
    )
    monkeypatch.setattr(platform_registry, "_entries", {"discord": fake_entry})

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(dict(yaml_config), sort_keys=True),
        encoding="utf-8",
    )

    return gateway_config.load_gateway_config()


@_PLATFORM_SHAPES
def test_plugin_yaml_hook_receives_top_level_platforms_and_gateway_platforms_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    yaml_config: Mapping[str, Any],
) -> None:
    hook_seen: list[dict[str, Any]] = []

    config = _load_config_with_fake_plugin(
        tmp_path,
        monkeypatch,
        yaml_config,
        hook_seen=hook_seen,
    )

    platform_config = config.platforms[Platform.DISCORD]

    assert shape_name
    assert len(hook_seen) == 1
    assert hook_seen[0]["require_mention"] is True
    assert hook_seen[0]["free_response_channels"][0].startswith("chan-")
    assert platform_config.enabled is True
    assert platform_config.extra["require_mention"] is True
    assert platform_config.extra["reply_prefix"].startswith("[")
    assert platform_config.extra["hook_seen_require_mention"] is True
    assert platform_config.extra["hook_seen_free_response_channels"][0].startswith("chan-")
    assert platform_config.extra["hook_seen_reply_to_mode"] == "off"


@_PLATFORM_SHAPES
def test_platform_extra_and_require_mention_survive_plugin_hook_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_with_fake_plugin(tmp_path, monkeypatch, yaml_config)

    platform_config = config.platforms[Platform.DISCORD]

    assert shape_name
    assert platform_config.extra["require_mention"] is True
    assert platform_config.extra["reply_prefix"] in {
        "[top]",
        "[platforms]",
        "[gateway]",
    }
    assert platform_config.extra["hook_seen_reply_to_mode"] == "off"


@pytest.mark.parametrize(
    "yaml_config",
    [
        {"discord": {"enabled": False}},
        {"platforms": {"discord": {"enabled": False}}},
        {"gateway": {"platforms": {"discord": {"enabled": False}}}},
    ],
)
def test_explicit_enabled_false_blocks_dummy_token_auto_enable_for_plugin_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_with_fake_plugin(
        tmp_path,
        monkeypatch,
        yaml_config,
        dummy_env={"DISCORD_BOT_TOKEN": "dummy-discord-token-for-test-only"},
    )

    platform_config = config.platforms[Platform.DISCORD]

    assert platform_config.enabled is False
    assert platform_config.token == "dummy-discord-token-for-test-only"
    assert Platform.DISCORD not in config.get_connected_platforms()


def test_dummy_token_and_home_channel_env_values_do_not_require_real_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config_with_fake_plugin(
        tmp_path,
        monkeypatch,
        {"platforms": {"discord": {"enabled": True}}},
        dummy_env={
            "DISCORD_BOT_TOKEN": "dummy-discord-token-for-test-only",
            "DISCORD_HOME_CHANNEL": "dummy-home-channel",
            "DISCORD_HOME_CHANNEL_NAME": "Dummy Home",
            "DISCORD_HOME_CHANNEL_THREAD_ID": "dummy-thread",
        },
    )

    platform_config = config.platforms[Platform.DISCORD]

    assert platform_config.enabled is True
    assert platform_config.token == "dummy-discord-token-for-test-only"
    assert platform_config.home_channel is not None
    assert platform_config.home_channel.platform is Platform.DISCORD
    assert platform_config.home_channel.chat_id == "dummy-home-channel"
    assert platform_config.home_channel.name == "Dummy Home"
    assert platform_config.home_channel.thread_id == "dummy-thread"
    assert Platform.DISCORD in config.get_connected_platforms()
