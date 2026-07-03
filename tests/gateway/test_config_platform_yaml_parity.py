"""Non-runtime platform YAML parity tests for gateway config loading.

These tests use temporary HERMES_HOME directories and dummy env values only.
They do not start the gateway, load real profile config, or touch credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from gateway.config import Platform


_PLATFORM_SHAPES = pytest.mark.parametrize(
    ("shape_name", "yaml_config"),
    [
        (
            "top_level_platform",
            {
                "discord": {
                    "enabled": True,
                    "require_mention": True,
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
                            "extra": {"reply_prefix": "[gateway]"},
                        }
                    }
                }
            },
        ),
    ],
)


def _load_config_from_temp_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_config: Mapping[str, Any],
    *,
    dummy_env: Mapping[str, str] | None = None,
):
    """Load gateway config from a temp home with an isolated dummy env view."""
    from gateway import config as gateway_config
    from hermes_cli import managed_scope

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(managed_scope, "apply_managed_overlay", lambda cfg: cfg)

    # load_gateway_config reads many platform env var names. Keep this test from
    # observing host credentials by making gateway.config.os.getenv see only the
    # explicit dummy env supplied by the test.
    env_values = dict(dummy_env or {})

    def fake_getenv(name: str, default: str | None = None) -> str | None:
        return env_values.get(name, default)

    monkeypatch.setattr(gateway_config.os, "getenv", fake_getenv)

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(dict(yaml_config), sort_keys=True),
        encoding="utf-8",
    )

    return gateway_config.load_gateway_config()


@_PLATFORM_SHAPES
def test_platform_config_accepts_top_level_platforms_and_gateway_platforms_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_from_temp_home(tmp_path, monkeypatch, yaml_config)

    platform_config = config.platforms[Platform.DISCORD]

    assert shape_name
    assert platform_config.enabled is True
    assert platform_config.extra["require_mention"] is True
    assert platform_config.extra["reply_prefix"].startswith("[")


@_PLATFORM_SHAPES
def test_platform_extra_and_require_mention_are_preserved_across_yaml_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_from_temp_home(tmp_path, monkeypatch, yaml_config)

    platform_config = config.platforms[Platform.DISCORD]

    assert shape_name
    assert platform_config.extra["require_mention"] is True
    assert platform_config.extra["reply_prefix"] in {
        "[top]",
        "[platforms]",
        "[gateway]",
    }


@pytest.mark.parametrize(
    "yaml_config",
    [
        {"discord": {"enabled": False}},
        {"platforms": {"discord": {"enabled": False}}},
        {"gateway": {"platforms": {"discord": {"enabled": False}}}},
    ],
)
def test_explicit_enabled_false_blocks_dummy_token_auto_enable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_from_temp_home(
        tmp_path,
        monkeypatch,
        yaml_config,
        dummy_env={"DISCORD_BOT_TOKEN": "dummy-discord-token-for-test-only"},
    )

    platform_config = config.platforms[Platform.DISCORD]

    assert platform_config.enabled is False
    assert platform_config.token == "dummy-discord-token-for-test-only"
    assert Platform.DISCORD not in config.get_connected_platforms()

