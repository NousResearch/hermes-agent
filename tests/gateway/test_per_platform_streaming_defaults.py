"""Per-platform streaming defaults + dashboard exposure.

Streaming is smooth on Telegram (native sendMessageDraft) but flickers on
edit-only platforms like Discord. The shipped defaults encode that:
display.platforms.telegram.streaming=true, .discord.streaming=false. These are
gap-fillers (user values win via deep-merge) and, because the dashboard schema
is generated from DEFAULT_CONFIG, they automatically appear as editable toggles
in the web UI.
"""

from __future__ import annotations

import pytest


def test_default_per_platform_streaming_flags():
    from hermes_cli.config import DEFAULT_CONFIG
    plats = DEFAULT_CONFIG["display"]["platforms"]
    assert plats["telegram"]["streaming"] is True
    assert plats["discord"]["streaming"] is False


def test_resolver_telegram_on_discord_off_when_global_enabled():
    """With global streaming on, the per-platform defaults make Telegram stream
    and Discord not — matching the platforms' actual streaming quality."""
    from hermes_cli.config import DEFAULT_CONFIG
    from gateway.config import StreamingConfig
    from gateway.display_config import (
        resolve_display_setting,
        resolve_gateway_streaming_enabled,
    )

    cfg = dict(DEFAULT_CONFIG)
    cfg["streaming"] = {"enabled": True, "transport": "auto"}
    scfg = StreamingConfig.from_dict(cfg["streaming"])

    def streams(plat):
        ov = resolve_display_setting(cfg, plat, "streaming")
        return resolve_gateway_streaming_enabled(scfg, ov)

    assert streams("telegram") is True
    assert streams("discord") is False
    # A platform with no default entry follows the global switch.
    assert streams("slack") is True


def test_user_override_wins_over_default():
    """A user who explicitly enables Discord streaming keeps their value — the
    default false must not clobber it (config deep-merge: user wins)."""
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge

    user = {"display": {"platforms": {"discord": {"streaming": True}}}}
    merged = _deep_merge(dict(DEFAULT_CONFIG), user)
    assert merged["display"]["platforms"]["discord"]["streaming"] is True
    # Partial override must not wipe the sibling telegram default.
    assert merged["display"]["platforms"]["telegram"]["streaming"] is True


def _resolved_streaming_enabled(streaming_config, platform_override):
    from gateway.config import StreamingConfig
    from gateway.display_config import resolve_gateway_streaming_enabled

    scfg = StreamingConfig.from_dict(streaming_config)
    return resolve_gateway_streaming_enabled(scfg, platform_override)


@pytest.mark.parametrize(
    ("streaming_config", "platform_override", "expected"),
    [
        ({"enabled": False, "transport": "auto"}, True, False),
        ({"enabled": False, "transport": "auto"}, False, False),
        ({"enabled": False, "transport": "auto"}, None, False),
        ({"enabled": True, "transport": "auto"}, True, True),
        ({"enabled": True, "transport": "auto"}, False, False),
        ({"enabled": True, "transport": "auto"}, None, True),
        ({"enabled": True, "transport": "off"}, True, False),
        ({"enabled": True, "transport": "off"}, None, False),
        ({"enabled": True, "transport": False}, True, False),
        ({"enabled": True, "transport": False}, None, False),
    ],
)
def test_gateway_streaming_global_gate_truth_table(
    streaming_config, platform_override, expected
):
    """Global streaming gates must be absolute; platform overrides only narrow."""
    assert _resolved_streaming_enabled(streaming_config, platform_override) is expected


def test_global_disabled_blocks_telegram_platform_default():
    """Regression for #53697: Telegram's default true must not bypass enabled=false."""
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge
    from gateway.config import StreamingConfig
    from gateway.display_config import (
        resolve_display_setting,
        resolve_gateway_streaming_enabled,
    )

    cfg = _deep_merge(
        dict(DEFAULT_CONFIG), {"streaming": {"enabled": False, "transport": "auto"}}
    )
    scfg = StreamingConfig.from_dict(cfg["streaming"])
    telegram_override = resolve_display_setting(cfg, "telegram", "streaming")

    assert telegram_override is True
    assert resolve_gateway_streaming_enabled(scfg, telegram_override) is False


def test_global_transport_off_is_hard_kill_switch_for_platform_defaults():
    """Global transport=off must disable streaming even when Telegram's
    per-platform default says streaming=true."""
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge
    from gateway.config import StreamingConfig
    from gateway.display_config import (
        resolve_display_setting,
        resolve_gateway_streaming_enabled,
    )

    cfg = _deep_merge(dict(DEFAULT_CONFIG), {"streaming": {"transport": "off"}})
    scfg = StreamingConfig.from_dict(cfg["streaming"])
    telegram_override = resolve_display_setting(cfg, "telegram", "streaming")

    assert telegram_override is True
    assert scfg.transport == "off"
    assert resolve_gateway_streaming_enabled(scfg, telegram_override) is False

    yaml_bool_cfg = _deep_merge(dict(DEFAULT_CONFIG), {"streaming": {"transport": False}})
    yaml_bool_scfg = StreamingConfig.from_dict(yaml_bool_cfg["streaming"])
    assert resolve_gateway_streaming_enabled(yaml_bool_scfg, telegram_override) is False


def test_dashboard_schema_exposes_per_platform_streaming():
    """Because the web settings schema is built from DEFAULT_CONFIG, the
    per-platform streaming toggles surface in the dashboard automatically."""
    pytest.importorskip("fastapi")  # web_server requires fastapi/uvicorn
    from hermes_cli.web_server import CONFIG_SCHEMA

    assert "display.platforms.telegram.streaming" in CONFIG_SCHEMA
    assert "display.platforms.discord.streaming" in CONFIG_SCHEMA
    assert CONFIG_SCHEMA["display.platforms.discord.streaming"]["type"] == "boolean"
    # Global streaming controls are exposed too.
    assert "streaming.enabled" in CONFIG_SCHEMA
    assert "streaming.transport" in CONFIG_SCHEMA
