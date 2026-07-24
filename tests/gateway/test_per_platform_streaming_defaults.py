"""Per-platform streaming defaults + dashboard exposure.

Telegram is a durable mobile inbox, so shipped defaults keep persistent
streaming/progress/interim chatter off unless explicitly enabled. Discord and
Slack edit-based streaming also defaults off. These are gap-fillers (user
values win via deep-merge) and, because the dashboard schema is generated from
DEFAULT_CONFIG, they automatically appear as editable controls in the web UI.
"""

from __future__ import annotations


def test_default_per_platform_streaming_flags():
    from hermes_cli.config import DEFAULT_CONFIG
    plats = DEFAULT_CONFIG["display"]["platforms"]
    assert plats["telegram"] == {
        "streaming": False,
        "tool_progress": "off",
        "interim_assistant_messages": False,
        "long_running_notifications": False,
        "cleanup_progress": True,
    }
    assert plats["discord"]["streaming"] is False
    assert plats["slack"]["streaming"] is False


def test_resolver_telegram_discord_and_slack_off_when_global_enabled():
    """Per-platform safety defaults beat the enabled global streaming switch."""
    from hermes_cli.config import DEFAULT_CONFIG
    from gateway.display_config import resolve_display_setting

    cfg = dict(DEFAULT_CONFIG)
    cfg["streaming"] = {"enabled": True, "transport": "auto"}

    def streams(plat):
        ov = resolve_display_setting(cfg, plat, "streaming")
        # global enabled; None override = follow global (True)
        return True if ov is None else bool(ov)

    assert streams("telegram") is False
    assert streams("discord") is False
    assert streams("slack") is False
    # A platform with no default entry still follows the global switch.
    assert streams("matrix") is True


def test_user_override_wins_over_default():
    """Explicit per-platform values win without clobbering sibling defaults."""
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge

    user = {"display": {"platforms": {
        "telegram": {
            "streaming": True,
            "tool_progress": "all",
            "interim_assistant_messages": True,
            "long_running_notifications": True,
            "cleanup_progress": False,
        },
        "discord": {"streaming": True},
        "slack": {"streaming": True},
    }}}
    merged = _deep_merge(dict(DEFAULT_CONFIG), user)
    assert merged["display"]["platforms"]["telegram"] == user["display"]["platforms"]["telegram"]
    assert merged["display"]["platforms"]["discord"]["streaming"] is True
    assert merged["display"]["platforms"]["slack"]["streaming"] is True


def test_dashboard_schema_exposes_per_platform_streaming():
    """Because the web settings schema is built from DEFAULT_CONFIG, the
    per-platform streaming toggles surface in the dashboard automatically."""
    import pytest
    pytest.importorskip("fastapi")  # web_server requires fastapi/uvicorn
    from hermes_cli.web_server import CONFIG_SCHEMA

    assert "display.platforms.telegram.streaming" in CONFIG_SCHEMA
    assert "display.platforms.discord.streaming" in CONFIG_SCHEMA
    assert "display.platforms.slack.streaming" in CONFIG_SCHEMA
    assert CONFIG_SCHEMA["display.platforms.discord.streaming"]["type"] == "boolean"
    assert CONFIG_SCHEMA["display.platforms.slack.streaming"]["type"] == "boolean"
    # Global streaming controls are exposed too.
    assert "streaming.enabled" in CONFIG_SCHEMA
    assert "streaming.transport" in CONFIG_SCHEMA
