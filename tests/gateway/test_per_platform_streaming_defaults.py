"""Per-platform streaming defaults + dashboard exposure.

Streaming is smooth on Telegram (native sendMessageDraft) but flickers on
edit-only platforms like Discord. The shipped defaults encode that:
display.platforms.telegram.streaming=true, .discord.streaming=false. These are
gap-fillers (user values win via deep-merge) and, because the dashboard schema
is generated from DEFAULT_CONFIG, they automatically appear as editable toggles
in the web UI.
"""

from __future__ import annotations


def test_default_per_platform_streaming_flags():
    from hermes_cli.config import DEFAULT_CONFIG
    plats = DEFAULT_CONFIG["display"]["platforms"]
    assert plats["telegram"]["streaming"] is True
    assert plats["discord"]["streaming"] is False


def test_resolver_telegram_on_discord_off_when_global_enabled():
    """With global streaming on, the per-platform defaults make Telegram stream,
    Discord not, and native-stream WeCom follow the global switch."""
    from hermes_cli.config import DEFAULT_CONFIG
    from gateway.display_config import resolve_display_setting

    cfg = dict(DEFAULT_CONFIG)
    cfg["streaming"] = {"enabled": True, "transport": "auto"}

    def streams(plat):
        ov = resolve_display_setting(cfg, plat, "streaming")
        # global enabled; None override = follow global (True)
        return True if ov is None else bool(ov)

    assert streams("telegram") is True
    assert streams("discord") is False
    assert streams("wecom") is True
    assert streams("wecom_callback") is True
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


def test_dashboard_schema_exposes_per_platform_streaming():
    """Because the web settings schema is built from DEFAULT_CONFIG, the
    per-platform streaming toggles surface in the dashboard automatically."""
    import pytest
    pytest.importorskip("fastapi")  # web_server requires fastapi/uvicorn
    from hermes_cli.web_server import CONFIG_SCHEMA

    assert "display.platforms.telegram.streaming" in CONFIG_SCHEMA
    assert "display.platforms.discord.streaming" in CONFIG_SCHEMA
    assert CONFIG_SCHEMA["display.platforms.discord.streaming"]["type"] == "boolean"
    # Global streaming controls are exposed too.
    assert "streaming.enabled" in CONFIG_SCHEMA
    assert "streaming.transport" in CONFIG_SCHEMA
