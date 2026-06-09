"""FGD-133: normal Telegram/Discord chats must not expose raw tool telemetry."""

from __future__ import annotations


def test_telegram_and_discord_suppress_raw_tool_progress_policy():
    from gateway.run import (
        _safe_gateway_tool_progress_message,
        _should_suppress_raw_tool_progress,
    )

    for platform in ("telegram", "discord"):
        assert _should_suppress_raw_tool_progress(platform) is True
        msg = _safe_gateway_tool_progress_message(platform)

        assert msg
        assert "terminal" not in msg
        assert "execute_code" not in msg
        assert "skill_view" not in msg
        assert "python" not in msg.lower()
        assert "command" not in msg.lower()
        assert "/Users/" not in msg


def test_non_chat_platforms_keep_existing_tool_progress_policy_surface():
    from gateway.run import _should_suppress_raw_tool_progress

    # FGD-133 only hardens normal Telegram/Discord user-visible chat output.
    # Other platform behavior remains governed by their display settings and
    # can be handled separately if a bounded issue approves it.
    assert _should_suppress_raw_tool_progress("api_server") is False
    assert _should_suppress_raw_tool_progress("local") is False


def test_discord_default_tool_progress_is_off_for_normal_chat():
    from gateway.display_config import resolve_display_setting

    assert resolve_display_setting({}, "discord", "tool_progress") == "off"
    assert resolve_display_setting({}, "telegram", "tool_progress") == "off"
