"""Regression tests for Discord message cascade guardrails."""

from gateway.platforms.discord import (
    _looks_like_gateway_status_echo,
    _sanitize_discord_outbound_content,
)


def test_outbound_sanitizer_removes_unwanted_marker_and_malformed_mentions():
    raw = "Ack " + chr(0x2589) + " <@123456789 trailing <@!987654321> ok <@&111222333>"

    sanitized = _sanitize_discord_outbound_content(raw)

    assert chr(0x2589) not in sanitized
    assert "<@123456789" not in sanitized
    assert "<@!987654321>" in sanitized
    assert "<@&111222333>" in sanitized


def test_outbound_sanitizer_removes_tool_preview_prefixes_from_reply_echoes():
    raw = "📖 read_file(['path'])\n{\"path\": \"x\"}\nActual response"

    sanitized = _sanitize_discord_outbound_content(raw)

    assert "read_file(['path'])" not in sanitized
    assert sanitized == "Actual response"


def test_gateway_status_echo_detection_catches_interrupt_notifications():
    assert _looks_like_gateway_status_echo(
        "⚡ Interrupting current task (iteration 3/1000). I'll respond to your message shortly."
    )
    assert _looks_like_gateway_status_echo("Operation interrupted: waiting for model response (6.0s elapsed).")
    assert not _looks_like_gateway_status_echo("please implement the Discord guardrail now")
