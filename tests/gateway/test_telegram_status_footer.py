"""Tests for Telegram status footer helpers."""

from gateway.config import Platform
from gateway.telegram_status_footer import (
    build_telegram_status_footer,
    maybe_append_telegram_status_footer,
)


class TestBuildTelegramStatusFooter:
    def test_formats_model_reasoning_and_usage(self):
        footer = build_telegram_status_footer(
            model="gpt-5.4",
            reasoning_effort="low",
            current_tokens=45000,
            context_window=128000,
        )

        assert footer == (
            "────────────\n"
            "🧠 gpt-5.4 · 💨 low\n"
            "📊 45k / 128k · 35%"
        )

    def test_uses_default_when_reasoning_is_missing(self):
        footer = build_telegram_status_footer(
            model="gpt-5.4",
            reasoning_effort=None,
            current_tokens=999,
            context_window=2000,
        )

        assert "💨 default" in footer
        assert "📊 999 / 2k · 50%" in footer


class TestMaybeAppendTelegramStatusFooter:
    def test_appends_for_telegram_dm_only(self):
        result = maybe_append_telegram_status_footer(
            "Hello world",
            platform=Platform.TELEGRAM,
            chat_type="dm",
            model="gpt-5.4",
            reasoning_effort="low",
            current_tokens=45000,
            context_window=128000,
        )

        assert result.endswith("📊 45k / 128k · 35%")
        assert "Hello world\n\n────────────" in result

    def test_skips_for_non_dm_or_non_telegram(self):
        group_result = maybe_append_telegram_status_footer(
            "Hello group",
            platform=Platform.TELEGRAM,
            chat_type="group",
            model="gpt-5.4",
            reasoning_effort="low",
            current_tokens=45000,
            context_window=128000,
        )
        discord_result = maybe_append_telegram_status_footer(
            "Hello discord",
            platform=Platform.DISCORD,
            chat_type="dm",
            model="gpt-5.4",
            reasoning_effort="low",
            current_tokens=45000,
            context_window=128000,
        )

        assert group_result == "Hello group"
        assert discord_result == "Hello discord"

    def test_skips_when_footer_already_present(self):
        content = (
            "Hello\n\n"
            "────────────\n"
            "🧠 gpt-5.4 · 💨 low\n"
            "📊 45k / 128k · 35%"
        )

        result = maybe_append_telegram_status_footer(
            content,
            platform=Platform.TELEGRAM,
            chat_type="dm",
            model="gpt-5.4",
            reasoning_effort="low",
            current_tokens=45000,
            context_window=128000,
        )

        assert result == content
