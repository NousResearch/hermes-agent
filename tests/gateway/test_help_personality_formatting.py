from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/help", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    return runner


class TestHelpFormatting:
    @pytest.mark.asyncio
    async def test_help_output_groups_commands_into_sections(self):
        runner = _make_runner()

        result = await runner._handle_help_command(_make_event(text="/help"))

        assert "📖 **Hermes Commands**" in result
        assert "**Session**" in result
        assert "**Configuration**" in result
        assert "**Context**" in result
        assert "**Maintenance**" in result
        assert "`/new` — Start a new session with a fresh conversation" in result
        assert "`/personality [name]` — List or switch personalities" in result
        assert "`/update` — Update Hermes Agent to the latest version" in result

    @pytest.mark.asyncio
    async def test_help_output_separates_skill_commands_section(self):
        runner = _make_runner()

        with patch("agent.skill_commands.get_skill_commands", return_value={
            "/gif-search": {"description": "Search for GIFs across providers"},
        }):
            result = await runner._handle_help_command(_make_event(text="/help"))

        assert "⚡ **Skill Commands** (1 installed)" in result
        assert "\n\n⚡ **Skill Commands** (1 installed)\n`/gif-search` — Search for GIFs across providers" in result


class TestPersonalityFormatting:
    @pytest.mark.asyncio
    async def test_personality_list_is_sorted_and_separated(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "agent:\n"
            "  personalities:\n"
            "    zebra: |\n"
            "      A personality with extra spacing\n"
            "      for tests.\n"
            "    alpha: 'Short and direct.'\n"
        )

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._handle_personality_command(
                _make_event(text="/personality")
            )

        assert result.startswith("🎭 **Available Personalities**")
        assert "\n\n`alpha`\nShort and direct." in result
        assert "\n\n`zebra`\nA personality with extra spacing for tests." in result
        assert result.index("`alpha`") < result.index("`zebra`")
        assert result.endswith("Use `/personality <name>` to switch.")

    @pytest.mark.asyncio
    async def test_personality_preview_truncates_long_prompt(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "agent:\n"
            "  personalities:\n"
            "    builder: '"
            + ("very long prompt " * 10)
            + "'\n"
        )

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._handle_personality_command(
                _make_event(text="/personality")
            )

        assert "`builder`" in result
        assert "..." in result
        assert "very long prompt very long prompt" in result
