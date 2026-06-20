import sys
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object, button=lambda *a, **k: (lambda fn: fn), Button=object
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import (  # noqa: E402
    _wrap_markdown_tables_for_discord,
    _render_table_block_for_discord,
    _render_table_compact,
    _display_width,
    _strip_cell_formatting,
)

_SIMPLE_TABLE = "| Name | Value |\n|------|-------|\n| foo  | bar   |"


class TestWrapMarkdownTablesForDiscord:
    def test_no_pipes_returns_identity(self):
        assert _wrap_markdown_tables_for_discord("hello world") == "hello world"

    def test_empty_string_returns_empty(self):
        assert _wrap_markdown_tables_for_discord("") == ""

    def test_incomplete_table_no_delimiter_unchanged(self):
        text = "| Name | Value |"
        assert _wrap_markdown_tables_for_discord(text) == text

    def test_complete_table_produces_code_block(self):
        result = _wrap_markdown_tables_for_discord(_SIMPLE_TABLE)
        assert result.startswith("```\n")
        assert result.endswith("\n```")
        assert "Name" in result
        assert "foo" in result

    def test_budget_none_always_box_drawing(self):
        result = _wrap_markdown_tables_for_discord(_SIMPLE_TABLE, budget=None)
        assert result.startswith("```\n")

    def test_budget_large_enough_uses_box_drawing(self):
        result = _wrap_markdown_tables_for_discord(_SIMPLE_TABLE, budget=2000)
        assert result.startswith("```\n")

    def test_budget_too_small_uses_compact_fallback(self):
        # box-drawing output is ~92 chars; budget=50 forces compact
        result = _wrap_markdown_tables_for_discord(_SIMPLE_TABLE, budget=50)
        assert not result.startswith("```")
        assert "Name: foo" in result
        assert "Value: bar" in result

    def test_table_inside_code_block_unchanged(self):
        text = "```\n| Name | Value |\n|------|-------|\n| foo  | bar   |\n```"
        assert _wrap_markdown_tables_for_discord(text) == text

    def test_table_with_surrounding_text_preserved(self):
        text = "Before\n" + _SIMPLE_TABLE + "\nAfter"
        result = _wrap_markdown_tables_for_discord(text)
        assert result.startswith("Before\n")
        assert result.endswith("\nAfter")
        assert "```" in result

    def test_adjacent_tables_not_merged(self):
        text = (
            "| A | B |\n|---|---|\n| x | y |\n"
            "| C | D |\n|---|---|\n| p | q |"
        )
        result = _wrap_markdown_tables_for_discord(text)
        # Both tables should be converted, each in its own code block
        assert result.count("```") == 4  # two opening + two closing fences

    def test_single_column_table_converted(self):
        text = "| Value |\n|-------|\n| foo   |"
        result = _wrap_markdown_tables_for_discord(text)
        assert result.startswith("```\n")
        assert "foo" in result

    def test_pipe_in_prose_followed_by_hr_not_converted(self):
        text = "Use | for alternatives\n---\nSome text after"
        result = _wrap_markdown_tables_for_discord(text)
        assert result == text


class TestRenderTableCompact:
    _BLOCK = [
        "| Name | Value |",
        "|------|-------|",
        "| foo  | bar   |",
        "| baz  | qux   |",
    ]

    def test_two_data_rows_one_line_each(self):
        result = _render_table_compact(self._BLOCK)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_first_row_contains_both_field_value_pairs(self):
        result = _render_table_compact(self._BLOCK)
        first = result.strip().split("\n")[0]
        assert "Name: foo" in first
        assert "Value: bar" in first

    def test_second_row_contains_second_data_row(self):
        result = _render_table_compact(self._BLOCK)
        second = result.strip().split("\n")[1]
        assert "Name: baz" in second
        assert "Value: qux" in second

    def test_columns_separated_by_dot(self):
        result = _render_table_compact(self._BLOCK)
        assert "  ·  " in result

    def test_no_code_block_fences(self):
        result = _render_table_compact(self._BLOCK)
        assert "```" not in result

    def test_single_line_block_returns_joined(self):
        assert _render_table_compact(["| A |"]) == "| A |"

    def test_header_only_table_returns_raw(self):
        block = ["| A | B |", "|---|---|"]
        result = _render_table_compact(block)
        # Should not return empty string
        assert result != ""
        assert "A" in result


class TestDisplayWidth:
    def test_ascii_char_is_width_1(self):
        assert _display_width("A") == 1

    def test_wide_cjk_char_is_width_2(self):
        assert _display_width("中") == 2

    def test_empty_string_is_zero(self):
        assert _display_width("") == 0

    def test_mixed_string(self):
        assert _display_width("A中") == 3


class TestStripCellFormatting:
    def test_strips_bold_asterisks(self):
        assert _strip_cell_formatting("**Bold**") == "Bold"

    def test_strips_italic_underscore(self):
        assert _strip_cell_formatting("_italic_") == "italic"

    def test_strips_strikethrough(self):
        assert _strip_cell_formatting("~~strike~~") == "strike"

    def test_plain_text_unchanged(self):
        assert _strip_cell_formatting("hello") == "hello"

    def test_preserves_snake_case(self):
        assert _strip_cell_formatting("user_id") == "user_id"

    def test_preserves_double_underscore(self):
        assert _strip_cell_formatting("__init__") == "__init__"


from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

import discord as _discord_mod  # noqa: E402 — after mock setup


class TestSendTableConversion:
    _TABLE = "| Name | Value |\n|------|-------|\n| foo  | bar   |"

    @pytest.mark.asyncio
    async def test_send_converts_table_to_code_block(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        sent = []

        async def fake_send(*, content, reference=None):
            sent.append(content)
            return SimpleNamespace(id=1)

        channel = SimpleNamespace(
            send=AsyncMock(side_effect=fake_send),
            fetch_message=AsyncMock(),
        )
        adapter._client = SimpleNamespace(
            get_channel=lambda _: channel,
            fetch_channel=AsyncMock(),
        )

        result = await adapter.send("123", self._TABLE)

        assert result.success is True
        assert len(sent) == 1
        assert sent[0].startswith("```\n")

    @pytest.mark.asyncio
    async def test_send_to_forum_converts_table_to_code_block(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

        thread_ch = SimpleNamespace(id=777, send=AsyncMock())
        thread = SimpleNamespace(id=777, message=SimpleNamespace(id=800), thread=thread_ch)
        forum_channel = _discord_mod.ForumChannel()
        forum_channel.id = 999
        forum_channel.name = "forum"

        create_args = []

        async def fake_create_thread(**kwargs):
            create_args.append(kwargs)
            return thread

        forum_channel.create_thread = AsyncMock(side_effect=fake_create_thread)

        result = await adapter._send_to_forum(forum_channel, self._TABLE)

        assert result.success is True
        assert len(create_args) == 1
        assert create_args[0]["content"].startswith("```\n")


class TestEditMessageTableConversion:
    _TABLE = "| Name | Value |\n|------|-------|\n| foo  | bar   |"

    @pytest.mark.asyncio
    async def test_intermediate_edit_skips_table_conversion(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        edited = []
        msg = SimpleNamespace(
            edit=AsyncMock(side_effect=lambda **kw: edited.append(kw["content"]))
        )
        channel = SimpleNamespace(fetch_message=AsyncMock(return_value=msg))
        adapter._client = SimpleNamespace(
            get_channel=lambda _: channel,
            fetch_channel=AsyncMock(),
        )

        await adapter.edit_message("123", "456", self._TABLE, finalize=False)

        assert len(edited) == 1
        assert not edited[0].startswith("```")
        assert "|" in edited[0]

    @pytest.mark.asyncio
    async def test_finalize_edit_converts_table_to_code_block(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        edited = []
        msg = SimpleNamespace(
            edit=AsyncMock(side_effect=lambda **kw: edited.append(kw["content"]))
        )
        channel = SimpleNamespace(fetch_message=AsyncMock(return_value=msg))
        adapter._client = SimpleNamespace(
            get_channel=lambda _: channel,
            fetch_channel=AsyncMock(),
        )

        await adapter.edit_message("123", "456", self._TABLE, finalize=True)

        assert len(edited) == 1
        assert edited[0].startswith("```\n")

    @pytest.mark.asyncio
    async def test_finalize_edit_budget_overflow_uses_compact_not_ellipsis(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        adapter.MAX_MESSAGE_LENGTH = 50  # box-drawing ~92 chars exceeds budget; compact ~24 fits

        edited = []
        msg = SimpleNamespace(
            edit=AsyncMock(side_effect=lambda **kw: edited.append(kw["content"]))
        )
        channel = SimpleNamespace(fetch_message=AsyncMock(return_value=msg))
        adapter._client = SimpleNamespace(
            get_channel=lambda _: channel,
            fetch_channel=AsyncMock(),
        )

        await adapter.edit_message("123", "456", self._TABLE, finalize=True)

        assert len(edited) == 1
        content = edited[0]
        assert not content.endswith("...")
        assert not content.startswith("```")
        assert "foo" in content
        assert "bar" in content
