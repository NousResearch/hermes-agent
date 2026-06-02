"""Tests for Discord markdown table formatting.

Discord does not render GFM pipe tables. ``_wrap_tables_in_code_fence``
detects tables, converts them to aligned plaintext, and wraps them in
triple-backtick fences so they render readably in Discord.
"""

from plugins.platforms.discord.adapter import _wrap_tables_in_code_fence


class TestWrapTablesInCodeFence:
    def test_unaligned_markdown_table_becomes_readable_plaintext(self):
        text = (
            "Model | Latency | Notes\n"
            "---|---|---\n"
            "GPT-5.5 | 120ms | fast\n"
            "Mini | 9ms | cheap"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result == (
            "```\n"
            "Model   | Latency | Notes\n"
            "------- | ------- | -----\n"
            "GPT-5.5 | 120ms   | fast\n"
            "Mini    | 9ms     | cheap\n"
            "```"
        )

    def test_table_inside_code_fence_is_untouched(self):
        text = (
            "```\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "```"
        )
        assert _wrap_tables_in_code_fence(text) == text

    def test_pipe_without_separator_is_untouched(self):
        text = "Use the | operator in bash for piping."
        assert _wrap_tables_in_code_fence(text) == text

    def test_ragged_rows_are_padded(self):
        text = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| 1 | 2 |\n"
            "| 3 | 4 | 5 |"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result == (
            "```\n"
            "A | B | C\n"
            "--- | --- | ---\n"
            "1 | 2 |\n"
            "3 | 4 | 5\n"
            "```"
        )

    def test_format_message_integration(self):
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = DiscordAdapter.__new__(DiscordAdapter)
        text = (
            "| X | Y |\n"
            "|---|---|\n"
            "| 1 | 2 |"
        )
        result = adapter.format_message(text)
        assert result == (
            "```\n"
            "X | Y\n"
            "--- | ---\n"
            "1 | 2\n"
            "```"
        )
