"""Tests for Discord markdown table formatting (issue #21168).

Discord does not render GFM pipe tables. ``_wrap_tables_in_code_fence``
detects tables, converts them to aligned plaintext, and wraps them in
triple-backtick fences so they render readably in Discord.
"""

from plugins.platforms.discord.adapter import _wrap_tables_in_code_fence


class TestWrapTablesInCodeFence:
    """Unit tests for the table-formatting helper."""

    def test_basic_table_is_aligned_and_wrapped(self):
        text = (
            "| Name | Value |\n"
            "|------|-------|\n"
            "| foo  | 1     |\n"
            "| bar  | 2     |"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result == (
            "```\n"
            "Name | Value\n"
            "---- | -----\n"
            "foo  | 1\n"
            "bar  | 2\n"
            "```"
        )

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

    def test_table_with_surrounding_text(self):
        text = (
            "Here is a comparison:\n"
            "\n"
            "| Model | Speed |\n"
            "|-------|-------|\n"
            "| A     | fast  |\n"
            "| B     | slow  |\n"
            "\n"
            "Hope that helps!"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result.startswith("Here is a comparison:\n")
        assert result.endswith("\nHope that helps!")
        assert "```\nModel | Speed" in result
        assert "B     | slow\n```" in result

    def test_table_inside_code_fence_is_untouched(self):
        text = (
            "```\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "```"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result == text

    def test_no_table_passthrough(self):
        text = "Just a regular message with no pipes or tables."
        assert _wrap_tables_in_code_fence(text) == text

    def test_pipe_without_separator_is_untouched(self):
        text = "Use the | operator in bash for piping."
        assert _wrap_tables_in_code_fence(text) == text

    def test_empty_input(self):
        assert _wrap_tables_in_code_fence("") == ""

    def test_table_without_leading_pipe(self):
        text = (
            "Name | Value\n"
            "------|-------\n"
            "foo  | 1\n"
            "bar  | 2"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result == (
            "```\n"
            "Name | Value\n"
            "---- | -----\n"
            "foo  | 1\n"
            "bar  | 2\n"
            "```"
        )

    def test_multiple_tables(self):
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "\n"
            "Some text\n"
            "\n"
            "| C | D |\n"
            "|---|---|\n"
            "| 3 | 4 |"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result.count("```") == 4
        assert "A | B\n--- | ---\n1 | 2" in result
        assert "C | D\n--- | ---\n3 | 4" in result

    def test_table_with_alignment_row(self):
        text = (
            "| Left | Center | Right |\n"
            "|:-----|:------:|------:|\n"
            "| a    | b      | c     |"
        )
        result = _wrap_tables_in_code_fence(text)
        assert result == (
            "```\n"
            "Left | Center | Right\n"
            "---- | ------ | -----\n"
            "a    | b      | c\n"
            "```"
        )

    def test_single_column_separator_not_matched(self):
        text = "-----\nnot a table"
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
        """Verify format_message calls the wrapping function."""
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
