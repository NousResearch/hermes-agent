"""Regression test for _STREAM_PAD — ensures streamed CLI output does not
trigger CommonMark's code-block rule (4+ leading spaces = <pre><code>).

Related: https://github.com/NousResearch/hermes-agent/pull/46557

Background
----------
The original `_STREAM_PAD` was hardcoded to 4 spaces, which meant every
streamed response line started with 4 leading spaces.  Per the CommonMark
spec, any text block with 4+ leading spaces is rendered as a code block.
This caused copied CLI output to render as ``<pre><code>`` in GitHub
issues, PR comments, and any other Markdown environment.

The fix reduces the prefix to 2 spaces, which stays below the 4-space
threshold while still providing visual indentation.
"""


from cli import _STREAM_PAD


class TestStreamPadLength:
    """_STREAM_PAD must be short enough to avoid Markdown code blocks."""

    def test_stream_pad_below_code_block_threshold(self):
        """CommonMark code-block rule triggers at 4+ leading spaces."""
        assert len(_STREAM_PAD) < 4, (
            f"_STREAM_PAD is {len(_STREAM_PAD)} spaces; "
            "values >= 4 trigger CommonMark code blocks when pasted "
            "into GitHub issues, PR comments, etc."
        )

    def test_stream_pad_is_two_spaces(self):
        """Regression: _STREAM_PAD should be exactly 2 spaces."""
        assert _STREAM_PAD == "  ", (
            f"_STREAM_PAD is {len(_STREAM_PAD)}-space string "
            f"({_STREAM_PAD!r}); expected exactly 2 spaces"
        )

    def test_stream_pad_is_spaces_only(self):
        """_STREAM_PAD must not contain tabs or other whitespace."""
        assert _STREAM_PAD.strip(" ") == "", (
            "_STREAM_PAD should contain only space characters"
        )


class TestStreamPadUsage:
    """Verify _STREAM_PAD produces output that stays below 4-space indent."""

    def test_streamed_line_prefix(self):
        """A typical streamed line should start with <4 leading spaces."""
        line = _STREAM_PAD + "hello world"
        # Count leading whitespace
        leading = len(line) - len(line.lstrip(" "))
        assert leading < 4, (
            f"Streamed line has {leading} leading spaces; "
            "this will render as a code block in Markdown"
        )
