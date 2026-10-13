"""The (×N) repeat marker must not break a fenced tool-progress block.

Appending it to the LAST line put it on the closing fence (```` ``` (×2) ````),
which the MarkdownV2 send-time closer (``(^[ \\t]*```)[ \\t]*\\r?$``) no longer
matches: the block never closes, the fence chars get escaped to literals, and
the bubble renders as raw backticks.
"""

import pytest

from gateway.run_turn_runner import _progress_repeat_marker

B3 = "```"


def _renders_as_code_block(text: str) -> bool:
    """True when the Telegram formatter keeps the fence verbatim (a real code block)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    out = TelegramAdapter.format_message(TelegramAdapter, text)
    return out.count(B3) >= 2


class TestProgressRepeatMarker:
    def test_header_block_marker_goes_on_header_not_closing_fence(self):
        block = f"\U0001F4BB terminal\n{B3}\ncd /x && ls -la\n{B3}"
        marked = _progress_repeat_marker(block, 2)
        assert marked.splitlines()[-1] == B3, "closing fence must stay bare"
        assert "(×2)" in marked.splitlines()[0]
        assert marked.count("(×2)") == 1

    def test_headerless_block_marker_gets_its_own_line(self):
        block = f"{B3}\ncd /x && ls -la\n{B3}"
        marked = _progress_repeat_marker(block, 2)
        assert marked.splitlines()[-1] == "(×2)"
        assert marked.splitlines()[-2] == B3, "closing fence stays bare"

    def test_plain_line_marker_stays_inline(self):
        assert _progress_repeat_marker("\U0001F40D Running code", 3) == "\U0001F40D Running code (×3)"

    def test_marked_fenced_block_still_renders_as_a_code_block(self):
        block = f"\U0001F4BB terminal\n{B3}\ncd /x && ls -la\n{B3}"
        marked = _progress_repeat_marker(block, 2)
        assert _renders_as_code_block(marked)
        # the pre-fix form is the regression this guards against
        assert not _renders_as_code_block(block + " (×2)")

    def test_headerless_marked_block_still_renders_as_a_code_block(self):
        block = f"{B3}\ncd /x && ls -la\n{B3}"
        assert _renders_as_code_block(_progress_repeat_marker(block, 4))
