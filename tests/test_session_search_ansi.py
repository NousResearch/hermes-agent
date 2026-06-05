"""session_search surfaces stored message text to the model; real ANSI control
bytes must be stripped from that surfaced copy (cf. tools/ansi_strip.py), while
printable text that merely *describes* escape codes must be preserved."""
from tools.session_search_tool import _clean_surfaced_text


def test_strips_real_escape_sequences():
    assert _clean_surfaced_text("\x1b[31mRed\x1b[0m") == "Red"
    assert _clean_surfaced_text("\x1b]0;window-title\x07keep") == "keep"


def test_preserves_plain_and_descriptive_text():
    # Printable look-alikes (no ESC/C1 byte) are NOT escape sequences. Keep them so
    # search results that discuss ANSI codes (docs, code samples) are not corrupted.
    assert _clean_surfaced_text("]11;rgb:2828/2c2c/3434 note") == "]11;rgb:2828/2c2c/3434 note"
    assert _clean_surfaced_text("see `\\x1b]11;rgb` in the docs") == "see `\\x1b]11;rgb` in the docs"
    assert _clean_surfaced_text("array[0] index") == "array[0] index"


def test_none_and_empty_safe():
    assert _clean_surfaced_text(None) is None
    assert _clean_surfaced_text("") == ""
