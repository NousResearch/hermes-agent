"""Regression tests for #72580: approval panel multi-line text wrapping.

_wrap_panel_text must split on newlines before wrapping. textwrap.wrap
treats embedded \\n as ordinary whitespace, so a multi-line command (e.g.
a heredoc pending approval) previously collapsed into a few long lines
with literal newlines, pushing the approve/deny choices off-screen.
"""

from cli import _wrap_clarify_panel_text, _wrap_panel_text


HEREDOC = (
    "python3 << 'EOF'\n"
    "import sqlite3\n"
    "conn = sqlite3.connect('db')\n"
    "cur.execute('SELECT * FROM sessions')\n"
    "print(cur.fetchall())\n"
    "conn.close()\n"
    "EOF"
)


def test_multiline_command_keeps_one_display_line_per_source_line():
    lines = _wrap_panel_text(HEREDOC, width=60)
    # 7 source lines, all short: one display line each, in order.
    assert lines == HEREDOC.split("\n")


def test_no_embedded_newlines_in_output():
    lines = _wrap_panel_text(HEREDOC, width=20)
    assert all("\n" not in line for line in lines)
    # Narrow width wraps long source lines but never merges source lines.
    assert len(lines) > len(HEREDOC.split("\n"))


def test_empty_lines_are_preserved():
    lines = _wrap_panel_text("a\n\nb", width=60)
    assert lines == ["a", "", "b"]


def test_all_line_ending_styles_are_normalized_without_losing_empty_lines():
    expected = ["a", "", "b", ""]
    for text in ("a\n\nb\n", "a\r\n\r\nb\r\n", "a\r\rb\r"):
        lines = _wrap_panel_text(text, width=60)
        assert lines == expected
        assert all("\r" not in line and "\n" not in line for line in lines)


def test_single_line_behaviour_unchanged():
    assert _wrap_panel_text("short", width=60) == ["short"]
    long_line = "x" * 130
    lines = _wrap_panel_text(long_line, width=60)
    assert lines == ["x" * 60, "x" * 60, "x" * 10]


def test_empty_input_returns_single_empty_line():
    assert _wrap_panel_text("", width=60) == [""]


def test_subsequent_indent_applies_to_continuations_only():
    # width floors at 8, so use text longer than 8 chars to force wrapping.
    lines = _wrap_panel_text("aa bb cc dd ee", width=4, subsequent_indent="> ")
    assert lines[0].startswith("aa")
    assert all(line.startswith("> ") for line in lines[1:])
    assert all(len(line) <= 8 for line in lines)


def test_narrow_width_floor():
    # width is floored at 8, matching the previous inline behaviour.
    lines = _wrap_panel_text("x" * 20, width=2)
    assert lines == ["x" * 8, "x" * 8, "x" * 4]


def test_clarify_panel_prose_kwargs_do_not_break_words():
    # The clarify panel delegates with break_long_words=False and
    # break_on_hyphens=False; long words must survive intact.
    word = "supercalifragilisticexpialidocious"
    lines = _wrap_panel_text(
        f"hello {word} world",
        width=10,
        break_long_words=False,
        break_on_hyphens=False,
    )
    assert any(word in line for line in lines)


# --- Clarify panel delegation path (_wrap_clarify_panel_text) ---


def test_clarify_wrapper_splits_multiline_question():
    # Rendering-path regression: a multi-line clarify question must keep
    # one display line per source line instead of collapsing.
    question = "Which database should I use?\nContext:\nthe sessions table is large"
    lines = _wrap_clarify_panel_text(question, width=72)
    assert lines == question.split("\n")
    assert all("\n" not in line for line in lines)


def test_clarify_wrapper_keeps_prose_wrapping():
    # Prose kwargs preserved through the delegation: words never break,
    # even when longer than the width.
    word = "supercalifragilisticexpialidocious"
    lines = _wrap_clarify_panel_text(f"Pick one: {word} or the other option", width=10)
    assert any(word in line for line in lines)
    assert all("\n" not in line for line in lines)


def test_clarify_wrapper_preserves_empty_lines():
    assert _wrap_clarify_panel_text("question?\n\nmore context", width=72) == [
        "question?",
        "",
        "more context",
    ]
