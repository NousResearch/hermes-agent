"""Copied assistant text must not carry terminal decoration into the clipboard.

The TUI renders answers with ANSI colour, box-drawing borders and icon glyphs
from a patched font. Those are display artifacts: pasting them into a document,
an email or a ticket produces mojibake and stray rules. Copying should yield the
text the user can read, not the bytes the terminal drew.
"""
from __future__ import annotations

from cli import _clean_copied_text


def test_ansi_escape_sequences_are_removed():
    assert _clean_copied_text("\x1b[31mvarning\x1b[0m") == "varning"


def test_osc_sequences_are_removed():
    assert _clean_copied_text("\x1b]0;title\x07hej") == "hej"


def test_box_drawing_glyphs_are_removed():
    assert "─" not in _clean_copied_text("── rubrik ──")
    assert "rubrik" in _clean_copied_text("── rubrik ──")


def test_private_use_icons_are_removed():
    """Nerd-font icons live in the private use area."""
    cleaned = _clean_copied_text(" fil.py")
    assert "" not in cleaned
    assert cleaned.strip() == "fil.py"


def test_indentation_is_preserved():
    """Leading whitespace is content in a code block, not decoration."""
    assert _clean_copied_text("def f():\n    return 1") == "def f():\n    return 1"


def test_decoration_between_words_becomes_a_space():
    """Removing a glyph must not weld two words together."""
    assert _clean_copied_text("vänster│höger") == "vänster höger"


def test_ordinary_text_survives_unchanged():
    assert _clean_copied_text("Kabeln är 5G2.5 och kostar 42 kr/m.") == "Kabeln är 5G2.5 och kostar 42 kr/m."


def test_meaningful_punctuation_is_not_decoration():
    """An arrow or em dash carries meaning in prose — only frame glyphs go."""
    assert _clean_copied_text("A → B") == "A → B"
    assert _clean_copied_text("Katrineholm — åäöÅÄÖ") == "Katrineholm — åäöÅÄÖ"


def test_crlf_is_normalised():
    assert _clean_copied_text("rad1\r\nrad2") == "rad1\nrad2"


def test_del_and_c1_controls_are_removed():
    """DEL (U+007F) and C1 controls (U+0080–U+009F) are not content."""
    assert _clean_copied_text("a\x7fb\x9cc") == "abc"
    assert _clean_copied_text("a\tb") == "a\tb"


def test_decoration_followed_by_existing_space_does_not_widen_the_run():
    """A removed glyph must not widen a following run of spaces."""
    assert _clean_copied_text("vänster│ höger") == "vänster höger"
    assert _clean_copied_text("vänster│   höger") == "vänster   höger"


def test_leading_and_trailing_blank_lines_are_trimmed():
    assert _clean_copied_text("\n\n  \nhej\n\n \n") == "hej"


def test_internal_blank_lines_are_kept():
    assert _clean_copied_text("stycke1\n\nstycke2") == "stycke1\n\nstycke2"


def test_trailing_whitespace_per_line_is_trimmed():
    assert _clean_copied_text("rad   \nnästa\t") == "rad\nnästa"
