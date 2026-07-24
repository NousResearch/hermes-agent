"""Regression tests for terminal-escape stripping in SessionDB.sanitize_title.

Two classes of residue are covered:

1. ESC-anchored sequences (`\x1b[31m…`, `\x1b]11;…\x07`). Before the fix,
   sanitize_title deleted the ESC anchor byte (0x1b is within the 0x0e-0x1f
   control-char class) but left the printable body, so "\x1b[31mRed\x1b[0m"
   became the visible-garbage title "[31mRed[0m". strip_ansi() now runs first.

2. Bare OSC residue (`]11;rgb:…`, `]0;title`) with no ESC byte. Some transports
   (e.g. Telegram) drop the 0x1b control byte in transit, leaving only the
   printable OSC body, which strip_ansi (ESC-anchored) cannot see. A narrow,
   digit-required, whitespace-terminated strip removes it without touching
   ordinary bracketed text like "[Note]" or "array[0]".
"""
import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize(
    "raw, expected",
    [
        # ESC-anchored
        ("\x1b[31mRed\x1b[0m", "Red"),
        ("\x1b[1mHello\x1b[0m World", "Hello World"),
        ("\x1b]11;rgb:2828/2c2c/3434\x07 deploy", "deploy"),
        # bare OSC residue (transport dropped the ESC byte)
        ("]11;rgb:2828/2c2c/3434 deploy", "deploy"),
        ("]0;window-title done", "done"),
        # must be preserved (no false positives)
        ("plain title", "plain title"),
        ("[Note] keeps brackets", "[Note] keeps brackets"),
        ("array[0] index", "array[0] index"),
        ("foo]3;bar baz", "foo]3;bar baz"),  # ]N; preceded by a word char -> not residue
        ("Résumé \U0001f680", "Résumé \U0001f680"),
    ],
)
def test_sanitize_title_strips_escapes_preserves_text(raw, expected):
    assert SessionDB.sanitize_title(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "\x1b]11;rgb:2828/2c2c/3434\x07",                  # ESC OSC colour, BEL-terminated
        "\x1b]8;;http://example.com\x1b\\\x1b]8;;\x1b\\",  # ESC OSC-8 hyperlink wrapper, no text
        "\x1b[0m\x1b[1m\x1b[31m",                          # ESC escapes only
        "]11;rgb:2828/2c2c/3434",                          # bare OSC only
    ],
)
def test_sanitize_title_all_escape_input_becomes_none(raw):
    # Stripped to empty -> normalized to None (existing contract).
    assert SessionDB.sanitize_title(raw) is None


def test_sanitize_title_leaves_no_escape_residue():
    raw = "\x1b[32mDeploy\x1b[0m \x1b]0;window-title\x07 done"
    out = SessionDB.sanitize_title(raw)
    assert out is not None
    assert "\x1b" not in out
    assert "[32m" not in out and "]0;" not in out
