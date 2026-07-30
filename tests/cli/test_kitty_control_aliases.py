"""Regression tests for Kitty keyboard protocol CSI-u control key sequences.

When ``enable_kitty_keyboard_protocol()`` pushes the terminal into CSI-u
mode, *every* key press is sent as a CSI-u sequence — not just
Shift+Enter and Ctrl+Enter.  prompt_toolkit's stock ``ANSI_SEQUENCES``
table contains only four CSI-u entries (codepoint 1, modifier 5–8).

Without ``install_kitty_control_aliases()``, sequences like ``\\x1b[99;5u``
(Ctrl+C) are unmapped.  The VT100 parser fires ``Keys.Escape`` on the
leading ESC byte, then inserts ``[99;5u`` as literal text — making
Ctrl+C, Ctrl+D, Escape, etc. unusable.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import (
    Vt100Parser,
    _IS_PREFIX_OF_LONGER_MATCH_CACHE,
)
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_kitty_control_aliases


def _parse(byte_seq: str):
    """Feed *byte_seq* through a fresh VT100 parser and return KeyPress list."""
    out: list = []
    parser = Vt100Parser(out.append)
    for ch in byte_seq:
        parser.feed(ch)
    parser.flush()
    return out


def _parse_bulk(byte_seq: str):
    """Feed *byte_seq* in one shot via ``feed_and_flush``."""
    out: list = []
    parser = Vt100Parser(out.append)
    parser.feed_and_flush(byte_seq)
    return out


# ---- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_aliases_installed():
    """Make every test idempotent — install the aliases once per test run."""
    install_kitty_control_aliases()


# ---- Mapping tests: Ctrl+A – Ctrl+Z --------------------------------------


@pytest.mark.parametrize(
    "codepoint,expected_key",
    [
        (97, Keys.ControlA),
        (98, Keys.ControlB),
        (99, Keys.ControlC),
        (100, Keys.ControlD),
        (101, Keys.ControlE),
        (102, Keys.ControlF),
        (103, Keys.ControlG),
        (104, Keys.ControlH),
        (105, Keys.ControlI),
        (106, Keys.ControlJ),
        (107, Keys.ControlK),
        (108, Keys.ControlL),
        (109, Keys.ControlM),
        (110, Keys.ControlN),
        (111, Keys.ControlO),
        (112, Keys.ControlP),
        (113, Keys.ControlQ),
        (114, Keys.ControlR),
        (115, Keys.ControlS),
        (116, Keys.ControlT),
        (117, Keys.ControlU),
        (118, Keys.ControlV),
        (119, Keys.ControlW),
        (120, Keys.ControlX),
        (121, Keys.ControlY),
        (122, Keys.ControlZ),
    ],
)
def test_csi_u_ctrl_letter_mapped(codepoint, expected_key):
    """Each Ctrl+letter CSI-u sequence must map to the correct Keys.ControlX."""
    seq = f"\x1b[{codepoint};5u"
    assert ANSI_SEQUENCES.get(seq) == expected_key, (
        f"CSI-u sequence {seq!r} should map to {expected_key}, "
        f"got {ANSI_SEQUENCES.get(seq)!r}"
    )


# ---- Mapping tests: Ctrl+0 – Ctrl+9 --------------------------------------


@pytest.mark.parametrize(
    "codepoint,expected_key",
    [
        (48, Keys.Control0),
        (49, Keys.Control1),
        (50, Keys.Control2),
        (51, Keys.Control3),
        (52, Keys.Control4),
        (53, Keys.Control5),
        (54, Keys.Control6),
        (55, Keys.Control7),
        (56, Keys.Control8),
        (57, Keys.Control9),
    ],
)
def test_csi_u_ctrl_number_mapped(codepoint, expected_key):
    """Each Ctrl+number CSI-u sequence must map to the correct Keys.ControlN."""
    seq = f"\x1b[{codepoint};5u"
    assert ANSI_SEQUENCES.get(seq) == expected_key, (
        f"CSI-u sequence {seq!r} should map to {expected_key}, "
        f"got {ANSI_SEQUENCES.get(seq)!r}"
    )


# ---- Mapping tests: Ctrl+special chars -----------------------------------


@pytest.mark.parametrize(
    "codepoint,expected_key,name",
    [
        (64, Keys.ControlAt, "Ctrl+@"),
        (91, Keys.Escape, "Ctrl+["),
        (92, Keys.ControlBackslash, "Ctrl+\\"),
        (93, Keys.ControlSquareClose, "Ctrl+]"),
        (94, Keys.ControlCircumflex, "Ctrl+^"),
        (95, Keys.ControlUnderscore, "Ctrl+_"),
        (32, Keys.ControlAt, "Ctrl+Space"),
    ],
)
def test_csi_u_ctrl_special_mapped(codepoint, expected_key, name):
    """Ctrl+special char CSI-u sequences must map to the correct Keys."""
    seq = f"\x1b[{codepoint};5u"
    assert ANSI_SEQUENCES.get(seq) == expected_key, (
        f"CSI-u sequence for {name} ({seq!r}) should map to {expected_key}, "
        f"got {ANSI_SEQUENCES.get(seq)!r}"
    )


# ---- Mapping test: Plain Escape ------------------------------------------


def test_escape_csi_u_mapped():
    """Plain Escape in kitty protocol is ESC[27u — must map to Keys.Escape."""
    assert ANSI_SEQUENCES.get("\x1b[27u") == Keys.Escape


# ---- Parser-level tests ---------------------------------------------------


def test_ctrl_c_parses_as_single_keypress():
    """Ctrl+C via CSI-u must produce exactly one KeyPress (Keys.ControlC),
    not Escape followed by literal '[99;5u' text."""
    result = _parse("\x1b[99;5u")
    assert len(result) == 1, f"Expected 1 key, got {len(result)}: {result!r}"
    assert result[0].key == Keys.ControlC


def test_escape_parses_as_single_keypress():
    """Escape via CSI-u must produce exactly one KeyPress (Keys.Escape),
    not Escape followed by literal '[27u' text."""
    result = _parse("\x1b[27u")
    assert len(result) == 1, f"Expected 1 key, got {len(result)}: {result!r}"
    assert result[0].key == Keys.Escape


def test_ctrl_c_bulk_feed_matches_char_feed():
    """Both char-by-char and bulk feed must produce the same result."""
    char_result = _parse("\x1b[99;5u")
    bulk_result = _parse_bulk("\x1b[99;5u")
    assert [kp.key for kp in char_result] == [kp.key for kp in bulk_result]


def test_ctrl_c_does_not_produce_garbage_text():
    """The old bug: ESC[99;5u parsed as Escape + '[99;5u' literal text.
    After the fix, the result must be a single Keys.ControlC keypress —
    no extra literal-text keypresses."""
    result = _parse("\x1b[99;5u")
    assert len(result) == 1, f"Expected 1 key, got {len(result)}: {result!r}"
    assert result[0].key == Keys.ControlC
    # No literal '[', '9', '9', ';', '5', 'u' keypresses should appear
    literal_chars = [kp for kp in result if kp.key in ("[", "9", ";", "5", "u")]
    assert not literal_chars, f"Literal text leaked: {literal_chars!r}"


def test_multiple_ctrl_keys_in_sequence():
    """Rapidly typed Ctrl+C then Ctrl+D must each parse independently."""
    result = _parse("\x1b[99;5u\x1b[100;5u")
    assert len(result) == 2
    assert result[0].key == Keys.ControlC
    assert result[1].key == Keys.ControlD


def test_ctrl_c_followed_by_plain_text():
    """Ctrl+C followed by 'hello' must produce Ctrl+C then the text chars."""
    result = _parse("\x1b[99;5uhello")
    assert result[0].key == Keys.ControlC
    assert [kp.key for kp in result[1:]] == ["h", "e", "l", "l", "o"]


def test_escape_followed_by_plain_text():
    """Escape followed by 'hello' must produce Escape then the text chars."""
    result = _parse("\x1b[27uhello")
    assert result[0].key == Keys.Escape
    assert [kp.key for kp in result[1:]] == ["h", "e", "l", "l", "o"]


def test_rapid_escape_and_ctrl_c():
    """Escape then Ctrl+C in rapid succession."""
    result = _parse("\x1b[27u\x1b[99;5u")
    assert len(result) == 2
    assert result[0].key == Keys.Escape
    assert result[1].key == Keys.ControlC


# ---- No-conflict tests ----------------------------------------------------


def test_no_conflict_with_shift_enter():
    """install_kitty_control_aliases must not overwrite Shift+Enter mappings
    installed by install_shift_enter_alias."""
    from hermes_cli.pt_input_extras import install_shift_enter_alias
    install_shift_enter_alias()
    install_kitty_control_aliases()
    alt_enter = (Keys.Escape, Keys.ControlM)
    assert ANSI_SEQUENCES.get("\x1b[13;2u") == alt_enter


def test_no_conflict_with_ctrl_enter():
    """install_kitty_control_aliases must not overwrite Ctrl+Enter mappings
    installed by install_ctrl_enter_alias."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    install_ctrl_enter_alias()
    install_kitty_control_aliases()
    alt_enter = (Keys.Escape, Keys.ControlM)
    assert ANSI_SEQUENCES.get("\x1b[13;5u") == alt_enter


def test_ctrl_m_does_not_conflict_with_ctrl_enter():
    """Ctrl+M (codepoint 109) and Ctrl+Enter (codepoint 13) are different
    sequences and must not interfere with each other."""
    assert ANSI_SEQUENCES.get("\x1b[109;5u") == Keys.ControlM
    # Ctrl+Enter is codepoint 13 — mapped by install_ctrl_enter_alias
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    install_ctrl_enter_alias()
    assert ANSI_SEQUENCES.get("\x1b[13;5u") == (Keys.Escape, Keys.ControlM)


def test_escape_csi_u_does_not_conflict_with_xterm_shift_enter():
    """ESC[27u (plain Escape) must not interfere with ESC[27;2;13~ (xterm
    Shift+Enter). The parser must correctly disambiguate them by prefix."""
    from hermes_cli.pt_input_extras import install_shift_enter_alias
    install_shift_enter_alias()
    install_kitty_control_aliases()

    # Both must parse correctly
    esc_result = _parse("\x1b[27u")
    shift_enter_result = _parse("\x1b[27;2;13~")

    assert len(esc_result) == 1
    assert esc_result[0].key == Keys.Escape

    assert len(shift_enter_result) == 2
    assert shift_enter_result[0].key == Keys.Escape
    assert shift_enter_result[1].key == Keys.ControlM


# ---- Idempotency / cache tests -------------------------------------------


def test_install_is_idempotent():
    """Running install twice should report 0 changes on the second call."""
    install_kitty_control_aliases()
    assert install_kitty_control_aliases() == 0


def test_prefix_cache_cleared_after_install():
    """The _IS_PREFIX_OF_LONGER_MATCH_CACHE must be cleared so that prefixes
    like '\\x1b[99' (previously cached as is_prefix_of_longer=False because
    no longer match existed) are recomputed and now return True."""
    install_kitty_control_aliases()
    assert _IS_PREFIX_OF_LONGER_MATCH_CACHE["\x1b[99"] is True


def test_prefix_cache_recomputes_correctly_after_clear():
    """After clearing the cache and installing aliases, the parser can
    correctly wait for the full CSI-u sequence instead of falling through
    to literal text on the first unmatched character."""
    # Simulate the old state: cache has stale entries
    _IS_PREFIX_OF_LONGER_MATCH_CACHE["\x1b[99"] = False
    _IS_PREFIX_OF_LONGER_MATCH_CACHE["\x1b[99;5"] = False

    # Install aliases — this should clear the cache
    install_kitty_control_aliases()

    # Now the parser should see \x1b[99 as a prefix of a longer match
    assert _IS_PREFIX_OF_LONGER_MATCH_CACHE["\x1b[99"] is True
