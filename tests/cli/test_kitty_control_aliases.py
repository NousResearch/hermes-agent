"""Regression tests for Kitty keyboard protocol CSI-u control key sequences.

When ``enable_kitty_keyboard_protocol()`` pushes the terminal into CSI-u
mode, *every* modified key is sent as a CSI-u sequence — not just
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


# ---- Mapping tests --------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_aliases_installed():
    """Make every test idempotent — install the aliases once per test run."""
    install_kitty_control_aliases()


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
    # After install, \x1b[99 should be a prefix of \x1b[99;5u
    # The cache should have been cleared, so this will recompute
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
