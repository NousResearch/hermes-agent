"""Verify Shift+Enter byte sequences parse to the same key tuple Alt+Enter
produces, so the existing Alt+Enter newline handler in `cli.py` fires for
terminals that emit a distinct Shift+Enter under the Kitty keyboard protocol
or xterm modifyOtherKeys mode.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_shift_enter_alias


SHIFT_ENTER_SEQUENCES = (
    "\x1b[13;2u",      # Kitty / CSI-u, modifier=2 (Shift)
    "\x1b[27;2;13~",   # xterm modifyOtherKeys=2
    "\x1b[27;2;13u",
)


@pytest.fixture(autouse=True)
def _ensure_alias_installed():
    """Make every test idempotent — install the alias once per test run."""
    install_shift_enter_alias()


def _parse(byte_seq: str):
    out = []
    parser = Vt100Parser(out.append)
    for ch in byte_seq:
        parser.feed(ch)
    parser.flush()
    return [kp.key for kp in out]


def test_install_registers_all_three_sequences():
    for seq in SHIFT_ENTER_SEQUENCES:
        assert seq in ANSI_SEQUENCES, f"missing mapping for {seq!r}"
        assert ANSI_SEQUENCES[seq] == (Keys.Escape, Keys.ControlM)


def test_install_overwrites_stock_modifyotherkeys_shift_enter():
    """Stock prompt_toolkit maps `\\x1b[27;2;13~` to plain Keys.ControlM —
    i.e. it drops the Shift modifier and treats Shift+Enter like Enter,
    which is the bug this helper exists to fix. The install must overwrite
    that entry."""
    seq = "\x1b[27;2;13~"
    ANSI_SEQUENCES[seq] = Keys.ControlM
    install_shift_enter_alias()
    assert ANSI_SEQUENCES[seq] == (Keys.Escape, Keys.ControlM)


def test_install_returns_zero_when_already_correct():
    """Idempotency — running install twice should not report a second change."""
    install_shift_enter_alias()
    assert install_shift_enter_alias() == 0


def test_csi_u_shift_enter_parses_as_alt_enter():
    """Kitty keyboard protocol Shift+Enter must parse to the same key tuple
    Alt+Enter produces, so the existing handler is reused."""
    alt_enter = _parse("\x1b\r")
    shift_enter = _parse("\x1b[13;2u")
    assert shift_enter == alt_enter, (
        f"Shift+Enter via CSI-u should parse identically to Alt+Enter; "
        f"got {shift_enter!r} vs {alt_enter!r}"
    )


def test_modify_other_keys_shift_enter_parses_as_alt_enter():
    """xterm modifyOtherKeys=2 Shift+Enter must parse identically to Alt+Enter."""
    alt_enter = _parse("\x1b\r")
    shift_enter = _parse("\x1b[27;2;13~")
    assert shift_enter == alt_enter


def test_plain_enter_remains_distinct_from_alt_enter():
    """Plain Enter must keep emitting a single key (submit), not a two-key
    Alt+Enter tuple — otherwise we would have broken submit."""
    enter = _parse("\r")
    alt_enter = _parse("\x1b\r")
    assert enter != alt_enter
    assert len(enter) == 1
    assert len(alt_enter) == 2
