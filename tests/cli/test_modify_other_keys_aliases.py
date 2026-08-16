"""Regression tests for issue #87711 — Ctrl+key / Alt+key combos broken
under modifyOtherKeys level 2.

When the CLI pushes ``ESC[>4;2m`` (modifyOtherKeys=2) to supported
terminals so Shift+Enter is distinguishable from Enter, the terminal
re-encodes EVERY Ctrl+key combo as ``ESC[27;5;<codepoint>~`` instead of
the raw control byte (``\\x01`` etc.). prompt_toolkit 3.x only ships a
mapping for ``ESC[27;5;13~`` (Ctrl+Enter = Ctrl+M); all other Ctrl+letter
combos are unmapped and leak as literal text or get swallowed.

``install_modify_other_keys_aliases()`` populates ``ANSI_SEQUENCES`` with
the full set so every Ctrl+combo continues to fire the same key binding
the raw byte would.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_modify_other_keys_aliases


@pytest.fixture(autouse=True)
def _ensure_alias_installed():
    """Make every test idempotent — install the alias once per test run."""
    install_modify_other_keys_aliases()


def _parse(byte_seq: str):
    """Feed bytes through prompt_toolkit's VT100 parser and return the
    list of KeyPress objects."""
    out = []
    parser = Vt100Parser(out.append)
    for ch in byte_seq:
        parser.feed(ch)
    parser.flush()
    return [kp.key for kp in out]


# ---------------------------------------------------------------------------
# Ctrl+letter: a-z
# ---------------------------------------------------------------------------

CTRL_LETTERS = [chr(c) for c in range(ord('a'), ord('z') + 1)]


@pytest.mark.parametrize("letter", CTRL_LETTERS)
def test_modify_other_keys_ctrl_letter_parses_as_raw_byte(letter):
    """Ctrl+<letter> under modifyOtherKeys must parse identically to the
    raw control byte that prompt_toolkit already understands."""
    raw_byte = chr(ord(letter) - ord('a') + 1)  # Ctrl+a = \x01, etc.
    raw_result = _parse(raw_byte)
    assert len(raw_result) == 1, f"raw byte {raw_byte!r} should produce 1 keypress"

    # modifyOtherKeys format
    mok_seq = f"\x1b[27;5;{ord(letter)}~"
    mok_result = _parse(mok_seq)
    assert mok_result == raw_result, (
        f"modifyOtherKeys Ctrl+{letter} ({mok_seq!r}) should parse identically "
        f"to raw {raw_byte!r}; got {mok_result!r} vs {raw_result!r}"
    )

    # CSI-u format
    csiu_seq = f"\x1b[{ord(letter)};5u"
    csiu_result = _parse(csiu_seq)
    assert csiu_result == raw_result, (
        f"CSI-u Ctrl+{letter} ({csiu_seq!r}) should parse identically "
        f"to raw {raw_byte!r}; got {csiu_result!r} vs {raw_result!r}"
    )


@pytest.mark.parametrize("letter", CTRL_LETTERS)
def test_modify_other_keys_ctrl_letter_single_keypress(letter):
    """Each Ctrl+letter sequence must produce exactly one keypress —
    a partial match would emit Escape plus literal text."""
    for seq in (f"\x1b[27;5;{ord(letter)}~", f"\x1b[{ord(letter)};5u"):
        result = _parse(seq)
        assert len(result) == 1, (
            f"{seq!r} should produce exactly 1 keypress, got {len(result)}: {result!r}"
        )


# ---------------------------------------------------------------------------
# Critical individual shortcuts
# ---------------------------------------------------------------------------

def test_ctrl_c_under_modify_other_keys():
    """Ctrl+C must produce Keys.ControlC, not literal text (#56684)."""
    assert _parse("\x1b[27;5;99~") == [Keys.ControlC]
    assert _parse("\x1b[99;5u") == [Keys.ControlC]


def test_ctrl_a_under_modify_other_keys():
    """Ctrl+A (line start) must still fire."""
    assert _parse("\x1b[27;5;97~") == [Keys.ControlA]
    assert _parse("\x1b[97;5u") == [Keys.ControlA]


def test_ctrl_e_under_modify_other_keys():
    """Ctrl+E (line end) must still fire."""
    assert _parse("\x1b[27;5;101~") == [Keys.ControlE]
    assert _parse("\x1b[101;5u") == [Keys.ControlE]


def test_ctrl_u_under_modify_other_keys():
    """Ctrl+U (kill line) must still fire."""
    assert _parse("\x1b[27;5;117~") == [Keys.ControlU]
    assert _parse("\x1b[117;5u") == [Keys.ControlU]


def test_ctrl_k_under_modify_other_keys():
    """Ctrl+K (kill to end) must still fire."""
    assert _parse("\x1b[27;5;107~") == [Keys.ControlK]
    assert _parse("\x1b[107;5u") == [Keys.ControlK]


def test_ctrl_r_under_modify_other_keys():
    """Ctrl+R (reverse search) must still fire."""
    assert _parse("\x1b[27;5;114~") == [Keys.ControlR]
    assert _parse("\x1b[114;5u") == [Keys.ControlR]


def test_ctrl_d_under_modify_other_keys():
    """Ctrl+D (EOF / delete) must still fire."""
    assert _parse("\x1b[27;5;100~") == [Keys.ControlD]
    assert _parse("\x1b[100;5u") == [Keys.ControlD]


def test_ctrl_w_under_modify_other_keys():
    """Ctrl+W (delete word) must still fire."""
    assert _parse("\x1b[27;5;119~") == [Keys.ControlW]
    assert _parse("\x1b[119;5u") == [Keys.ControlW]


def test_ctrl_z_under_modify_other_keys():
    """Ctrl+Z (suspend) must still fire."""
    assert _parse("\x1b[27;5;122~") == [Keys.ControlZ]
    assert _parse("\x1b[122;5u") == [Keys.ControlZ]


def test_ctrl_l_under_modify_other_keys():
    """Ctrl+L (clear screen) must still fire."""
    assert _parse("\x1b[27;5;108~") == [Keys.ControlL]
    assert _parse("\x1b[108;5u") == [Keys.ControlL]


# ---------------------------------------------------------------------------
# Ctrl+digit: 0-9
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("digit", [str(d) for d in range(10)])
def test_modify_other_keys_ctrl_digit(digit):
    """Ctrl+digit under modifyOtherKeys must map to Keys.Control<digit>."""
    codepoint = ord(digit)
    expected = getattr(Keys, f"Control{digit}")

    mok_seq = f"\x1b[27;5;{codepoint}~"
    assert _parse(mok_seq) == [expected], (
        f"modifyOtherKeys Ctrl+{digit} ({mok_seq!r}) should parse as {expected}"
    )

    csiu_seq = f"\x1b[{codepoint};5u"
    assert _parse(csiu_seq) == [expected], (
        f"CSI-u Ctrl+{digit} ({csiu_seq!r}) should parse as {expected}"
    )


# ---------------------------------------------------------------------------
# Ctrl+symbol
# ---------------------------------------------------------------------------

def test_ctrl_left_bracket_under_modify_other_keys():
    """Ctrl+[ = Escape under modifyOtherKeys."""
    assert _parse("\x1b[27;5;91~") == _parse("\x1b")
    assert _parse("\x1b[91;5u") == _parse("\x1b")


def test_ctrl_backslash_under_modify_other_keys():
    """Ctrl+\\ = ControlBackslash under modifyOtherKeys."""
    assert _parse("\x1b[27;5;92~") == [Keys.ControlBackslash]
    assert _parse("\x1b[92;5u") == [Keys.ControlBackslash]


def test_ctrl_right_bracket_under_modify_other_keys():
    """Ctrl+] = ControlSquareClose under modifyOtherKeys."""
    assert _parse("\x1b[27;5;93~") == [Keys.ControlSquareClose]
    assert _parse("\x1b[93;5u") == [Keys.ControlSquareClose]


def test_ctrl_space_under_modify_other_keys():
    """Ctrl+Space = ControlAt (NUL) under modifyOtherKeys."""
    assert _parse("\x1b[27;5;32~") == _parse("\x00")
    assert _parse("\x1b[32;5u") == _parse("\x00")


# ---------------------------------------------------------------------------
# Alt+letter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("letter", [chr(c) for c in range(ord('a'), ord('z') + 1)])
def test_modify_other_keys_alt_letter_parses_as_escape_letter(letter):
    """Alt+<letter> under modifyOtherKeys must parse to the same tuple
    that a bare ESC+<letter> produces."""
    bare_result = _parse(f"\x1b{letter}")
    assert len(bare_result) == 2, f"bare ESC+{letter} should produce 2 keypresses"

    mok_seq = f"\x1b[27;3;{ord(letter)}~"
    mok_result = _parse(mok_seq)
    assert mok_result == bare_result, (
        f"modifyOtherKeys Alt+{letter} ({mok_seq!r}) should parse identically "
        f"to bare ESC+{letter}; got {mok_result!r} vs {bare_result!r}"
    )

    csiu_seq = f"\x1b[{ord(letter)};3u"
    csiu_result = _parse(csiu_seq)
    assert csiu_result == bare_result, (
        f"CSI-u Alt+{letter} ({csiu_seq!r}) should parse identically "
        f"to bare ESC+{letter}; got {csiu_result!r} vs {bare_result!r}"
    )


# ---------------------------------------------------------------------------
# Idempotency and non-clobbering
# ---------------------------------------------------------------------------

def test_install_is_idempotent():
    install_modify_other_keys_aliases()
    assert install_modify_other_keys_aliases() == 0


# ---------------------------------------------------------------------------
# Shift+letter → uppercase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("letter", [chr(c) for c in range(ord('a'), ord('z') + 1)])
def test_modify_other_keys_shift_letter_produces_uppercase(letter):
    """Shift+<letter> under modifyOtherKeys must produce the uppercase
    character, not leak as literal escape text — the 'caps locked' bug."""
    upper = letter.upper()
    # modifyOtherKeys format
    mok_seq = f"\x1b[27;2;{ord(letter)}~"
    assert _parse(mok_seq) == [upper], (
        f"modifyOtherKeys Shift+{letter} ({mok_seq!r}) should produce '{upper}'"
    )
    # CSI-u format
    csiu_seq = f"\x1b[{ord(letter)};2u"
    assert _parse(csiu_seq) == [upper], (
        f"CSI-u Shift+{letter} ({csiu_seq!r}) should produce '{upper}'"
    )


def test_does_not_clobber_shift_enter_alias():
    """install_modify_other_keys_aliases must not overwrite mappings
    installed by install_shift_enter_alias (modifier=2, not 5)."""
    from hermes_cli.pt_input_extras import install_shift_enter_alias
    install_shift_enter_alias()
    assert ANSI_SEQUENCES["\x1b[27;2;13~"] == (Keys.Escape, Keys.ControlM)
    assert ANSI_SEQUENCES["\x1b[13;2u"] == (Keys.Escape, Keys.ControlM)


def test_does_not_clobber_ctrl_enter_alias():
    """install_modify_other_keys_aliases must not overwrite mappings
    installed by install_ctrl_enter_alias (which maps Ctrl+Enter)."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    install_ctrl_enter_alias()
    # Ctrl+Enter (modifier=5, codepoint=13) is mapped to (Escape, ControlM)
    assert ANSI_SEQUENCES["\x1b[27;5;13~"] == (Keys.Escape, Keys.ControlM)
    assert ANSI_SEQUENCES["\x1b[13;5u"] == (Keys.Escape, Keys.ControlM)


def test_ctrl_enter_still_works_under_modify_other_keys():
    """Ctrl+Enter must produce the Alt+Enter newline tuple, not plain Ctrl+M.
    This is the install_ctrl_enter_alias behavior — our new function must
    not clobber it."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    install_ctrl_enter_alias()
    install_modify_other_keys_aliases()

    alt_enter = _parse("\x1b\r")
    ctrl_enter_mok = _parse("\x1b[27;5;13~")
    ctrl_enter_csiu = _parse("\x1b[13;5u")
    assert ctrl_enter_mok == alt_enter
    assert ctrl_enter_csiu == alt_enter


def test_plain_enter_remains_distinct():
    """Plain Enter must keep producing a single keypress (submit), not
    the two-key Alt+Enter tuple."""
    enter = _parse("\r")
    alt_enter = _parse("\x1b\r")
    assert enter != alt_enter
    assert len(enter) == 1
    assert len(alt_enter) == 2
