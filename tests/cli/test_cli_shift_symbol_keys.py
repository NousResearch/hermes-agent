"""Regression tests for Shift+<symbol> under xterm modifyOtherKeys=2.

``_enable_extended_enter_keys()`` pushes modifyOtherKeys level 2 (CSI >4;2m)
to Ghostty/iTerm2/WezTerm/VS Code/tmux/Windows Terminal. Those terminals then
re-encode modified keys as ``ESC[27;<mod>;<codepoint>~``.

``install_modify_other_keys_aliases()`` mapped Shift+letter but not
Shift+symbol, so on Ghostty 1.3.x every shifted punctuation key leaked into
the prompt buffer as literal ``^[[27;2;NN~`` text — '?', '!', ':', '"', '_',
'+', '<' and friends were untypeable while the mode was active.
"""

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_modify_other_keys_aliases


def _parse_keys(data: str):
    events = []
    parser = Vt100Parser(events.append)
    parser.feed_and_flush(data)
    return [event.key for event in events]


# US-layout shifted punctuation, as Ghostty reports it: the codepoint is
# already shifted by the terminal, so 63 is '?' and not '/'.
SHIFTED_SYMBOLS = [
    (63, "?"),   # Shift+/
    (33, "!"),   # Shift+1
    (64, "@"),   # Shift+2
    (58, ":"),   # Shift+;
    (34, '"'),   # Shift+'
    (95, "_"),   # Shift+-
    (43, "+"),   # Shift+=
    (60, "<"),   # Shift+,
    (62, ">"),   # Shift+.
    (40, "("),   # Shift+9
    (126, "~"),  # Shift+`
]


def test_shift_symbols_parse_as_the_symbol():
    install_modify_other_keys_aliases()

    for codepoint, char in SHIFTED_SYMBOLS:
        assert _parse_keys(f"\x1b[27;2;{codepoint}~") == [char], (
            f"Shift+{char} (codepoint {codepoint}) did not parse as {char!r}"
        )


def test_no_printable_codepoint_leaks_as_literal_text():
    """Every printable ASCII must be mapped — a leak yields many key events."""
    install_modify_other_keys_aliases()

    leaked = [
        codepoint
        for codepoint in range(0x20, 0x7F)
        if len(_parse_keys(f"\x1b[27;2;{codepoint}~")) != 1
    ]
    assert leaked == [], f"codepoints still leaking as literal text: {leaked}"


def test_shift_letters_still_map_to_uppercase():
    """The pre-existing Shift+letter behaviour must be unchanged."""
    install_modify_other_keys_aliases()

    assert _parse_keys("\x1b[27;2;97~") == ["A"]   # lowercase codepoint
    assert _parse_keys("\x1b[27;2;65~") == ["A"]   # already-shifted codepoint
    assert _parse_keys("\x1b[27;2;122~") == ["Z"]


def test_space_and_tab_and_backspace_are_not_clobbered():
    """Codepoints 32/9/127 have dedicated meanings that must win."""
    install_modify_other_keys_aliases()

    assert _parse_keys("\x1b[27;2;32~") == [" "]
    assert _parse_keys("\x1b[27;2;9~") == [Keys.BackTab]
    assert _parse_keys("\x1b[27;2;127~") == [Keys.ControlH]


def test_csi_u_spelling_is_left_alone():
    """Kitty reports the UNSHIFTED codepoint, so chr() would be wrong there.

    Registering ESC[47;2u -> '/' would type '/' for Shift+/ on kitty. Printable
    keys arrive as plain text under the Kitty protocol, so there is nothing to
    map; this asserts we did not opportunistically add them.
    """
    install_modify_other_keys_aliases()

    for codepoint in (47, 49, 50, 59, 39, 45, 61, 44, 46, 57, 96):
        assert f"\x1b[{codepoint};2u" not in ANSI_SEQUENCES


def test_install_is_idempotent():
    install_modify_other_keys_aliases()
    snapshot = dict(ANSI_SEQUENCES)

    install_modify_other_keys_aliases()

    assert ANSI_SEQUENCES == snapshot
