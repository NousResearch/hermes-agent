"""Regression tests for kitty/xterm modified cursor-key sequences.

Kitty reports the arrow cluster in the ``CSI 1;<mod> <final>`` cursor-key form
with the keypad flag riding in the modifier suffix — e.g. ``\\x1b[1;129A`` (Up),
``\\x1b[1;129B`` (Down), ``\\x1b[1;129C`` (Right), ``\\x1b[1;129D`` (Left),
129 = 0x80 keypad flag | 0x01. Without mappings these leak into the prompt as
literal text like ``[1;129A[1;129C[1;129D[1;129B``.
"""

from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_modified_cursor_key_aliases


def _parse_keys(data: str):
    events = []
    parser = Vt100Parser(events.append)
    parser.feed_and_flush(data)
    return [(event.key, event.data) for event in events]


def test_kitty_keypad_flagged_arrows_parse_as_navigation_keys():
    install_modified_cursor_key_aliases()

    assert _parse_keys("\x1b[1;129A") == [(Keys.Up, "\x1b[1;129A")]
    assert _parse_keys("\x1b[1;129B") == [(Keys.Down, "\x1b[1;129B")]
    assert _parse_keys("\x1b[1;129C") == [(Keys.Right, "\x1b[1;129C")]
    assert _parse_keys("\x1b[1;129D") == [(Keys.Left, "\x1b[1;129D")]
    assert _parse_keys("\x1b[1;129H") == [(Keys.Home, "\x1b[1;129H")]
    assert _parse_keys("\x1b[1;129F") == [(Keys.End, "\x1b[1;129F")]


def test_keypad_flagged_arrows_do_not_leak_as_literal_text():
    install_modified_cursor_key_aliases()

    # The exact reported leak must NOT print the raw escape bytes to the
    # prompt; each sequence is consumed as a single navigation key.
    result = _parse_keys("\x1b[1;129A\x1b[1;129C\x1b[1;129D\x1b[1;129B")
    assert result == [
        (Keys.Up, "\x1b[1;129A"),
        (Keys.Right, "\x1b[1;129C"),
        (Keys.Left, "\x1b[1;129D"),
        (Keys.Down, "\x1b[1;129B"),
    ]


def test_install_is_idempotent_and_default_safe():
    install_modified_cursor_key_aliases()
    # Second call adds nothing new since sequences are already present.
    assert install_modified_cursor_key_aliases() == 0