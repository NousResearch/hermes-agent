"""Ghostty's modifyOtherKeys=2 shifted-symbol emissions must type their text (#114242).

Ghostty reports the SHIFTED codepoint in ``ESC[27;2;<cp>~`` — ``{`` arrives as
``ESC[27;2;123~`` where 123 IS ``{`` — so mapping the codepoint back to its character is
layout-safe there, while the general table deliberately leaves Shift+symbol unmapped because
other emitters report the unshifted codepoint ("leaking beats wrong input").
"""

import asyncio

import pytest
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser

from hermes_cli import pt_input_extras


GHOSTTY_ENV = {"TERM_PROGRAM": "ghostty"}
NON_GHOSTTY_ENVIRONS = [
    {},
    {"TERM": "xterm-256color"},
    {"TERM_PROGRAM": "iTerm.app", "TERM": "xterm-256color"},
    {"TERM_PROGRAM": "ghostty-something"},  # prefix must not match
]


@pytest.fixture(autouse=True)
def _restore_sequences():
    """Snap ANSI_SEQUENCES and the parser prefix cache back after each test."""
    saved = dict(ANSI_SEQUENCES)
    yield
    ANSI_SEQUENCES.clear()
    ANSI_SEQUENCES.update(saved)
    from prompt_toolkit.input.vt100_parser import _IS_PREFIX_OF_LONGER_MATCH_CACHE

    _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()


def _parse(sequence):
    presses = []
    parser = Vt100Parser(presses.append)
    for char in sequence:
        parser.feed(char)
    parser.flush()
    return presses  # full KeyPress: both key and data matter for what gets typed


@pytest.mark.parametrize(
    "ch", ["{", "}", ":", '"', "|", "~", "!", "<", ">", "@", "+", "?"]
)
def test_ghostty_shifted_symbols_parse_as_their_text(ch):
    """Under Ghostty the shifted-symbol sequence must resolve to exactly the character."""
    pt_input_extras.install_modify_other_keys_aliases()
    assert pt_input_extras.install_ghostty_shifted_symbol_aliases(GHOSTTY_ENV) > 0
    pt_input_extras.install_keypress_data_normalization()

    presses = _parse(f"\x1b[27;2;{ord(ch)}~")
    assert len(presses) == 1, f"expected one keypress for {ch!r}, got {presses!r}"
    assert presses[0].key == ch
    # self-insert types event.data — it must carry the character, not the raw CSI bytes.
    assert presses[0].data == ch


@pytest.mark.parametrize("env", NON_GHOSTTY_ENVIRONS)
def test_non_ghostty_environments_stay_unmapped(env):
    """Fail-closed: without a Ghostty environment nothing is installed (leaking beats wrong input)."""
    assert pt_input_extras.install_ghostty_shifted_symbol_aliases(env) == 0
    assert "\x1b[27;2;123~" not in ANSI_SEQUENCES
    assert "\x1b[27;2;125~" not in ANSI_SEQUENCES


@pytest.mark.parametrize(
    "env",
    [
        {"TERM": "xterm-ghostty"},
        {"TERM": "XTERM-GHOSTTY"},  # case-insensitive, like cli._is_ghostty_terminal
        {"TERM_PROGRAM": "  ghostty  "},
    ],
)
def test_ghostty_detection_variants_install(env):
    assert pt_input_extras.install_ghostty_shifted_symbol_aliases(env) > 0
    assert ANSI_SEQUENCES[f"\x1b[27;2;{ord('{')}~"] == "{"


def test_letters_are_left_to_the_uppercase_aliases():
    """The symbol table must not stage entries for letters — Shift+letter is already owned."""
    from prompt_toolkit.keys import Keys

    staged = pt_input_extras._ghostty_shifted_symbol_aliases(dict(ANSI_SEQUENCES), Keys)
    assert f"\x1b[27;2;{ord('H')}~" not in staged
    assert f"\x1b[27;2;{ord('h')}~" not in staged
    assert staged[f"\x1b[27;2;{ord('{')}~"] == "{"


def test_install_is_idempotent():
    assert pt_input_extras.install_ghostty_shifted_symbol_aliases(GHOSTTY_ENV) > 0
    assert pt_input_extras.install_ghostty_shifted_symbol_aliases(GHOSTTY_ENV) == 0


def test_shifted_symbols_type_into_the_buffer():
    """End to end: the composer must receive the characters, not the raw escape sequences."""
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import BufferControl, Layout, Window
    from prompt_toolkit.output import DummyOutput

    pt_input_extras.install_modify_other_keys_aliases()
    pt_input_extras.install_ghostty_shifted_symbol_aliases(GHOSTTY_ENV)
    pt_input_extras.install_keypress_data_normalization()

    async def probe(sequence):
        buf = Buffer()
        kb = KeyBindings()
        kb.add("c-q")(lambda event: event.app.exit(result=buf.text))
        with create_pipe_input() as inp:
            app = Application(
                layout=Layout(Window(BufferControl(buf))),
                key_bindings=kb,
                input=inp,
                output=DummyOutput(),
            )
            return await asyncio.wait_for(
                app.run_async(pre_run=lambda: inp.send_text(sequence + "\x11")),
                timeout=5,
            )

    typed = f"\x1b[27;2;{ord('{')}~\x1b[27;2;{ord('}')}~\x1b[27;2;{ord(':')}~"
    assert asyncio.run(probe(typed)) == "{}:"
