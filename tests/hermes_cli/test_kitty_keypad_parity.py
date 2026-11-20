"""Modified kitty keypad keys must behave like their non-keypad twins (#97290)."""

import asyncio

import pytest
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser

from hermes_cli import pt_input_extras


@pytest.fixture(autouse=True)
def _aliases_installed():
    pt_input_extras.install_shift_enter_alias()
    pt_input_extras.install_ctrl_enter_alias()
    pt_input_extras.install_modify_other_keys_aliases()
    pt_input_extras.install_keypress_data_normalization()


def _parse(sequence):
    presses = []
    parser = Vt100Parser(presses.append)
    for char in sequence:
        parser.feed(char)
    parser.flush()
    return [press.key for press in presses]


def test_modified_keypad_mirrors_its_non_keypad_twin():
    """Every supported modifier inherits its twin, including lock bits, only in CSI-u."""
    twins = {
        57414: "\x1b[13;{mod}u",
        **{code: "\x1b[1;{mod}" + suffix for code, suffix in
           zip((57417, 57418, 57419, 57420, 57423, 57424), "DCABHF")},
        **{code: f"\x1b[{number};{{mod}}~" for code, number in
           ((57421, 5), (57422, 6), (57425, 2), (57426, 3))},
        **{57399 + digit: f"\x1b[{ord(str(digit))};{{mod}}u" for digit in range(10)},
    }
    for code, template in twins.items():
        for base_mod in range(2, 9):
            for offset in (0, 64, 128, 192):
                modifier = base_mod + offset
                sequence = f"\x1b[{code};{modifier}u"
                twin = template.format(mod=modifier)
                if twin in ANSI_SEQUENCES:
                    assert _parse(sequence) == _parse(twin), (code, modifier)
                else:
                    assert sequence not in ANSI_SEQUENCES
                assert f"\x1b[27;{modifier};{code}~" not in ANSI_SEQUENCES

    # First-writer-wins applies to the twin too, even when this builder stages a default.
    from prompt_toolkit.keys import Keys
    installed = dict(ANSI_SEQUENCES)
    installed["\x1b[13;3u"] = Keys.ControlA
    aliases = pt_input_extras._modify_other_keys_aliases(installed, Keys)
    assert aliases["\x1b[57414;3u"] == installed["\x1b[13;3u"]


@pytest.mark.parametrize("prefix", ["alpha", "alpha  ", "alpha!?.,", "[Pasted text #1: 15 lines]"])
def test_keypad_alt_enter_inserts_newlines_at_the_cursor(prefix):
    """Extended Enter must reach the newline handler, never self-insert raw CSI text."""
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import BufferControl, Layout, Window
    from prompt_toolkit.output import DummyOutput

    from hermes_cli.cli_tui_mixin import CLITuiMixin

    async def probe(sequence):
        buf = Buffer(document=Document(prefix + "suffix", len(prefix)))
        kb = KeyBindings()
        kb.add("escape", "enter")(CLITuiMixin()._tui_insert_newline)
        kb.add("c-q")(lambda event: event.app.exit(result=buf.text))
        with create_pipe_input() as inp:
            app = Application(layout=Layout(Window(BufferControl(buf))), key_bindings=kb,
                              input=inp, output=DummyOutput())
            return await asyncio.wait_for(app.run_async(
                pre_run=lambda: inp.send_text(sequence * 2 + "\x11")), timeout=5)

    for sequence in ("\x1b[57414;3u", "\x1b[57414;131u", "\x1b[13;3u", "\x1b\r"):
        assert asyncio.run(probe(sequence)) == prefix + "\n\nsuffix"
