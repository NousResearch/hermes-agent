"""Verify macOS Command/Option navigation and deletion sequences.

Modern terminals report Command as the super modifier and Option as Alt. Hermes
must preserve the distinction after prompt_toolkit parses CSI-u sequences.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_macos_editing_aliases


def _parse(byte_seq: str):
    out = []
    parser = Vt100Parser(out.append)
    for ch in byte_seq:
        parser.feed(ch)
    parser.flush()
    return [kp.key for kp in out]


@pytest.fixture(autouse=True)
def _install_aliases():
    install_macos_editing_aliases()


@pytest.mark.parametrize("seq", ("\x1b[1;9D", "\x1b[1;10D"))
def test_command_left_goes_to_start_of_line(seq):
    assert _parse(seq) == _parse("\x01")


@pytest.mark.parametrize("seq", ("\x1b[1;9C", "\x1b[1;10C"))
def test_command_right_goes_to_end_of_line(seq):
    assert _parse(seq) == _parse("\x05")


@pytest.mark.parametrize("seq", ("\x1b[1;3D", "\x1b[1;4D"))
def test_option_left_moves_one_word_backward(seq):
    assert _parse(seq) == [Keys.ControlLeft]


@pytest.mark.parametrize("seq", ("\x1b[1;3C", "\x1b[1;4C"))
def test_option_right_moves_one_word_forward(seq):
    assert _parse(seq) == [Keys.ControlRight]


@pytest.mark.parametrize("seq", ("\x1b[127;3u", "\x1b[127;4u", "\x1b[27;3;127~"))
def test_option_backspace_deletes_one_word_backward(seq):
    assert _parse(seq) == [Keys.Escape, Keys.ControlH]


@pytest.mark.parametrize("seq", ("\x1b[3;3~", "\x1b[3;4~"))
def test_option_forward_delete_deletes_one_word_forward(seq):
    assert _parse(seq) == [Keys.ControlDelete]


def test_install_is_idempotent():
    install_macos_editing_aliases()
    assert install_macos_editing_aliases() == 0


@pytest.mark.parametrize("seq", ("\x1b[1;73D", "\x1b[1;137D"))  # Command+Left with CapsLock / NumLock
def test_command_left_with_lock_state_still_goes_to_start_of_line(seq):
    assert _parse(seq) == _parse("\x01")
