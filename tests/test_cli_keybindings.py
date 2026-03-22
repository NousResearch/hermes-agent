"""Tests for Hermes CLI terminal keyboard handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import InMemoryClipboard
from prompt_toolkit.document import Document
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from hermes_cli import terminal_keyboard as tkbd


def test_register_word_delete_keybindings_wires_terminal_sequences():
    kb = KeyBindings()

    tkbd.register_word_delete_keybindings(kb)

    assert len(kb.bindings) == 3

    with patch.object(tkbd, "BACKWARD_KILL_WORD") as backward_kill_word:
        event = object()
        for binding in kb.bindings:
            binding.handler(event)

    assert backward_kill_word.call_count == 3


def test_parse_terminal_keyboard_capabilities_detects_kitty_and_modify_other_keys():
    capabilities = tkbd.parse_capabilities(
        "\x1b[?1u\x1b[>4;2m\x1b[?64;1;2;6;9;15;18;21;22c"
    )

    assert capabilities.kitty_supported is True
    assert capabilities.modify_other_keys_supported is True


def test_parse_terminal_keyboard_detection_result_preserves_non_probe_input():
    result = tkbd.parse_detection_result(
        "\x1b[?1uhello\x1b[>4;2m\x1b[?64;1;2;6;9c"
    )

    assert result.capabilities.kitty_supported is True
    assert result.capabilities.modify_other_keys_supported is True
    assert result.pending_input == "hello"


def test_select_terminal_keyboard_mode_prefers_kitty():
    capabilities = tkbd.TerminalKeyboardCapabilities(
        kitty_supported=True,
        modify_other_keys_supported=True,
    )

    assert tkbd.select_mode(capabilities) == "kitty"


def test_select_terminal_keyboard_mode_falls_back_to_modify_other_keys():
    capabilities = tkbd.TerminalKeyboardCapabilities(
        kitty_supported=False,
        modify_other_keys_supported=True,
    )

    assert tkbd.select_mode(capabilities) == "modify_other_keys"


def test_select_terminal_keyboard_mode_returns_none_when_unsupported():
    assert tkbd.select_mode(tkbd.TerminalKeyboardCapabilities()) is None


@pytest.mark.parametrize(
    ("mode", "enable", "expected_sequence"),
    [
        ("kitty", True, tkbd.KITTY_KEYBOARD_ENABLE),
        ("kitty", False, tkbd.KITTY_KEYBOARD_DISABLE),
        ("modify_other_keys", True, tkbd.MODIFY_OTHER_KEYS_ENABLE),
        ("modify_other_keys", False, tkbd.MODIFY_OTHER_KEYS_DISABLE),
    ],
)
def test_set_terminal_keyboard_mode_emits_expected_sequence(
    mode,
    enable,
    expected_sequence,
):
    written = []

    tkbd.set_mode(
        mode,
        enable=enable,
        writer=written.append,
    )

    assert written == [expected_sequence]


def test_install_ctrl_backspace_sequences_registers_modified_keycodes():
    sequences = {}

    tkbd.install_ctrl_backspace_sequences(
        sequences,
        terminfo_backspace=None,
    )

    for sequence in tkbd.CTRL_BACKSPACE_ESCAPE_SEQUENCES:
        assert sequences[sequence] == tkbd.CTRL_BACKSPACE_KEYS


@pytest.mark.parametrize(
    ("terminfo_backspace", "ctrl_backspace_byte"),
    [
        (b"\x08", "\x7f"),
        (b"\x7f", "\x08"),
    ],
)
def test_install_ctrl_backspace_sequences_uses_terminfo_backspace_swap(
    terminfo_backspace,
    ctrl_backspace_byte,
):
    sequences = {}

    tkbd.install_ctrl_backspace_sequences(
        sequences,
        terminfo_backspace=terminfo_backspace,
    )

    assert sequences[ctrl_backspace_byte] == tkbd.CTRL_BACKSPACE_KEYS


def test_vt100_parser_maps_ctrl_backspace_csi_u_sequence():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed("\x1b[127;5u")
    parser.flush()

    assert [press.key for press in key_presses] == list(tkbd.CTRL_BACKSPACE_KEYS)


def test_vt100_parser_maps_ctrl_b_kitty_sequence():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed(tkbd.kitty_key_sequence(ord("b"), 5))
    parser.flush()

    assert [press.key for press in key_presses] == [Keys.ControlB]


def test_vt100_parser_maps_alt_v_kitty_sequence():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed(tkbd.kitty_key_sequence(ord("v"), 3))
    parser.flush()

    assert [press.key for press in key_presses] == [Keys.Escape, "v"]


def test_vt100_parser_maps_alt_enter_modify_other_keys_sequence():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed(tkbd.modify_other_keys_sequence(13, 3))
    parser.flush()

    assert [press.key for press in key_presses] == [Keys.Escape, Keys.ControlM]


def test_vt100_parser_maps_ctrl_enter_kitty_sequence_to_control_j():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed(tkbd.kitty_key_sequence(13, 5))
    parser.flush()

    assert [press.key for press in key_presses] == [Keys.ControlJ]


def test_vt100_parser_maps_ctrl_enter_modify_other_keys_sequence_to_control_j():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed(tkbd.modify_other_keys_sequence(13, 5))
    parser.flush()

    assert [press.key for press in key_presses] == [Keys.ControlJ]


def test_backward_kill_word_uses_prompt_toolkit_word_boundaries():
    buffer = Buffer()
    buffer.set_document(Document("alpha beta", cursor_position=len("alpha beta")))
    clipboard = InMemoryClipboard()
    event = SimpleNamespace(
        current_buffer=buffer,
        arg=1,
        is_repeat=False,
        app=SimpleNamespace(
            clipboard=clipboard,
            output=SimpleNamespace(bell=lambda: None),
        ),
    )

    tkbd.BACKWARD_KILL_WORD(event)

    assert buffer.text == "alpha "
    assert buffer.cursor_position == len("alpha ")
    assert clipboard.get_data().text == "beta"
