"""Tests for Hermes CLI prompt-toolkit keybindings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import InMemoryClipboard
from prompt_toolkit.document import Document
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding import KeyBindings

import cli


def test_register_word_delete_keybindings_wires_terminal_sequences():
    kb = KeyBindings()

    cli._register_word_delete_keybindings(kb)

    assert len(kb.bindings) == 3

    with patch.object(cli, "_delete_previous_word") as delete_previous_word:
        event = object()
        for binding in kb.bindings:
            binding.handler(event)

    assert delete_previous_word.call_count == 3


def test_parse_terminal_keyboard_capabilities_detects_kitty_and_modify_other_keys():
    capabilities = cli._parse_terminal_keyboard_capabilities(
        "\x1b[?1u\x1b[>4;2m\x1b[?64;1;2;6;9;15;18;21;22c"
    )

    assert capabilities.kitty_supported is True
    assert capabilities.modify_other_keys_supported is True


def test_select_terminal_keyboard_mode_prefers_kitty():
    capabilities = cli._TerminalKeyboardCapabilities(
        kitty_supported=True,
        modify_other_keys_supported=True,
    )

    assert cli._select_terminal_keyboard_mode(capabilities) == "kitty"


def test_select_terminal_keyboard_mode_falls_back_to_modify_other_keys():
    capabilities = cli._TerminalKeyboardCapabilities(
        kitty_supported=False,
        modify_other_keys_supported=True,
    )

    assert cli._select_terminal_keyboard_mode(capabilities) == "modify_other_keys"


@pytest.mark.parametrize(
    ("mode", "enable", "expected_sequence"),
    [
        ("kitty", True, cli._KITTY_KEYBOARD_ENABLE),
        ("kitty", False, cli._KITTY_KEYBOARD_DISABLE),
        ("modify_other_keys", True, cli._MODIFY_OTHER_KEYS_ENABLE),
        ("modify_other_keys", False, cli._MODIFY_OTHER_KEYS_DISABLE),
    ],
)
def test_set_terminal_keyboard_mode_emits_expected_sequence(
    mode,
    enable,
    expected_sequence,
):
    written = []

    cli._set_terminal_keyboard_mode(
        mode,
        enable=enable,
        writer=written.append,
    )

    assert written == [expected_sequence]


def test_install_ctrl_backspace_sequences_registers_modified_keycodes():
    sequences = {}

    cli._install_ctrl_backspace_input_sequences(
        sequences,
        terminfo_backspace=None,
    )

    for sequence in cli._CTRL_BACKSPACE_ESCAPE_SEQUENCES:
        assert sequences[sequence] == cli._CTRL_BACKSPACE_KEYS


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

    cli._install_ctrl_backspace_input_sequences(
        sequences,
        terminfo_backspace=terminfo_backspace,
    )

    assert sequences[ctrl_backspace_byte] == cli._CTRL_BACKSPACE_KEYS


def test_vt100_parser_maps_ctrl_backspace_csi_u_sequence():
    key_presses = []
    parser = Vt100Parser(key_presses.append)

    parser.feed("\x1b[127;5u")
    parser.flush()

    assert [press.key for press in key_presses] == list(cli._CTRL_BACKSPACE_KEYS)


def test_delete_previous_word_uses_prompt_toolkit_word_boundaries():
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

    cli._delete_previous_word(event)

    assert buffer.text == "alpha "
    assert buffer.cursor_position == len("alpha ")
    assert clipboard.get_data().text == "beta"
