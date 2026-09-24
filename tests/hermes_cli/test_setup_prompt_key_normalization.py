"""The setup wizard's numbered-input prompt must not type raw CSI bytes.

``_read_numbered_input`` runs when curses is unavailable, so it installs the
CSI-u aliases itself — setup can run without the classic CLI, which normally
installs them at startup. Those aliases only remap the *key*; prompt_toolkit
keeps the raw escape text in ``KeyPress.data`` and self-insert types it. The
prompt reads a menu number, so a leaked ``\\x1b[32;2u`` makes the answer
non-numeric and aborts the menu (#88071 on a sibling input path).
"""

from __future__ import annotations

import sys

import pytest

if sys.platform == "win32":
    pytest.skip("curses is not available on Windows", allow_module_level=True)

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from hermes_cli.curses_ui import (
    _read_numbered_input,
    reset_menu_navigation_handler,
    set_menu_navigation_handler,
)


def _type(payload: str) -> str:
    """Drive the real numbered-input prompt with ``payload`` then Enter."""
    token = set_menu_navigation_handler(lambda _event: None)
    try:
        with create_pipe_input() as inp:
            inp.send_text(payload + "\r")
            with create_app_session(input=inp, output=DummyOutput()):
                return _read_numbered_input("  Choice [1-9]: ")
    finally:
        reset_menu_navigation_handler(token)


@pytest.mark.parametrize(
    ("label", "payload", "expected"),
    [
        ("plain space", "1 2", "1 2"),
        ("shift+space xterm", "1\x1b[27;2;32~2", "1 2"),
        ("shift+space kitty", "1\x1b[32;2u2", "1 2"),
        ("shift+letter kitty", "\x1b[97;2u", "A"),
        ("keypad digit kitty", "\x1b[57404u", "5"),
    ],
)
def test_numbered_input_types_the_mapped_character(label, payload, expected):
    # Arrange: the prompt installs the CSI-u aliases itself; the mapped
    # character — never the escape text — must reach the buffer.
    # Act
    typed = _type(payload)

    # Assert
    assert typed == expected, f"{label}: got {typed!r}, expected {expected!r}"
