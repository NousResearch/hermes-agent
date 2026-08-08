"""Verify Cmd+Backspace / Cmd+ForwardDelete byte sequences from CSI-u
terminals reach prompt_toolkit's readline kill bindings instead of leaking
into the buffer as literal text.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

from hermes_cli.pt_input_extras import install_cmd_backspace_alias


# Cmd rides as the super modifier bit (8), so modifier = 9 (super) or
# 10 (super+shift).
CMD_BACKSPACE_SEQUENCES = (
    "\x1b[127;9u",       # Kitty CSI-u
    "\x1b[127;10u",      # Kitty CSI-u, +shift
    "\x1b[27;9;127~",    # xterm modifyOtherKeys
)

# Forward-delete is a CSI tilde key, not a CSI-u codepoint.
CMD_FWD_DELETE_SEQUENCES = (
    "\x1b[3;9~",
    "\x1b[3;10~",
)


@pytest.fixture(autouse=True)
def _ensure_alias_installed():
    install_cmd_backspace_alias()


def _parse(byte_seq: str):
    out = []
    parser = Vt100Parser(out.append)
    for ch in byte_seq:
        parser.feed(ch)
    parser.flush()
    return [kp.key for kp in out]


@pytest.mark.parametrize("seq", CMD_BACKSPACE_SEQUENCES)
def test_cmd_backspace_parses_as_ctrl_u(seq):
    """Cmd+Backspace must reach unix-line-discard, exactly as Ctrl+U does."""
    assert _parse(seq) == _parse("\x15")


@pytest.mark.parametrize("seq", CMD_FWD_DELETE_SEQUENCES)
def test_cmd_forward_delete_parses_as_ctrl_k(seq):
    """Cmd+ForwardDelete must reach kill-line, exactly as Ctrl+K does."""
    assert _parse(seq) == _parse("\x0b")


@pytest.mark.parametrize("seq", CMD_BACKSPACE_SEQUENCES + CMD_FWD_DELETE_SEQUENCES)
def test_sequences_emit_exactly_one_keypress(seq):
    """The whole sequence is consumed. A partial match would emit Escape
    plus the remainder as literal text — the bug this alias exists to fix."""
    assert len(_parse(seq)) == 1


def test_install_is_idempotent():
    install_cmd_backspace_alias()
    assert install_cmd_backspace_alias() == 0


def test_unmodified_keys_keep_their_own_bindings():
    """Aliasing the Cmd variants must not disturb the bare keys."""
    assert ANSI_SEQUENCES["\x1b[3~"] == Keys.Delete
    assert ANSI_SEQUENCES["\x7f"] == Keys.ControlH


def test_ctrl_forward_delete_is_not_remapped():
    """Ctrl+ForwardDelete (modifier 5) is delete-word on Linux/Windows and
    must keep prompt_toolkit's own binding, not become kill-line."""
    assert ANSI_SEQUENCES["\x1b[3;5~"] == Keys.ControlDelete
