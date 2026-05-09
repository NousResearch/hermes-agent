"""Enhanced terminal protocol key sequence mappings for prompt_toolkit.

Handles CSI-u (Kitty keyboard protocol) and xterm modifyOtherKeys sequences
that modern terminals send for key combinations like Ctrl+Enter.

Reference:
- CSI-u: https://sw.kovidgoyal.net/kitty/keyboard-protocol/
- modifyOtherKeys: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
"""

from __future__ import annotations

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent


def install_ctrl_enter_alias(kb: KeyBindings) -> None:
    """Register enhanced Ctrl+Enter sequences as newline triggers.

    Modern terminals (Kitty, iTerm2, WezTerm, xterm with modifyOtherKeys)
    may send Ctrl+Enter as CSI escape sequences instead of raw LF (c-j).

    This function maps those sequences to (Escape, ControlM) so they
    activate the same handler as Alt+Enter (which prompt_toolkit already
    maps to newline in Hermes).

    Sequences handled:
        - \\x1b[13;5u       — CSI-u (Kitty protocol)
        - \\x1b[27;5;13~    — xterm modifyOtherKeys (mode 2)
        - \\x1b[27;5;13u    — xterm modifyOtherKeys (mode 1, Kitty variant)
    """

    @kb.add('escape', '[', '1', '3', ';', '5', 'u')
    def _csi_u_ctrl_enter(event: KeyPressEvent) -> None:
        """CSI-u Ctrl+Enter (Kitty protocol)."""
        event.current_buffer.insert_text('\n')

    @kb.add('escape', '[', '2', '7', ';', '5', ';', '1', '3', '~')
    def _mok2_ctrl_enter(event: KeyPressEvent) -> None:
        """xterm modifyOtherKeys mode 2 Ctrl+Enter."""
        event.current_buffer.insert_text('\n')

    @kb.add('escape', '[', '2', '7', ';', '5', ';', '1', '3', 'u')
    def _mok1_ctrl_enter(event: KeyPressEvent) -> None:
        """xterm modifyOtherKeys mode 1 Ctrl+Enter (Kitty variant)."""
        event.current_buffer.insert_text('\n')
