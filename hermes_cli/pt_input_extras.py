"""Augmentations to prompt_toolkit's input-parsing tables.

Imported once at CLI startup. Each helper installs a small mapping into
prompt_toolkit's `ANSI_SEQUENCES` so byte sequences emitted by modern
keyboard protocols (Kitty / xterm `modifyOtherKeys`) decode to existing
key tuples Hermes already binds.

Kept in a standalone module — separate from `cli.py` — so the registrations
can be unit-tested without importing the whole CLI runtime.
"""

from __future__ import annotations


def install_shift_enter_alias() -> int:
    """Map Shift+Enter byte sequences to the (Escape, ControlM) key tuple
    that Alt+Enter produces, so the existing Alt+Enter newline handler
    fires for terminals that emit a distinct Shift+Enter.

    Sequences mapped:
      - "\\x1b[13;2u"     — Kitty keyboard protocol / CSI-u, modifier=2 (Shift)
      - "\\x1b[27;2;13~"  — xterm modifyOtherKeys=2, modifier=2 (Shift)
      - "\\x1b[27;2;13u"  — alternate ordering some emitters use

    The CSI-u sequence is not in stock prompt_toolkit. The modifyOtherKeys
    variant `\\x1b[27;2;13~` IS in stock prompt_toolkit but mapped to plain
    `Keys.ControlM` — i.e. Shift+Enter behaves identically to Enter, which
    is the very bug this helper exists to fix. We therefore overwrite
    those two specific keys (and `\\x1b[27;2;13u`) unconditionally; other
    `\\x1b[27;...;13~` sequences (Ctrl+Enter, Alt+Enter via modifyOtherKeys
    variants 5/6/etc.) are left untouched.

    Default macOS Terminal and stock Windows Terminal still send the same
    byte for Enter and Shift+Enter, so there is no fix for those terminals
    at the application layer — the sequences above never reach Hermes.

    Returns the number of sequences whose mapping was changed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    alt_enter = (Keys.Escape, Keys.ControlM)
    changed = 0
    for seq in ("\x1b[13;2u", "\x1b[27;2;13~", "\x1b[27;2;13u"):
        if ANSI_SEQUENCES.get(seq) != alt_enter:
            ANSI_SEQUENCES[seq] = alt_enter
            changed += 1
    return changed


def install_ctrl_enter_alias() -> int:
    """Map Ctrl+Enter byte sequences to the (Escape, ControlM) key tuple
    that Alt+Enter produces, so the existing Alt+Enter newline handler
    fires for terminals that emit a distinct Ctrl+Enter.

    Sequences mapped:
      - "\\x1b[13;5u"     — Kitty keyboard protocol / CSI-u, modifier=5 (Ctrl)
      - "\\x1b[27;5;13~"  — xterm modifyOtherKeys=2, modifier=5 (Ctrl)
      - "\\x1b[27;5;13u"  — alternate ordering some emitters use

    Stock prompt_toolkit doesn't map any of these. Without this alias,
    Kitty/mintty/xterm-with-modifyOtherKeys users over SSH never get a
    Ctrl+Enter newline — the keystroke arrives as a raw CSI sequence that
    falls through to the default character-insert handler. See #22379.

    Returns the number of sequences whose mapping was changed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    alt_enter = (Keys.Escape, Keys.ControlM)
    changed = 0
    for seq in ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[27;5;13u"):
        if ANSI_SEQUENCES.get(seq) != alt_enter:
            ANSI_SEQUENCES[seq] = alt_enter
            changed += 1
    return changed


def install_cmd_backspace_alias() -> int:
    """Map Cmd+Backspace / Cmd+ForwardDelete to the readline kill bindings
    prompt_toolkit already ships (``unix-line-discard`` / ``kill-line``).

    Terminals that rewrite Cmd+Backspace to Ctrl+U (``\\x15``) already work.
    Kitty keyboard protocol and xterm modifyOtherKeys terminals instead
    report Cmd as the *super* modifier bit (8), producing sequences
    prompt_toolkit does not map — the raw bytes then fall through to
    literal insertion.

    Cmd+Backspace → ``Keys.ControlU`` (kill backward to start of line).
    Codepoint 127 with modifier 9 (super) / 10 (super+shift):
      - ``\\x1b[127;9u`` / ``\\x1b[127;10u``  — Kitty CSI-u
      - ``\\x1b[27;9;127~``                   — xterm modifyOtherKeys

    Cmd+ForwardDelete → ``Keys.ControlK`` (kill to end of line). The
    forward-delete key is a CSI *tilde* key, not a CSI-u codepoint, so the
    modifier rides in the standard ``CSI 3 ; mod ~`` form:
      - ``\\x1b[3;9~`` / ``\\x1b[3;10~``

    Returns the number of sequences whose mapping was changed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    aliases = {
        "\x1b[127;9u": Keys.ControlU,
        "\x1b[127;10u": Keys.ControlU,
        "\x1b[27;9;127~": Keys.ControlU,
        "\x1b[3;9~": Keys.ControlK,
        "\x1b[3;10~": Keys.ControlK,
    }
    changed = 0
    for seq, key in aliases.items():
        if ANSI_SEQUENCES.get(seq) != key:
            ANSI_SEQUENCES[seq] = key
            changed += 1
    return changed


def install_ignored_terminal_sequences() -> int:
    """Map terminal-emitted noise sequences to ``Keys.Ignore`` so they
    are consumed by the VT100 parser before they reach key bindings or
    the input buffer.

    Currently covers focus reports:
      - ``\\x1b[I`` — terminal regained focus (focus in)
      - ``\\x1b[O`` — terminal lost focus (focus out)

    Ghostty, iTerm2, and some xterm builds can emit these sequences when
    the user switches tabs / windows or when a multiplexer toggles focus
    tracking upstream. prompt_toolkit does not map these by default, so
    its parser falls back to literal key presses (ESC, ``[``, ``I``/``O``)
    and inserts ``[I``/``[O`` into the prompt buffer after the ESC byte
    is handled.

    Registering them as ``Keys.Ignore`` is parser-level — strictly
    cleaner than post-hoc regex stripping in the input sanitizer because
    the bytes never reach the buffer. ``setdefault`` is used so any user
    or downstream registration wins.

    Returns the number of sequences whose mapping was changed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    changed = 0
    for seq in ("\x1b[I", "\x1b[O"):
        if seq not in ANSI_SEQUENCES:
            ANSI_SEQUENCES[seq] = Keys.Ignore
            changed += 1
    return changed
