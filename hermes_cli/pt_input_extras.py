"""Augmentations to prompt_toolkit's input-parsing tables.

Imported once at CLI startup. Each helper installs a small mapping into
prompt_toolkit's `ANSI_SEQUENCES` so byte sequences emitted by modern
keyboard protocols (Kitty / xterm `modifyOtherKeys`) decode to existing
key tuples Hermes already binds.

Kept in a standalone module — separate from `cli.py` — so the registrations
can be unit-tested without importing the whole CLI runtime.
"""

from __future__ import annotations

import sys


def enable_kitty_keyboard_protocol() -> bool:
    """Enable the Kitty keyboard protocol so the terminal sends distinct
    byte sequences for Shift+Enter, Ctrl+Enter, etc.

    Sends the CSI > 1 u sequence to push the terminal into kitty protocol
    mode (level 1 — disambiguate escape keys). This makes Shift+Enter emit
    \\x1b[13;2u instead of a bare CR, which the existing shift_enter_alias
    handler then maps to a newline.

    Only sent on POSIX (not Windows). Safe no-op on terminals that don't
    support it — they ignore the sequence.

    The protocol is automatically disabled when the application exits
    (prompt_toolkit sends the pop sequence on teardown), but we also
    register an atexit fallback for safety.
    """
    if sys.platform == "win32":
        return False
    try:
        seq = "\x1b[>1u"
        sys.stdout.write(seq)
        sys.stdout.flush()
        # Register pop on exit (prompt_toolkit should handle this, but be safe)
        import atexit
        atexit.register(lambda: (sys.stdout.write("\x1b[<1u"), sys.stdout.flush()))
        return True
    except Exception:
        return False


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


def install_kitty_control_aliases() -> int:
    """Map Kitty keyboard protocol CSI-u sequences for all control keys
    to their corresponding prompt_toolkit ``Keys`` entries.

    When :func:`enable_kitty_keyboard_protocol` pushes the terminal into
    CSI-u mode, *every* key press is sent as a CSI-u sequence — not just
    Shift+Enter and Ctrl+Enter.  prompt_toolkit's stock ``ANSI_SEQUENCES``
    table contains only four CSI-u entries (codepoint 1, modifier 5–8).
    The existing :func:`install_shift_enter_alias` and
    :func:`install_ctrl_enter_alias` cover Enter (codepoint 13), but all
    other Ctrl+key combinations (Ctrl+C = ``\\x1b[99;5u``, Ctrl+D =
    ``\\x1b[100;5u``, …) and plain Escape (``\\x1b[27u``) are unmapped.

    Unmapped CSI-u sequences fall through prompt_toolkit's VT100 parser to
    literal text insertion: the leading ESC fires as ``Keys.Escape``, then
    ``[99;5u`` appears as garbage text in the prompt buffer.  In vi mode
    this is especially disruptive because Escape is the mode-switch key.

    This helper registers CSI-u sequences for:

    - **Ctrl+A–Z** (codepoints 97–122, modifier 5 = Ctrl)
    - **Ctrl+0–9** (codepoints 48–57, modifier 5 = Ctrl)
    - **Ctrl+special chars**: ``@`` (64), ``[`` (91), ``\\`` (92),
      ``]`` (93), ``^`` (94), ``_`` (95) — all modifier 5 = Ctrl
    - **Plain Escape** (codepoint 27, no modifier)

    so the parser routes them to the correct ``Keys.ControlX`` /
    ``Keys.Escape`` entries instead of inserting literal text.

    Sequences for which prompt_toolkit has no corresponding ``Keys`` entry
    (e.g. Alt+letter = modifier 3, Ctrl+Shift+letter = modifier 6) cannot
    be mapped and are left untouched.  The parser will still fall through
    to literal text for those — a limitation of prompt_toolkit's key model.

    Also clears the VT100 parser's ``_IS_PREFIX_OF_LONGER_MATCH_CACHE`` so
    the new entries are immediately effective even if a parser has already
    been instantiated earlier in the process lifecycle.

    Returns the number of sequences whose mapping was changed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.input.vt100_parser import (
            _IS_PREFIX_OF_LONGER_MATCH_CACHE,
        )
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    # Map Unicode codepoint → prompt_toolkit Keys for all Ctrl-modified
    # keys that prompt_toolkit has an enum entry for.
    #
    # In the Kitty keyboard protocol, the CSI-u codepoint is the Unicode
    # code point of the key's base character.  For Ctrl+A the base char is
    # 'a' (codepoint 97), so the sequence is ESC[97;5u.  For Ctrl+1 the
    # base char is '1' (codepoint 49), so the sequence is ESC[49;5u.
    #
    # Modifier 5 = Ctrl (bit 2 set in the modifier bitmask).
    _CTRL_CODEPOINTS: dict[int, Keys] = {
        # Ctrl+A – Ctrl+Z (codepoints 97–122)
        97: Keys.ControlA,
        98: Keys.ControlB,
        99: Keys.ControlC,
        100: Keys.ControlD,
        101: Keys.ControlE,
        102: Keys.ControlF,
        103: Keys.ControlG,
        104: Keys.ControlH,
        105: Keys.ControlI,
        106: Keys.ControlJ,
        107: Keys.ControlK,
        108: Keys.ControlL,
        109: Keys.ControlM,
        110: Keys.ControlN,
        111: Keys.ControlO,
        112: Keys.ControlP,
        113: Keys.ControlQ,
        114: Keys.ControlR,
        115: Keys.ControlS,
        116: Keys.ControlT,
        117: Keys.ControlU,
        118: Keys.ControlV,
        119: Keys.ControlW,
        120: Keys.ControlX,
        121: Keys.ControlY,
        122: Keys.ControlZ,
        # Ctrl+0 – Ctrl+9 (codepoints 48–57)
        48: Keys.Control0,
        49: Keys.Control1,
        50: Keys.Control2,
        51: Keys.Control3,
        52: Keys.Control4,
        53: Keys.Control5,
        54: Keys.Control6,
        55: Keys.Control7,
        56: Keys.Control8,
        57: Keys.Control9,
        # Ctrl+special characters
        64: Keys.ControlAt,          # Ctrl+@
        91: Keys.Escape,             # Ctrl+[ — same as Escape
        92: Keys.ControlBackslash,   # Ctrl+\
        93: Keys.ControlSquareClose,  # Ctrl+]
        94: Keys.ControlCircumflex,  # Ctrl+^
        95: Keys.ControlUnderscore,  # Ctrl+_
        32: Keys.ControlAt,          # Ctrl+Space — same key as Ctrl+@ (NUL)
    }

    changed = 0
    for codepoint, key in _CTRL_CODEPOINTS.items():
        seq = f"\x1b[{codepoint};5u"
        if ANSI_SEQUENCES.get(seq) != key:
            ANSI_SEQUENCES[seq] = key
            changed += 1

    # Plain Escape: codepoint 27, no modifier.
    #
    # In kitty protocol level 1, the terminal sends ESC[27u for a bare
    # Escape key press instead of a raw ESC byte.  Without this mapping the
    # parser fires Keys.Escape on the leading ESC byte and then inserts
    # "[27u" as literal text.
    esc_seq = "\x1b[27u"
    if ANSI_SEQUENCES.get(esc_seq) != Keys.Escape:
        ANSI_SEQUENCES[esc_seq] = Keys.Escape
        changed += 1

    # Invalidate the prefix cache so the parser picks up the new entries.
    # _IsPrefixOfLongerMatchCache.__missing__ caches results on first
    # access; if any prefix (e.g. "\x1b[99") was queried before our
    # additions, it would have a stale is_prefix_of_longer=False that
    # prevents the parser from waiting for the full sequence.
    _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()

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
