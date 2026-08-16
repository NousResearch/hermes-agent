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


def install_modify_other_keys_aliases() -> int:
    """Map Ctrl+key and Alt+key sequences emitted under ``modifyOtherKeys`` level 2
    and Kitty CSI-u to the same ``Keys``.* values that the raw control bytes
    already map to.

    When the terminal is in ``modifyOtherKeys=2`` mode (pushed by
    ``_enable_extended_enter_keys`` so Shift+Enter is distinguishable from
    Enter), the terminal re-encodes *every* Ctrl+key combo as
    ``ESC[27;5;<codepoint>~`` instead of the raw control byte (``\\x01`` etc.).
    Kitty keyboard protocol emits ``ESC[<codepoint>;5u``.

    Stock prompt_toolkit 3.x only maps ``ESC[27;5;13~`` (Ctrl+Enter = Ctrl+M);
    all other Ctrl+letter combos are unmapped and leak as literal text or get
    swallowed — breaking Ctrl+A, Ctrl+C, Ctrl+D, Ctrl+E, Ctrl+K, Ctrl+R,
    Ctrl+U, Ctrl+W, Ctrl+Z, etc. (#56684, #87711).

    This function populates ``ANSI_SEQUENCES`` for the full set:

    * **Ctrl+letter** (a–z): ``ESC[27;5;<codepoint>~`` and ``ESC[<codepoint>;5u``
      → ``Keys.ControlA`` .. ``Keys.ControlZ``
    * **Ctrl+digit** (0–9): same formats → ``Keys.Control0`` .. ``Keys.Control9``
    * **Ctrl+symbol** (``[`` ``\\`` ``]`` ``^`` ``_`` `` `` ``@``):
      same formats → the same ``Keys`` value the raw control byte maps to.
    * **Alt+letter** (a–z, A–Z): ``ESC[27;3;<codepoint>~`` and
      ``ESC[<codepoint>;3u`` → ``(Keys.Escape, <letter>)`` — matching how
      prompt_toolkit handles a bare ``ESC`` followed by a character.

    Existing mappings (including those installed by
    ``install_shift_enter_alias`` / ``install_ctrl_enter_alias``) are never
    overwritten — ``setdefault`` semantics.

    Returns the number of sequences whose mapping was newly installed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    # -- Ctrl+letter / Ctrl+digit / Ctrl+symbol → Keys.Control* ----
    # codepoint -> Keys value.  The raw control byte for Ctrl+<ch> is
    # chr(ord(ch) & 0x1f) (i.e. ord(ch) - 96 for lowercase).  We map the
    # *extended* sequence to the same Keys value that the raw byte maps to,
    # so prompt_toolkit's existing key bindings fire identically.
    ctrl_key_map: dict[int, object] = {}

    # a-z: Ctrl+A = \x01 = Keys.ControlA, ..., Ctrl+Z = \x1a = Keys.ControlZ
    for ch in range(ord('a'), ord('z') + 1):
        raw = chr(ch & 0x1F)  # 0x01..0x1a
        existing = ANSI_SEQUENCES.get(raw)
        if existing is not None:
            ctrl_key_map[ch] = existing

    # 0-9: Ctrl+digit codepoints don't have a useful raw-byte mapping
    # (e.g. chr(ord('0') & 0x1F) = 0x10 = ControlP, not Control0), so map
    # them directly to Keys.Control0..Keys.Control9.
    for d in range(10):
        ctrl_key_map[ord('0') + d] = getattr(Keys, f"Control{d}")

    # Symbols that produce control chars:
    # Ctrl+@   (64)  = \x00 = Keys.ControlAt
    # Ctrl+[   (91)  = \x1b = Keys.Escape
    # Ctrl+\   (92)  = \x1c = Keys.ControlBackslash
    # Ctrl+]   (93)  = \x1d = Keys.ControlSquareClose
    # Ctrl+^   (94)  = \x1e = Keys.ControlCircumflex
    # Ctrl+_   (95)  = \x1f = Keys.ControlUnderscore
    # Ctrl+Space(32) = \x00 = Keys.ControlAt (prompt_toolkit maps \x00 → ControlAt)
    for codepoint in (64, 91, 92, 93, 94, 95, 32):
        raw = chr(codepoint & 0x1F)
        existing = ANSI_SEQUENCES.get(raw)
        if existing is not None:
            ctrl_key_map[codepoint] = existing

    changed = 0

    def _install_paired(modifier: int, mapping: dict) -> None:
        """Install both modifyOtherKeys (ESC[27;N;CP~) and CSI-u (ESC[CP;Nu)
        mappings for the given modifier and codepoint→key mapping."""
        nonlocal changed
        for codepoint, key_val in mapping.items():
            for seq in (
                f"\x1b[27;{modifier};{codepoint}~",
                f"\x1b[{codepoint};{modifier}u",
            ):
                if seq not in ANSI_SEQUENCES:
                    ANSI_SEQUENCES[seq] = key_val
                    changed += 1

    # Ctrl+letter / Ctrl+digit / Ctrl+symbol (modifier 5)
    _install_paired(5, ctrl_key_map)

    # -- Alt+letter → (Escape, <letter>) ----
    # Under modifyOtherKeys, Alt+a = ESC[27;3;97~. Without mapping, this
    # leaks as literal text. prompt_toolkit handles bare Alt+letter as
    # (Escape, <letter>), so we map the extended sequences to the same tuple.
    alt_map: dict[int, tuple] = {}
    for ch in range(ord('a'), ord('z') + 1):
        letter = chr(ch)
        upper = chr(ch - 32)  # uppercase variant
        alt_map[ch] = (Keys.Escape, letter)
        alt_map[ch - 32] = (Keys.Escape, upper)
    _install_paired(3, alt_map)

    # -- Shift+letter → uppercase letter ----
    # Under modifyOtherKeys=2, some terminals re-encode Shift+a as
    # ESC[27;2;97~. Without mapping, this leaks as literal escape +
    # "[27;2;97~" in the prompt buffer — the "caps locked" / "every key
    # combo is broken" symptom (#87711).
    # Map Shift+letter to the uppercase character so typing works normally.
    # This is safe across all Latin keyboard layouts: Shift always uppercases
    # letters.  Shift+digit symbols are layout-specific (US: '!', AZERTY: '¹',
    # etc.) so they are NOT mapped here — if the terminal sends those under
    # modifyOtherKeys, they will leak, but that's better than wrong input.
    # Map both the lowercase and uppercase codepoints — some terminals send
    # the already-shifted codepoint (65 for 'A') with modifier=2.
    shift_map: dict[int, str] = {}
    for ch in range(ord('a'), ord('z') + 1):
        upper_char = chr(ch - 32)  # 'A'..'Z'
        shift_map[ch] = upper_char
        shift_map[ch - 32] = upper_char
    _install_paired(2, shift_map)

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
