"""Augmentations to prompt_toolkit's input-parsing tables.

Imported once at CLI startup. Each helper installs a small mapping into
prompt_toolkit's `ANSI_SEQUENCES` so byte sequences emitted by modern
keyboard protocols (Kitty / xterm `modifyOtherKeys`) decode to existing
key tuples Hermes already binds.

Kept in a standalone module — separate from `cli.py` — so the registrations
can be unit-tested without importing the whole CLI runtime.
"""

from __future__ import annotations


def _clear_vt100_prefix_cache() -> None:
    """Drop prompt_toolkit's memoized "is this a prefix of a longer match?"
    answers after mutating ``ANSI_SEQUENCES``.

    The cache is module-global and populated lazily per distinct prefix, so
    parsers created before an install (or primed by earlier tests) would
    otherwise keep stale ``False`` answers and misparse newly registered
    sequences. Call after any install that changed the table.
    """
    try:
        from prompt_toolkit.input.vt100_parser import (
            _IS_PREFIX_OF_LONGER_MATCH_CACHE,
        )
        _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()
    except Exception:
        pass


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
    if changed:
        _clear_vt100_prefix_cache()
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
    if changed:
        _clear_vt100_prefix_cache()
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
    if changed:
        _clear_vt100_prefix_cache()
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
    Ctrl+U, Ctrl+W, Ctrl+Z, etc. (#56684, #86866, #87390).

    This function populates ``ANSI_SEQUENCES`` for the full set:

    * **Ctrl+letter** (a–z): ``ESC[27;5;<codepoint>~`` and ``ESC[<codepoint>;5u``
      → ``Keys.ControlA`` .. ``Keys.ControlZ``
    * **Ctrl+digit** (0–9): same formats → ``Keys.Control0`` .. ``Keys.Control9``
    * **Ctrl+symbol** (``[`` ``\\`` ``]`` ``^`` ``_`` `` `` ``@``):
      same formats → the same ``Keys`` value the raw control byte maps to.
    * **Alt+letter** (a–z, A–Z): ``ESC[27;3;<codepoint>~`` and
      ``ESC[<codepoint>;3u`` → ``(Keys.Escape, <letter>)`` — matching how
      prompt_toolkit handles a bare ``ESC`` followed by a character.
    * **Shift+letter** (a–z): → the uppercase character.
    * **Multi-modifier letters** (Shift+Alt=4, Ctrl+Shift=6, Ctrl+Alt=7,
      Ctrl+Alt+Shift=8): normalized onto the same targets — Ctrl-bearing
      combos behave as the Ctrl key (Alt adds an ``Escape`` prefix),
      matching how dte/kakoune normalize these protocols.
    * **Lock-bit variants**: every CSI-u mapping above is also installed
      with the CapsLock (64) and NumLock (128) bits ORed into the modifier
      parameter — kitty/ghostty include them while a lock is on, and
      without the variants every key combo dies with the lock enabled
      (``ESC[99;133u`` instead of ``ESC[99;5u``, #89651).
    * **Esc key**: ``ESC[27u`` / ``ESC[27;<mod>u`` (Kitty disambiguate mode
      reports Esc this way, #56684) → ``Keys.Escape``.
    * **Modified Enter/Tab/Backspace/Space**: Alt+Enter → the Alt+Enter
      newline tuple; Shift+Tab → ``BackTab``; Ctrl+Tab → plain Tab;
      Ctrl/Alt+Backspace → ``(Escape, ControlH)`` (backward-kill-word,
      matching the Ink TUI and Desktop, #78285); Shift+Backspace → plain
      backspace; Shift+Space → a plain space (#86866); Alt+Space →
      ``(Escape, " ")``.
    * **Kitty functional keys** (Private Use Area codepoints): keypad keys
      → their non-keypad equivalents (KP_ENTER → Enter, KP_4 → '4',
      KP_LEFT → Left, …); F13–F24 → ``Keys.F13``..``F24``; lock/media/
      modifier-event keys → ``Keys.Ignore`` so they are consumed instead of
      leaking as literal text. kitty emits these CSI-u forms even in legacy
      mode for keys that have no legacy encoding.

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

    # Kitty CSI-u encodes CapsLock/NumLock state as extra modifier bits
    # (caps=64, num=128) ORed into the parameter: with NumLock on, Ctrl+C
    # arrives as ESC[99;133u (5 + 128) instead of ESC[99;5u. Terminals
    # that report these bits (kitty, ghostty) break every key combo while
    # a lock is on (#89651) unless the lock variants are mapped too. The
    # xterm modifyOtherKeys encoding never carries the lock bits, so only
    # the CSI-u form needs them.
    _CSI_U_LOCK_BIT_VARIANTS = (0, 64, 128, 192)

    def _install_paired(modifier: int, mapping: dict) -> None:
        """Install both modifyOtherKeys (ESC[27;N;CP~) and CSI-u (ESC[CP;Nu)
        mappings for the given modifier and codepoint→key mapping."""
        nonlocal changed
        for codepoint, key_val in mapping.items():
            seqs = [f"\x1b[27;{modifier};{codepoint}~"]
            for lock_bits in _CSI_U_LOCK_BIT_VARIANTS:
                seqs.append(f"\x1b[{codepoint};{modifier + lock_bits}u")
            for seq in seqs:
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

    # -- Multi-modifier letters: Shift+Alt (4), Ctrl+Shift (6),
    # Ctrl+Alt (7), Ctrl+Alt+Shift (8) ----
    # The Kitty protocol always reports the UNSHIFTED codepoint; some
    # modifyOtherKeys emitters send the shifted one — map both cases.
    # Ctrl-bearing combos normalize onto the Ctrl key (Alt adds an Escape
    # prefix), Shift+Alt onto (Escape, UPPER) — the same normalization
    # dte/kakoune apply to these protocols. Without these, Ctrl+Shift+R
    # etc. leak as literal text under either protocol.
    shift_alt_map: dict[int, tuple] = {}
    ctrl_shift_map: dict[int, object] = {}
    ctrl_alt_map: dict[int, tuple] = {}
    for ch in range(ord('a'), ord('z') + 1):
        upper_char = chr(ch - 32)
        ctrl_key = ctrl_key_map.get(ch)
        for cp in (ch, ch - 32):
            shift_alt_map[cp] = (Keys.Escape, upper_char)
            if ctrl_key is not None:
                ctrl_shift_map[cp] = ctrl_key
                ctrl_alt_map[cp] = (Keys.Escape, ctrl_key)
    _install_paired(4, shift_alt_map)
    _install_paired(6, ctrl_shift_map)
    _install_paired(7, ctrl_alt_map)
    _install_paired(8, ctrl_alt_map)  # Ctrl+Alt+Shift — same normalization

    # -- The Esc KEY under Kitty disambiguate mode: ESC[27u (+ modifiers) --
    # Disambiguate mode reports the Esc key as CSI-u so it is
    # distinguishable from the ESC byte that starts escape sequences
    # (#56684 — previously leaked "[27u" as literal text into the prompt).
    # Modifiers run from 1 to 16: kitty reports Cmd as the super bit
    # (mod 9+) — same reason install_cmd_backspace_alias maps 9/10 — and
    # the lock-bit variants of the modifier-less form (1+64/128/192) are
    # how a lone Esc keypress arrives with a lock on. Lock bits (caps/num)
    # get the same variant treatment as _install_paired.
    for seq in ["\x1b[27u"] + [
        f"\x1b[27;{m + lock_bits}u"
        for m in range(1, 17)
        for lock_bits in _CSI_U_LOCK_BIT_VARIANTS
    ]:
        if seq not in ANSI_SEQUENCES:
            ANSI_SEQUENCES[seq] = Keys.Escape
            changed += 1

    # -- Modified Enter / Tab / Backspace / Space ----
    # Shift+Enter / Ctrl+Enter are installed by install_shift_enter_alias /
    # install_ctrl_enter_alias (which run first and win via setdefault).
    _install_paired(2, {
        9: Keys.BackTab,        # Shift+Tab — same as the legacy ESC[Z
        127: Keys.ControlH,     # Shift+Backspace — plain backspace
        32: " ",                # Shift+Space — still a space (#86866)
    })
    _install_paired(3, {
        13: (Keys.Escape, Keys.ControlM),   # Alt+Enter — newline tuple
        127: (Keys.Escape, Keys.ControlH),  # Alt+Backspace — backward-kill-word
        32: (Keys.Escape, " "),             # Alt+Space
    })
    _install_paired(5, {
        9: Keys.ControlI,                   # Ctrl+Tab — degrade to Tab
        127: (Keys.Escape, Keys.ControlH),  # Ctrl+Backspace — backward-kill-word,
                                            # matching Ink TUI + Desktop (#78285)
    })

    # -- Kitty functional keys (Private Use Area codepoints) ----
    # kitty emits these CSI-u encodings even in LEGACY mode for keys that
    # have no legacy encoding, so unmapped they leak as literal text in any
    # kitty session regardless of which modes were pushed.
    functional_map: dict[int, object] = {}
    for d in range(10):                       # KP_0..KP_9 → digits
        functional_map[57399 + d] = str(d)
    functional_map.update({                   # KP operators / punctuation
        57409: ".", 57410: "/", 57411: "*", 57412: "-",
        57413: "+", 57414: Keys.ControlM, 57415: "=", 57416: ",",
    })
    functional_map.update({                   # KP navigation → non-keypad keys
        57417: Keys.Left, 57418: Keys.Right, 57419: Keys.Up,
        57420: Keys.Down, 57421: Keys.PageUp, 57422: Keys.PageDown,
        57423: Keys.Home, 57424: Keys.End, 57425: Keys.Insert,
        57426: Keys.Delete,
    })
    for n in range(13, 25):                   # F13..F24
        functional_map[57376 + (n - 13)] = getattr(Keys, f"F{n}")
    # No prompt_toolkit equivalent (lock keys, PrintScreen, Menu, F25-F35,
    # KP_BEGIN, media keys, bare modifier events): consume as Ignore
    # instead of leaking literal text.
    for code in (
        list(range(57358, 57364))       # locks, PrintScreen, Pause, Menu
        + list(range(57388, 57399))     # F25..F35
        + [57427]                       # KP_BEGIN
        + list(range(57428, 57455))     # media keys + modifier key events
    ):
        functional_map.setdefault(code, Keys.Ignore)
    for code, key_val in functional_map.items():
        seq = f"\x1b[{code}u"
        if seq not in ANSI_SEQUENCES:
            ANSI_SEQUENCES[seq] = key_val
            changed += 1

    # New longer sequences can flip "is this a prefix of a longer match?"
    # answers the VT100 parser already cached — drop the cache so parsers
    # created before this install (or in earlier tests) can't misparse.
    if changed:
        _clear_vt100_prefix_cache()

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
    if changed:
        _clear_vt100_prefix_cache()
    return changed
