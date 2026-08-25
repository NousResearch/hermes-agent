"""Augmentations to prompt_toolkit's input-parsing tables."""

from __future__ import annotations

# kitty CSI-u ORs lock-key state into the modifier parameter of every key
# event while a lock is on: CapsLock=64, NumLock=128, both=192 (#88221,
# #89651).  Every fixed-modifier CSI-u (and legacy CSI-tilde / CSI-letter)
# registration therefore needs lock-offset twins, or those events leak into
# the prompt as literal text.  The xterm modifyOtherKeys ``ESC[27;N;CP~``
# encoding never carries lock bits, so it never gets the twins.
_LOCK_BIT_OFFSETS = (0, 64, 128, 192)


def _lock_variants(modifier: int) -> tuple[int, ...]:
    """Return ``modifier`` plus its CapsLock/NumLock/both twins."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS)


def _lock_twins(modifier: int) -> tuple[int, ...]:
    """Return only the lock twins of ``modifier`` (never the base value)."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS[1:])


def _clear_vt100_prefix_cache() -> None:
    """Drop prompt_toolkit's memoized prefix-match answers after mutating ``ANSI_SEQUENCES``.

    The cache is module-global and lazily filled per prefix, so parsers created before an install
    would keep stale ``False`` answers and misparse newly registered sequences.
    """
    try:
        from prompt_toolkit.input.vt100_parser import _IS_PREFIX_OF_LONGER_MATCH_CACHE
        _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()
    except Exception:
        pass


def _install(build, *, overwrite: bool) -> int:
    """Install ``build(ANSI_SEQUENCES, Keys) -> {seq: key}`` into prompt_toolkit's table; return
    the number of entries changed (0 when prompt_toolkit is unavailable).

    Sequences mapped:
      - "\\x1b[13;2u"     — Kitty keyboard protocol / CSI-u, modifier=2 (Shift)
        (plus its CapsLock/NumLock lock twins via ``_lock_variants``)
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
    changed = 0
    seqs = [f"\x1b[13;{m}u" for m in _lock_variants(2)]
    seqs += ["\x1b[27;2;13~", "\x1b[27;2;13u"]
    for seq in seqs:
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
        (plus its CapsLock/NumLock lock twins via ``_lock_variants``)
      - "\\x1b[27;5;13~"  — xterm modifyOtherKeys=2, modifier=5 (Ctrl)
      - "\\x1b[27;5;13u"  — alternate ordering some emitters use

    Stock prompt_toolkit maps only the tilde form ``\\x1b[27;5;13~`` (to
    plain ``Keys.ControlM``, which this deliberately overwrites — same
    bug-fix rationale as install_shift_enter_alias). Without this alias,
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
    seqs = [f"\x1b[13;{m}u" for m in _lock_variants(5)]
    seqs += ["\x1b[27;5;13~", "\x1b[27;5;13u"]
    for seq in seqs:
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
    Codepoint 127 with modifier 9 (super) / 10 (super+shift), each with
    its CapsLock/NumLock lock twins via ``_lock_variants``:
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

    aliases: dict[str, object] = {}
    for base in (9, 10):  # super / super+shift
        for mod in _lock_variants(base):
            aliases[f"\x1b[127;{mod}u"] = Keys.ControlU
            aliases[f"\x1b[3;{mod}~"] = Keys.ControlK
    aliases["\x1b[27;9;127~"] = Keys.ControlU
    changed = 0
    for seq, key in aliases.items():
        if ANSI_SEQUENCES.get(seq) != key:
            ANSI_SEQUENCES[seq] = key
            changed += 1
    if changed:
        _clear_vt100_prefix_cache()
    return changed


def install_keypress_data_normalization() -> int:
    """Normalize KeyPress data for extended-key aliases that map to a single plain character
    (Shift+Space → ``' '``, Shift+letter → uppercase, keypad digits/operators).

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
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
    except Exception:
        return 0

    _orig_call_handler = _vt100_mod.Vt100Parser._call_handler
    if getattr(_orig_call_handler, "_hermes_char_data_normalized", False):
        return 0

    def _patched_call_handler(self, key, insert_text):
        # A single plain character mapped from an extended sequence must carry the mapped
        # character as its data — self-insert inserts event.data and the raw CSI would leak.
        if (isinstance(key, str) and len(key) == 1 and not isinstance(key, _PtKeys)
                and isinstance(insert_text, str) and insert_text.startswith("\x1b")):
            insert_text = key
        return _orig_call_handler(self, key, insert_text)

    _patched_call_handler._hermes_char_data_normalized = True
    _vt100_mod.Vt100Parser._call_handler = _patched_call_handler
    return 1


def _install_enter_alias(modifier: int) -> int:
    """Map <modifier>+Enter (Kitty CSI-u ``ESC[13;<m>u`` plus lock twins, xterm ``ESC[27;<m>;13~``
    / ``;13u``) to (Escape, ControlM) so the Alt+Enter newline handler fires.

    # Kitty CSI-u encodes CapsLock/NumLock state as extra modifier bits
    # (caps=64, num=128) ORed into the parameter: with NumLock on, Ctrl+C
    # arrives as ESC[99;133u (5 + 128) instead of ESC[99;5u. Terminals
    # that report these bits (kitty, ghostty) break every key combo while
    # a lock is on (#89651) unless the lock variants are mapped too. The
    # xterm modifyOtherKeys encoding never carries the lock bits, so only
    # the CSI-u form needs them.
    def _install_paired(modifier: int, mapping: dict) -> None:
        """Install both modifyOtherKeys (ESC[27;N;CP~) and CSI-u (ESC[CP;Nu)
        mappings for the given modifier and codepoint→key mapping.

        The tilde form is skipped for modifier 1 ("no modifier") — xterm
        never emits modifier-1 tilde sequences.
        """
        nonlocal changed
        for codepoint, key_val in mapping.items():
            seqs = [] if modifier == 1 else [f"\x1b[27;{modifier};{codepoint}~"]
            for mod in _lock_variants(modifier):
                seqs.append(f"\x1b[{codepoint};{mod}u")
            for seq in seqs:
                if seq not in ANSI_SEQUENCES:
                    ANSI_SEQUENCES[seq] = key_val
                    changed += 1

    # Ctrl+<ch>: the extended sequence maps to whatever Keys value the raw control byte
    # chr(ord(ch) & 0x1f) already maps to, so existing bindings fire identically. Covers a-z and
    # the control-producing symbols @ [ \ ] ^ _ and Space (\x00 -> ControlAt).
    letters = range(ord('a'), ord('z') + 1)
    ctrl_key_map: dict[int, object] = {
        cp: key for cp in (*letters, 64, 91, 92, 93, 94, 95, 32)
        if (key := ANSI_SEQUENCES.get(chr(cp & 0x1F))) is not None
    }
    # Ctrl+digit has no useful raw byte (chr(ord('0') & 0x1F) is ControlP), so map directly.
    ctrl_key_map.update({ord('0') + d: getattr(Keys, f"Control{d}") for d in range(10)})
    _install_paired(5, ctrl_key_map)

    # Letter combos. Alt+a -> (Escape, 'a') like bare Alt. Shift+a -> 'A' (safe on every Latin
    # layout; Shift+digit symbols are layout-specific and deliberately NOT mapped — leaking beats
    # wrong input). Kitty reports the UNSHIFTED codepoint, some modifyOtherKeys emitters the shifted
    # one — map both. Ctrl-bearing combos normalize onto the Ctrl key (Alt adds an Escape prefix),
    # Shift+Alt onto (Escape, UPPER) — the same normalization dte/kakoune apply.
    for ch in letters:
        upper_char = chr(ch - 32)
        ctrl_key = ctrl_key_map.get(ch)
        _install_paired(3, {ch: (Keys.Escape, chr(ch)), ch - 32: (Keys.Escape, upper_char)})
        for cp in (ch, ch - 32):
            _install_paired(2, {cp: upper_char})
            _install_paired(4, {cp: (Keys.Escape, upper_char)})
            if ctrl_key is not None:
                _install_paired(6, {cp: ctrl_key})
                for modifier in (7, 8):  # Ctrl+Alt and Ctrl+Alt+Shift — same normalization
                    _install_paired(modifier, {cp: (Keys.Escape, ctrl_key)})

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
        f"\x1b[27;{mod}u"
        for m in range(1, 17)
        for mod in _lock_variants(m)
    ]:
        if seq not in ANSI_SEQUENCES:
            ANSI_SEQUENCES[seq] = Keys.Escape
            changed += 1

    # Modified Enter/Tab/Backspace/Space (Shift/Ctrl+Enter are owned by the enter aliases, which run
    # first and win). Modifier 1 = unmodified keys kitty CSI-u-encodes on their own when a lock bit
    # is set (plain Backspace arrives as ESC[127;129u rather than \x7f).
    alt_backspace = (Keys.Escape, Keys.ControlH)  # backward-kill-word, matching Ink TUI + Desktop
    _install_paired(2, {9: Keys.BackTab, 127: Keys.ControlH, 32: " "})
    _install_paired(3, {13: (Keys.Escape, Keys.ControlM), 127: alt_backspace, 32: (Keys.Escape, " ")})
    _install_paired(5, {9: Keys.ControlI, 127: alt_backspace})  # Ctrl+Tab degrades to Tab
    _install_paired(1, {9: Keys.ControlI, 13: Keys.ControlM, 32: " ", 127: Keys.ControlH})

    # -- Unmodified keys with a lock bit set (kitty modifier 1 = "none") --
    # With a lock on, kitty stamps the lock bit onto keys pressed with NO
    # real modifier too, so plain Backspace arrives as ESC[127;129u
    # (1 + 128) rather than \x7f. _install_paired(1, ...) registers the
    # bare mod-1 spelling and its lock twins. Only keys kitty CSI-u-encodes
    # on their own are listed; plain text characters are still delivered
    # as UTF-8, lock bits or not.
    _install_paired(1, {
        9: Keys.ControlI,     # Tab
        13: Keys.ControlM,    # Enter
        32: " ",              # Space
        127: Keys.ControlH,   # Backspace
    })

    # -- Lock-key modifier bits (NumLock=128, CapsLock=64) on the legacy
    # CSI-letter / CSI-tilde forms kitty keeps using under the disambiguate
    # push: kitty encodes lock state into the modifier parameter, so a
    # plain Down with NumLock on arrives as ESC[1;129B (NumLock), ESC[1;65B
    # (CapsLock) or ESC[1;193B (both) instead of the legacy ESC[B — and a
    # modified one shifts the same way (Alt+Left → ESC[1;131D). Those fall
    # through the parser and leak as literal text ("[1;129B") in the input
    # line. Derive the lock twins from whatever the table already maps for
    # the base modifier (stock prompt_toolkit entries included), so every
    # modifier the terminal can report keeps working under a lock.
    for m in range(1, 17):
        # CSI-letter navigation: Up/Down/Right/Left/End/Home + F1-F4
        for trailer in "ABCDFHPQRS":
            base_seq = f"\x1b[1;{m}{trailer}" if m > 1 else f"\x1b[{trailer}"
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None and m == 1:
                # Plain F1-F4 live in the table as SS3 (ESC O P) forms.
                key = ANSI_SEQUENCES.get(f"\x1bO{trailer}")
            if key is None:
                continue
            for mod in _lock_twins(m):
                seq = f"\x1b[1;{mod}{trailer}"
                if seq not in ANSI_SEQUENCES:
                    ANSI_SEQUENCES[seq] = key
                    changed += 1
        # CSI-tilde navigation: Insert/Delete/PageUp/PageDown/Home/End
        for num in (1, 2, 3, 4, 5, 6, 7, 8):
            base_seq = f"\x1b[{num};{m}~" if m > 1 else f"\x1b[{num}~"
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None:
                continue
            for mod in _lock_twins(m):
                seq = f"\x1b[{num};{mod}~"
                if seq not in ANSI_SEQUENCES:
                    ANSI_SEQUENCES[seq] = key
                    changed += 1

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
        # Lock twins: with a lock on these arrive as ESC[<code>;129u etc.
        for mod in _lock_twins(1):
            seq = f"\x1b[{code};{mod}u"
            if seq not in ANSI_SEQUENCES:
                ANSI_SEQUENCES[seq] = key_val
                changed += 1

    keypad_twins = {
        57414: "\x1b[13;{mod}u",  # Enter
        **{code: "\x1b[1;{mod}" + suffix for code, suffix in
           zip((57417, 57418, 57419, 57420, 57423, 57424), "DCABHF")},
        **{code: f"\x1b[{number};{{mod}}~" for code, number in
           ((57421, 5), (57422, 6), (57425, 2), (57426, 3))},
    }
    for code, key_val in _kitty_functional_map(Keys).items():
        _put(f"\x1b[{code}u", key_val)
        for mod in _lock_twins(1):  # with a lock on these arrive as ESC[<code>;129u etc.
            _put(f"\x1b[{code};{mod}u", key_val)
        twin = keypad_twins.get(code)
        if isinstance(key_val, str) and not isinstance(key_val, Keys) and len(key_val) == 1:
            twin = f"\x1b[{ord(key_val)};{{mod}}u"
        # Modified keypad keys inherit existing non-keypad semantics, not new bindings.
        # Some twins (Alt+Enter, lock variants) are still in this builder's pending aliases.
        for modifier in range(2, 9):
            for mod in _lock_variants(modifier):
                equivalent = None
                if twin is not None:
                    source = twin.format(mod=mod)
                    equivalent = ANSI_SEQUENCES.get(source, aliases.get(source))
                elif key_val is Keys.Ignore:
                    equivalent = Keys.Ignore
                if equivalent is not None:
                    _put(f"\x1b[{code};{mod}u", equivalent)
    return aliases


def install_ignored_terminal_sequences() -> int:
    """Map focus reports ``ESC[I`` / ``ESC[O`` (Ghostty, iTerm2, some xterms) to ``Keys.Ignore``.

    Parser-level handling beats post-hoc regex stripping because the bytes never reach the buffer.
    ``setdefault`` lets user/downstream registrations win.
    """
    return _install(lambda _seqs, keys: {"\x1b[I": keys.Ignore, "\x1b[O": keys.Ignore}, overwrite=False)
