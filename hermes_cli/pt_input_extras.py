"""Augmentations to prompt_toolkit's input-parsing tables."""

from __future__ import annotations

import os


def _kitty_reports_unshifted_codepoints() -> bool:
    """True when the CSI-u path will be driven with UNSHIFTED codepoints.

    The kitty keyboard protocol always reports the base (unshifted) codepoint
    plus a Shift modifier, while xterm modifyOtherKeys emitters report the
    already-shifted one.  That single difference decides which half of the
    Shift+punctuation table can ever fire, and therefore whether a US-layout
    assumption is load-bearing — see ``_shift_punctuation_base_map_is_safe``.

    Ghostty is excluded deliberately: it is pushed modifyOtherKeys only (its
    kitty disambiguate mode strips Alt from Backspace), so it reports shifted
    codepoints even though its TERM mentions neither protocol.
    """
    env = os.environ
    if (env.get("TERM_PROGRAM") or "").strip() == "ghostty":
        return False
    term = (env.get("TERM") or "").strip().lower()
    if term == "xterm-ghostty":
        return False
    return bool(env.get("KITTY_WINDOW_ID") or "kitty" in term)


def _configured_layout() -> str:
    """The primary configured keyboard layout, or "" when nothing says.

    Cheap and env/file based on purpose: this runs on the CLI startup path, so
    it must not spawn a subprocess.
    """
    layout = (os.environ.get("XKB_DEFAULT_LAYOUT") or "").strip().lower()
    if not layout:
        for path, key in (
            ("/etc/default/keyboard", "XKBLAYOUT"),
            ("/etc/vconsole.conf", "KEYMAP"),
        ):
            try:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        name, _, value = line.partition("=")
                        if name.strip() == key:
                            layout = value.strip().strip('"').strip("'").lower()
                            break
            except OSError:
                continue
            if layout:
                break
    # "us,gr" means us is primary; "us-acentos"/"us.utf-8" are console keymaps.
    return layout.split(",")[0].strip()


def _configured_layout_and_variant() -> tuple[str, str]:
    """Split the primary layout into ``(layout, variant)``.

    xkb spells a variant as ``us(dvorak)``; the variant is the block name inside
    the layout's symbol file, so keeping it lets the derived map describe the
    layout the user is actually typing on rather than that file's default.
    """
    primary = _configured_layout()
    if primary.endswith(")") and "(" in primary:
        layout, _, variant = primary.partition("(")
        return layout.strip(), variant[:-1].strip()
    return primary, ""


def _us_punctuation_layout() -> bool:
    """True only when Shift+<punct> is POSITIVELY known to follow US layout.

    Unknown answers return False, because the caller treats False as "do not
    guess".
    """
    primary = _configured_layout()
    if not primary:
        return False
    # Only bare "us" (optionally with a console encoding suffix such as
    # "us.utf-8") is safe by NAME. "us-" prefixed console keymaps are NOT:
    # us-acentos turns ' and " into dead accent keys, so Shift+' is not '"'
    # there. Anything else falls through to the xkb table, which answers from
    # the layout's actual definition instead of its name.
    if primary == "us" or primary.startswith("us."):
        return True
    # The name is not "us", but the layout may still leave Shift+punctuation
    # exactly where US puts it — Greek does.  Ask its xkb table rather than
    # rejecting a layout that is in fact compatible.
    return _layout_keeps_us_shift_punctuation(primary)


# Shift+<punct> on a US layout, keyed by the base character.  Single source of
# truth: the alias table below builds both of its halves from this, and
# ``_layout_keeps_us_shift_punctuation`` checks a foreign layout against it, so
# the two can never drift apart.
_SHIFT_PUNCTUATION_BY_CHAR = {
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
    "`": "~",
}
_SHIFT_PUNCTUATION = {ord(b): s for b, s in _SHIFT_PUNCTUATION_BY_CHAR.items()}

# xkb key names for the keys the Shift+punctuation map covers, plus the keysym
# names those keys carry at shift level 2 on US.  These answer the question the
# layout NAME cannot: does this layout leave Shift+punctuation where US puts it?
# Greek DOES — it declares the number row ``any, any`` or with the identical US
# symbols and only overrides the AltGr levels — while AZERTY does not (``AE01``
# is ``ampersand, 1``).  Judging by name alone would reject Greek needlessly,
# and that is exactly the difference between a kitty user getting the fix or not.
_XKB_KEY_BY_BASE = {
    "1": "AE01",
    "2": "AE02",
    "3": "AE03",
    "4": "AE04",
    "5": "AE05",
    "6": "AE06",
    "7": "AE07",
    "8": "AE08",
    "9": "AE09",
    "0": "AE10",
    "-": "AE11",
    "=": "AE12",
    "[": "AD11",
    "]": "AD12",
    "\\": "BKSL",
    ";": "AC10",
    "'": "AC11",
    ",": "AB08",
    ".": "AB09",
    "/": "AB10",
    "`": "TLDE",
}
_XKB_KEYSYM_CHAR = {
    "exclam": "!",
    "at": "@",
    "numbersign": "#",
    "dollar": "$",
    "percent": "%",
    "asciicircum": "^",
    "ampersand": "&",
    "asterisk": "*",
    "parenleft": "(",
    "parenright": ")",
    "underscore": "_",
    "plus": "+",
    "braceleft": "{",
    "braceright": "}",
    "bar": "|",
    "colon": ":",
    "quotedbl": '"',
    "less": "<",
    "greater": ">",
    "question": "?",
    "asciitilde": "~",
}
_XKB_SYMBOLS_DIRS = ("/usr/share/X11/xkb/symbols", "/usr/local/share/X11/xkb/symbols")
_XKB_KEY_RE = None


# X11 keysym names for every printable ASCII character.  Needed to turn an xkb
# key definition back into the characters it produces, so the Shift+punctuation
# map can be DERIVED from the user's actual layout instead of assumed from US.
_KEYSYM_NAMES = (
    "space exclam quotedbl numbersign dollar percent ampersand apostrophe "
    "parenleft parenright asterisk plus comma minus period slash "
    "colon semicolon less equal greater question at "
    "bracketleft backslash bracketright asciicircum underscore grave "
    "braceleft bar braceright asciitilde"
).split()
_KEYSYM_CHARS = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_XKB_SYM_TO_CHAR = dict(zip(_KEYSYM_NAMES, _KEYSYM_CHARS))
_XKB_SYM_TO_CHAR.update({c: c for c in "0123456789"})
_XKB_SYM_TO_CHAR.update({c: c for c in "abcdefghijklmnopqrstuvwxyz"})
_XKB_SYM_TO_CHAR.update({c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})


def _xkb_symbol_char(name: str) -> str | None:
    """Resolve one xkb keysym name to its character, or None if not printable.

    Dead keys (``dead_acute``), Greek/Cyrillic letters and anything outside
    printable ASCII deliberately resolve to None: a key whose shifted value we
    cannot represent as a character is one we must leave alone.
    """
    char = _XKB_SYM_TO_CHAR.get(name)
    if char is not None:
        return char
    if len(name) >= 5 and name[0] == "U":
        try:
            code = int(name[1:], 16)
        except ValueError:
            return None
        if 0x20 <= code < 0x7F:
            return chr(code)
    return None


def _derive_shift_punctuation(layout: str, variant: str = "") -> dict[int, str] | None:
    """Build ``{base codepoint: shifted char}`` from *layout*'s own xkb table.

    This is what makes the kitty path layout-correct rather than layout-guessed.
    Kitty reports the UNSHIFTED codepoint, so knowing what a given physical key
    produces at shift level 2 on THIS layout is exactly the missing information —
    and xkb already holds it.  On AZERTY ``AE01`` is ``[ampersand, 1]``, so the
    correct entry is ``ord('&') -> '1'``; on Greek ``AE01`` is ``[1, exclam]``,
    giving ``ord('1') -> '!'``.  Neither is the US table.

    Only the named variant block is read (``basic`` by default), because later
    blocks in the same file describe *other* variants and mixing them would
    invent a layout nobody is using.  Keys whose level-1 or level-2 symbol is
    not printable ASCII — dead keys, Greek letters — are skipped, so they keep
    leaking rather than typing something wrong.  Returns None when no xkb data
    is available at all.
    """
    if not layout or "/" in layout or "." in layout or ".." in layout:
        return None
    block = _xkb_variant_block(layout, variant or "basic")
    if block is None:
        return None
    _compile_xkb_key_re()
    wanted = {v: k for k, v in _XKB_KEY_BY_BASE.items()}  # xkb key name -> base char
    derived: dict[int, str] = {}
    for match in _XKB_KEY_RE.finditer(block):
        if match.group("key") not in wanted:
            continue
        syms = [s.strip() for s in match.group("syms").split(",")]
        if len(syms) < 2:
            continue
        level1, level2 = syms[0], syms[1]
        if level1 == "any" or level2 == "any":
            # Inherits the base (US) layout for this key.
            us_base = wanted[match.group("key")]
            derived[ord(us_base)] = _SHIFT_PUNCTUATION_BY_CHAR[us_base]
            continue
        base_char = _xkb_symbol_char(level1)
        shifted_char = _xkb_symbol_char(level2)
        if base_char is None or shifted_char is None or base_char == shifted_char:
            continue
        # Dvorak and friends put LETTERS on punctuation positions, so the scan
        # picks up pairs like s -> S. Correct, but already covered by the
        # letter table above and out of place in a punctuation map; skip them
        # so this map means only what its name says.
        if base_char.isalpha() and shifted_char.isalpha():
            continue
        derived[ord(base_char)] = shifted_char
    return derived or None


def _xkb_variant_block(layout: str, variant: str) -> str | None:
    """Return the text of ``xkb_symbols "<variant>"`` from *layout*'s file."""
    for directory in _XKB_SYMBOLS_DIRS:
        try:
            with open(
                f"{directory}/{layout}", encoding="utf-8", errors="replace"
            ) as handle:
                text = handle.read()
        except OSError:
            continue
        marker = f'xkb_symbols "{variant}"'
        start = text.find(marker)
        if start < 0:
            return None
        nxt = text.find("xkb_symbols", start + len(marker))
        return text[start:] if nxt < 0 else text[start:nxt]
    return None


def _compile_xkb_key_re() -> None:
    global _XKB_KEY_RE
    if _XKB_KEY_RE is None:
        import re

        _XKB_KEY_RE = re.compile(
            r"key\s*<(?P<key>[A-Z0-9]+)>\s*\{[^}]*?\[(?P<syms>[^\]]*)\]"
        )


def _layout_keeps_us_shift_punctuation(layout: str) -> bool:
    """True when *layout*'s xkb definition leaves Shift+punctuation at US values.

    Answers by reading the layout's own symbol table rather than trusting its
    name.  A key that is absent, or declared ``any, any``, inherits the base
    layout and therefore matches US; a key that names a level-2 keysym must
    name the US one.  Anything unreadable or unresolvable is treated as a
    mismatch, because the caller's contract is "do not guess".

    Deliberately conservative across variants: if ANY block in the file gives a
    target key a level-2 symbol that disagrees with US, the layout is rejected.
    A needless rejection costs a leaked escape sequence; a wrong acceptance
    costs wrongly typed characters.
    """
    global _XKB_KEY_RE
    if not layout or "/" in layout or "." in layout:
        return False
    if _XKB_KEY_RE is None:
        import re

        _XKB_KEY_RE = re.compile(
            r"key\s*<(?P<key>[A-Z0-9]+)>\s*\{\s*\[(?P<syms>[^\]]*)\]"
        )
    wanted = {_XKB_KEY_BY_BASE[b]: v for b, v in _SHIFT_PUNCTUATION_BY_CHAR.items()}
    seen: set[str] = set()
    for directory in _XKB_SYMBOLS_DIRS:
        try:
            with open(
                f"{directory}/{layout}", encoding="utf-8", errors="replace"
            ) as fh:
                text = fh.read()
        except OSError:
            continue
        for match in _XKB_KEY_RE.finditer(text):
            key = match.group("key")
            expected = wanted.get(key)
            if expected is None:
                continue
            syms = [s.strip() for s in match.group("syms").split(",")]
            if len(syms) < 2:
                return False
            level2 = syms[1]
            if level2 == "any":
                seen.add(key)  # inherits the base layout => US
                continue
            if _XKB_KEYSYM_CHAR.get(level2) != expected:
                return False  # redefined away from US
            seen.add(key)
        return bool(seen)  # file found; verdict stands
    return False  # no xkb data available


def _shift_punctuation_base_map() -> dict[int, str] | None:
    """The base-codepoint half of the Shift+punctuation map, or None.

    Three cases, in order:

    * **modifyOtherKeys terminals** report the already-shifted codepoint, so
      this half is never consulted. The US table is harmless there and costs
      nothing, so install it and keep the behaviour identical to before.
    * **A literal "us" layout** — the US table is simply correct.
    * **kitty on anything else** — the terminal reports the UNSHIFTED codepoint,
      so what Shift produces depends on the layout. Derive it from that
      layout's own xkb definition; on AZERTY that yields ``&`` -> ``1``, on
      Greek ``1`` -> ``!``. Keys whose shifted value is a dead key or a
      non-ASCII letter are omitted and keep leaking, which is the correct
      failure. If xkb data is unavailable, return None and guess nothing.
    """
    if not _kitty_reports_unshifted_codepoints():
        return dict(_SHIFT_PUNCTUATION)
    layout, variant = _configured_layout_and_variant()
    if not layout:
        return None
    if layout == "us" or layout.startswith("us."):
        return dict(_SHIFT_PUNCTUATION)
    return _derive_shift_punctuation(layout, variant)


def _shift_punctuation_base_map_is_safe() -> bool:
    """Whether the base-codepoint half of the Shift+punctuation map may install.

    The map has two halves and they are NOT equally safe:

    * ``punct_map[ord(shifted)] = shifted`` is an IDENTITY mapping — whatever
      shifted codepoint the terminal reports is echoed back.  Correct on every
      keyboard layout, so it installs unconditionally.
    * ``punct_map[base_cp] = shifted`` translates ``2`` into ``@`` from a US
      table.  It is a guess, and on a non-US layout it types the WRONG
      character rather than leaking an escape sequence.

    Which half fires is decided entirely by the terminal, and only kitty
    reaches the guessing half — so a kitty user on a Greek/AZERTY/German
    keyboard is the one who would eat wrong input.  The original Shift+letter
    patch refused symbols for exactly this reason ("they will leak, but that's
    better than wrong input"); leaking is still the better failure, so the
    guess installs only where it cannot be wrong.
    """
    return not _kitty_reports_unshifted_codepoints() or _us_punctuation_layout()




# kitty CSI-u ORs lock-key state into the modifier parameter of every key event while a lock is
# on: CapsLock=64, NumLock=128, both=192. Every fixed-modifier CSI-u (and legacy CSI-tilde /
# CSI-letter) registration therefore needs lock-offset twins, or those events leak into the prompt
# as literal text. The xterm modifyOtherKeys ``ESC[27;N;CP~`` encoding never carries lock bits.
# See #88221, #89651.
_LOCK_BIT_OFFSETS = (0, 64, 128, 192)


def _lock_variants(modifier: int) -> tuple[int, ...]:
    """``modifier`` plus its CapsLock/NumLock/both twins."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS)


def _lock_twins(modifier: int) -> tuple[int, ...]:
    """Only the lock twins of ``modifier`` (never the base value)."""
    return _lock_variants(modifier)[1:]


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

    ``overwrite=True`` replaces differing entries; ``overwrite=False`` behaves like ``setdefault``
    so existing/user registrations win. Clears the VT100 prefix cache when anything changed.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0
    changed = 0
    for seq, key in build(ANSI_SEQUENCES, Keys).items():
        if (ANSI_SEQUENCES.get(seq) != key) if overwrite else (seq not in ANSI_SEQUENCES):
            ANSI_SEQUENCES[seq] = key
            changed += 1
    if changed:
        _clear_vt100_prefix_cache()
    return changed


def install_keypress_data_normalization() -> int:
    """Normalize KeyPress data for extended-key aliases that map to a single plain character
    (Shift+Space → ``' '``, Shift+letter → uppercase, keypad digits/operators).

    Root cause of #88071: ``Vt100Parser._call_handler`` builds ``KeyPress(key, match.group(0))`` — the *key*
    is correctly remapped by ``ANSI_SEQUENCES``, but the *data* field still carries the full raw escape text
    (e.g. ``"\\x1b[32;2u"``). prompt_toolkit's default character-insert binding (``self-insert``,
    ``basic.py``) inserts ``event.data``, so the raw CSI bytes land in the prompt buffer. For a plain space
    both fields are ``' '`` so it is invisible; for any mapped extended sequence the escape text is what
    gets inserted.
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
    except Exception:
        return 0

    # ``_call_handler`` is prompt_toolkit-private. Fetch it defensively: cli.py calls every
    # installer inside one blanket ``except Exception: pass``, so an AttributeError from a
    # future rename would be swallowed there and silently take the installers that run after
    # this one down with it. Degrade to a no-op instead.
    _orig_call_handler = getattr(_vt100_mod.Vt100Parser, "_call_handler", None)
    if _orig_call_handler is None:
        return 0
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
    try:
        _vt100_mod.Vt100Parser._call_handler = _patched_call_handler
    except Exception:
        return 0
    return 1


def _install_enter_alias(modifier: int) -> int:
    """Map <modifier>+Enter (Kitty CSI-u ``ESC[13;<m>u`` plus lock twins, xterm ``ESC[27;<m>;13~``
    / ``;13u``) to (Escape, ControlM) so the Alt+Enter newline handler fires.

    Stock prompt_toolkit maps the tilde form to plain ControlM (i.e. Shift+Enter == Enter, the very
    bug this fixes), so these keys are overwritten unconditionally.
    """
    def build(_seqs, keys):
        alt_enter = (keys.Escape, keys.ControlM)
        seqs = [f"\x1b[13;{m}u" for m in _lock_variants(modifier)] + [f"\x1b[27;{modifier};13~", f"\x1b[27;{modifier};13u"]
        return dict.fromkeys(seqs, alt_enter)

    return _install(build, overwrite=True)


def install_shift_enter_alias() -> int:
    """Map Shift+Enter to (Escape, ControlM). macOS Terminal and stock Windows Terminal send the
    same byte for Enter and Shift+Enter, so nothing can be done for them here.
    """
    return _install_enter_alias(2)


def install_ctrl_enter_alias() -> int:
    """Map Ctrl+Enter to (Escape, ControlM); otherwise Kitty/mintty/xterm over SSH insert raw CSI.

    Stock prompt_toolkit maps only the tilde form ``\\x1b[27;5;13~`` (to plain ``Keys.ControlM``, which this
    deliberately overwrites — same bug-fix rationale as install_shift_enter_alias). Without this alias,
    Kitty/mintty/xterm-with-modifyOtherKeys users over SSH never get a Ctrl+Enter newline — the keystroke
    arrives as a raw CSI sequence that falls through to the default character-insert handler. See #22379.
    """
    return _install_enter_alias(5)


def install_cmd_backspace_alias() -> int:
    """Map Cmd+Backspace -> ControlU and Cmd+ForwardDelete -> ControlK.

    Kitty/modifyOtherKeys report Cmd as the super bit (8), yielding unmapped sequences that insert
    literally. Forward-delete is not a CSI-u codepoint, so it uses the CSI tilde form ``ESC[3;9~``.
    """
    def build(_seqs, keys):
        mods = [mod for base in (9, 10) for mod in _lock_variants(base)]  # super / super+shift
        aliases = {f"\x1b[127;{mod}u": keys.ControlU for mod in mods}
        aliases.update({f"\x1b[3;{mod}~": keys.ControlK for mod in mods})
        aliases["\x1b[27;9;127~"] = keys.ControlU
        return aliases

    return _install(build, overwrite=True)


# Kitty functional keys (Private Use Area codepoints) that have prompt_toolkit equivalents.
# kitty emits these CSI-u encodings even in LEGACY mode, so unmapped they leak as literal text.
_KITTY_FUNCTIONAL_NAMED = {
    57409: ".", 57410: "/", 57411: "*", 57412: "-", 57413: "+", 57414: "ControlM",  # KP ops
    57415: "=", 57416: ",",
    57417: "Left", 57418: "Right", 57419: "Up", 57420: "Down", 57421: "PageUp",  # KP nav
    57422: "PageDown", 57423: "Home", 57424: "End", 57425: "Insert", 57426: "Delete",
}
# No prompt_toolkit equivalent: locks/PrintScreen/Pause/Menu, F25-F35, KP_BEGIN, media keys and
# bare modifier events — consumed as Ignore instead of leaking literal text.
_KITTY_FUNCTIONAL_IGNORED = (*range(57358, 57364), *range(57388, 57399), 57427, *range(57428, 57455))


def _kitty_functional_map(Keys) -> dict[int, object]:
    fm: dict[int, object] = {57399 + d: str(d) for d in range(10)}  # KP_0..KP_9
    fm.update({cp: getattr(Keys, v) if v[0].isupper() else v for cp, v in _KITTY_FUNCTIONAL_NAMED.items()})
    fm.update({57376 + (n - 13): getattr(Keys, f"F{n}") for n in range(13, 25)})  # F13..F24
    for code in _KITTY_FUNCTIONAL_IGNORED:
        fm.setdefault(code, Keys.Ignore)
    return fm


def install_modify_other_keys_aliases() -> int:
    """Map modifyOtherKeys-2 / Kitty CSI-u Ctrl/Alt+key sequences to their raw-byte ``Keys``.

    Once ``modifyOtherKeys=2`` is pushed (to distinguish Shift+Enter) the terminal re-encodes
    EVERY Ctrl combo as ``ESC[27;5;<cp>~``; stock prompt_toolkit maps only Ctrl+Enter, so
    Ctrl+A/C/D/... leak as text. Installs Ctrl/Alt/Shift letters, digits, symbols, multi-modifier
    combos, lock-bit variants, CSI-u Esc, modified Enter/Tab/Backspace/Space and Kitty functional
    keys. ``setdefault`` semantics: existing mappings (incl. the Shift/Ctrl+Enter aliases) win.

    (#56684, #86866, #87390).
    * **Ctrl+letter** (a–z): ``ESC[27;5;<codepoint>~`` and ``ESC[<codepoint>;5u`` → ``Keys.ControlA`` .. *
    **Ctrl+digit** (0–9): same formats → ``Keys.Control0`` .. * **Ctrl+symbol** (``[`` ``\\`` ``]`` ``^``
    ``_`` `` `` ``@``): same formats → the same ``Keys`` value the raw control byte maps to. *
    **Alt+letter** (a–z, A–Z): ``ESC[27;3;<codepoint>~`` and ``ESC[<codepoint>;3u`` → ``(Keys.Escape,
    <letter>)`` — matching how prompt_toolkit handles a bare ``ESC`` followed by a character. *
    **Shift+letter** (a–z): → the uppercase character. * **Multi-modifier letters** (Shift+Alt=4,
    Ctrl+Shift=6, Ctrl+Alt=7, Ctrl+Alt+Shift=8): normalized onto the same targets — Ctrl-bearing combos
    behave as the Ctrl key (Alt adds an ``Escape`` prefix), matching how dte/kakoune normalize these
    protocols. * **Lock-bit variants**: every CSI-u mapping above is also installed with the CapsLock (64)
    and NumLock (128) bits ORed into the modifier parameter — kitty/ghostty include them while a lock is on,
    and without the variants every key combo dies with the lock enabled (``ESC[99;133u`` instead of
    ``ESC[99;5u``, #89651). * **Esc key**: ``ESC[27u`` / ``ESC[27;<mod>u`` (Kitty disambiguate mode reports
    Esc this way, #56684) → ``Keys.Escape``. * **Modified Enter/Tab/Backspace/Space**: Alt+Enter → the
    Alt+Enter newline tuple; Shift+Tab → ``BackTab``; Ctrl+Tab → plain Tab; Ctrl/Alt+Backspace → ``(Escape,
    ControlH)`` (backward-kill-word, matching the Ink TUI and Desktop, #78285); Shift+Backspace → plain
    backspace; Shift+Space → a plain space (#86866); Alt+Space → ``(Escape, " ")``. * **Kitty functional
    keys** (Private Use Area codepoints): keypad keys → their non-keypad equivalents (KP_ENTER → Enter, KP_4
    → '4', KP_LEFT → Left, …); F13–F24 → ``Keys.F13``..``F24``; lock/media/ modifier-event keys →
    ``Keys.Ignore`` so they are consumed instead of leaking as literal text. kitty emits these CSI-u forms
    even in legacy mode for keys that have no legacy encoding.
    """
    return _install(_modify_other_keys_aliases, overwrite=False)


def _modify_other_keys_aliases(ANSI_SEQUENCES: dict, Keys) -> dict[str, object]:
    # Collected first-writer-wins (matching setdefault order), installed once at the end.
    aliases: dict[str, object] = {}
    _put = aliases.setdefault

    # Kitty CSI-u encodes CapsLock/NumLock state as extra modifier bits (caps=64, num=128) ORed into the
    # parameter: with NumLock on, Ctrl+C arrives as ESC[99;133u (5 + 128) instead of ESC[99;5u. Terminals
    # that report these bits (kitty, ghostty) break every key combo while a lock is on (#89651) unless the
    # lock variants are mapped too. The xterm modifyOtherKeys encoding never carries the lock bits, so only
    # the CSI-u form needs them.
    def _install_paired(modifier: int, mapping: dict) -> None:
        """Both modifyOtherKeys (ESC[27;N;CP~, never for mod 1) and CSI-u (ESC[CP;Nu + lock twins)."""
        for codepoint, key_val in mapping.items():
            if modifier != 1:
                _put(f"\x1b[27;{modifier};{codepoint}~", key_val)
            for mod in _lock_variants(modifier):
                _put(f"\x1b[{codepoint};{mod}u", key_val)

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

    # The Esc KEY under Kitty disambiguate mode: ESC[27u (+ modifiers 1-16 incl. super 9+, and
    # lock twins of the modifier-less form, which is how a lone Esc arrives with a lock on).
    _put("\x1b[27u", Keys.Escape)
    for mod in (mod for m in range(1, 17) for mod in _lock_variants(m)):
        _put(f"\x1b[27;{mod}u", Keys.Escape)

    # Modified Enter/Tab/Backspace/Space (Shift/Ctrl+Enter are owned by the enter aliases, which run
    # first and win). Modifier 1 = unmodified keys kitty CSI-u-encodes on their own when a lock bit
    # is set (plain Backspace arrives as ESC[127;129u rather than \x7f).
    alt_backspace = (Keys.Escape, Keys.ControlH)  # backward-kill-word, matching Ink TUI + Desktop
    _install_paired(2, {9: Keys.BackTab, 127: Keys.ControlH, 32: " "})
    _install_paired(3, {13: (Keys.Escape, Keys.ControlM), 127: alt_backspace, 32: (Keys.Escape, " ")})
    _install_paired(5, {9: Keys.ControlI, 127: alt_backspace})  # Ctrl+Tab degrades to Tab
    _install_paired(1, {9: Keys.ControlI, 13: Keys.ControlM, 32: " ", 127: Keys.ControlH})

    # -- Shift+punctuation → the shifted character ----
    # Shifted punctuation (tilde, @, ^, _, {}, |, …) is essential for coding prompts.
    # Terminals re-encode Shift+1 as ESC[27;2;49~ (base codepoint) and Shift+{ as
    # ESC[27;2;123~ (already-shifted codepoint), the same way they send Shift+a as codepoint
    # 97 or 65 — so map both forms.
    punct_map: dict[int, str] = {}
    # Identity half — echoes back whatever shifted codepoint the terminal reported. Correct on
    # every layout by construction, so it always installs.
    for _base_cp, _shifted in _SHIFT_PUNCTUATION.items():
        punct_map[ord(_shifted)] = _shifted
    # Base half — the UNSHIFTED codepoint, which only the kitty protocol reports. What that key
    # produces under Shift is a property of the user's LAYOUT, so derive it from xkb rather than
    # assuming US: a US table would type the WRONG CHARACTER on a Greek/German/AZERTY keyboard
    # instead of merely leaking, and leaking is the better failure. Falls back to the US table
    # only where it cannot be wrong — modifyOtherKeys terminals never reach this half, and a
    # layout that is provably US agrees with it. See _shift_punctuation_base_map_is_safe.
    _punct_base = _shift_punctuation_base_map()
    if _punct_base:
        punct_map.update(_punct_base)
    _install_paired(2, punct_map)

    # Lock twins for the legacy CSI-letter / CSI-tilde forms kitty keeps using under the
    # disambiguate push (Down with NumLock on = ESC[1;129B; Alt+Left = ESC[1;131D). Derived from
    # whatever the table already maps for the base modifier, stock entries included.
    for m in range(1, 17):
        legacy = [(f"\x1b[1;{m}{t}" if m > 1 else f"\x1b[{t}", f"\x1b[1;{{mod}}{t}", f"\x1bO{t}") for t in "ABCDFHPQRS"]
        legacy += [(f"\x1b[{n};{m}~" if m > 1 else f"\x1b[{n}~", f"\x1b[{n};{{mod}}~", None) for n in range(1, 9)]
        for base_seq, twin_fmt, ss3_seq in legacy:  # CSI-letter nav/F1-F4, then CSI-tilde nav keys
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None and m == 1 and ss3_seq:
                key = ANSI_SEQUENCES.get(ss3_seq)  # plain F1-F4 live as SS3 forms
            for mod in _lock_twins(m) if key is not None else ():
                _put(twin_fmt.format(mod=mod), key)

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
