"""Augmentations to prompt_toolkit's input-parsing tables.

Imported once at CLI startup. Each helper installs a small mapping into
prompt_toolkit's `ANSI_SEQUENCES` so byte sequences emitted by modern
keyboard protocols (Kitty / xterm `modifyOtherKeys`) decode to existing
key tuples Hermes already binds.

Kept in a standalone module — separate from `cli.py` — so the registrations
can be unit-tested without importing the whole CLI runtime.
"""

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


def install_keypress_data_normalization() -> int:
    """Normalize KeyPress data for extended-key aliases that map to a
    single plain character (Shift+Space → ``' '``, Shift+letter → the
    uppercase letter, keypad digits → ``'0'``..``'9'``, keypad operators).

    Root cause of #88071: ``Vt100Parser._call_handler`` builds
    ``KeyPress(key, match.group(0))`` — the *key* is correctly remapped by
    ``ANSI_SEQUENCES``, but the *data* field still carries the full raw
    escape text (e.g. ``"\\x1b[32;2u"``). prompt_toolkit's default
    character-insert binding (``self-insert``, ``basic.py``) inserts
    ``event.data``, so the raw CSI bytes land in the prompt buffer. For a
    plain space both fields are ``' '`` so it is invisible; for any mapped
    extended sequence the escape text is what gets inserted.

    This patches ``Vt100Parser._call_handler`` so that when a sequence maps
    to a single plain character, the KeyPress data is that character rather
    than the raw sequence — the bytes never reach the buffer. Idempotent;
    repeated calls are no-ops.

    Returns 1 when the patch was applied, 0 when already applied or the
    import failed.
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
    except Exception:
        return 0

    if getattr(
        _vt100_mod.Vt100Parser._call_handler, "_hermes_char_data_normalized", False
    ):
        return 0

    _orig_call_handler = _vt100_mod.Vt100Parser._call_handler

    def _patched_call_handler(self, key, insert_text):
        # A single plain character (not a Keys member, not a tuple) mapped
        # from an extended sequence must carry the mapped character as its
        # data — self-insert inserts event.data and the raw CSI would leak.
        if (
            isinstance(key, str)
            and len(key) == 1
            and not isinstance(key, _PtKeys)
            and isinstance(insert_text, str)
            and insert_text.startswith("\x1b")
        ):
            insert_text = key
        return _orig_call_handler(self, key, insert_text)

    _patched_call_handler._hermes_char_data_normalized = True
    _vt100_mod.Vt100Parser._call_handler = _patched_call_handler
    return 1


def install_shift_enter_alias() -> int:
    """Map Shift+Enter byte sequences to the (Escape, ControlM) key tuple
    that Alt+Enter produces, so the existing Alt+Enter newline handler
    fires for terminals that emit a distinct Shift+Enter.

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

    alt_enter = (Keys.Escape, Keys.ControlM)
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


def _install_literal_key_data_patch() -> bool:
    """Make character-valued ``ANSI_SEQUENCES`` entries insert themselves.

    prompt_toolkit's VT100 parser builds every key press as
    ``KeyPress(key=<table value>, data=<matched bytes>)``, and the default
    ``Keys.Any`` binding inserts ``event.data`` — the *bytes*, not the key.
    That is invisible for entries resolving to a ``Keys`` member (bindings
    match on ``key``, and ``data`` is unused), which is every entry stock
    prompt_toolkit ships.

    It is not invisible for the character-valued entries registered above.
    Mapping ``ESC[27;2;72~`` → ``"H"`` makes the parser emit
    ``KeyPress(key="H", data="\\x1b[27;2;72~")``, so the prompt still
    receives the raw escape sequence as literal text — the mapping alone
    fixes what the key *is* but not what it *types*.

    This narrows ``data`` to the character for that CSI→char case. Stock
    Keys-valued table entries and the plain-typing path (where ``key`` is
    already ``data``) keep prior behavior. Multi-key ANSI matches that
    intentionally blank trailing ``insert_text`` are also left alone.

    Idempotent; returns True when this call installed the patch.
    """
    try:
        from prompt_toolkit.input.vt100_parser import Vt100Parser
        from prompt_toolkit.keys import Keys
    except Exception:
        return False

    # `_call_handler` is prompt_toolkit-private. Fetch it defensively so a
    # future rename degrades to the same no-op as a missing module, rather
    # than raising through install_modify_other_keys_aliases() into cli.py's
    # blanket `except Exception: pass` — which would silently skip the
    # installers that run after it.
    original_call_handler = getattr(Vt100Parser, "_call_handler", None)
    if original_call_handler is None:
        return False

    # The idempotency marker rides on the wrapper rather than the class, so
    # it cannot outlive the wrapper it describes: if anything later replaces
    # `_call_handler`, the marker goes with it and we wrap the replacement
    # instead of skipping on a stale flag.
    if getattr(original_call_handler, "_hermes_literal_key_data", False):
        return False

    def _call_handler(self, key, insert_text):  # type: ignore[no-untyped-def]
        # Only rewrite when the parser handed us a real payload that is the
        # raw escape bytes. Stock _call_handler blanks insert_text for every
        # key after the first in a multi-key ANSI match (Alt+letter →
        # (Escape, "a") with "" on the letter). Reviving that blank would
        # insert the letter and break meta chords (Shift+Alt+a → "A").
        #
        # Keys is a str Enum, so Keys members are isinstance(..., str). The
        # `not isinstance(key, Keys)` check is load-bearing — do not drop it
        # even though no current Keys value has len == 1.
        if (
            isinstance(key, str)
            and not isinstance(key, Keys)
            and len(key) == 1
            and insert_text
            and insert_text != key
        ):
            insert_text = key
        return original_call_handler(self, key, insert_text)

    _call_handler._hermes_literal_key_data = True  # type: ignore[attr-defined]

    try:
        Vt100Parser._call_handler = _call_handler  # type: ignore[assignment]
    except Exception:
        return False
    return True


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
    for ch in range(ord("a"), ord("z") + 1):
        raw = chr(ch & 0x1F)  # 0x01..0x1a
        existing = ANSI_SEQUENCES.get(raw)
        if existing is not None:
            ctrl_key_map[ch] = existing

    # 0-9: Ctrl+digit codepoints don't have a useful raw-byte mapping
    # (e.g. chr(ord('0') & 0x1F) = 0x10 = ControlP, not Control0), so map
    # them directly to Keys.Control0..Keys.Control9.
    for d in range(10):
        ctrl_key_map[ord("0") + d] = getattr(Keys, f"Control{d}")

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

    # Ctrl+letter / Ctrl+digit / Ctrl+symbol (modifier 5)
    _install_paired(5, ctrl_key_map)

    # -- Alt+letter → (Escape, <letter>) ----
    # Under modifyOtherKeys, Alt+a = ESC[27;3;97~. Without mapping, this
    # leaks as literal text. prompt_toolkit handles bare Alt+letter as
    # (Escape, <letter>), so we map the extended sequences to the same tuple.
    alt_map: dict[int, tuple] = {}
    for ch in range(ord("a"), ord("z") + 1):
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
    for ch in range(ord("a"), ord("z") + 1):
        upper_char = chr(ch - 32)  # 'A'..'Z'
        shift_map[ch] = upper_char
        shift_map[ch - 32] = upper_char
    _install_paired(2, shift_map)

    # -- Shift+punctuation → the shifted character ----
    # Shifted punctuation (tilde, @, ^, _, {}, |, etc.) is essential for
    # coding prompts. Terminals re-encode Shift+1 as ESC[27;2;49~ (base
    # codepoint) and Shift+{ as ESC[27;2;123~ (already-shifted codepoint)
    # the same way they send Shift+a as codepoint 97 or 65 — map both
    # forms. Each base key maps to its US-layout shifted character.
    punct_shift = _SHIFT_PUNCTUATION
    punct_map: dict[int, str] = {}
    # Identity half — echoes back whatever shifted codepoint the terminal
    # reported. Layout-independent by construction, so it always installs.
    for _base_cp, shifted in punct_shift.items():
        punct_map[ord(shifted)] = shifted
    # Base half — the UNSHIFTED codepoint, which only the kitty protocol
    # reports. What that key produces under Shift is a property of the user's
    # layout, so DERIVE it from xkb rather than assuming US; a US table would
    # type the wrong character on AZERTY instead of leaking. Falls back to the
    # US table only where it cannot be wrong: modifyOtherKeys terminals (which
    # never reach this half) and a layout that is literally "us".
    base_map = _shift_punctuation_base_map()
    if base_map:
        punct_map.update(base_map)
    _install_paired(2, punct_map)

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
    for ch in range(ord("a"), ord("z") + 1):
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
        f"\x1b[27;{mod}u" for m in range(1, 17) for mod in _lock_variants(m)
    ]:
        if seq not in ANSI_SEQUENCES:
            ANSI_SEQUENCES[seq] = Keys.Escape
            changed += 1

    # -- Modified Enter / Tab / Backspace / Space ----
    # Shift+Enter / Ctrl+Enter are installed by install_shift_enter_alias /
    # install_ctrl_enter_alias (which run first and win via setdefault).
    _install_paired(
        2,
        {
            9: Keys.BackTab,  # Shift+Tab — same as the legacy ESC[Z
            127: Keys.ControlH,  # Shift+Backspace — plain backspace
            32: " ",  # Shift+Space — still a space (#86866)
        },
    )
    _install_paired(
        3,
        {
            13: (Keys.Escape, Keys.ControlM),  # Alt+Enter — newline tuple
            127: (Keys.Escape, Keys.ControlH),  # Alt+Backspace — backward-kill-word
            32: (Keys.Escape, " "),  # Alt+Space
        },
    )
    _install_paired(
        5,
        {
            9: Keys.ControlI,  # Ctrl+Tab — degrade to Tab
            127: (Keys.Escape, Keys.ControlH),  # Ctrl+Backspace — backward-kill-word,
            # matching Ink TUI + Desktop (#78285)
        },
    )

    # -- Unmodified keys with a lock bit set (kitty modifier 1 = "none") --
    # With a lock on, kitty stamps the lock bit onto keys pressed with NO
    # real modifier too, so plain Backspace arrives as ESC[127;129u
    # (1 + 128) rather than \x7f. _install_paired(1, ...) registers the
    # bare mod-1 spelling and its lock twins. Only keys kitty CSI-u-encodes
    # on their own are listed; plain text characters are still delivered
    # as UTF-8, lock bits or not.
    _install_paired(
        1,
        {
            9: Keys.ControlI,  # Tab
            13: Keys.ControlM,  # Enter
            32: " ",  # Space
            127: Keys.ControlH,  # Backspace
        },
    )

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
    for d in range(10):  # KP_0..KP_9 → digits
        functional_map[57399 + d] = str(d)
    functional_map.update({  # KP operators / punctuation
        57409: ".",
        57410: "/",
        57411: "*",
        57412: "-",
        57413: "+",
        57414: Keys.ControlM,
        57415: "=",
        57416: ",",
    })
    functional_map.update({  # KP navigation → non-keypad keys
        57417: Keys.Left,
        57418: Keys.Right,
        57419: Keys.Up,
        57420: Keys.Down,
        57421: Keys.PageUp,
        57422: Keys.PageDown,
        57423: Keys.Home,
        57424: Keys.End,
        57425: Keys.Insert,
        57426: Keys.Delete,
    })
    for n in range(13, 25):  # F13..F24
        functional_map[57376 + (n - 13)] = getattr(Keys, f"F{n}")
    # No prompt_toolkit equivalent (lock keys, PrintScreen, Menu, F25-F35,
    # KP_BEGIN, media keys, bare modifier events): consume as Ignore
    # instead of leaking literal text.
    for code in (
        list(range(57358, 57364))  # locks, PrintScreen, Pause, Menu
        + list(range(57388, 57399))  # F25..F35
        + [57427]  # KP_BEGIN
        + list(range(57428, 57455))  # media keys + modifier key events
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

    # New longer sequences can flip "is this a prefix of a longer match?"
    # answers the VT100 parser already cached — drop the cache so parsers
    # created before this install (or in earlier tests) can't misparse.
    if changed:
        _clear_vt100_prefix_cache()

    # Character-valued entries above (Shift+letter, Shift+Space, keypad
    # digits) need the parser to type the character rather than the escape
    # sequence that produced it. Without this, Ghostty/iTerm/WezTerm/kitty
    # Shift+letter still leaks `[27;2;<code>~` into the prompt (#87390).
    _install_literal_key_data_patch()

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
