"""Decode the extended-key grammars instead of enumerating their cross-product.

``pt_input_extras`` registers every extended-key sequence as a literal string in
``ANSI_SEQUENCES``, which needs ~2,500 entries because CSI-u is a grammar and a dict of
literal byte-strings can only represent one by materialising every codepoint times every
modifier times every lock-bit variant times both spellings. Any cell nobody generated is
a key that leaks its raw escape sequence into the prompt.

    CSI code[:shifted[:base]] ; modifiers[:event-type] ; text-codepoints u

Parsing it costs a regex and some bit arithmetic. Lock bits stop being a concept
(``modifier = param - 1``, then mask), modifiers cannot have gaps, and event types and
alternate keys become fields rather than families of entries. prompt_toolkit already
does dynamic matching here for the same reason --- its ``_get_match`` special-cases CPR
and mouse sequences with regexes, "because it contains integer variables" --- so this
installs into that seam rather than fighting it.

The alias table still wins every lookup: its remaining entries encode application
decisions the protocol has no opinion about, such as Shift+Enter inserting a newline.
This answers only what nobody mapped.

``push()`` is what ``cli.py`` now sends, so the flag set has one definition rather
than a literal in each file. ``negotiated()`` is not wired: reading the terminal's
answer to ``CSI ? u`` needs a round-trip on the tty that startup does not do, and
flag 4 is additive --- a terminal that lacks it omits the field. Asking for it is
only safe with this module installed, since the sub-parameter spelling it produces
has no entry in ``ANSI_SEQUENCES``; the >1u push was reverted once already, in
87074, when Ctrl+C arrived as an escape nothing could read (#56684).
"""

from __future__ import annotations

import re
from typing import NamedTuple

# CSI code[:shifted[:base]] ; modifiers[:event] ; text u
_CSI_U_RE = re.compile(
    r"^\x1b\["
    r"(\d+)(?::(\d*))?(?::(\d*))?"      # key code, shifted key, base-layout key
    r"(?:;(\d*)(?::(\d+))?)?"           # modifiers, event type
    r"(?:;([\d:]*))?"                   # associated text codepoints
    r"u$"
)
# Deliberately loose: a false "yes" delays a flush by a byte, a false "no" breaks the
# key --- the same trade prompt_toolkit's own CPR/mouse prefix regexes make.
_CSI_U_PREFIX_RE = re.compile(r"^\x1b(\[[\d;:]*)?$")

# Modifier bits, per the kitty spec: the parameter is 1 + the sum of these.
SHIFT, ALT, CTRL, SUPER, HYPER, META, CAPS_LOCK, NUM_LOCK = 1, 2, 4, 8, 16, 32, 64, 128
# Locks describe the keyboard's state, not the chord the user pressed. The table
# approach must enumerate them; here they are simply not in the mask.
_REAL_MODIFIERS = SHIFT | ALT | CTRL | SUPER | HYPER | META

PRESS, REPEAT, RELEASE = 1, 2, 3

# kitty reports these in the Unicode Private Use Area. One entry per key --- the
# modifier dimension is applied afterwards rather than multiplied in.
_FUNCTIONAL = {
    # The keys themselves, which reach CSI-u under flag 8 or through a multiplexer
    # that left it set. Unmapped they fall to chr(), so Up types a PUA glyph.
    57344: "Escape", 57345: "Enter", 57346: "Tab", 57347: "Backspace",
    57348: "Insert", 57349: "Delete",
    57350: "Left", 57351: "Right", 57352: "Up", 57353: "Down",
    57354: "PageUp", 57355: "PageDown", 57356: "Home", 57357: "End",
    # The keypad, which stands in for the same keys.
    57414: "Enter", 57415: "=", 57416: ",",
    57417: "Left", 57418: "Right", 57419: "Up", 57420: "Down",
    57421: "PageUp", 57422: "PageDown", 57423: "Home", 57424: "End",
    57425: "Insert", 57426: "Delete",
    57409: ".", 57410: "/", 57411: "*", 57412: "-", 57413: "+",
}
_FUNCTIONAL.update({57364 + (n - 1): f"F{n}" for n in range(1, 13)})    # F1..F12
_FUNCTIONAL.update({57399 + d: str(d) for d in range(10)})          # KP_0..KP_9
_FUNCTIONAL.update({57376 + (n - 13): f"F{n}" for n in range(13, 25)})  # F13..F24
# Locks, PrintScreen/Pause/Menu, F25-F35, media and bare modifier events: no
# prompt_toolkit equivalent, so consume them rather than let them leak as text.
_IGNORED = frozenset((*range(57358, 57364), *range(57388, 57399), 57427, *range(57428, 57455)))

# Legacy codepoints that name a key rather than a character.
_NAMED = {8: "Backspace", 9: "Tab", 13: "Enter", 27: "Escape", 32: " ",
          127: "Backspace"}   # 8 or 127 depending on the terminal's backspace mode

# prompt_toolkit spells the unmodified forms of these differently from
# "Shift"/"Control" + name, so they need naming once.
_PLAIN_KEY = {
    "Enter": "ControlM", "Tab": "ControlI", "Backspace": "ControlH",
    "Escape": "Escape",
    # KP_Begin (keypad 5, NumLock off): prompt_toolkit consumes ESC[E and says
    # nothing about the modified forms, so neither do we.
    "Begin": "Ignore",
}


class ParsedKey(NamedTuple):
    """One decoded extended-key event."""

    codepoint: int
    shifted: int | None       # flag 4: what this key produces with Shift, per the terminal
    base: int | None          # flag 4: the same physical key on the base layout
    mods: int                 # raw modifier bits (locks included)
    event: int                # 1 press, 2 repeat, 3 release
    text: str                 # flag 16: the text this event produces, if reported
    already_shifted: bool = False  # xterm modifyOtherKeys: codepoint IS the shifted key
    named: str | None = None       # legacy nav forms name a key rather than a codepoint

    @property
    def real_mods(self) -> int:
        """Modifier bits with lock state masked out."""
        return self.mods & _REAL_MODIFIERS

    @property
    def is_release(self) -> bool:
        return self.event == RELEASE


def _is_unshifted_text(pk: ParsedKey) -> bool:
    """True for a printable key held with Shift alone and no alternate key reported."""
    return (pk.named is None and pk.shifted is None and not pk.already_shifted
            and pk.real_mods == SHIFT and 32 < pk.codepoint < 0x110000)


def _scalar(cp: int) -> bool:
    """True if chr(cp) is a character rather than a crash or a lone surrogate."""
    return cp <= 0x10FFFF and not 0xD800 <= cp <= 0xDFFF


def _mod_bits(param: str | None, *, xterm: bool = False) -> int | None:
    """Modifier bits from a parameter, or None if it is malformed.

    The parameter is 1 + the bits, so 0 is not a valid encoding of "none" --- and
    left unchecked it yields -1, whose mask invents every modifier at once. xterm
    assigns bit 8 to Meta where the kitty protocol assigns Super; prompt_toolkit
    renders both Meta and Alt as an Escape prefix, so fold it there.
    """
    if not param:
        return 0
    n = int(param)
    if n < 1:
        return None
    bits = n - 1
    if bits > NUM_LOCK | (NUM_LOCK - 1):   # nothing above bit 7 is a modifier
        return None
    if xterm and bits & SUPER:
        bits = (bits & ~SUPER) | ALT
    return bits


def _event(param: str | None) -> int | None:
    """Event type, or None if it is not one of the three the protocol defines."""
    if not param:
        return PRESS
    n = int(param)
    return n if n in (PRESS, REPEAT, RELEASE) else None


def parse(seq: str) -> ParsedKey | None:
    """Decode a complete CSI-u sequence, or return None if it is not one."""
    m = _CSI_U_RE.match(seq)
    if m is None:
        return None
    code, shifted, base, mods, event, text = m.groups()
    bits, ev = _mod_bits(mods), _event(event)
    if bits is None or ev is None:
        return None
    cps = [int(p) for p in (text or "").split(":") if p]
    keys = [int(code)] + [int(x) for x in (shifted, base) if x]
    if not all(_scalar(c) for c in cps + keys):
        return None
    return ParsedKey(
        codepoint=int(code),
        shifted=int(shifted) if shifted else None,
        base=int(base) if base else None,
        mods=bits,
        event=ev,
        text="".join(chr(c) for c in cps),
    )


def is_prefix(seq: str) -> bool:
    """True if ``seq`` could still grow into an extended-key sequence."""
    return bool(_CSI_U_PREFIX_RE.match(seq) or _LEGACY_PREFIX_RE.match(seq))


# ---- The legacy grammars: same modifier arithmetic, different terminators ----

# xterm modifyOtherKeys level 2: ESC[27;<modifier>;<codepoint>~
_MOK_RE = re.compile(r"^\x1b\[27;(\d+);(\d+)~$")
# CSI-letter nav / F1-F4: ESC[<n>;<modifier>[:<event>]<final>, and bare ESC[<final>
_CSI_LETTER_RE = re.compile(r"^\x1b\[(\d*)(?:;(\d+)(?::(\d+))?)?([A-FHPQRSZ])$")
# CSI-tilde nav: ESC[<n>;<modifier>[:<event>]~
_CSI_TILDE_RE = re.compile(r"^\x1b\[(\d+)(?:;(\d+)(?::(\d+))?)?~$")
# SS3, the application-cursor-mode spelling: ESC O [<modifier>] <final>.
_SS3_RE = re.compile(r"^\x1bO(\d+)?([A-FHPQRSXjklmnopqrstuvwxy])$")
_LEGACY_PREFIX_RE = re.compile(r"^\x1b(\[[\d;:]*|O\d*)?$")

# SS3 keypad finals, which carry characters rather than key names.
_SS3_KEYPAD = {
    "X": "=", "j": "*", "k": "+", "l": ",", "m": "-", "n": ".", "o": "/",
    "p": "0", "q": "1", "r": "2", "s": "3", "t": "4",
    "u": "5", "v": "6", "w": "7", "x": "8", "y": "9",
}

# Final byte -> key name, for the CSI-letter family.
_LETTER_KEY = {
    "A": "Up", "B": "Down", "C": "Right", "D": "Left",
    "E": "Begin", "F": "End", "H": "Home",
    "P": "F1", "Q": "F2", "R": "F3", "S": "F4",
}
# Leading number -> key name, for the CSI-tilde family. F1-F12 are here so that a
# release (ESC[15;1:3~) can be recognised at all, and therefore declined.
_TILDE_KEY = {
    1: "Home", 2: "Insert", 3: "Delete", 4: "End",
    5: "PageUp", 6: "PageDown", 7: "Home", 8: "End",
    11: "F1", 12: "F2", 13: "F3", 14: "F4", 15: "F5", 17: "F6",
    18: "F7", 19: "F8", 20: "F9", 21: "F10", 23: "F11", 24: "F12",
}


def parse_legacy(seq: str) -> ParsedKey | None:
    """Decode modifyOtherKeys and the legacy CSI nav forms."""
    m = _MOK_RE.match(seq)
    if m is not None:
        bits, cp = _mod_bits(m.group(1), xterm=True), int(m.group(2))
        if bits is None or not _scalar(cp):
            return None
        # xterm reports the ALREADY-SHIFTED codepoint (Shift+2 arrives as '@'), which
        # is what makes this form layout-safe where the CSI-u one has to decline.
        return ParsedKey(cp, None, None, bits, PRESS, "", already_shifted=True)

    m = _CSI_LETTER_RE.match(seq)
    if m is not None:
        num, mods, event, final = m.groups()
        bits, ev = _mod_bits(mods), _event(event)
        if bits is None or ev is None:
            return None
        if num not in ("", "1"):              # ESC[<other>;<m>A is not this grammar
            return None
        if final == "Z":                      # BackTab, which is Shift+Tab by definition
            return ParsedKey(9, None, None, bits | SHIFT, ev, "", named="Tab")
        # A release is ESC[1;6:3A, which no alias table carries: unparsed, it leaks.
        return ParsedKey(0, None, None, bits, ev, "", named=_LETTER_KEY[final])

    m = _CSI_TILDE_RE.match(seq)
    if m is not None:
        num, mods, event = int(m.group(1)), m.group(2), m.group(3)
        bits, ev = _mod_bits(mods), _event(event)
        name = _TILDE_KEY.get(num)
        if name is None or bits is None or ev is None:
            return None
        return ParsedKey(0, None, None, bits, ev, "", named=name)

    m = _SS3_RE.match(seq)
    if m is not None:
        mods, final = m.group(1), m.group(2)
        bits = _mod_bits(mods)
        if bits is None:
            return None
        ch = _SS3_KEYPAD.get(final)
        if ch is not None:                     # keypad: a character, not a key name
            return ParsedKey(ord(ch), None, None, bits, PRESS, "")
        name = _LETTER_KEY.get(final)
        if name is None:
            return None
        return ParsedKey(0, None, None, bits, PRESS, "", named=name)

    return None


def resolve(pk: ParsedKey, Keys):
    """Turn a decoded event into a prompt_toolkit key value.

    Returns a ``Keys`` member, a plain character, a tuple (for Alt chords), or
    None to mean "no opinion --- let the alias table try".
    """
    # Repeat acts like press --- dropping repeats would break held-arrow scrolling ---
    # and a release types nothing, which is what makes a stale flag 2 harmless.
    if pk.is_release:
        return Keys.Ignore
    if pk.codepoint in _IGNORED:
        return Keys.Ignore

    mods = pk.real_mods
    ctrl, alt, shift = mods & CTRL, mods & ALT, mods & SHIFT

    # prompt_toolkit cannot express Super/Hyper/Meta, and acting on the base key is
    # worse than doing nothing: Super+Delete would delete. The table still binds the
    # ones the application cares about (Cmd+Backspace as kill-line) and runs first.
    if mods & (SUPER | HYPER | META):
        return Keys.Ignore

    # What the terminal says it types, on any layout. Ctrl/Alt are commands, not text.
    # KeyPress takes one character, so a dead key or an IME commit goes back as a
    # tuple, which prompt_toolkit delivers one press at a time.
    if pk.text and not ctrl and not alt:
        return pk.text if len(pk.text) == 1 else tuple(pk.text)

    # Legacy nav forms name their key directly instead of carrying a codepoint.
    if pk.named is not None:
        return _resolve_named(pk.named, ctrl, alt, shift, Keys)

    name = _FUNCTIONAL.get(pk.codepoint) or _NAMED.get(pk.codepoint)
    is_char = name is None and 32 < pk.codepoint < 0x110000
    base_char = chr(pk.codepoint) if is_char else None

    if name is not None and name not in _PLAIN_KEY and not name.isalnum():
        # Keypad operator (. / * - +) --- a character wearing a key's name.
        base_char, name = name, None
    elif name is not None and name.isdigit():
        base_char, name = name, None

    if name is not None:
        return _resolve_named(name, ctrl, alt, shift, Keys)
    if base_char is None:
        return None

    return _resolve_char(pk, base_char, ctrl, alt, shift, Keys)


def _resolve_named(name: str, ctrl: int, alt: int, shift: int, Keys):
    """Nav/function/whitespace keys, where prompt_toolkit uses composed names."""
    if not (ctrl or alt or shift):
        member = _PLAIN_KEY.get(name, name)
        return getattr(Keys, member, None)

    # Shift+Tab is BackTab, which does not follow the Shift<Name> pattern.
    if name == "Tab" and shift and not ctrl and not alt:
        return Keys.BackTab

    # xterm names Shift+F1..F12 as F13..F24 and the table follows it. Consuming the
    # Shift here keeps Ctrl+Shift+F1 composing as ControlF13.
    if shift and name.startswith("F") and name[1:].isdigit() and int(name[1:]) <= 12:
        name = f"F{int(name[1:]) + 12}"
        shift = 0

    prefix = ("Control" if ctrl else "") + ("Shift" if shift else "")
    key = getattr(Keys, f"{prefix}{name}", None) if prefix else None
    if key is None:
        key = getattr(Keys, _PLAIN_KEY.get(name, name), None)
    if key is None:
        return None
    return (Keys.Escape, key) if alt else key


def _resolve_char(pk: ParsedKey, base_char: str, ctrl: int, alt: int, shift: int, Keys):
    """Character-producing keys."""
    if ctrl:
        # Ctrl+C on Cyrillic arrives as 1089 and binds to nothing, but reports base
        # 99. Only when reported: re-deriving clobbers the caller's character, and a
        # keypad PUA codepoint's low bits land on an unrelated control byte.
        if pk.base and chr(pk.base).isascii():
            base_char = chr(pk.base)
        lowered = base_char.lower()
        member = getattr(Keys, f"Control{lowered.upper()}", None) if lowered.isalpha() else None
        if member is None and lowered.isdigit():
            # Ctrl+0..9 have their own Keys members; they do not produce control
            # bytes, so the fallback below cannot find them.
            member = getattr(Keys, f"Control{lowered}", None)
        if member is None:
            ctrl_byte = chr(ord(lowered) & 0x1F)
            member = _CONTROL_BYTE.get(ctrl_byte)
            member = getattr(Keys, member, None) if member else None
        if member is None:
            return None
        return (Keys.Escape, member) if alt else member

    if shift:
        if pk.already_shifted:
            # The codepoint IS the shifted key --- layout-safe without xkb. Letters are
            # normalised up, since terminals disagree, but not when that lengthens them.
            upper = base_char.upper() if base_char.isalpha() else base_char
            char = upper if len(upper) == 1 else base_char
        elif pk.shifted:
            # Flag 4: the terminal reported what this key produces shifted. The
            # layout problem, answered by the terminal that owns the answer.
            char = chr(pk.shifted)
        elif base_char.isalpha():
            # The last inference here, and labelled as one: right for Latin/Greek/Cyrillic,
            # conditional for Turkic, unanswerable elsewhere. Flag 4 replaces it.
            char = _shift_char(base_char)
        else:
            return None  # kitty punctuation without alternate keys: the table's job
        if char is None:
            return None
        return (Keys.Escape, char) if alt else char

    return (Keys.Escape, base_char) if alt else base_char


_CONTROL_BYTE = {
    "\x00": "ControlAt", "\x1c": "ControlBackslash", "\x1d": "ControlSquareClose",
    "\x1e": "ControlCircumflex", "\x1f": "ControlUnderscore",
}


# SpecialCasing.txt gives 'i' a conditional uppercase: 'I', but 'İ' under tr/az.
# str.upper() is locale-independent and returns 'I' --- length 1, so the guard below
# cannot catch it and a Turkish user silently gets the wrong letter.

_TURKIC_UPPER = {"i": "İ", "ı": "I"}   # i -> İ, dotless ı -> I
_TURKIC_LAYOUTS = ("tr", "az")
_turkic_cache: bool | None = None


def _turkic_locale() -> bool:
    """True when the active keyboard layout or locale is Turkish/Azeri."""
    global _turkic_cache
    if _turkic_cache is None:
        import os
        vals = [os.environ.get(v, "") for v in
                ("XKB_DEFAULT_LAYOUT", "LC_ALL", "LC_CTYPE", "LANG")]
        _turkic_cache = any(
            v.split(",")[0].split("_")[0].split(".")[0].lower() in _TURKIC_LAYOUTS
            for v in vals if v
        )
    return _turkic_cache


def _shift_char(base_char: str) -> str | None:
    """Uppercase for a key press, or None when it cannot be answered safely."""
    special = _TURKIC_UPPER.get(base_char)
    if special is not None:
        return special if _turkic_locale() else base_char.upper()
    upper = base_char.upper()
    # One press yields at most one character, and 'ß'.upper() is 'SS' --- a layout whose
    # Shift level holds something else is exactly what must not be guessed.
    return upper if len(upper) == 1 else None


# Negotiation. Hermes pushes flag 1 alone, then infers what flag 4 would report.

DISAMBIGUATE, EVENT_TYPES, ALTERNATE_KEYS, ALL_AS_ESCAPES, ASSOCIATED_TEXT = 1, 2, 4, 8, 16

# Flag 4 alone. Flag 8 turns an unhandled sequence from a visible leak into a key that
# silently types nothing; flag 2 doubles the stream with releases nothing here acts on;
# flag 16 reaches only Ctrl chords, whose text resolve() discards. Both are still decoded
# when a terminal sends them anyway.
WANTED = DISAMBIGUATE | ALTERNATE_KEYS

QUERY = "\x1b[?u"          # "which flags do you support?" -> ESC[?<flags>u
_QUERY_REPLY_RE = re.compile(r"^\x1b\[\?(\d+)u$")


def push(flags: int = WANTED) -> str:
    """Escape sequence enabling ``flags``, pushed onto the terminal's own stack."""
    return f"\x1b[>{flags & 0x1F}u"


def pop() -> str:
    """Restore the terminal's previous keyboard mode.

    Not cosmetic: leaving the protocol enabled is why a plain ``nano`` in the
    same pane afterwards receives raw ``;129u`` fragments (#56759). The stack
    exists precisely so a program can hand the terminal back as it found it.
    """
    return "\x1b[<u"


def parse_query_reply(reply: str) -> int | None:
    """Read a CSI ? flags u reply, or None if that is not one."""
    m = _QUERY_REPLY_RE.match(reply)
    return int(m.group(1)) if m else None


def negotiated(reply: int | None, wanted: int = WANTED) -> int:
    """Flags to actually push, given the terminal's answer to CSI ? u.

    The reply carries the flags CURRENTLY SET, not the ones supported --- the
    protocol has no way to ask that. So the only signal is whether an answer came
    back at all, and masking against it would read a fresh terminal's ``CSI ? 0 u``
    as "supports nothing" and disable the very flag we are negotiating for.

    A terminal that does not answer gets the conservative floor rather than an
    assumption, which is the failure in #100169: an inherited ``WT_SESSION`` was
    treated as identification and the protocol pushed at a core that mis-encodes it.
    """
    if reply is None:
        return DISAMBIGUATE
    return wanted


# ---- Installation ----

def install(*, overwrite: bool = False) -> bool:
    """Teach prompt_toolkit's VT100 parser to decode CSI-u dynamically.

    Patches the same two seams prompt_toolkit already uses for CPR and mouse
    sequences. Idempotent; returns True when this call installed the parser.
    Degrades to a no-op --- never raises --- if the private API moves, because
    cli.py installs every input patch inside one blanket ``except Exception``.
    """
    try:
        import prompt_toolkit.input.vt100_parser as vt
        from prompt_toolkit.keys import Keys
    except Exception:
        return False

    get_match = getattr(vt.Vt100Parser, "_get_match", None)
    cache = getattr(vt, "_IS_PREFIX_OF_LONGER_MATCH_CACHE", None)
    if get_match is None or cache is None:
        return False
    if getattr(get_match, "_hermes_csi_u", False) and not overwrite:
        return False

    def _get_match(self, prefix: str):
        # The table wins: its entries are APPLICATION decisions the protocol has no opinion
        # about, such as Shift+Enter aliased onto Alt+Enter for a newline.
        matched = get_match(self, prefix)
        if matched is not None:
            return matched
        pk = parse(prefix) or parse_legacy(prefix)
        if pk is None:
            return None
        # The table has already missed, so None puts raw bytes in the prompt: a chord
        # nobody can name is consumed instead. A keystroke is not --- Shift+1 has an
        # answer we cannot compute, and it is the table's to give.
        return resolve(pk, Keys) or (None if _is_unshifted_text(pk) else Keys.Ignore)

    _get_match._hermes_csi_u = True

    original_missing = type(cache).__missing__

    def __missing__(self, prefix: str) -> bool:
        if is_prefix(prefix):
            self[prefix] = True
            return True
        return original_missing(self, prefix)

    try:
        vt.Vt100Parser._get_match = _get_match
        type(cache).__missing__ = __missing__
        cache.clear()
    except Exception:
        return False
    return True
