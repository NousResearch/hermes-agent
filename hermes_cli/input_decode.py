"""Decode modifyOtherKeys / Kitty CSI-u escape sequences leaked into bare ``input()`` reads.

The classic CLI pushes extended-key reporting globally for the process —
xterm ``modifyOtherKeys`` level 2 (``CSI >4;2m``) and, on allowlisted
terminals, the Kitty keyboard protocol (``CSI >1u``); see
``cli.py::_enable_extended_enter_keys``.  While that mode is active the
terminal re-encodes every *modified* key:

* modifyOtherKeys=2: ``ESC[27;<mod>;<codepoint>~``
* Kitty CSI-u:       ``ESC[<codepoint>;<mod>u``

``prompt_toolkit`` decodes both via its ``ANSI_SEQUENCES`` table (which
``hermes_cli.pt_input_extras`` fills in), but Python's builtin ``input()``
does **not** — it returns the raw escape text.  Every bare ``input()`` prompt
reached while the mode is active therefore receives garbage whenever the user
types a shifted character (issue #97975).

``decode_modified_key_sequences`` is the pure string transform that fixes
that: it maps each sequence back to the literal character the user pressed
when the modifier chain is Shift-only, and passes everything else through
unchanged.  ``safe_input`` wraps builtin ``input()`` with that transform.
"""

from __future__ import annotations

import builtins
import re
from typing import Callable

__all__ = ["decode_modified_key_sequences", "safe_input"]

# Shift-only modifier values per protocol.  The modifier parameter is a bitmask
# plus one: shift=1, alt=2, ctrl=4, super=8, hyper=16, meta=32, caps=64, num=128.
# Both protocols therefore report a plain Shift combo as 2 (0b01 + 1).  Kitty
# additionally ORs the *event type* into the low bits only in *alternate*
# report mode (press=+0/repeat=+1/release=+2), so there the release/repeat
# forms of Shift+key arrive as 3 and 4.
#
# Under modifyOtherKeys the terminal applies CapsLock/NumLock *before*
# encoding (shift+caps on 'T' sends codepoint 116 = 't'), so the caps/num
# bits are not lock noise — 65/129 are genuinely un-shifted keys and must
# pass through untouched.  Decoding only happens for the exact shift-only
# modifiers 2 (both protocols) and 3/4 (Kitty alternate report mode).
_SHIFT_ONLY_MODS = frozenset({2, 3, 4})
# modifyOtherKeys never encodes event type in the modifier, so there only the
# exact shift value (2) is safe to decode — mod 3 is genuinely Alt.
_EXACT_SHIFT_MODS = frozenset({2})

# modifyOtherKeys level-2: ESC [ 27 ; <mod> ; <codepoint> ~
_MODIFY_OTHER_KEYS_RE = re.compile(r"\x1b\[27;(\d+);(\d+)~")
# Kitty CSI-u: ESC [ <codepoint> ; <mod> u   (``;mod`` omitted when mod==1)
_KITTY_CSI_U_RE = re.compile(r"\x1b\[(\d+)(?:;(\d+))?u")


def _shift_char(
    match: re.Match, mod_index: int, cp_index: int, allowed_mods: frozenset
) -> str:
    """Return the literal character for a shift-only modified-key sequence.

    The codepoint parameter is *already the shifted character*: under
    modifyOtherKeys=2 Shift+T sends ``ESC[27;2;84~`` where 84 == ``ord('T')``
    (the terminal applies the layout before encoding).  The same holds for
    Kitty disambiguate mode.  So decoding a shift-only event is simply
    ``chr(codepoint)``.

    Any other modifier combination (ctrl, alt, super, ...) is returned
    verbatim: ``input()`` has no key-binding layer that could interpret a
    decoded control character meaningfully, and silently dropping a keypress
    is worse than letting the prompt echo it the way stock ``input()`` does.
    Un-parsable codepoints (outside the Unicode range, surrogates) likewise
    pass through.
    """
    mod_raw = match.group(mod_index)
    if mod_raw is None or int(mod_raw) not in allowed_mods:
        return match.group(0)
    try:
        codepoint = int(match.group(cp_index))
    except (ValueError, OverflowError):
        return match.group(0)
    # Functional keys (Kitty private use plane 57344-63743, incl. Shift+Enter
    # at 57427 reported by Ghostty), un-mapped control codepoints (CR, ESC,
    # Tab, ...), surrogates and out-of-range values pass through: they stand
    # for editor functions, not literal text, and chr(13) must never replace a
    # byte sequence that meant "accept the prompt" under stock input().
    if codepoint < 32 or codepoint == 127:
        return match.group(0)
    if 0xD800 <= codepoint <= 0xDFFF or 57344 <= codepoint <= 63743:
        return match.group(0)
    try:
        char = chr(codepoint)
    except (ValueError, OverflowError):
        return match.group(0)
    return char


def decode_modified_key_sequences(text: str) -> str:
    """Replace leaked extended-keyboard sequences with the typed characters.

    * ``ESC[27;<mod>;<codepoint>~`` (modifyOtherKeys level 2) decodes to
      ``chr(codepoint)`` when ``<mod>`` is exactly 2 — the shared encoding
      both protocols use for a plain Shift combo.
    * ``ESC[<codepoint>;<mod>u`` (Kitty CSI-u) additionally decodes ``<mod>``
      3/4, the press/repeat-shifted forms Kitty's *alternate* report mode
      produces by ORing the event type into the modifier.  Alt+T under
      modifyOtherKeys (also mod 3) has no event bits to disambiguate from and
      therefore passes through — never decode Alt there.
    * Functional keys (Kitty private-use plane 57344-63743, incl. Shift+Enter
      at 57427), control codepoints (<32/127), lock-bit combos (CapsLock is
      applied to the codepoint before encoding), other modifiers, and all
      plain text are left byte-for-byte untouched.
    """
    if not text or "\x1b[" not in text:
        return text
    text = _MODIFY_OTHER_KEYS_RE.sub(
        lambda m: _shift_char(m, 1, 2, _EXACT_SHIFT_MODS), text
    )
    text = _KITTY_CSI_U_RE.sub(
        lambda m: _shift_char(m, 2, 1, _SHIFT_ONLY_MODS), text
    )
    return text


def safe_input(prompt: str = "", *, _reader: Callable[[str], str] | None = None) -> str:
    """Builtin ``input()`` that decodes leaked modified-key sequences.

    Drop-in replacement for ``input()`` at bare interactive prompt sites
    (consent gates, setup wizards, credential/endpoint prompts, plugin
    configurators).  Terminals in modifyOtherKeys / Kitty extended-key mode
    re-encode shifted keystrokes as escape sequences that ``input()`` hands
    back as raw text (#97975); this wrapper runs those sequences back to the
    characters the user actually typed.

    Behavior on empty input, EOF and Ctrl+C is identical to builtin
    ``input()`` — the decode is a no-op on both.

    ``_reader`` exists for tests only; it replaces the builtin read call.
    """
    raw = (_reader or builtins.input)(prompt)
    return decode_modified_key_sequences(raw)
