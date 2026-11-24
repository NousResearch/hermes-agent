"""Python regex dialect -> RE2 dialect, as text (stdlib-only, no engine).

``translate(pattern, flags)`` respells what RE2 lacks or reads differently: Unicode `\\d \\w \\s`
(bare and inside bracket classes, through ``_translate_class``), exact IGNORECASE folding,
`\\Z`, `\\xhh \\uXXXX \\UXXXXXXXX`, octal escapes, ``[[:alpha:]]``, ``{,n}``. It tracks MULTILINE and
IGNORECASE through scoped inline flags. It never changes what a match consumes: a non-multiline
`$` is NOT translated here (``_edge_plan`` turns a tail-position `$` into a trailing guard
before this runs); one that reaches this module raises ``NonTailDollar``.
"""
from __future__ import annotations

from ._errors import error
from ._translate_class import (
    SHORTHANDS as _SHORTHANDS, Unsupported as _Unsupported, bracket as _bracket,
    folded_literal as _folded, shorthand as _shorthand,
)
from ._translate_scan import (
    class_end as _class_end, group_open as _group_open, has_surrogate as _has_surrogate, octal_end as _octal_end,
    property_end as _property_end, repeat_end as _repeat_end,
)

_FLAG_LETTERS = frozenset("aiLmsux-")


class NonTailDollar(Exception):
    """A non-multiline `$` the edge planner did not take over; ``args[0]`` is its index."""


def _group_flags(pattern: str, i: int):
    """For ``pattern[i:]`` starting with ``(?``: the inline-flag letters and the char that
    closes them (``)`` or ``:``), else ``None``."""
    j = i + 2
    while j < len(pattern) and pattern[j] in _FLAG_LETTERS:
        j += 1
    if j > i + 2 and j < len(pattern) and pattern[j] in "):":
        return pattern[i + 2:j], pattern[j]
    return None


def _with(scope: tuple, letters: str) -> tuple:
    """The ``(multiline, ignorecase)`` scope after the inline-flag ``letters``."""
    on, _, off = letters.partition("-")
    return (False if "m" in off else (True if "m" in on else scope[0]),
            False if "i" in off else (True if "i" in on else scope[1]))


_HEX = frozenset("0123456789abcdefABCDEF")
_DIGITS = frozenset("0123456789")
_WIDTH = {"x": 2, "u": 4, "U": 8}
_FOLD_ASCII = ("iI", "kKsS")  # ASCII letters RE2 folds differently from stdlib: [Unicode mode, ASCII mode]


def _may_fold(pattern: str, flags: int) -> bool:
    """Whether IGNORECASE translation can change ``pattern`` (it has no backslash and no `$`)."""
    if "[" in pattern or not pattern.isascii():
        return True
    return any(letter in pattern for letter in _FOLD_ASCII[bool(flags & 256)])


def _same(pattern: str, flags: int) -> bool:
    """Whether ``pattern`` reads the same in both dialects, so nothing needs translating.
    ("[:" : RE2 would read [[:alpha:]] as a POSIX class; "{," : RE2 reads {,n} as a literal.)"""
    if "\\" in pattern or "$" in pattern or "[:" in pattern or "{," in pattern:
        return False
    if not pattern.isascii() and _has_surrogate(pattern):
        return False  # a lone surrogate in a bracket class has to be written as an escape
    return not ((flags & 2 or "(?" in pattern) and _may_fold(pattern, flags))


def _escape(pattern: str, i: int, icase: bool, ascii_only: bool, latin1: bool):
    """``(RE2 source, index after it)`` for the escape at ``pattern[i]`` (not the last character)."""
    nxt = pattern[i + 1]
    if nxt in _WIDTH:
        width = _WIDTH[nxt]
        digits = pattern[i + 2:i + 2 + width]
        folded = icase and len(digits) == width and _HEX.issuperset(digits) and _folded(
            int(digits, 16), ascii_only, latin1)
        return folded or "\\x{" + digits + "}", i + 2 + width
    if nxt in _DIGITS:
        end = _octal_end(pattern, i)
        if end:  # stdlib octal escape: a literal (RE2 reads some digit runs differently)
            code = int(pattern[i + 1:end], 8)
            return (icase and _folded(code, ascii_only, latin1)) or f"\\x{{{code:X}}}", end
        if pattern[i + 2:i + 3] in _DIGITS and pattern[i + 2:i + 3]:
            # stdlib: a two-digit group reference; RE2 would silently read it as octal
            raise error(f"invalid escape sequence: {pattern[i:i + 3]} (RE2 has no backreference)", pattern)
    if nxt in _SHORTHANDS:
        return _shorthand(nxt, ascii_only, icase, latin1), i + 2
    if nxt == "Z":
        return "\\z", i + 2
    if nxt in "pP":  # RE2's property escape: copied whole, so IGNORECASE never folds its name
        end = _property_end(pattern, i)
        return pattern[i:end], end
    if nxt >= "\x80":  # stdlib: an escaped non-ASCII character is itself; RE2 rejects the escape
        return (icase and _folded(ord(nxt), ascii_only, latin1)) or nxt, i + 2
    return "\\" + nxt, i + 2


def _class(pattern: str, i: int, icase: bool, ascii_only: bool, latin1: bool):
    """``(RE2 source, index after it)`` for the bracket class at ``pattern[i]``."""
    end = _class_end(pattern, i)
    body = pattern[i:end]
    plain = not (icase or "\\" in body or "]" in body[1:3] or "[" in body[1:])
    if plain and (body.isascii() or not _has_surrogate(body)):
        return body, end  # plain members: the same in both dialects
    try:
        return _bracket(body, ascii_only, icase, latin1), end
    except (_Unsupported, ValueError, IndexError) as exc:
        raise error(f"bad or unsupported bracket class {body!r}: {exc}", pattern) from None


def _open(pattern: str, i: int, scopes: list):
    """``(the group opener, index after it)`` for ``pattern[i] == '('``; updates the flag scopes."""
    inline = _group_flags(pattern, i) if pattern.startswith("(?", i) else None
    if inline and inline[1] == ")":
        scopes[-1] = _with(scopes[-1], inline[0])
    else:
        scopes.append(_with(scopes[-1], inline[0]) if inline else scopes[-1])
    after = abs(_group_open(pattern, i))  # the opener (flags, group name) is copied untranslated
    return pattern[i:after], after


def translate(pattern: str, flags: int, latin1: bool = False, dollar=None) -> str:
    """RE2 source for the Python-dialect ``pattern``. ``latin1``: a bytes pattern, decoded 1:1.

    A non-multiline `$` raises ``NonTailDollar`` unless ``dollar`` gives its spelling (used only
    by callers that want RE2's verdict on the rest of the pattern).
    """
    if _same(pattern, flags):
        return pattern
    ascii_only = bool(flags & 256)
    out, i, n = [], 0, len(pattern)
    scopes = [(bool(flags & 8), bool(flags & 2))]  # (multiline, ignorecase) per open group
    while i < n:
        ch = pattern[i]
        multiline, icase = scopes[-1]
        if ch == "\\" and i + 1 < n:
            text, i = _escape(pattern, i, icase, ascii_only, latin1)
        elif ch == "[":
            text, i = _class(pattern, i, icase, ascii_only, latin1)
        elif ch == "(":
            text, i = _open(pattern, i, scopes)
        elif ch == "{" and pattern.startswith(",", i + 1) and _repeat_end(pattern, i):
            text, i = "{0", i + 1  # stdlib: {,n} is {0,n}; RE2 would read it as literal text
        elif ch == "$" and not multiline:
            if dollar is None:
                raise NonTailDollar(i)
            text, i = dollar, i + 1
        else:
            if ch == ")" and len(scopes) > 1:
                scopes.pop()
            fold = icase and (ch in _FOLD_ASCII[ascii_only] or ch >= "\x80")
            text, i = (fold and _folded(ord(ch), ascii_only, latin1)) or ch, i + 1
        out.append(text)
    return "".join(out)
