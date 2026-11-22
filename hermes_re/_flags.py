"""Flag constants with stdlib `re`'s integer values, and the flag check (stdlib-only, no engine).

``VERBOSE``, ``LOCALE`` and ``DEBUG`` (and ``X``, ``L``) exist so that code naming them gets the
clear ``ValueError`` from ``compile()`` instead of an ``AttributeError``; RE2 cannot honour them.
"""
from __future__ import annotations

import enum
import operator


class RegexFlag(enum.IntFlag):
    # Same bit values as stdlib re, so a literal `flags=2` keeps meaning IGNORECASE.
    NOFLAG = 0
    IGNORECASE = I = 2
    LOCALE = L = 4      # unsupported: compile() raises ValueError
    MULTILINE = M = 8
    DOTALL = S = 16
    UNICODE = U = 32    # the default for str patterns
    VERBOSE = X = 64    # unsupported: compile() raises ValueError
    DEBUG = 128         # unsupported: compile() raises ValueError
    ASCII = A = 256


NOFLAG = RegexFlag.NOFLAG
IGNORECASE = I = RegexFlag.IGNORECASE
LOCALE = L = RegexFlag.LOCALE
MULTILINE = M = RegexFlag.MULTILINE
DOTALL = S = RegexFlag.DOTALL
UNICODE = U = RegexFlag.UNICODE
VERBOSE = X = RegexFlag.VERBOSE
DEBUG = RegexFlag.DEBUG
ASCII = A = RegexFlag.ASCII

SSIZE_MIN, SSIZE_MAX = -(1 << 63), (1 << 63) - 1
_INT_MIN, _INT_MAX = -(1 << 31), (1 << 31) - 1


_INLINE = {"i": int(IGNORECASE), "m": int(MULTILINE), "s": int(DOTALL), "x": int(VERBOSE), "a": int(ASCII),
           "u": int(UNICODE), "L": int(LOCALE)}


def inline_globals(text: str) -> int:
    """The flags of the leading global flag groups (``(?i)(?m)...``), which stdlib reports in
    ``Pattern.flags``. ``text`` has its comment groups removed (stdlib skips them too)."""
    bits, i, n = 0, 0, len(text)
    while text.startswith("(?", i):
        j = i + 2
        while j < n and text[j] in _INLINE:
            j += 1
        if j == i + 2 or j >= n or text[j] != ")":
            break
        for letter in text[i + 2:j]:
            bits |= _INLINE[letter]
        i = j + 1
    return bits


def ssize(value) -> int:
    """``value`` as stdlib's C ``Py_ssize_t`` argument (``pos``, ``endpos``, ``count``,
    ``maxsplit``): the same TypeError for a non-integer and OverflowError out of range."""
    value = operator.index(value)
    if not SSIZE_MIN <= value <= SSIZE_MAX:
        raise OverflowError("Python int too large to convert to C ssize_t")
    return value


SUPPORTED = int(IGNORECASE | MULTILINE | DOTALL | UNICODE | ASCII)
_UNSUPPORTED = ((int(VERBOSE), "re.VERBOSE"), (int(LOCALE), "re.LOCALE"), (int(DEBUG), "re.DEBUG"))


def checked(kind: type, flags) -> int:
    """``flags`` as the integer a Pattern of type ``kind`` (``str`` or ``bytes``) stores, or the
    ``ValueError`` for a flag combination that cannot be honoured (and stdlib's TypeError /
    OverflowError for a flags value that is no C int).

    As in stdlib, a ``str`` pattern without ASCII carries the UNICODE bit in ``Pattern.flags``.
    """
    if not isinstance(flags, int):  # as stdlib, whose flag tests (`flags & X`) need an int
        raise TypeError(f"unsupported operand type(s) for &: {type(flags).__name__!r} and 'int'")
    flags = int(flags)
    names = [name for bit, name in _UNSUPPORTED if flags & bit]
    if names:
        raise ValueError(f"flags RE2 cannot honour: {', '.join(names)}; rewrite the pattern without them")
    if kind is str:
        if flags & ASCII and flags & UNICODE:
            raise ValueError("ASCII and UNICODE flags are incompatible")
        flags = flags if flags & ASCII else flags | int(UNICODE)
    elif flags & UNICODE:
        raise ValueError("cannot use UNICODE flag with a bytes pattern")
    if not _INT_MIN <= flags <= _INT_MAX:  # after the ValueErrors, in stdlib's order
        raise OverflowError("Python int too large to convert to C int")
    return flags  # bits without a meaning are kept and ignored, as stdlib does
