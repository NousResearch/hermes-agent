"""Flag constants with stdlib `re`'s integer values, and the flag check (stdlib-only, no engine).

``VERBOSE``, ``LOCALE`` and ``DEBUG`` (and ``X``, ``L``) exist so that code naming them gets the
clear ``ValueError`` from ``compile()`` instead of an ``AttributeError``; RE2 cannot honour them.
"""
from __future__ import annotations

import enum


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

SUPPORTED = int(IGNORECASE | MULTILINE | DOTALL | UNICODE | ASCII)
_UNSUPPORTED = ((int(VERBOSE), "re.VERBOSE"), (int(LOCALE), "re.LOCALE"), (int(DEBUG), "re.DEBUG"))


def checked(kind: type, flags) -> int:
    """``flags`` as the integer a Pattern of type ``kind`` (``str`` or ``bytes``) stores, or the
    ``ValueError`` for a flag combination that cannot be honoured.

    As in stdlib, a ``str`` pattern without ASCII carries the UNICODE bit in ``Pattern.flags``.
    """
    flags = int(flags)
    if flags & ~SUPPORTED:
        names = [name for bit, name in _UNSUPPORTED if flags & bit] or [hex(flags & ~SUPPORTED)]
        raise ValueError(f"flags RE2 cannot honour: {', '.join(names)}; rewrite the pattern without them")
    if kind is str:
        if flags & ASCII and flags & UNICODE:
            raise ValueError("ASCII and UNICODE flags are incompatible")
        return flags if flags & ASCII else flags | int(UNICODE)
    if flags & UNICODE:
        raise ValueError("cannot use UNICODE flag with a bytes pattern")
    return flags
