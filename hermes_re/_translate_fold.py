"""Case folding as stdlib `re` defines it. Stdlib-only, no regex engine.

Stdlib IGNORECASE on ``str``: a pattern character ``p`` that is CASED matches a text character
``c`` when ``lower(c)`` is ``lower(p)`` or one of the lowercase letters that share an uppercase
with it (i / dotless i, s / long s, ...). ``lower`` is the simple one-character mapping. An
uncased ``p`` matches only itself, except inside a bracket class that also has a cased member:
then every member is compared through ``lower(c)``. With the ASCII flag only A-Z / a-z fold.

Everything is computed from the running interpreter's ``str.lower`` / ``str.upper`` / ``str.title``,
so it follows that interpreter's Unicode version, whatever version RE2's own tables are.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

_LIMIT = 0x20000  # cased characters live in planes 0 and 1 (t3_table.py checks all planes)
_orbits = None  # {code point: tuple of the OTHER code points matching it through lower()}
_keys = None  # sorted code points having an orbit
_cased = None  # sorted code points stdlib treats as cased


def _lower(ch: str) -> str:
    low = ch.lower()
    return low[0]  # CPython's one-character lowercase: U+0130 lowers to "i" + combining dot -> "i"


def _upper(ch: str) -> str:
    return ch.upper()[0]  # CPython's one-character uppercase is the first of the full mapping


def _build():
    global _orbits, _keys, _cased
    lowers, groups, cased = {}, {}, []
    for base in range(0, _LIMIT, 256):
        chunk = "".join(map(chr, range(base, base + 256)))
        if chunk.lower() == chunk and chunk.upper() == chunk:
            continue  # no case mapping in this block
        for ch in chunk:
            low = _lower(ch)
            if low != ch:
                lowers.setdefault(low, []).append(ch)
            if low != ch or _upper(ch) != ch:
                cased.append(ord(ch))
            groups.setdefault(ch.upper(), set()).add(low)
    for up, lows in groups.items():  # a block skipped above still joins its own group
        if len(up) == 1 and up.upper() == up:
            lows.add(_lower(up))
    fixes = {}
    for lows in groups.values():
        if len(lows) > 1:
            for low in lows:
                fixes.setdefault(low, set()).update(lows - {low})
    orbits = {}
    for ch in set(lowers) | set(fixes) | {c for v in lowers.values() for c in v}:
        low = _lower(ch)
        members = set()
        for target in {low} | fixes.get(low, set()):
            members.add(target)
            members.update(lowers.get(target, ()))
        members.discard(ch)
        if members:
            orbits[ord(ch)] = tuple(sorted(map(ord, members)))
    _orbits, _keys, _cased = orbits, sorted(orbits), sorted(cased)
    return orbits


def is_cased(cp: int, ascii_only: bool) -> bool:
    return any_cased(cp, cp, ascii_only)


def any_cased(lo: int, hi: int, ascii_only: bool) -> bool:
    """Whether ``lo..hi`` holds a character stdlib treats as cased."""
    if ascii_only:
        return lo <= 122 and hi >= 65 and (lo <= 90 or hi >= 97)
    if _cased is None:
        _build()
    index = bisect_left(_cased, lo)
    return index < len(_cased) and _cased[index] <= hi


def orbit(cp: int, ascii_only: bool) -> tuple:
    """The other code points whose ``lower()`` matches ``cp``'s (empty when there are none)."""
    if ascii_only:
        if 65 <= cp <= 90:
            return (cp + 32,)
        if 97 <= cp <= 122:
            return (cp - 32,)
        return ()
    return (_orbits or _build()).get(cp, ())


def range_orbit(lo: int, hi: int, ascii_only: bool) -> set:
    """Code points outside ``lo..hi`` matching some member of the range through ``lower()``."""
    out = set()
    if ascii_only:
        for a, b, shift in ((65, 90, 32), (97, 122, -32)):
            out.update(cp + shift for cp in range(max(lo, a), min(hi, b) + 1))
    else:
        orbits = _orbits or _build()
        for index in range(bisect_left(_keys, lo), bisect_right(_keys, hi)):
            out.update(orbits[_keys[index]])
    return {cp for cp in out if not lo <= cp <= hi}


def upper_hits(lo: int, hi: int) -> set:
    """Stdlib quirk for a class range that reaches past the BMP under IGNORECASE: it also accepts
    ``c`` when ``upper(lower(c))`` falls in the range (``[\\u02bc-\\U00010000]`` matches U+0149,
    whose uppercase starts with U+02BC)."""
    if _cased is None:
        _build()
    return {cp for cp in _cased if lo <= ord(_upper(_lower(chr(cp)))) <= hi}
