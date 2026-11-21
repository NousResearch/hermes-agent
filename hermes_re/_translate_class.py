r"""Bracket classes and shorthand classes with stdlib `re` meaning, as RE2 source .
Stdlib-only; no regex engine.

``shorthand(letter, ascii_only, icase)`` translates a bare ``\d \D \w \W \s \S``;
``bracket(body, ascii_only, icase)`` translates a whole ``[...]``.

Unicode `\w` is RE2's ``[\p{L}\p{N}_]`` corrected by a static delta table for the running Python's
Unicode version (``_tables``), likewise `\d` over ``\p{Nd}``; `\s` is enumerated.
Under IGNORECASE the class is folded here, from Python's own case tables, and emitted inside
``(?-i:...)`` so RE2 neither folds it again nor folds the ``\p{..}`` properties.
"""
from __future__ import annotations

import functools
import unicodedata

from . import _translate_fold as _fold
from . import _translate_ranges as _r
from ._tables import TABLES as _TABLES

_ASCII = {"d": ((48, 57),), "w": ((48, 57), (65, 90), (95, 95), (97, 122)), "s": ((9, 13), (32, 32))}
SHORTHANDS = frozenset("dDwWsS")
_CONTROL = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11, "a": 7, "b": 8}
_OCTAL = frozenset("01234567")
_unicode = None


class Unsupported(ValueError):
    """Class syntax this translator does not handle; the message says what."""


def _unicode_cells() -> dict:
    """Positive `d w s` cell sets for the running interpreter's Unicode version."""
    global _unicode
    table = _TABLES.get(unicodedata.unidata_version)
    if table is None:  # unknown Unicode version: uncorrected RE2 tables (gen_tables.py --check flags it)
        table = {"W_ADD": (), "W_SUB": (), "D_ADD": (), "D_SUB": (), "SPACE": _TABLES["16.0.0"]["SPACE"]}
    w_add, w_sub, d_add, d_sub = (tuple(table[key]) for key in ("W_ADD", "W_SUB", "D_ADD", "D_SUB"))
    keep_w = _r.complement(w_sub)
    # What each cell can contain: ASCII membership is fixed, W_ADD lies in cell O by definition,
    # W_SUB in L/N, D_SUB in Nd, D_ADD outside Nd.
    upper = _r.intersect(((128, _r.MAX),), _r.complement(_r.union(w_add, d_sub)))
    care = (_r.union(upper, ((65, 90), (97, 122))),
            _r.union(_r.intersect(((128, _r.MAX),), _r.complement(_r.union(w_add, d_add))), ((48, 57),)),
            upper,
            _r.complement(_r.union(_r.union(w_sub, d_sub), ((48, 57), (65, 90), (97, 122)))))
    _unicode = {
        "w": (keep_w, keep_w, keep_w, _r.union(w_add, ((95, 95),))),
        "d": (d_add, _r.complement(d_sub), d_add, d_add),
        "s": _r.literal(tuple(table["SPACE"])),
        "care": care,
    }
    return _unicode


def _category(letter: str, ascii_only: bool) -> tuple:
    low = letter.lower()
    cells = _r.literal(_ASCII[low]) if ascii_only else (_unicode or _unicode_cells())[low]
    return cells if letter == low else _r.cells_complement(cells)


def _emit(cells: tuple, latin1: bool = False) -> str:
    if latin1:  # bytes patterns are always ASCII-mode, so the four masks are one range set
        return _r.emit_bytes(cells[0])
    return _r.emit(cells, (_unicode or _unicode_cells())["care"])


def _wrap(source: str, icase: bool) -> str:
    return "(?-i:" + source + ")" if icase else source


@functools.lru_cache(maxsize=64)
def shorthand(letter: str, ascii_only: bool, icase: bool, latin1: bool = False) -> str:
    return _wrap(_emit(_category(letter, ascii_only), latin1), icase)


def _escape(body: str, i: int):
    """``(code point or None, shorthand letter or None, next index)`` for the escape at ``i``."""
    ch = body[i + 1]
    if ch in "dDwWsS":
        return None, ch, i + 2
    width = {"x": 2, "u": 4, "U": 8}.get(ch)
    if width:
        return int(body[i + 2:i + 2 + width], 16), None, i + 2 + width
    if ch in _OCTAL:
        j = i + 1
        while j < min(i + 4, len(body)) and body[j] in _OCTAL:
            j += 1
        return int(body[i + 1:j], 8), None, j
    if ch in _CONTROL:
        return _CONTROL[ch], None, i + 2
    if ch.isascii() and ch.isalnum():
        raise Unsupported(f"escape \\{ch} inside a bracket class")
    return ord(ch), None, i + 2


def _parse(body: str):
    """``(negated, literal ranges, shorthand letters)`` of ``[...]``."""
    i, end = 1, len(body) - 1
    negated = body[1] == "^"
    if negated:
        i = 2
    ranges, letters = [], []
    while i < end:
        if body[i] == "\\":
            low, letter, i = _escape(body, i)
            if letter:
                letters.append(letter)
                continue
        else:
            low, i = ord(body[i]), i + 1
        if i + 1 < end and body[i] == "-":
            if body[i + 1] == "\\":
                high, letter, i = _escape(body, i + 1)
                if letter:
                    raise Unsupported("a shorthand class as a range end")
            else:
                high, i = ord(body[i + 1]), i + 2
            if high < low:
                raise Unsupported("bad character range")
            ranges.append((low, high))
        else:
            ranges.append((low, low))
    return negated, ranges, letters


def members(body: str, ascii_only: bool, icase: bool):
    """``(negated, literal ranges as a range set, shorthand letters)`` of ``[...]``, the literal
    ranges with stdlib's IGNORECASE folding applied."""
    negated, ranges, letters = _parse(body)
    if icase and ranges and any(_fold.any_cased(lo, hi, ascii_only) for lo, hi in ranges):
        extra = set()
        for lo, hi in ranges:
            extra |= _fold.range_orbit(lo, hi, ascii_only)
            if hi > 0xFFFF and not ascii_only:
                extra |= _fold.upper_hits(lo, hi)
        ranges += [(cp, cp) for cp in extra]
    return negated, _r.normalize(ranges), letters


@functools.lru_cache(maxsize=2048)
def bracket(body: str, ascii_only: bool, icase: bool, latin1: bool = False) -> str:
    negated, ranges, letters = members(body, ascii_only, icase)
    cells = _r.literal(ranges)
    for letter in letters:
        cells = _r.cells_union(cells, _category(letter, ascii_only))
    if negated:
        cells = _r.cells_complement(cells)
    return _wrap(_emit(cells, latin1), icase)


@functools.lru_cache(maxsize=1024)
def folded_literal(cp: int, ascii_only: bool, latin1: bool = False) -> str:
    """RE2 source for the literal ``cp`` under IGNORECASE with stdlib's folding; ``""`` when RE2's
    own folding of that character is already the same (the caller then emits it unchanged)."""
    if cp < 128:
        if chr(cp) not in ("kKsS" if ascii_only else "iI"):
            return ""  # RE2 folds the other ASCII letters exactly as stdlib does (t3_pairs.py)
    elif not _fold.orbit(cp, False):
        return ""  # uncased for Python: RE2 has no fold for it either (t3_re2_only.py)
    if latin1:
        return ""  # bytes: RE2's Latin-1 folding is ASCII-only, like stdlib's (t3_misc.py)
    members = [(cp, cp)] + [(other, other) for other in _fold.orbit(cp, ascii_only)]
    return "(?-i:" + _emit(_r.literal(_r.normalize(members))) + ")"
