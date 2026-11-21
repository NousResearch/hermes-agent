r"""Cheap spellings of one-character edge guards, and their exact test in Python.

An edge guard on a Unicode class (``(?<!\w)``, ``(?!\w)``, the guards a `\b` at a pattern edge is
respelled into) used to put that class into every program of the pattern: for `\w` about
0.3 MiB and 0.8 ms per program. Here the guard gets a class that is EXACT for an ASCII neighbour
and accepts every non-ASCII neighbour, which costs a handful of instructions; the runner
(``_edge_run``) then looks at the neighbour itself when it is not ASCII and asks ``test``.

Why that is exact: the cheap program accepts every path the exact one accepts, in the same
priority order, and differs only in the class decision on a guard character. If the guard
characters of the match it reports pass the real test, the exact program reports the same match.
If one does not, the runner knows the report is wrong and handles it (``_edge_run.Edge``).

Stdlib only, no engine, no `re`.
"""
from __future__ import annotations

import functools
import unicodedata
from bisect import bisect_right

from ._tables import TABLES
from ._translate_class import Unsupported, members
from ._translate_ranges import MAX, normalize

ANY = "(?s:.)"
_COSTLY = frozenset("wWdD")  # the shorthands whose Unicode class is large
_IS = {"w": lambda ch: ch.isalnum() or ch == "_", "d": str.isdecimal, "s": str.isspace}


def _flag_nodes(branches, top=True):
    """Every node that sets or clears IGNORECASE inline, as ``(node, is a top-level (?i))``."""
    for index, seq in enumerate(branches):
        for position, node in enumerate(seq):
            if "i" in node.flags[0] or "i" in node.flags[1]:
                yield node, top and index == 0 and position == 0 and node.kind == "flags" and "i" in node.flags[0]
            if node.branches:
                yield from _flag_nodes(node.branches, False)


def setting(branches, flags: int):
    """The IGNORECASE state that holds for the WHOLE pattern (``True`` / ``False``), or ``None``
    when guards must keep their exact classes: the ASCII flag (the classes are small already),
    IGNORECASE switched inside the pattern, or an interpreter whose Unicode version has no
    table (the exact class is then RE2's own, which the Python test would not reproduce)."""
    if flags & 256 or unicodedata.unidata_version not in TABLES:
        return None
    found = list(_flag_nodes(branches))
    if not found:
        return bool(flags & 2)
    return True if len(found) == 1 and found[0][1] else None


def _inside(body: str, icase: bool):
    """``(test(ch) for the bracket class ``body``, its shorthand letters)``."""
    negated, ranges, letters = members(body, False, icase)
    lows = [lo for lo, _ in ranges]
    kinds = tuple((_IS[letter.lower()], letter.islower()) for letter in letters)

    def inside(ch: str) -> bool:
        code = ord(ch)
        index = bisect_right(lows, code) - 1
        found = (index >= 0 and code <= ranges[index][1]) or any(test(ch) is want for test, want in kinds)
        return found is not negated
    return inside, letters


def _escape(code: int) -> str:
    return f"\\x{code:02x}" if code < 256 else f"\\U{code:08x}"


@functools.lru_cache(maxsize=256)
def approx(items: tuple, negative: bool, icase: bool):
    """For a guard over the one-character classes ``items`` (``(("pos" | "neg", members), ...)``;
    ``negative``: the guard holds when NO item matches): ``(cheap class in the Python dialect,
    test(ch) -> bool)``, or ``None`` when the exact class is cheap enough to keep."""
    try:
        parts = [_inside("[" + ("^" if kind == "neg" else "") + member + "]", icase) for kind, member in items]
    except (Unsupported, ValueError, IndexError):
        return None  # the caller's exact spelling raises the proper error
    if not any(letter in _COSTLY for _, letters in parts for letter in letters):
        return None
    tests = tuple(inside for inside, _ in parts)

    def test(ch: str) -> bool:
        return any(inside(ch) for inside in tests) is not negative
    ranges = normalize([(code, code) for code in range(128) if test(chr(code))] + [(128, MAX)])
    if ranges == ((0, MAX),):
        return ANY, test
    body = "".join(_escape(lo) if lo == hi else _escape(lo) + "-" + _escape(hi) for lo, hi in ranges)
    return ("(?-i:[" + body + "])" if icase else "[" + body + "]"), test
