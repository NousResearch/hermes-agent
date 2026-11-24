r"""Code-point range sets and the four-cell set algebra behind class translation.

A range set is a sorted tuple of inclusive, non-adjacent ``(lo, hi)`` pairs. RE2 has no class
intersection or subtraction, only union and whole-class negation, so a set is kept as four range
sets ("masks"), one per CELL of a partition of all code points that RE2 can name both ways:

    cell   RE2 member     RE2 complement
    L      \p{L}          \P{L}
    Nd     \p{Nd}         \P{Nd}
    Nx     \p{Nl}\p{No}   \P{N}\p{Nd}
    O      (none)         \p{L}\p{N}

``cells = (mL, mNd, mNx, mO)`` means "the union over cells of (cell AND mask)". Union and
complement are per-cell range operations, and ``cell AND mask`` is spelled
``[^<complement of cell><complement of mask>]`` (double negation).

``care`` holds, per cell, the code points that CAN be in that cell (ASCII membership is fixed;
the delta tables say more). Outside it a mask bit is a don't-care, which lets most sets collapse
to ONE class. Stdlib-only; no regex engine.
"""
from __future__ import annotations

MAX = 0x10FFFF
ALL = ((0, MAX),)
NONE = ()
ANY = "[\\x{0}-\\x{10FFFF}]"
NEVER = "[^\\x{0}-\\x{10FFFF}]"
_CELL = (r"\p{L}", r"\p{Nd}", r"\p{Nl}\p{No}")
_NOT_CELL = (r"\P{L}", r"\P{Nd}", r"\P{N}\p{Nd}", r"\p{L}\p{N}")


def normalize(pairs) -> tuple:
    out = []
    for lo, hi in sorted(pairs):
        if out and lo <= out[-1][1] + 1:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def union(a: tuple, b: tuple) -> tuple:
    if not a or a == b:
        return b
    if not b:
        return a
    return normalize(a + b)


def complement(a: tuple) -> tuple:
    out, start = [], 0
    for lo, hi in a:
        if lo > start:
            out.append((start, lo - 1))
        start = hi + 1
    if start <= MAX:
        out.append((start, MAX))
    return tuple(out)


def intersect(a: tuple, b: tuple) -> tuple:
    if a == b or b == ALL:
        return a
    if a == ALL:
        return b
    return complement(union(complement(a), complement(b)))


def _char(cp: int) -> str:
    if 48 <= cp <= 57 or 65 <= cp <= 90 or 97 <= cp <= 122:
        return chr(cp)
    return f"\\x{{{cp:X}}}"


def render(a: tuple) -> str:
    """Range set as RE2 class members."""
    return "".join(_char(lo) if lo == hi else _char(lo) + ("-" if hi > lo + 1 else "") + _char(hi)
                   for lo, hi in a)


def literal(ranges: tuple) -> tuple:
    return (ranges, ranges, ranges, ranges)


def cells_union(a: tuple, b: tuple) -> tuple:
    return tuple(union(x, y) for x, y in zip(a, b))


def cells_complement(a: tuple) -> tuple:
    return tuple(complement(x) for x in a)


def _split(cells: tuple, care: tuple):
    """Per cell: what must be in (``mask AND care``) and what may be in (``mask OR NOT care``);
    and the code points that are in the set whichever cell they belong to."""
    must = [intersect(m, k) for m, k in zip(cells, care)]
    free = [union(m, complement(k)) for m, k in zip(cells, care)]
    common = intersect(intersect(free[0], free[1]), intersect(free[2], free[3]))
    return must, free, common


def _single(cells: tuple, care: tuple):
    """Members of ONE positive class equal to the set, or ``None`` when it needs several."""
    must, _, common = _split(cells, care)
    outside, whole, need = complement(common), [], NONE
    for index in range(4):
        if index < 3 and must[index] == care[index]:
            whole.append(_CELL[index])
        elif intersect(must[index], outside):
            return None
        else:
            need = union(need, must[index])
    props = "".join(whole).replace(r"\p{Nd}\p{Nl}\p{No}", r"\p{N}")
    return min(render(need), render(common), key=len) + props


def _alternation(cells: tuple, care: tuple) -> str:
    must, free, common = _split(cells, care)
    outside = complement(common)
    terms = ["[" + render(common) + "]"] if common else []
    parts = {}
    for index in range(4):
        if intersect(must[index], outside):
            parts[index] = min(render(complement(free[index])), render(complement(must[index])), key=len)
    if 1 in parts and parts.get(2) == parts[1]:
        terms.append("[^" + r"\P{N}" + parts.pop(1) + "]")
        del parts[2]
    terms += ["[^" + _NOT_CELL[index] + excluded + "]" for index, excluded in sorted(parts.items())]
    return terms[0] if len(terms) == 1 else "(?:" + "|".join(terms) + ")"


def emit(cells: tuple, care: tuple) -> str:
    """RE2 source matching exactly one character of the set."""
    if cells[0] == cells[1] == cells[2] == cells[3]:
        only = cells[0]
        if only == ALL:
            return ANY
        if not only:
            return NEVER
        inside, outside = render(only), render(complement(only))
        return "[" + inside + "]" if len(inside) <= len(outside) else "[^" + outside + "]"
    for opener, masks, empty in (("[", cells, NEVER), ("[^", cells_complement(cells), ANY)):
        members = _single(masks, care)
        if members is not None:
            return opener + members + "]" if members else empty
    return _alternation(cells, care)


_BYTES = ((0, 0xFF),)


def emit_bytes(only: tuple) -> str:
    """Like ``emit`` for a Latin-1 (bytes) pattern: nothing above 0xFF may appear in the source."""
    inside = intersect(only, _BYTES)
    outside = intersect(complement(only), _BYTES)
    if not outside:
        return "[\\x{0}-\\x{FF}]"
    if not inside:
        return "[^\\x{0}-\\x{FF}]"
    a, b = render(inside), render(outside)
    return "[" + a + "]" if len(a) <= len(b) else "[^" + b + "]"
