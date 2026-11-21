"""Rewrite `\\b` into edge guards before the edge planner runs.

``rewrite(pattern, flags)`` returns the pattern with each `\\b` it can decide replaced:

  both neighbours definite, one word one not   -> removed (always a boundary)
  both neighbours definite, same kind          -> a never-matching class (never a boundary)
  leading position, next character definite    -> (?<!\\w) or (?<=\\w)
  trailing position, previous one definite     -> (?!\\w)  or (?=\\w)
  word char, `\\b`, then ONE optional non-word atom at the trailing position
                                               -> (?:ATOM|(?!\\w))
  anything else                                -> left as RE2's ASCII `\\b`

The lookarounds it writes are exactly the shapes ``_edge_plan`` emulates, at positions where
``_edge_plan`` looks for them, and `\\w` in them is the facade's Unicode `\\w`.
"""
from __future__ import annotations

from ._edge_parse import Node, Unsupported, parse
from ._edge_plan import DEAD
from ._edge_word import N, W, definite, edge, inline_flags, is_boundary

ENABLED = True
_GUARD = {("lead", W): r"(?<!\w)", ("lead", N): r"(?<=\w)", ("trail", W): r"(?!\w)", ("trail", N): r"(?=\w)"}
_ONCE = ("", "?", "??")          # quantifiers that cannot repeat the group
_OPTIONAL = {"?": "(?:{0}|(?!\\w))", "??": "(?:(?!\\w)|{0})", "*": "(?:{0}+|(?!\\w))", "*?": "(?:(?!\\w)|{0}+?)"}


class Stats:
    __slots__ = ("always", "guard", "kept", "kept_at", "never", "not_at", "positive", "tail", "total")

    def __init__(self):
        self.total = self.guard = self.always = self.never = self.kept = 0
        self.positive = self.tail = 0  # of `guard`: next to a NON-word character | optional-tail rule
        self.kept_at = []  # (offset, left, right, at lead, at trail) of each `\\b` left to RE2
        self.not_at = []   # offset of each `\\B`: always left to RE2


def _optional_tail(pattern, node, icase):
    """The replacement template for `\\b` + this optional atom, when the atom always starts
    with a non-word character and cannot match nothing itself."""
    template = _OPTIONAL.get(pattern[node.aend:node.end])
    if template is None or node.kind not in ("char", "set", "group"):
        return None
    if node.kind == "group":
        parts = [edge(pattern, branch, icase) for branch in node.branches]
        values = set().union(*(v for v, _ in parts))
        if any(nothing for _, nothing in parts):
            return None
    else:
        values, _ = edge(pattern, [node], icase)
    return template if values == {N} else None


def rewrite(pattern: str, flags: int):
    """``(source for the edge planner / RE2, Stats or None when `\\b` needs no attention)``."""
    if not ENABLED or flags & 256 or not ("\\b" in pattern or "\\B" in pattern):  # ASCII flag: RE2's `\b` is exact
        return pattern, None
    try:
        branches, _ = parse(pattern)
    except Unsupported:
        return pattern, None
    inline = inline_flags(branches)
    if "a" in inline or "L" in inline:
        return pattern, None
    icase = bool(flags & 2) or "i" in inline
    unknown = Node("other", 0)  # stands for "any text" around a repeated group
    stats, edits = Stats(), []

    def walk(seqs, before, after, lead, trail):
        for seq in seqs:
            first = next((k for k, n in enumerate(seq) if n.kind != "flags"), None) if lead else None
            last = len(seq) - 1 if trail else None
            for k, node in enumerate(seq):
                if node.kind == "look":  # inside a lookaround: decide only what is local to it
                    walk(node.branches, [unknown], [unknown], False, False)
                    continue
                if node.kind == "group":
                    quant = pattern[node.aend:node.end]
                    loop = [] if quant in _ONCE else [unknown]
                    walk(node.branches, before + seq[:k] + loop, loop + seq[k + 1:] + after,
                         k == first and not quant, k == last and quant in _ONCE)
                    continue
                if not is_boundary(node):
                    if node.kind == "anchor" and node.sub == "B":
                        stats.not_at.append(node.start)
                    continue
                stats.total += 1
                left = definite(pattern, before + seq[:k], icase, True)
                right = definite(pattern, seq[k + 1:] + after, icase)
                if left and right:
                    if left != right:
                        stats.always += 1
                        edits.append((node.start, node.end, ""))
                    else:
                        stats.never += 1
                        edits.append((node.start, node.end, DEAD))
                elif right and k == first:
                    stats.guard += 1
                    stats.positive += right == N
                    edits.append((node.start, node.end, _GUARD["lead", right]))
                elif left and k == last:
                    stats.guard += 1
                    stats.positive += left == N
                    edits.append((node.start, node.end, _GUARD["trail", left]))
                elif (left == W and last is not None and k + 1 == last and _optional_tail(pattern, seq[last], icase)
                      and "\\b" not in pattern[seq[last].start:seq[last].end]):
                    tail = seq[last]
                    stats.guard += 1
                    stats.tail += 1
                    template = _optional_tail(pattern, tail, icase)
                    edits.append((node.start, tail.end, template.format(pattern[tail.start:tail.aend])))
                else:
                    stats.kept += 1
                    stats.kept_at.append((node.start, left, right, k == first, k == last))
    walk(branches, [], [], True, True)
    for start, end, text in sorted(edits, reverse=True):
        pattern = pattern[:start] + text + pattern[end:]
    return pattern, stats
