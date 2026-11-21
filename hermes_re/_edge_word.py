"""Unicode-exact `\\b` wherever a neighbour is definitely (non-)word.

RE2's `\\b` is ASCII-only; stdlib's is Unicode on str. A `\\b` whose neighbour atom can only be a
word character (or only a non-word character) needs to look at ONE side only, so it is the same
as a one-character lookbehind/lookahead on `\\w`, which the edge-guard machinery emulates.
Stdlib only, no `re`.
"""
from __future__ import annotations

W, N, U = "W", "N", "U"   # definitely word | definitely non-word | unknown or mixed
_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a", "b": "\b"}
_HEX = {"x": 2, "u": 4, "U": 8}
_SHORT = {"w": W, "d": W, "s": N, "W": N, "D": U, "S": U}
_RANGE_LIMIT = 4096
# The only case-insensitive equivalence class that mixes word and non-word characters:
# U+0345 (combining ypogegrammeni, not a word character) folds with the three iotas.
_FOLD_MIXED = frozenset("\u0345\u0399\u03b9\u1fbe")


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _tokens(member: str):
    """Class-member text as ``("cat", letter)``, ``("chr", char or None)`` and ``("dash", "-")``
    tokens."""
    i, n = 0, len(member)
    while i < n:
        ch = member[i]
        if ch != "\\" or i + 1 >= n:
            yield ("dash" if ch == "-" else "chr"), ch  # an unescaped "-" may form a range
            i += 1
            continue
        nxt = member[i + 1]
        if nxt in _SHORT:
            yield "cat", nxt
            i += 2
        elif nxt in _HEX:
            end = i + 2 + _HEX[nxt]
            yield "chr", chr(int(member[i + 2:end], 16))
            i = end
        elif nxt == "N":
            end = member.find("}", i) + 1 or n
            yield "chr", None
            i = end
        elif nxt.isdigit():
            end = i + 1
            while end < min(i + 4, n) and member[end] in "01234567":
                end += 1
            yield "chr", chr(int(member[i + 1:end] or "0", 8))
            i = max(end, i + 2)
        else:
            yield "chr", _C_ESCAPES.get(nxt, nxt)
            i += 2


def _char(ch, icase):
    if ch is None or (icase and ch in _FOLD_MIXED):
        return U
    return W if _is_word(ch) else N


def wordness(node, icase) -> str:
    """W / N / U for a one-character atom (`char` or `set` node)."""
    if node.kind == "char":
        token = next(_tokens(node.sub), ("chr", None))
        return _SHORT[token[1]] if token[0] == "cat" else _char(token[1], icase)
    kind, member = node.sub
    if kind == "dot":
        return U
    toks = list(_tokens(member))
    if kind == "neg":  # the complement is definite only when the class holds every (non-)word char
        cats = {value for what, value in toks if what == "cat"}
        return N if "w" in cats else W if "W" in cats else U
    found, i = set(), 0
    while i < len(toks):
        what, value = toks[i]
        if what == "cat":
            found.add(_SHORT[value])
        elif what == "chr" and i + 2 < len(toks) and toks[i + 1][0] == "dash" and toks[i + 2][0] == "chr":
            lo, hi = value, toks[i + 2][1]
            if lo is None or hi is None or ord(hi) - ord(lo) > _RANGE_LIMIT:
                found.add(U)
            else:
                found.update(_char(chr(c), icase) for c in range(ord(lo), ord(hi) + 1))
            i += 2
        else:
            found.add(_char(value, icase))
        i += 1
    return found.pop() if len(found) == 1 else U


def _may_be_empty(pattern, node) -> bool:
    quant = pattern[node.aend:node.end]
    if not quant:
        return False
    if quant[0] in "*?":
        return True
    return quant[0] == "{" and quant[1:].lstrip("0")[:1] in ",}"


def edge(pattern, nodes, icase, backwards=False):
    """``(set of W/N/U for the first [last] character the nodes can match, may match nothing)``.

    An end anchor ahead (`$`, `\\Z`) means the next character is a newline or the end of the
    text, both non-word; likewise a start anchor behind."""
    found = set()
    stop = "^A" if backwards else "$Zz"
    for node in (reversed(nodes) if backwards else nodes):
        kind = node.kind
        if kind == "look" and node.sub == ("<=" if backwards else "=") and node.end == node.aend:
            for branch in node.branches:  # a positive lookaround describes the neighbour itself
                values, nothing = edge(pattern, branch, icase, backwards)
                found |= values | ({U} if nothing else set())
            return found, False
        if kind in ("look", "flags"):
            continue  # zero-width: the neighbour character is further along
        if kind == "anchor":
            if node.end != node.aend:
                found.add(U)
                return found, False
            if node.sub in stop:
                found.add(N)
                return found, False
            continue
        if kind == "other":
            found.add(U)
            return found, False
        empty = False
        if kind == "group":
            for branch in node.branches:
                values, nothing = edge(pattern, branch, icase, backwards)
                found |= values
                empty = empty or nothing
        else:
            found.add(wordness(node, icase))
        if not (empty or _may_be_empty(pattern, node)):
            return found, False
    return found, True


def definite(pattern, nodes, icase, backwards=False):
    """W or N when the neighbour character exists and is definite, else None."""
    values, nothing = edge(pattern, nodes, icase, backwards)
    if nothing or len(values) != 1 or U in values:
        return None
    return next(iter(values))


def is_boundary(node) -> bool:
    return node.kind == "anchor" and node.sub == "b" and node.end == node.aend


def inline_flags(branches) -> set:
    """Every flag letter some inline group turns on."""
    found = set()

    def walk(seqs):
        for seq in seqs:
            for node in seq:
                found.update(node.flags[0])
                if node.branches:
                    walk(node.branches)
    walk(branches)
    return found
