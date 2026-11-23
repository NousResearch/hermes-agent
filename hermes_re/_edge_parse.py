"""A small parser for the Python regex dialect (stdlib only, no `re`).

It builds just enough structure for the edge-guard rewrite: top-level alternatives, groups,
lookarounds, single-character atoms and quantifiers, each with its source span, so anything
the rewrite does not touch is copied verbatim from the pattern text.
"""
from __future__ import annotations

from ._translate_scan import property_end, repeat_end

_FLAG_LETTERS = frozenset("aiLmsux")
_QUANT = frozenset("*+?")
_HEX = {"x": 2, "u": 4, "U": 8}
_SHORT = {"d": ("pos", r"\d"), "w": ("pos", r"\w"), "s": ("pos", r"\s"),
          "D": ("neg", r"\d"), "W": ("neg", r"\w"), "S": ("neg", r"\s")}
_ANCHORS = frozenset("AZzbB")
_CLASS_SPECIAL = frozenset("\\]^-[")


class Unsupported(Exception):
    """The pattern uses syntax this parser does not model; the caller leaves it to RE2."""


class Node:
    """kind: char | set | anchor | group | look | flags | other.

    char   sub = text usable as a bracket-class member
    set    sub = ("pos" | "neg", member text): a shorthand, a bracket class or a dot
    anchor sub = one of ^ $ A Z z b B
    group  open = text up to the body, cap = capturing?, flags = (on, off) letters
    look   sub = "=", "!", "<=" or "<!"
    Source spans: start..aend is the atom, aend..end its quantifier.
    """
    __slots__ = ("aend", "branches", "cap", "end", "flags", "kind", "ncap", "open", "start", "sub")

    def __init__(self, kind, start, sub=None):
        self.kind, self.start, self.sub = kind, start, sub
        self.aend = self.end = start
        self.branches, self.ncap, self.open, self.cap, self.flags = None, 0, "", False, ("", "")


class Parser:
    def __init__(self, pattern: str):
        self.p, self.i, self.n = pattern, 0, len(pattern)

    def parse(self):
        branches, ncap = self.alternation()
        if self.i != self.n:
            raise Unsupported("unbalanced parenthesis")
        return branches, ncap

    def alternation(self):
        branches, ncap = [], 0
        while True:
            seq = []
            while self.i < self.n and self.p[self.i] not in "|)":
                node = self.atom()
                self.quantifier(node)
                ncap += node.ncap
                seq.append(node)
            branches.append(seq)
            if self.i < self.n and self.p[self.i] == "|":
                self.i += 1
                continue
            return branches, ncap

    def quantifier(self, node):
        p, i = self.p, self.i
        node.aend = i
        if i < self.n and p[i] in _QUANT:
            i += 1
        elif i < self.n and p[i] == "{":
            i = repeat_end(p, i) or i
        if i > node.aend and i < self.n and p[i] in "?+":
            i += 1
        node.end = self.i = i

    def atom(self):
        p, i = self.p, self.i
        ch = p[i]
        if ch == "(":
            return self.group()
        if ch == "[":
            return self.bracket()
        if ch == "\\":
            return self.escape()
        self.i = i + 1
        if ch == ".":
            return Node("set", i, ("dot", ""))
        if ch in "^$":
            return Node("anchor", i, ch)
        return Node("char", i, "\\" + ch if ch in _CLASS_SPECIAL else ch)

    def escape(self):
        p, i = self.p, self.i
        if i + 1 >= self.n:
            raise Unsupported("trailing backslash")
        nxt = p[i + 1]
        if nxt in _SHORT:
            self.i = i + 2
            return Node("set", i, _SHORT[nxt])
        if nxt in _ANCHORS:
            self.i = i + 2
            return Node("anchor", i, nxt)
        if nxt in _HEX:
            self.i = i + 2 + _HEX[nxt]
        elif nxt in "pP":  # RE2's property escape: one opaque atom
            self.i = property_end(p, i)
            return Node("other", i)
        elif nxt == "N" and p.startswith("{", i + 2):
            self.i = p.find("}", i) + 1 or self.n
        elif nxt == "0":
            j = i + 2
            while j < min(i + 4, self.n) and p[j] in "01234567":
                j += 1
            self.i = j
        elif "0" <= nxt <= "9":  # a backreference (or a 3-digit octal escape; not worth telling apart)
            j = i + 2
            while j < min(i + 4, self.n) and "0" <= p[j] <= "9":
                j += 1
            self.i = j
            return Node("other", i)
        else:
            self.i = i + 2
        return Node("char", i, p[i:self.i])

    def bracket(self):
        p, start = self.p, self.i
        i = start + 1
        negated = i < self.n and p[i] == "^"
        if negated:
            i += 1
        inner = i
        if i < self.n and p[i] == "]":  # a leading "]" is a literal
            i += 1
        while i < self.n and p[i] != "]":
            i += 2 if p[i] == "\\" else 1
        if i >= self.n:
            raise Unsupported("unterminated character class")
        self.i = i + 1
        member = p[inner:i]
        if member.startswith("]"):
            member = "\\" + member
        if member.endswith("-") and not member.endswith("\\-"):
            member = member[:-1] + "\\-"
        if member.startswith(("-", "^")):  # a leading "^" would negate the class the member is spliced into
            member = "\\" + member
        return Node("set", start, ("neg" if negated else "pos", member))

    def group(self):
        p, start = self.p, self.i
        node = Node("group", start)
        i = start + 1
        if not p.startswith("?", i):
            node.cap = True
        elif p.startswith(("?=", "?!"), i):
            node.kind, node.sub, i = "look", p[i + 1], i + 2
        elif p.startswith(("?<=", "?<!"), i):
            node.kind, node.sub, i = "look", p[i + 1:i + 3], i + 3
        elif p.startswith("?:", i):
            i += 2
        elif p.startswith("?P<", i):
            node.cap = True
            i = p.find(">", i) + 1
            if not i:
                raise Unsupported("unterminated group name")
        elif p.startswith(("?#", "?P="), i):  # comment or named backreference: opaque
            end = p.find(")", i)
            if end < 0:
                raise Unsupported("unterminated group")
            self.i = end + 1
            node.kind = "other"
            return node
        elif p.startswith(("?>", "?("), i):  # atomic group, conditional: RE2 rejects them
            node.kind = "other"
            i = p.find(")", i) + 1 if p.startswith("?(", i) else i + 2
            if not i:
                raise Unsupported("unterminated conditional")
        else:  # inline flags: (?i) or (?i-m:...)
            j = i + 1
            while j < self.n and (p[j] in _FLAG_LETTERS or p[j] == "-"):
                j += 1
            if j >= self.n or p[j] not in "):" or j == i + 1:
                raise Unsupported("unknown group syntax")
            on, _, off = p[i + 1:j].partition("-")
            node.flags = (on, off)
            if p[j] == ")":
                node.kind = "flags"
                self.i = j + 1
                return node
            i = j + 1
        node.open = p[start:i]
        self.i = i
        node.branches, inner = self.alternation()
        if self.i >= self.n or p[self.i] != ")":
            raise Unsupported("unbalanced parenthesis")
        self.i += 1
        node.ncap = inner + (1 if node.cap and node.kind == "group" else 0)
        return node


def parse(pattern: str):
    """``(branches, capture-group count)``; raises Unsupported."""
    return Parser(pattern).parse()
