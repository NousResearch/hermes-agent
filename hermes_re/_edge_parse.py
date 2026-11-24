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
_OCTAL = frozenset("01234567")
_EDGE_SPECIAL = frozenset("]^-[&~|")  # a literal that means something else at the edge of a joined class


def _class_tokens(text: str):
    """Bracket-class member text as ``(source, is an escape)`` tokens."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "\\" or i + 1 >= n:
            yield text[i], False
            i += 1
            continue
        nxt, end = text[i + 1], i + 2
        if nxt in _HEX:
            end += _HEX[nxt]
        elif nxt == "N" and text.startswith("{", end):
            end = text.find("}", end) + 1 or n
        elif nxt in _OCTAL:
            while end < min(i + 4, n) and text[end] in _OCTAL:
                end += 1
        yield text[i:end], True
        i = end


def class_safe(text: str) -> str:
    r"""Member text that keeps its meaning when spliced next to other members in one class.

    An octal escape is respelled ``\xHH``: a digit after it would otherwise extend it (``\0`` then
    ``1`` reads ``\01``). A literal that means something else at the edge of the joined class
    (a leading ``^`` or ``]``, a ``-`` that would form a range, a doubled set operator) is escaped.
    """
    out = []
    tokens = list(_class_tokens(text))
    for k, (tok, escaped) in enumerate(tokens):
        if escaped and tok[1] in _OCTAL:
            value = int(tok[1:], 8)
            out.append(f"\\x{value:02x}" if value <= 0o377 else tok)  # a larger value stays an error
        elif not escaped and tok in _EDGE_SPECIAL and k in (0, len(tokens) - 1):
            out.append("\\" + tok)
        else:
            out.append(tok)
    return "".join(out)


class Unsupported(Exception):
    """The pattern uses syntax this parser does not model; the caller leaves it to RE2."""


class Node:
    """kind: char | set | anchor | group | look | flags | other.

    char   sub = text usable as a bracket-class member
    set    sub = ("pos" | "neg", member text): a shorthand, a bracket class or a dot
    anchor sub = one of ^ $ A Z z b B
    group  open = text up to the body, cap = capturing?, flags = (on, off) letters
    (``open`` is non-empty exactly for the nodes with a body: groups, lookarounds, atomic groups
    and conditionals)
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
        """Read the whole pattern with an explicit stack of open groups (no recursion per group:
        stdlib compiles nesting far deeper than the interpreter's recursion limit would allow)."""
        p, n = self.p, self.n
        stack = []  # per open group: (group node, then the enclosing branches, sequence, capture count)
        branches, seq, ncap = [], [], 0
        while True:
            if self.i < n and p[self.i] not in "|)":
                node = self.atom()
                if node.open:  # a group whose body follows: read it before the group's quantifier
                    stack.append((node, branches, seq, ncap))
                    branches, seq, ncap = [], [], 0
                    continue
                self.quantifier(node)
                ncap += node.ncap
                seq.append(node)
                continue
            branches.append(seq)
            seq = []
            if self.i < n and p[self.i] == "|":
                self.i += 1
                continue
            if not stack:
                if self.i != n:
                    raise Unsupported("unbalanced parenthesis")
                return branches, ncap
            if self.i >= n:
                raise Unsupported("unbalanced parenthesis")
            self.i += 1  # the group's ")"
            node, branches_out, seq, ncap_out = stack.pop()
            node.branches = branches
            node.ncap = ncap + (1 if node.cap and node.kind == "group" else 0)
            self.quantifier(node)
            branches, ncap = branches_out, ncap_out + node.ncap
            seq.append(node)

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
        return Node("char", i, class_safe(ch))

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
        elif nxt in _OCTAL and len(p[i + 2:i + 4]) == 2 and set(p[i + 2:i + 4]) <= _OCTAL:  # \177: octal
            self.i = i + 4
        elif "0" <= nxt <= "9":  # a backreference
            j = i + 2
            while j < min(i + 4, self.n) and "0" <= p[j] <= "9":
                j += 1
            self.i = j
            return Node("other", i)
        else:
            self.i = i + 2
        return Node("char", i, class_safe(p[i:self.i]))

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
        return Node("set", start, ("neg" if negated else "pos", class_safe(p[inner:i])))

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
        node.open = p[start:i]  # non-empty: ``parse`` reads the body and the closing ")"
        self.i = i
        return node


def drive(walk):
    """Run a recursive generator walk with an explicit stack instead of the interpreter's.

    ``walk`` yields the sub-walk it would have called and receives its return value; the
    nesting depth of a pattern then costs heap, not frames (stdlib compiles 495 nested groups
    from a fresh interpreter, which recursive helpers here would exceed)."""
    stack, value = [walk], None
    while True:
        try:
            sub = stack[-1].send(value)
        except StopIteration as done:
            stack.pop()
            if not stack:
                return done.value
            value = done.value
            continue
        stack.append(sub)
        value = None


def parse(pattern: str):
    """``(branches, capture-group count)``; raises Unsupported."""
    return Parser(pattern).parse()
