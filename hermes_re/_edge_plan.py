r"""Rewrite edge guards into consumed neighbours (stdlib only, no engine).

An EDGE GUARD is a zero-width test at the start or the end of the match that RE2 cannot express
zero-width: a one-character lookbehind at the start, a lookahead at the end, and a non-multiline
`$` at the end (it is the lookahead ``(?=\n?\Z)``). The guard is replaced by text that CONSUMES
the neighbour; the runner (``_edge_run``) then reports the match without it.

The sources produced here are still in the Python dialect: ``_translate`` turns them into RE2
source afterwards, so guard classes get the exact Unicode classes and the exact case folding.

``plan(pattern, flags)`` returns ``None`` when the pattern has no edge guard, else a Plan with
the sources to compile and the bookkeeping needed to report the user's own match:

  scan   every path consumes ONE character before its body (the guard character)
  start  the same bodies with the leading guards resolved for "at the start of the text"
  full() the same two sources forced to end at the end of the text (for fullmatch)
"""
from __future__ import annotations

from ._edge_guard import approx, setting
from ._edge_parse import Unsupported, drive, parse

ANY = "(?s:.)"                   # the character an unguarded alternative steps over
DEAD = r"[^\s\S]"  # never matches, keeps the group numbering; valid in str and bytes patterns alike
DOLLAR = r"(?:\n?\z)"            # what a non-multiline `$` stands before: an optional final newline
_END_ANCHORS = frozenset("$Zz")
_TOKENS = ("(?=", "(?!", "(?<=", "(?<!")
_FLIP = {r"\d": r"\D", r"\w": r"\W", r"\s": r"\S"}  # a negated shorthand as a class member
_ONCE = {"": "", "{1}": "", "?": "?", "{0,1}": "?", "??": "??", "{0,1}?": "??"}  # quantifier -> its short form


def wanted(pattern: str) -> bool:
    """Cheap pre-check, so most patterns never reach the parser."""
    return "$" in pattern or ("(?" in pattern and any(token in pattern for token in _TOKENS))


def _scoped(node, ml: bool) -> bool:
    """The MULTILINE state inside ``node`` (a group may carry ``(?m:`` or ``(?-m:``)."""
    on, off = node.flags
    return False if "m" in off else True if "m" in on else ml


def _is_dollar(node, ml: bool) -> bool:
    """A non-multiline `$` without a quantifier: the trailing guard ``(?=\\n?\\Z)``."""
    return node.kind == "anchor" and node.sub == "$" and not ml and node.end == node.aend


def _items(node):
    """One ``(kind, text)`` per alternative of a lookaround whose alternatives are single
    one-character atoms or anchors, else ``None``."""
    out = []
    for seq in node.branches:
        if len(seq) != 1 or seq[0].end != seq[0].aend:
            return None
        atom = seq[0]
        if atom.kind == "char":
            out.append(("pos", atom.sub))
        elif atom.kind == "set":
            out.append(atom.sub)
        elif atom.kind == "anchor":
            out.append(("anchor", atom.sub))
        else:
            return None
    return out


def _negated(items):
    """A bracket class for "one character that no item matches", else ``None``."""
    if len(items) == 1 and items[0][0] == "neg":
        return "[" + items[0][1] + "]"
    members = [member if kind == "pos" else _FLIP.get(member) if kind == "neg" else None for kind, member in items]
    if items and None not in members:
        return "[^" + "".join(members) + "]"
    return None


class Plan:
    __slots__ = ("_full", "groups", "gspec", "lead", "lead_tests", "markers", "scan", "simple", "start", "trail",
                 "trail_tests")

    def full(self):
        """``(scan, start)`` sources that must end at the end of the text."""
        return self._full()


class _Emitter:
    """One pass over the parsed pattern producing one source flavour.

    lead: None (no consumed prefix) | "scan" | "start".  trail: rewrite trailing guards.
    tail: text inserted where the user's match ends ("" or the end-of-text anchor).
    """

    def __init__(self, pattern, multiline, lead, trail, tail="", icase=None):
        self.p, self.ml, self.lead, self.trail, self.tail = pattern, multiline, lead, trail, tail
        self.icase = icase   # None: guards keep their exact classes; else the pattern's IGNORECASE state
        self.lead_tests, self.trail_tests = [], []  # the Python tests of the guards spelled cheaply
        self.real = 0        # capture groups emitted so far, hidden markers included
        self.gspec = [None]  # user group number -> (real number, encloses lead, encloses trail)
        self.markers = []    # real numbers of the hidden "user match ends here" groups
        self.nlead = self.ntrail = 0
        self.alive = False   # start flavour: some alternative can match at the start of text

    def top(self, branches):
        return "|".join(drive(self.seq(b, self.lead is not None, self.trail, self.ml)) for b in branches)

    def marker(self):
        """The hidden empty group whose offset is where the user's match ends."""
        self.ntrail += 1
        self.real += 1
        self.markers.append(self.real)
        return "()" + self.tail

    def content(self, node):
        return self.p[node.start + len(node.open):node.end - 1]

    def count(self, n, lead=False, trail=False):
        for _ in range(n):
            self.real += 1
            self.gspec.append((self.real, lead, trail))

    def verbatim(self, node):
        self.count(node.ncap)
        return self.p[node.start:node.end]

    def unguarded(self, out):
        if self.lead == "scan":
            out.append(ANY)
        else:
            self.alive = True

    def seq(self, nodes, lead, trail, ml):
        """One alternative. ``lead`` / ``trail``: it sits at the leading / trailing edge of the
        pattern. ``ml``: the MULTILINE state at this point. A walk for ``drive``: it yields the
        walks of the edge groups it descends into."""
        out = []
        first = next((k for k, n in enumerate(nodes) if n.kind != "flags"), None) if lead else None
        if lead and first is None:  # empty alternative (or only inline flags)
            out.extend(self.verbatim(n) for n in nodes)
            self.unguarded(out)
            return "".join(out) + (self.tail if trail else "")
        last = len(nodes) - 1 if trail and nodes else None
        for k, node in enumerate(nodes):
            plain = node.end == node.aend  # no quantifier
            if k == first:
                guard = self.lead_guard(node, ml) if node.kind == "look" and plain else None
                if guard is not None:
                    self.nlead += 1
                    text = guard[0] if self.lead == "scan" else guard[1]
                    if text is None:
                        text = DEAD
                    elif self.lead == "start":
                        self.alive = True
                    out.append(text + (self.tail if k == last else ""))
                    continue
                if node.kind == "group" and plain:
                    out.append((yield self.group(node, True, k == last, ml)))
                    continue
                self.unguarded(out)
            if k == last:
                suffix = self.trail_guard(node, ml) if node.kind == "look" and plain else None
                if suffix is not None:
                    out.append(self.marker() + suffix)
                    self.count(node.ncap)
                elif _is_dollar(node, ml):  # `$` is the trailing guard (?=\n?\Z)
                    out.append(self.marker() + DOLLAR)
                elif node.kind == "group" and self.p[node.aend:node.end] in _ONCE:
                    out.append((yield self.group(node, False, True, ml)))
                else:
                    out.append(self.verbatim(node) + self.tail)
                continue
            out.append(self.verbatim(node))
        if trail and not nodes:
            out.append(self.tail)
        return "".join(out)

    def group(self, node, lead, trail, ml):
        if node.cap:
            self.count(1, lead, trail)
        ml = _scoped(node, ml)
        bodies = []
        for branch in node.branches:
            bodies.append((yield self.seq(branch, lead, trail, ml)))
        text = node.open + "|".join(bodies) + ")"
        quant = _ONCE[self.p[node.aend:node.end]]
        if quant == "?":   # X? is (?:X|): spell it out so the tail lands on both paths
            return "(?:" + text + "|" + self.tail + ")"
        if quant == "??":
            return "(?:" + self.tail + "|" + text + ")"
        return text

    def cheap(self, items, negative, tests):
        """The cheap class for a guard over ``items`` (``_edge_guard.approx``), its test noted in
        ``tests``; ``None`` when the guard keeps its exact spelling."""
        if self.icase is None or any(kind not in ("pos", "neg") for kind, _ in items):
            return None
        found = approx(tuple(items), negative, self.icase)
        if found is None:
            return None
        if found[1] not in tests:
            tests.append(found[1])
        return found[0]

    def lead_guard(self, node, ml):
        """``(scan text, start text or None when it cannot hold at the start of text)``."""
        if node.sub not in ("<=", "<!"):
            return None
        items = _items(node)
        if not items:
            return None
        negative = node.sub == "<!"
        if any(kind == "anchor" for kind, _ in items):
            if len(items) != 1 or items[0][1] not in "^A":
                return None
            anchor = self.content(node)
            if not negative:           # (?<=^) is ^ itself
                return ANY + anchor, anchor
            if items[0][1] == "A" or not ml:
                return ANY, None       # (?<!^): any previous character will do
            return r"[^\n]", None
        if negative:
            cls = None if any(kind == "dot" for kind, _ in items) else _negated(items)
            return (self.cheap(items, True, self.lead_tests) or cls, "") if cls else None
        return self.cheap(items, False, self.lead_tests) or "(?:" + self.content(node) + ")", None

    def consumed(self, branches, ml):
        """The body of a positive trailing lookahead as consumed text: verbatim, except that a
        non-multiline `$` ending it becomes the optional final newline it stands before. A walk
        for ``drive``."""
        parts = []
        for nodes in branches:
            last = nodes[-1] if nodes else None
            head = self.p[nodes[0].start:last.start] if nodes else ""
            if last is None:
                parts.append("")
            elif _is_dollar(last, ml):
                parts.append(head + DOLLAR)
            elif last.kind == "group" and self.p[last.aend:last.end] in _ONCE:
                inner = yield self.consumed(last.branches, _scoped(last, ml))
                parts.append(head + last.open + inner + ")" + self.p[last.aend:last.end])
            else:
                parts.append(head + self.p[last.start:last.end])
        return "|".join(parts)

    def trail_guard(self, node, ml):
        """The consumed suffix standing for a trailing lookahead, else ``None``."""
        if node.sub not in ("=", "!"):
            return None
        items = _items(node)
        if node.sub == "=":
            if items or node.branches != [[]]:  # (?=Y) at the trailing edge with any RE2-compatible Y
                return (items and self.cheap(items, False, self.trail_tests)) or (
                    "(?:" + drive(self.consumed(node.branches, ml)) + ")")
            return None
        if not items:
            return None
        anchors = {text for kind, text in items if kind == "anchor"}
        rest = [item for item in items if item[0] != "anchor"]
        if anchors - _END_ANCHORS or any(kind == "dot" for kind, _ in rest):
            return None
        if "$" in anchors:  # `$` also holds before a final newline (before any newline when multiline)
            if not all(kind == "pos" for kind, _ in rest) or (rest and not ml):
                return None
            if not ml:  # (?!$): a character follows, and it is not the final newline
                return r"(?:[^\n]|\n" + ANY + ")"
            return "[^" + "".join(member for _, member in rest) + r"\n]"
        if not rest:
            return ANY
        cls = _negated(rest)
        if cls is None:
            return None
        cls = self.cheap(rest, True, self.trail_tests) or cls
        return cls if anchors else "(?:" + cls + r"|\z)"


def _multiline(branches, flags) -> bool:
    """The pattern-wide MULTILINE state: the flag, or a global inline ``(?m)`` (stdlib accepts a
    global inline flag only at the start of the pattern). Scoped ``(?m:...)`` groups are followed
    by the emitter on its way down the edges."""
    return bool(flags & 8) or any(
        node.kind == "flags" and "m" in node.flags[0] for seq in branches for node in seq)


def _fixed(pattern: str, node) -> bool:
    """Whether ``node`` matches the same number of characters on every path (an explicit
    stack: the groups can nest as deep as stdlib accepts)."""
    todo = [node]
    while todo:
        node = todo.pop()
        quant = pattern[node.aend:node.end]
        if node.kind in ("char", "set"):
            if quant and not (quant[0] == "{" and quant[-1] == "}" and "," not in quant):
                return False
            continue
        if quant:
            return False
        if node.kind in ("anchor", "flags"):
            continue
        if node.kind != "group" or len(node.branches) != 1:
            return False
        todo.extend(node.branches[0])
    return True


def _simple(pattern: str, branches, ml: bool) -> bool:
    """Whether a match that begins at one place can end at one place only and always passes the
    same guards: ONE alternative of fixed width between its guards. For such a pattern a guard
    that fails on a neighbour rules out every match beginning there (see ``_edge_run``).

    The runner retries such a pattern one character further, and each attempt reads the text the
    scan program consumes, the trailing lookahead included: its body must be of bounded width too,
    or ``(?<=\\w)a(?=.*z)`` reads the rest of the text once per refused neighbour."""
    if len(branches) != 1:
        return False
    body = [node for node in branches[0] if node.kind != "flags"]
    if body and body[0].kind == "look":
        body = body[1:]
    if body and body[-1].kind == "look":
        body = body[:-1] + [node for seq in body[-1].branches for node in seq]
    elif body and _is_dollar(body[-1], ml):
        body = body[:-1]
    return all(_fixed(pattern, node) for node in body)


def plan(pattern: str, flags: int, cheap: bool = False):
    """The Plan of ``pattern``, or ``None`` when it has no edge guard. ``cheap``: spell guards on
    large Unicode classes cheaply and leave their exact decision to the runner
    (``Plan.lead_tests`` / ``trail_tests``; both empty when nothing was respelled)."""
    try:
        branches, ncap = parse(pattern)
    except Unsupported:
        return None
    ml = _multiline(branches, flags)
    icase = setting(branches, flags) if cheap else None
    scan = _Emitter(pattern, ml, "scan", True, "", icase)
    text = scan.top(branches)
    if not (scan.nlead or scan.ntrail):
        return None
    lead, trail = ("scan" if scan.nlead else None), bool(scan.ntrail)
    if not (lead and trail):  # the first pass assumed guards at both edges
        scan = _Emitter(pattern, ml, lead, trail, "", icase)
        text = scan.top(branches)
    out = Plan()
    out.scan, out.start, out.lead, out.trail = text, None, bool(lead), trail
    out.groups, out.gspec, out.markers = ncap, tuple(scan.gspec[1:]), tuple(scan.markers)
    out.lead_tests, out.trail_tests = tuple(scan.lead_tests), tuple(scan.trail_tests)
    out.simple = bool(out.lead_tests or out.trail_tests) and _simple(pattern, branches, ml)
    if lead:
        start = _Emitter(pattern, ml, "start", trail, "", icase)
        text = start.top(branches)
        out.start = text if start.alive else None

    def full():  # trail=True even without a trailing guard: it places the end anchor
        first = _Emitter(pattern, ml, lead, True, r"\z", icase).top(branches)
        second = None
        if out.start is not None:
            second = _Emitter(pattern, ml, "start", True, r"\z", icase).top(branches)
        return first, second
    out._full = full
    return out
