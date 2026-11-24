r"""Syntax check of one Python-dialect source BEFORE it is translated for RE2 (stdlib-only, no engine).

``compile()`` does not create the RE2 program (that happens on first use), so everything RE2
would refuse has to be found here, from the text alone: unbalanced groups and classes, bad
escapes, quantifier misuse, repeat counts RE2 does not take, and every construct RE2 lacks and
the edge planner did not rewrite (lookarounds, backreferences, possessive quantifiers, atomic
groups, conditionals, named characters, Python-only inline flags).

``check`` also returns what the group structure says, so ``Pattern.groups`` and
``Pattern.groupindex`` need no engine: ``(number of capture groups, ((name, number), ...))``.

Where stdlib and RE2 disagree the stricter reading wins (a repeated anchor, a duplicate group
name, ``\Q...\E``, ``\C``). ``stdlib_syntax`` refuses, on the caller's text before any edge
rewrite, the syntax RE2 would take but stdlib refuses: ``\pL`` / ``\p{Name}``, ``\z`` before
Python 3.14, ``(?<name>...)``, ``(?U)`` and every other unknown extension or flag letter, a
global inline flag after the start, and the flag combinations ``_parse_flags`` rejects.

What is NOT decided here and can still fail when the program is built: a program larger than
the memory budget.
"""
from __future__ import annotations

import sys
import unicodedata

from ._errors import error
from ._translate_scan import has_surrogate, octal_end, property_end, repeat_end

MAX_REPEAT = 1000  # RE2's limit for one repeat count and for the product of nested counts
# stdlib's parser recurses once per group: from a fresh interpreter at the default recursion limit
# it compiles 495 nested groups and raises RecursionError at 496 (measured on 3.11 and 3.14). The
# facade's own walks use explicit stacks, so this is the only depth limit, and it refuses with
# ``error`` where stdlib would crash.
MAX_NESTING = 495
_NOTHING, _ATOM, _ANCHOR = 0, 1, 2  # what a quantifier would apply to at the scan position
_HEX = frozenset("0123456789abcdefABCDEF")
_WIDTH = {"x": 2, "u": 4, "U": 8}
_ANCHORS = frozenset("bBAZz")
_PLAIN_ESCAPES = frozenset("dDwWsSafnrtv")
_RE2_FLAGS = frozenset("ims")
_PYTHON_FLAGS = frozenset("auLx")
_NAME_CATEGORIES = frozenset(("Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc"))
_CATEGORIES = frozenset((
    "Any", "C", "Cc", "Cf", "Co", "Cs", "L", "Ll", "Lm", "Lo", "Lt", "Lu", "M", "Mc", "Me", "Mn", "N", "Nd",
    "Nl", "No", "P", "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "S", "Sc", "Sk", "Sm", "So", "Z", "Zl", "Zp", "Zs"))
_GUARD = " that is not a supported edge guard (one guard per edge of a top-level alternative: a one-character " \
         "lookbehind at its start, a lookahead at its end)"
_LACKING = (
    ("?=", "lookahead `(?=...)`" + _GUARD + "; RE2 has no lookahead"),
    ("?!", "negative lookahead `(?!...)`" + _GUARD + "; RE2 has no negative lookahead"),
    ("?<=", "lookbehind `(?<=...)`" + _GUARD + "; RE2 has no lookbehind"),
    ("?<!", "negative lookbehind `(?<!...)`" + _GUARD + "; RE2 has no negative lookbehind"),
    ("?>", "atomic group `(?>...)` (RE2 has no atomic group)"),
    ("?P=", "named backreference `(?P=name)` (RE2 has no named backreference)"),
    ("?(", "conditional `(?(...)...)` (RE2 has no conditional)"),
    ("?#", "comment group in a rewritten source"),
)

_LOOKS = ("?=", "?!", "?<=", "?<!")


class _Walk:
    """The scan state: position, open groups, capture groups seen, and the repeat bookkeeping."""
    __slots__ = ("before", "i", "latin1", "lone", "looks", "most", "n", "names", "ncap", "original", "p", "prod",
                 "seen", "stack")

    def __init__(self, text: str, latin1: bool, original, looks: bool = False):
        self.p, self.n, self.i, self.latin1, self.original = text, len(text), 0, latin1, original
        self.before, self.looks = _NOTHING, looks
        # A lone surrogate cannot be encoded for RE2. Inside a bracket class the class translator
        # writes it as an escape (it then never matches); anywhere else it is refused. Not judged
        # when reading lookarounds: an edge guard turns its character into a class member.
        self.lone = not latin1 and not looks and not text.isascii() and has_surrogate(text)
        self.stack = []     # per open group: the largest repeat product outside it so far
        self.most = 1       # largest product of nested repeat counts in the current group
        self.prod = 1       # the same for the atom just read
        self.ncap, self.names, self.seen = 0, [], set()

    def fail(self, message: str):
        return error(f"{message} (position {self.i})", self.original)

    def atom(self, width: int) -> None:
        self.i += width
        self.before, self.prod = _ATOM, 1

    def literal(self, ch: str) -> None:
        if self.lone and "\ud800" <= ch <= "\udfff":
            raise self.fail("pattern contains a lone surrogate, which RE2 cannot match")
        self.atom(1)

    def push(self, after: int) -> None:
        self.stack.append(self.most)
        self.most, self.before, self.i = 1, _NOTHING, after


def _hex(w: _Walk, letter: str) -> None:
    width = _WIDTH[letter]
    digits = w.p[w.i + 2:w.i + 2 + width]
    if len(digits) != width or not _HEX.issuperset(digits):
        raise w.fail(f"incomplete escape \\{letter}{digits}")
    if letter != "x" and (w.latin1 or int(digits, 16) > 0x10FFFF):
        raise w.fail(f"bad escape \\{letter}{digits}")
    w.atom(2 + width)


def _property(w: _Walk) -> None:
    """RE2's ``\\pL`` / ``\\p{Name}`` / ``\\p{^Name}``. A name that is not a general category is
    taken for a script name and left for RE2 to judge when the program is built."""
    end = property_end(w.p, w.i)
    name = w.p[w.i + 2:end]
    if name.startswith("{"):
        name = name[1:-1].removeprefix("^") if name.endswith("}") else ""
    known = name in _CATEGORIES if len(name) < 3 else (name.isascii() and name.replace("_", "a").isalpha())
    if not known:
        raise w.fail(f"bad Unicode property escape {w.p[w.i:end]!r}")
    w.atom(end - w.i)


def _escape(w: _Walk) -> None:
    if w.i + 1 >= w.n:
        raise w.fail("bad escape (end of pattern)")
    nxt = w.p[w.i + 1]
    if nxt in _ANCHORS:
        w.i += 2
        w.before = _ANCHOR
    elif nxt in _PLAIN_ESCAPES:
        w.atom(2)
    elif not (nxt.isascii() and nxt.isalnum()):  # an escaped punctuation or non-ASCII character is itself
        w.i += 1
        w.literal(nxt)
    elif nxt in _WIDTH:
        _hex(w, nxt)
    elif nxt.isdigit():
        end = octal_end(w.p, w.i)
        if not end:
            raise w.fail(f"backreference \\{nxt} (RE2 has no backreference)")
        w.atom(end - w.i)
    elif nxt in "pP":
        _property(w)
    elif nxt == "N":
        raise w.fail("named character `\\N{...}` is not supported: write the character or `\\uXXXX`")
    else:
        raise w.fail(f"bad escape \\{nxt}")


def _class(w: _Walk) -> None:
    p, n = w.p, w.n
    j = w.i + 1
    if j < n and p[j] == "^":
        j += 1
    first = j
    if j < n and p[j] == "]":
        j += 1
    while j < n and p[j] != "]":
        j += 2 if p[j] == "\\" else 1
    if j >= n:
        raise w.fail("unterminated character set")
    if "\\" not in p[first:j]:  # with a backslash the class translator reads (and checks) it
        k = first
        while k < j:
            if k + 2 < j and p[k + 1] == "-":
                if p[k + 2] < p[k]:
                    raise w.fail(f"bad character range {p[k:k + 3]}")
                k += 3
            else:
                k += 1
    w.atom(j + 1 - w.i)


def _flags(w: _Walk) -> None:
    """``(?flags)`` or ``(?flags:`` with RE2's letters; anything else after ``(?`` is refused."""
    p, j = w.p, w.i + 2
    while j < w.n and (p[j] in _RE2_FLAGS or p[j] == "-"):
        j += 1
    letters = p[w.i + 2:j]
    closer = p[j:j + 1]
    _, dash, off = letters.partition("-")
    if closer in _PYTHON_FLAGS and closer:
        raise w.fail(f"inline flag `{closer}` cannot be honoured by RE2; rewrite the pattern without it")
    if closer not in (")", ":") or not letters or "-" in off or (dash and not off) or not closer:
        raise w.fail(f"unknown or malformed group extension {p[w.i:j + 1]!r}")
    if closer == ":":
        w.push(j + 1)
    else:  # flags only: nothing a following quantifier could repeat
        w.i, w.before = j + 1, _NOTHING


def _named(w: _Walk, start: int) -> None:
    end = w.p.find(">", start)
    if end < 0:
        raise w.fail("missing >, unterminated name")
    name = w.p[start:end]
    good = name.isidentifier() and (name.isascii() or (
        not w.latin1 and all(unicodedata.category(ch) in _NAME_CATEGORIES for ch in name)))
    if not good:
        raise w.fail(f"bad character in group name {name!r}" if name else "missing group name")
    if name in w.seen:
        raise w.fail(f"redefinition of group name {name!r}")
    w.seen.add(name)
    w.ncap += 1
    w.names.append((name, w.ncap))
    w.push(end + 1)


def _open(w: _Walk) -> None:
    p, i = w.p, w.i
    if not p.startswith("?", i + 1):
        w.ncap += 1
        w.push(i + 1)
    elif p.startswith("?:", i + 1):
        w.push(i + 3)
    elif p.startswith("?P<", i + 1):
        _named(w, i + 4)
    elif w.looks and p.startswith(_LOOKS, i + 1):  # a lookaround read as a group: its body is checked
        w.push(i + (4 if p.startswith("?<", i + 1) else 3))
    else:
        for token, message in _LACKING:
            if p.startswith(token, i + 1):
                raise w.fail("unsupported " + message)
        _flags(w)


def _close(w: _Walk) -> None:
    if not w.stack:
        raise w.fail("unbalanced parenthesis")
    inner = w.most
    w.most = max(w.stack.pop(), inner)
    w.i += 1
    w.before, w.prod = _ATOM, inner


def _counts(w: _Walk, end: int):
    """``(min, max or None)`` of the repeat ``w.p[w.i:end]``, checked against RE2's limits."""
    low, comma, high = w.p[w.i + 1:end - 1].partition(",")
    least = int(low or 0)
    most = int(high) if high else (None if comma else least)
    if least > MAX_REPEAT or (most or 0) > MAX_REPEAT:
        raise w.fail(f"repeat count over {MAX_REPEAT}: {w.p[w.i:end]} (RE2's limit)")
    if most is not None and most < least:
        raise w.fail("min repeat greater than max repeat")
    w.prod *= max(least if most is None else most, 1)
    if (least >= 2 or (most or 0) >= 2) and w.prod > MAX_REPEAT:
        raise w.fail(f"nested repeat counts multiply to {w.prod}, over {MAX_REPEAT}: {w.p[w.i:end]} (RE2's limit)")
    w.most = max(w.most, w.prod)
    return least, most


def _repeat(w: _Walk) -> None:
    """A quantifier at ``w.i``: ``*``, ``+``, ``?`` or a ``{...}`` that stdlib reads as a repeat."""
    p = w.p
    end = w.i + 1
    if p[w.i] == "{":
        end = repeat_end(p, w.i)
        if not end:
            w.atom(1)  # a literal brace
            return
    if w.before != _ATOM:
        raise w.fail("nothing to repeat" + (" (stdlib does not repeat an anchor)" if w.before == _ANCHOR else ""))
    if p[w.i] == "{":
        _counts(w, end)
    if p.startswith("?", end):
        end += 1  # lazy
    elif p.startswith("+", end):
        raise w.fail(f"possessive quantifier {p[w.i:end + 1]} (RE2 has no possessive quantifier)")
    nxt = p[end:end + 1]
    if nxt and (nxt in "*+?" or (nxt == "{" and repeat_end(p, end))):
        raise w.fail("multiple repeat")
    w.i, w.before = end, _NOTHING


def _bar(w: _Walk) -> None:
    w.i += 1
    w.before = _NOTHING


def _anchor(w: _Walk) -> None:
    w.i += 1
    w.before = _ANCHOR


def _surrogate(w: _Walk) -> None:
    w.literal(w.p[w.i])


_HANDLERS = {"\\": _escape, "[": _class, "(": _open, ")": _close, "*": _repeat, "+": _repeat, "?": _repeat,
             "{": _repeat, "|": _bar, "^": _anchor, "$": _anchor}


def check(text: str, latin1: bool, original, looks: bool = False):
    """``(capture groups, ((name, number), ...))`` for the Python-dialect ``text``, or ``error``
    (carrying ``original``, the caller's pattern) for syntax RE2 cannot run.

    ``looks``: read lookarounds as groups instead of refusing them. The edge planner rewrites a
    guard's body into a class or a neighbour test, where a malformed atom (``(?<!*)``, a bad
    escape) would turn into something runnable; a source with guards is therefore read once
    this way BEFORE the rewrite, so the syntax stdlib refuses inside a guard is refused too."""
    w = _Walk(text, latin1, original, looks)
    handlers, n = _HANDLERS, w.n
    if w.lone:
        handlers = dict(handlers)
        handlers.update((ch, _surrogate) for ch in set(text) if "\ud800" <= ch <= "\udfff")
    while w.i < n:
        handler = handlers.get(text[w.i])
        if handler is None:
            w.i += 1
            w.before, w.prod = _ATOM, 1
        else:
            handler(w)
    if w.stack:
        raise error("missing ), unterminated subpattern", original)
    return w.ncap, tuple(w.names)


# stdlib's inline flag letters, the type letters of them, and the extensions that may follow `(?`
_STD_FLAGS = frozenset("aiLmsux")
_STD_TYPES = frozenset("auL")
_STD_EXTENSIONS = frozenset(":P=!<>(")
_STD_ESCAPES = frozenset("pP") if sys.version_info >= (3, 14) else frozenset("pPz")  # 3.14 added `\z`


def _std_flags(text: str, i: int, latin1: bool, fail):
    """Read the flags of the group extension at ``text[i:]`` (just after ``(?``) as stdlib's
    ``_parse_flags`` does: ``(index after it, whether it opens a scoped group)``."""
    n, on, off = len(text), "", ""
    while i < n and text[i] in _STD_FLAGS:
        if text[i] == ("u" if latin1 else "L"):
            raise fail(f"bad inline flags: cannot use '{text[i]}' flag with a {'bytes' if latin1 else 'str'} pattern")
        on += text[i]
        if len(_STD_TYPES.intersection(on)) > 1:
            raise fail("bad inline flags: flags 'a', 'u' and 'L' are incompatible")
        i += 1
    closer = text[i:i + 1]
    if on and closer == ")":
        return i + 1, False
    if closer == "-":
        i += 1
        while i < n and text[i] in _STD_FLAGS:
            if text[i] in _STD_TYPES:
                raise fail("bad inline flags: cannot turn off flags 'a', 'u' and 'L'")
            off += text[i]
            i += 1
        if not off:
            raise fail("missing flag")
        closer = text[i:i + 1]
    if closer != ":":
        raise fail("missing :" if off else "missing -, : or )")
    if set(on) & set(off):
        raise fail("bad inline flags: flag turned on and off")
    return i + 1, True


def _std_extension(text: str, i: int, latin1: bool, fail):
    """The group extension at ``text[i]`` (an ``(`` followed by ``?``): ``(index after its
    opening, whether it is a global flag group)``, or ``error`` where stdlib raises."""
    char = text[i + 2:i + 3]
    if char in _STD_FLAGS or char == "-":
        end, scoped = _std_flags(text, i + 2, latin1, fail)
        return end, not scoped
    if char == "<" and text[i + 3:i + 4] not in ("=", "!"):
        raise fail("unknown extension ?<" + text[i + 3:i + 4])
    if char not in _STD_EXTENSIONS:
        raise fail("unknown extension ?" + char)
    return i + 1, False


def stdlib_syntax(text: str, latin1: bool, original) -> None:
    """Raise ``error`` (carrying ``original``) for syntax in ``text`` (Python dialect, comment
    groups removed) that stdlib refuses and RE2 would run. Everything else is left to ``check``."""
    i, n, lead = 0, len(text), True  # lead: nothing but global flag groups read so far
    depth = 0  # groups open at ``i``

    def fail(message):
        return error(f"{message} (position {i})", original)
    while i < n:
        if depth > MAX_NESTING:  # checked as the next character is read: one past the deepest "("
            raise fail(f"more than {MAX_NESTING} nested groups (stdlib exceeds its recursion limit)")
        ch = text[i]
        if ch == "\\":
            if text[i + 1:i + 2] in _STD_ESCAPES:
                raise fail(f"bad escape {text[i:i + 2]}")
            i += 2
        elif ch == "[":  # a class: skipped (the class translator reads its escapes)
            j = i + 2 if text.startswith("^", i + 1) else i + 1
            j += text.startswith("]", j)
            while j < n and text[j] != "]":
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        elif ch == "(" and text.startswith("?", i + 1):
            end, is_global = _std_extension(text, i, latin1, fail)
            if is_global and not lead:
                raise fail("global flags not at the start of the expression")
            depth += not is_global  # a global flag group is closed already
            i, lead = end, lead and is_global
            continue
        else:
            depth += (ch == "(") - (ch == ")")
            i += 1
        lead = False
