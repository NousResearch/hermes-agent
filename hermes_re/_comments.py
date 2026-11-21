"""`(?#...)` comment groups: removed from the pattern text before anything else reads it.

Stdlib reads a comment as: ``(?#``, then everything up to the FIRST ``)`` (so it may hold ``[``
or ``(`` but cannot be nested), and it must be terminated. Inside a bracket class and after a
backslash ``(?#`` is ordinary text. A comment is not an atom: a quantifier after it applies to
what stands before it (``a(?#x)*`` is ``a*``).

Removing the text is therefore right, except where it would JOIN the two sides into a token
stdlib never saw. Three rules keep the result the same as stdlib's reading:

  ``a{2(?#c)}``, ``\\0(?#c)1``   the comment ends a brace run or a numeric escape that the
                                 next character would extend: an empty group ``(?:)`` takes
                                 the comment's place, so the two sides stay apart
  ``((?#c)?:a)``                 stdlib: ``?`` has nothing to repeat -> ``error``
  ``a*(?#c)?``                   stdlib: multiple repeat -> ``error``

``strip`` returns the comment-free text and the cuts needed to map an offset in it back to
the caller's pattern (``origin``). Stdlib-only, no engine.
"""
from __future__ import annotations

from bisect import bisect_right

from ._errors import error
from ._translate_scan import class_end, group_open, repeat_end

SEPARATOR = "(?:)"
_REPEATS = frozenset("*+?{")
_DIGITS = frozenset("0123456789")
_HEX = frozenset("0123456789abcdefABCDEF")
_NAME = frozenset("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
_START, _PAREN, _ATOM, _REPEAT = 0, 1, 2, 3  # what stands just before the scan position
# An OPEN RUN is a token that the next character could still extend: after `\x`, `\u`, `\U` the
# number of hex digits still missing, after backslash + digit up to two more digits, and _BRACE
# after a literal `{` followed by digits and commas only.
_BRACE = -1
_ESCAPE_RUN = {"x": 2, "u": 4, "U": 8, **dict.fromkeys("0123456789", 2)}
_LAST = float("inf")  # sorts after every shift, so a bisect lands after the cut AT an offset


def _comment_end(pattern: str, i: int, original) -> int:
    """Index after the run of comments starting at ``pattern[i:] == '(?#...'``."""
    while pattern.startswith("(?#", i):
        end = pattern.find(")", i)
        if end < 0:
            raise error("missing ), unterminated comment", original)
        i = end + 1
    return i


def _junction(pattern: str, i: int, before: int, run: int, original) -> str:
    """What replaces a comment that ends at ``i``; ``before`` and ``run`` describe the text in
    front of it."""
    nxt = pattern[i:i + 1]
    if nxt in _REPEATS and nxt and not (nxt == "{" and not repeat_end(pattern, i)):
        if nxt == "?" and before == _PAREN:
            raise error("nothing to repeat (a comment separates `(` from `?`)", original)
        if before == _REPEAT:
            raise error("multiple repeat (a comment separates two quantifiers)", original)
        return ""
    joins = nxt in _HEX if run > 0 else (nxt in _DIGITS or nxt in ",}") if run == _BRACE else False
    return SEPARATOR if joins and nxt else ""


def _opener(pattern: str, i: int, original) -> int:
    """Index after the group opener at ``pattern[i] == '('``."""
    end = abs(group_open(pattern, i))
    if end == i + 2:  # `(?` and then nothing the scanner knows: a comment may not finish the opener
        j = end
        while j < len(pattern) and pattern[j] in _NAME:
            j += 1
        if pattern.startswith("(?#", j):
            raise error("missing -, : or ) (a comment inside a group opener)", original)
    return end


def _plain(ch: str, run: int):
    """``(what stands before the next position, open run)`` after the ordinary character ``ch``."""
    if ch == "{":
        run = _BRACE
    elif run == _BRACE:
        run = _BRACE if ch in _DIGITS or ch == "," else 0
    elif run > 0:
        run = run - 1 if ch in _HEX else 0
    return (_REPEAT if ch in "*+?" else _START if ch == "|" else _ATOM), run


def strip(pattern: str, original=None):
    """``(text without comment groups, cuts)``. ``cuts`` is a tuple of ``(offset in text,
    shift)`` pairs for ``origin``; it is empty when the pattern has no comment."""
    if "(?#" not in pattern:
        return pattern, ()
    original = pattern if original is None else original
    out, cuts, size, before, run = [], [], 0, _START, 0
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if pattern.startswith("(?#", i):
            i = _comment_end(pattern, i, original)
            piece = _junction(pattern, i, before, run, original)
            if piece:
                before, run = _ATOM, 0
            cuts.append((size + len(piece), i - size - len(piece)))
        elif ch == "\\":
            piece, i, before, run = pattern[i:i + 2], i + 2, _ATOM, _ESCAPE_RUN.get(pattern[i + 1:i + 2], 0)
        elif ch == "[":
            end = class_end(pattern, i)
            piece, i, before, run = pattern[i:end], end, _ATOM, 0
        elif ch == "(":
            end = _opener(pattern, i, original)
            piece, i, before, run = pattern[i:end], end, (_PAREN if end == i + 1 else _START), 0
        elif ch == "{" and repeat_end(pattern, i):
            end = repeat_end(pattern, i)
            piece, i, before, run = pattern[i:end], end, _REPEAT, 0
        else:
            piece, i = ch, i + 1
            before, run = _plain(ch, run)
        out.append(piece)
        size += len(piece)
    return "".join(out), tuple(cuts)


def origin(cuts: tuple, offset: int) -> int:
    """The offset in the caller's pattern of ``offset`` in the stripped text."""
    if not cuts:
        return offset
    index = bisect_right(cuts, (offset, _LAST)) - 1
    return offset + cuts[index][1] if index >= 0 else offset
