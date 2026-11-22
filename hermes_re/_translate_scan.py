"""Small text scans over a Python-dialect pattern shared by the translator, the validator, the
comment stripper and `lastindex` (stdlib-only, no engine): where a bracket class ends, where a
group's body starts, where a repeat / octal / property escape ends, and the order in which
capture groups close.
"""
from __future__ import annotations

_FLAG_LETTERS = frozenset("aiLmsux-")
_DIGITS = frozenset("0123456789")
_OCTDIGITS = frozenset("01234567")


def class_end(pattern: str, i: int) -> int:
    """Index just past the `]` closing the class that opens at ``pattern[i] == '['``."""
    n = len(pattern)
    j = i + 1
    if j < n and pattern[j] == "^":
        j += 1
    if j < n and pattern[j] == "]":
        j += 1
    while j < n and pattern[j] != "]":
        j += 2 if pattern[j] == "\\" else 1
    return min(j + 1, n)


def has_surrogate(text: str) -> bool:
    """Whether ``text`` holds a lone surrogate (which UTF-8, and so RE2, cannot carry)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def repeat_end(pattern: str, i: int) -> int:
    """For ``pattern[i] == '{'``: the index after a stdlib repeat (``{m}``, ``{m,}``, ``{,n}``,
    ``{m,n}``, ``{,}``), or 0 when stdlib reads the brace as a literal."""
    n = len(pattern)
    j = i + 1
    while j < n and pattern[j] in _DIGITS:
        j += 1
    if j < n and pattern[j] == ",":
        j += 1
        while j < n and pattern[j] in _DIGITS:
            j += 1
    return j + 1 if j < n and pattern[j] == "}" and j > i + 1 else 0


def octal_end(pattern: str, i: int) -> int:
    """For ``pattern[i:]`` = backslash + digit: the index after a stdlib OCTAL escape, or 0 when
    stdlib reads it as a group reference. (``\\0`` + up to two octal digits, or three octal digits.)"""
    if pattern[i + 1] == "0":
        end = i + 2
        while end < min(i + 4, len(pattern)) and pattern[end] in _OCTDIGITS:
            end += 1
        return end
    run = pattern[i + 1:i + 4]
    return i + 4 if len(run) == 3 and _OCTDIGITS.issuperset(run) and run[0] in "0123" else 0


def property_end(pattern: str, i: int) -> int:
    """For ``pattern[i:]`` = ``\\p`` or ``\\P`` (RE2's Unicode property escape): the index after
    ``\\pL`` or ``\\p{Name}``."""
    if not pattern.startswith("{", i + 2):
        return min(i + 3, len(pattern))
    end = pattern.find("}", i)
    return end + 1 if end != -1 else len(pattern)


def comment_end(pattern: str, i: int) -> int:
    """For ``pattern[i:]`` = ``(?#...``: the index after its closing ``)``, or ``-1`` when it is
    not terminated. As in stdlib, a backslash takes the next character with it, so ``\\)`` does
    not end the comment."""
    j, n = i + 3, len(pattern)
    while j < n:
        ch = pattern[j]
        if ch == ")":
            return j + 1
        j += 2 if ch == "\\" else 1
    return -1


def group_open(pattern: str, i: int) -> int:
    """Index after the opener of the group at ``pattern[i] == '('``; negative when the group
    has no body (inline flags ``(?i)``, comment, named backreference): then ``-index after it``."""
    if not pattern.startswith("(?", i):
        return i + 1
    n = len(pattern)
    rest = pattern[i + 2:i + 4]
    if rest[:1] == "#":
        end = comment_end(pattern, i)
        return -(end if end != -1 else n)
    if rest == "P=":
        end = pattern.find(")", i)
        return -(end + 1 if end != -1 else n)
    if rest[:1] in ":=!>":
        return i + 3
    if rest in ("<=", "<!"):
        return i + 4
    if rest == "P<" or rest[:1] == "<":
        end = pattern.find(">", i)
        return end + 1 if end != -1 else n
    if rest[:1] == "(":  # conditional (?(1)a|b)
        end = pattern.find(")", i)
        return end + 1 if end != -1 else n
    j = i + 2
    while j < n and pattern[j] in _FLAG_LETTERS:
        j += 1
    if j < n and pattern[j] == ")":
        return -(j + 1)
    return j + 1 if j < n and pattern[j] == ":" else i + 2


class _Ranks:
    """The scan state of ``close_ranks``; one method per character that needs attention, each
    returning the index to go on from."""
    __slots__ = ("closed", "opened", "pattern", "ranks", "stack")

    def __init__(self, pattern: str):
        self.pattern, self.ranks, self.stack, self.opened, self.closed = pattern, [0], [], 0, 0

    def escape(self, i: int) -> int:
        return i + 2

    def bracket(self, i: int) -> int:
        return class_end(self.pattern, i)

    def open(self, i: int) -> int:
        pattern = self.pattern
        after = group_open(pattern, i)
        if after < 0:
            return -after
        capture = not pattern.startswith("(?", i) or pattern.startswith("(?P<", i) or (
            pattern.startswith("(?<", i) and pattern[i + 3:i + 4] not in ("=", "!"))
        if capture:
            self.opened += 1
            self.ranks.append(0)
        self.stack.append(self.opened if capture else 0)
        return after

    def close(self, i: int) -> int:
        if self.stack:
            index = self.stack.pop()
            if index:
                self.closed += 1
                self.ranks[index] = self.closed
        return i + 1


_RANK_STEPS = {"\\": _Ranks.escape, "[": _Ranks.bracket, "(": _Ranks.open, ")": _Ranks.close}


def close_ranks(pattern: str) -> tuple:
    """For each capture group (index 1..n, entry 0 unused): the order in which its closing
    parenthesis appears in the pattern. Among groups ending at one offset, the one that closes
    last is stdlib's ``lastindex``."""
    state = _Ranks(pattern)
    i, n = 0, len(pattern)
    while i < n:
        step = _RANK_STEPS.get(pattern[i])
        i = step(state, i) if step else i + 1
    return tuple(state.ranks)
