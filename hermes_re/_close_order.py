"""Compile-time refusal of the repeats whose `lastindex` RE2's spans cannot decide (stdlib-only, no engine).

``Match.lastindex`` is the capture group that closed LAST. RE2 reports where each group's last
capture is, not when it closed, and ``Match.lastindex`` reads the order from the spans: of the
groups ending at the latest end, the outermost ones compete, an empty one closed after one that
took text, and two empty ones closed in pattern order. That last rule fails only inside a repeat:
``(?:(?P<y>)b|a(?P<x>))+`` on ``"ab"`` captures ``x`` empty at offset 1 at the END of one iteration
and ``y`` empty at offset 1 at the START of the next, so ``y`` closed last although its `)` comes
first. ``refuse`` raises ``error`` for every repeat where that can happen: two capture groups in
its body, neither inside the other, the earlier one able to close empty before anything of an
iteration is consumed, the later one able to close empty with nothing left to consume in the
iteration, and the later one not certain to close after the earlier one in the same iteration
(a group further on in the same sequence, outside any optional part or alternative).

Everything runs on explicit stacks and each candidate pair is judged from capture numbers: a pattern
nested as deeply as stdlib takes must not hit Python's recursion limit here, nor a long sequence of
groups in one repeat cost a pass per pair.
"""
from __future__ import annotations

from ._errors import error
from ._translate_scan import class_end, group_open, octal_end, property_end, repeat_end

_ZERO_WIDTH = frozenset("bBAZz")
_WIDTH = {"x": 4, "u": 6, "U": 10}
_SIMPLE = {"*": (0, None), "+": (1, None), "?": (0, 1)}


class _Node:
    """``kind``: ``atom`` / ``group`` / ``alt`` / ``seq`` / ``rep``. A group's ``index`` is its
    capture number (0 for a non-capturing group) and ``last`` the highest capture number inside
    it; ``first`` is the lowest capture number the node's text can hold (the next one to open
    where it starts); ``look`` marks a lookaround (zero-width); ``parent`` links the tree."""
    __slots__ = ("first", "index", "items", "kind", "last", "least", "look", "most", "nullable", "parent")

    def __init__(self, kind, first, items=(), nullable=False, index=0, least=1, most: int | None = 1, look=False):
        self.kind, self.first, self.items, self.nullable, self.index = kind, first, list(items), nullable, index
        self.least, self.most, self.look, self.last = least, most, look, index
        self.parent = None
        for item in self.items:
            item.parent = self


def _escape_end(text: str, i: int) -> int:
    nxt = text[i + 1:i + 2]
    if nxt in _WIDTH:
        return i + _WIDTH[nxt]
    if nxt.isdigit():
        return octal_end(text, i) or i + 2
    if nxt in ("p", "P"):
        return property_end(text, i)
    return i + 2


def _quantified(text: str, i: int, node: _Node):
    """``(node, index after it)`` with the quantifier at ``text[i]`` applied, if there is one."""
    ch = text[i:i + 1]
    if ch == "{":
        end = repeat_end(text, i)
        if not end:
            return node, i
        low, comma, high = text[i + 1:end - 1].partition(",")
        least, most = int(low or 0), (int(high) if high else (None if comma else int(low or 0)))
    elif ch and ch in _SIMPLE:
        end = i + 1
        least, most = _SIMPLE[ch]
    else:
        return node, i
    rep = _Node("rep", node.first, [node], least == 0 or node.nullable, least=least, most=most)
    return rep, end + text.startswith("?", end)


def _seq(items: list, following: int) -> _Node:
    return _Node("seq", items[0].first if items else following, items, all(item.nullable for item in items))


def _alt(branches: list) -> _Node:
    return _Node("alt", branches[0].first, branches, any(branch.nullable for branch in branches))


def _read(text: str):
    """``(tree, repeats)`` of a source ``_validate.check`` has accepted (so it is well formed);
    ``repeats`` lists every repeat that can run its body more than once."""
    # per open group: (its group node, finished branches, items of the current branch)
    frames, branches, items, repeats = [], [], [], []
    i, n, groups = 0, len(text), 0
    while i < n:
        ch = text[i]
        if ch == "|":
            branches.append(_seq(items, groups + 1))
            items, i = [], i + 1
            continue
        if ch == "(":
            after = group_open(text, i)
            if after < 0:  # inline flags or a comment: nothing to match, nothing to repeat
                i = -after
                continue
            index = 0
            if not text.startswith("(?", i) or text.startswith("(?P<", i):
                groups += 1
                index = groups
            look = text.startswith(("(?=", "(?!", "(?<=", "(?<!"), i)
            frames.append((_Node("group", index or groups + 1, index=index, look=look), branches, items))
            branches, items, i = [], [], after
            continue
        if ch == ")":
            node, outer_branches, outer_items = frames.pop()
            body = _alt([*branches, _seq(items, groups + 1)])
            node.items, node.nullable, node.last = [body], node.look or body.nullable, groups
            body.parent = node
            branches, items, i = outer_branches, outer_items, i + 1
        elif ch == "[":
            node, i = _Node("atom", groups + 1), class_end(text, i)
        elif ch == "\\":
            node, i = _Node("atom", groups + 1, nullable=text[i + 1:i + 2] in _ZERO_WIDTH), _escape_end(text, i)
        else:
            node, i = _Node("atom", groups + 1, nullable=ch in "^$"), i + 1
        node, i = _quantified(text, i, node)
        if node.kind == "rep" and (node.most is None or node.most > 1):
            repeats.append(node)
        items.append(node)
    return _alt([*branches, _seq(items, groups + 1)]), repeats


def _candidates(body: _Node):
    """``(starts, ends)``: the capture groups in ``body`` (outside lookarounds) that can close
    empty with nothing consumed before them in an iteration / nothing left to consume after."""
    starts, ends, todo = [], [], [(body, True, True)]
    while todo:
        node, first, final = todo.pop()
        if node.kind == "group" and node.look:
            continue
        if node.kind == "group" and node.index and node.nullable:
            if first:
                starts.append(node)
            if final:
                ends.append(node)
        if node.kind == "seq":
            items = node.items
            # the first item that must consume, and the last: items before / after them are clear
            hard = [k for k, item in enumerate(items) if not item.nullable]
            low, high = (hard[0], hard[-1]) if hard else (len(items), -1)
            todo.extend((item, first and k <= low, final and k >= high) for k, item in enumerate(items))
        else:
            todo.extend((item, first, final) for item in node.items)
    return starts, ends


def _certain_from(group: _Node, body: _Node) -> int:
    """The lowest capture number from which on every group before ``group`` (and not around it) is
    certain to be followed by ``group`` in the same pass: the groups in the earlier items of the
    sequences above ``group``, up to the first alternative or optional part (or the repeat body)."""
    low, node = group.index, group
    while node is not body:
        parent = node.parent
        if parent.kind == "alt" and len(parent.items) > 1 or parent.kind == "rep" and parent.least == 0:
            break
        if parent.kind == "seq":
            low = parent.first
        node = parent
    return low


def _check_repeat(rep: _Node, original) -> None:
    body = rep.items[0]
    starts, ends = _candidates(body)
    if not starts or not ends:
        return
    starts.sort(key=lambda group: group.index)
    for other in ends:
        low = _certain_from(other, body)
        for group in starts:
            if group.index >= low:
                break
            if group.last < other.index:  # before it, not around it, and not certain to precede it
                raise error(
                    f"capture groups {group.index} and {other.index} can both close empty at one offset in two "
                    f"iterations of a repeat, and RE2 cannot tell which closed last (Match.lastindex): give one "
                    f"of them text to match or move it out of the repeat", original)


def refuse(text: str, groups: int, original) -> None:
    """Raise ``error`` (carrying ``original``) when a repeat of the Python-dialect ``text`` (comment
    groups removed, accepted by ``_validate.check``) holds two capture groups whose closing order
    the spans cannot give (module docstring)."""
    if groups < 2 or not any(ch in text for ch in "*+{"):
        return
    for rep in _read(text)[1]:
        _check_repeat(rep, original)
