"""The ``Match`` object: stdlib `re`'s Match API over a list of spans in buffer offsets.

``spans[k]`` is group ``k`` as ``(start, end)`` offsets into the buffer RE2 scanned (``(-1, -1)``
when the group did not take part), already in the caller's group numbering. For a non-ASCII
``str`` subject the offsets are UTF-8 byte offsets and are converted to character indices lazily.
"""
from __future__ import annotations

import operator
import types

from . import _offsets
from ._subject import DIRECT, UTF8, flat
from ._template import template as _template
from ._translate_scan import close_ranks as _close_ranks


class Match:
    __slots__ = ("_data", "_mode", "_offsets", "_shift", "_spans", "endpos", "pos", "re", "string")
    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, pattern, string, data, mode, spans, pos, endpos, offsets=None, shift=0):
        self.re, self.string, self._mode, self._spans = pattern, string, mode, spans
        # A binary subject is sliced afresh, as stdlib does, so a Match never pins a view of the
        # caller's buffer (a bytearray stays resizable while the Match is alive).
        self._data = data if mode != DIRECT or isinstance(string, str) else None
        self._shift = shift  # character index of the buffer's first character (a window of the subject)
        size = len(string) if isinstance(string, (str, bytes, bytearray)) else len(flat(string))
        self.pos = 0 if pos <= 0 else min(pos, size)
        self.endpos = size if endpos >= size else max(endpos, 0)
        self._offsets = offsets  # sparse offset index of the subject (None for byte == char offsets)

    def _index(self, group):
        if group.__class__ is int or hasattr(type(group), "__index__"):  # as stdlib: True is group 1
            if 0 <= operator.index(group) <= self.re.groups:
                return operator.index(group)
        else:
            index = self.re._names.get(group)
            if index is not None:
                return index
        raise IndexError("no such group")

    def span(self, group=0):
        a, e = self._spans[self._index(group)]
        if a < 0:
            return a, e
        shift = self._shift
        if self._mode == DIRECT:
            return (a + shift, e + shift) if shift else (a, e)
        offsets = self._offsets
        if offsets is None:  # uncached subject scanned from the start: index on first span request
            offsets = self._offsets = _offsets.Offsets(self._data)
        start = offsets.char(a) + shift
        return start, (start + _offsets.to_chars(self._data, a, e) if e > a else start)

    def start(self, group=0):
        return self.span(group)[0]

    def end(self, group=0):
        return self.span(group)[1]

    def _text(self, group, default=None):
        a, e = self._spans[self._index(group)]
        if a < 0:
            return default
        if self._mode == DIRECT:
            string, shift = self.string, self._shift
            if isinstance(string, (str, bytes)):
                return string[a + shift:e + shift] if shift else string[a:e]
            # as stdlib: any other bytes-like subject yields bytes, its offsets counting bytes
            return bytes(flat(string)[a:e])
        if self._mode == UTF8:
            return str(self._data[a:e], "utf-8")
        a, e = self.span(group)
        return self.string[a:e]

    def group(self, *groups):
        if len(groups) < 2:
            return self._text(groups[0] if groups else 0)
        return tuple(self._text(g) for g in groups)

    def __getitem__(self, group):
        return self._text(group)

    def groups(self, default=None):
        return tuple(self._text(g, default) for g in range(1, self.re.groups + 1))

    def groupdict(self, default=None):
        return {name: self._text(index, default) for name, index in self.re._names.items()}

    def expand(self, template):
        fill = _template(template, self.re, False)
        return fill(self) if callable(fill) else fill

    def __copy__(self):  # as stdlib: a Match is immutable, so a copy is itself
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        raise TypeError("cannot pickle 're.Match' object")

    @property
    def lastindex(self):
        # The group that closed last, read from the spans. Its end is the latest end, ``last``. A
        # group inside a trailing lookahead closes when the lookahead runs, at the end of the match
        # (``stop``), however far past it the group's text reaches. Among the groups ending at
        # ``last`` a group closes after every group nested in it, so only the outermost ones
        # compete; they do not overlap, so at most one of them took text (two that did would
        # overlap at ``last``), and every EMPTY one closed after it (an empty group that closed
        # first would put the other's start at ``last``), even in an earlier `(` position (a later
        # iteration of a repeat). Two outermost empty groups close in pattern order: the compile
        # step refuses the repeats where a later iteration could reverse that (``_close_order``).
        pattern = self.re
        ranks = pattern._ranks
        if ranks is None:
            source = pattern.pattern
            ranks = pattern._ranks = _close_ranks(source if pattern._is_str else source.decode("latin-1"))
        spans, stop = self._spans, self._spans[0][1]
        ends = [min(spans[index][1], stop) for index in range(pattern.groups + 1)]
        last = max(ends[1:], default=-1)
        if last < 0:
            return None
        tied = [index for index in range(1, pattern.groups + 1) if ends[index] == last]
        best, best_key = None, None
        for index in tied:
            # a tied group with a lower number and a later `)` encloses this one
            if any(other < index and ranks[other] > ranks[index] for other in tied):
                continue
            key = (spans[index][0] >= last, ranks[index])
            if best_key is None or key > best_key:
                best, best_key = index, key
        return best

    @property
    def lastgroup(self):
        index = self.lastindex
        for name, number in self.re._names.items():
            if number == index:
                return name
        return None

    @property
    def regs(self):
        return tuple(self.span(g) for g in range(self.re.groups + 1))

    def __repr__(self):
        return f"<re.Match object; span={self.span()}, match={self.group()!r}>"
