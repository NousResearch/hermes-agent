"""The ``Match`` object: stdlib `re`'s Match API over a list of spans in buffer offsets.

``spans[k]`` is group ``k`` as ``(start, end)`` offsets into the buffer RE2 scanned (``(-1, -1)``
when the group did not take part), already in the caller's group numbering. For a non-ASCII
``str`` subject the offsets are UTF-8 byte offsets and are converted to character indices lazily.
"""
from __future__ import annotations

import types

from . import _offsets
from ._subject import DIRECT, UTF8
from ._template import template as _template
from ._translate_scan import close_ranks as _close_ranks


class Match:
    __slots__ = ("_data", "_mode", "_offsets", "_shift", "_spans", "endpos", "pos", "re", "string")
    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, pattern, string, data, mode, spans, pos, endpos, offsets=None, shift=0):
        self.re, self.string, self._data, self._mode, self._spans = pattern, string, data, mode, spans
        self._shift = shift  # character index of the buffer's first character (a window of the subject)
        size = len(string)
        self.pos = 0 if pos <= 0 else min(pos, size)
        self.endpos = size if endpos >= size else max(endpos, 0)
        self._offsets = offsets  # sparse offset index of the subject (None for byte == char offsets)

    def _index(self, group):
        if group.__class__ is int:
            if 0 <= group <= self.re.groups:
                return group
        else:
            index = self.re.groupindex.get(group)
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
            shift = self._shift
            text = self.string[a + shift:e + shift] if shift else self.string[a:e]
            # as stdlib: any other bytes-like subject (bytearray, memoryview, mmap) yields bytes
            return text if isinstance(text, (str, bytes)) else bytes(text)
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
        return {name: self._text(index, default) for name, index in self.re.groupindex.items()}

    def expand(self, template):
        fill = _template(template, self.re)
        return fill(self) if callable(fill) else fill

    @property
    def lastindex(self):
        # The group that closed last: of those ending latest, the one whose `)` is last in the pattern.
        # A group inside a trailing lookahead closes when the lookahead runs, at the end of the
        # match (``stop``), however far past it the group's text reaches.
        pattern = self.re
        ranks = pattern._ranks
        if ranks is None:
            source = pattern.pattern
            ranks = pattern._ranks = _close_ranks(source if pattern._is_str else source.decode("latin-1"))
        best, best_key = None, (-1, -1)
        stop = self._spans[0][1]
        for index in range(1, pattern.groups + 1):
            end = min(self._spans[index][1], stop)
            if end >= 0 and (end, ranks[index]) > best_key:
                best, best_key = index, (end, ranks[index])
        return best

    @property
    def lastgroup(self):
        index = self.lastindex
        for name, number in self.re.groupindex.items():
            if number == index:
                return name
        return None

    @property
    def regs(self):
        return tuple(self.span(g) for g in range(self.re.groups + 1))

    def __repr__(self):
        return f"<re.Match object; span={self.span()}, match={self.group()!r}>"
