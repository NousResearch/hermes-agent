r"""The compiled ``Pattern``: stdlib `re`'s Pattern API over one or more RE2 programs.

``Pattern(...)`` does everything that needs no engine at once: comment groups are removed, edge
guards are planned, every source is checked and translated (EAGER: a pattern the facade or RE2
cannot run raises ``error`` from ``compile()``), and ``groups`` / ``groupindex`` are filled. The
RE2 programs are built on FIRST USE (``_build``): RE2 programs cannot be persisted and a
Unicode-heavy one takes hundreds of milliseconds, which a module-level ``compile()`` must not
cost at import. What can still fail at first use: ``error`` for a program over the memory
budget, ``ModuleNotFoundError`` when google-re2 cannot be imported (``_engine.engine_missing``).
``_cache.build`` / ``build_all`` force the build, for tests.

A built pattern has one of two shapes:
  plain    one RE2 program (``_rx``); RE2's spans are the caller's spans.
  guarded  an ``_edge_run.Edge`` runner (``_edge``): the pattern has an edge guard (leading
           lookbehind, trailing lookahead, tail `$`) that RE2 matches by consuming a neighbour.
A ``str`` pattern with a `\b` that can be made Unicode-exact also has a WIDE form (``_wide``):
the same pattern with that `\b` respelled as edge guards on `\w`. RE2's own ASCII `\b` is
exact on ASCII subjects, so the wide form is built on, and used for, the first non-ASCII subject.

Cost: every RE2 program built for a pattern is added to ``_cost`` (``_grew``), which the compile
cache's memory bound reads (``_budget``).

Threads: ``_build`` works on local values and stores ``_built`` last, so a thread that sees
``_built`` sees a complete pattern; two threads may both build (the second result replaces an
equal first one). ``_live`` holds every Pattern weakly, for ``build_all``.

Fork: RE2 guards each program's DFA cache with a mutex, and a thread of the parent may hold one
at the moment of ``fork()``; in the child that mutex is never released. ``_after_fork`` (run in
the child, registered when the first program is built) therefore makes every live Pattern unbuilt again and
parks the inherited programs where nothing touches or frees them; the child builds its own on
first use.
"""
from __future__ import annotations

import os
import sys
import types
import weakref

from . import _budget, _comments, _edge_bound, _engine
from ._edge_run import step
from ._errors import error
from ._flags import RegexFlag, checked
from ._form import Form
from ._match import Match
from ._offsets import Offsets
from ._subject import DIRECT, UTF8, WINDOW, held, subject, window
from ._template import template as _template

_LAZY = frozenset(("_anchors", "_edge", "_rx"))  # slots filled by the build
_live = weakref.WeakSet()  # every Pattern alive: the compile cache's and the ones modules hold
_parked = []  # programs inherited through fork(): kept alive, never used, never freed
_fork_hook = []  # holds one item once ``_after_fork`` is registered with the interpreter


def live() -> list:
    """Every Pattern object alive now."""
    for _ in range(8):  # another thread creating a Pattern can interrupt the iteration
        try:
            return list(_live)
        except RuntimeError:
            continue
    return list(_live)


def _iterate(scan, unanchored, data, direct, start, end):
    """Successive RE2 span lists of a plain pattern; an empty match advances one character."""
    while start <= end:
        spans = scan(unanchored, data, start, end)
        first, last = spans[0]
        if first < 0:
            return
        yield spans
        if last > first:
            start = last
        elif direct or last >= end:
            start = last + 1
        else:
            start = step(data, last)


def _aligned(edge, rx, unanchored, data, start, end):
    """The first match at or after ``start`` that does not sit INSIDE a character, else ``None``.

    RE2 decides `\\B` by its ASCII rule byte by byte, and between two bytes of one UTF-8 character
    neither neighbour is a word byte: `\\B` holds there. A pattern that can match the empty string
    through `\\B` is then reported in the middle of a character. Such a match is not a position
    of the caller's text: skip it. (Only an EMPTY match can sit there; nothing consumes from a
    continuation byte.)"""
    size = len(data)
    while start <= end:
        if edge is not None:
            spans = edge.find(data, False, start, end)
        else:
            spans = rx.Match(unanchored, data, start, end)
            if spans[0][0] < 0:
                return None
        if spans is None:
            return None
        at, stop = spans[0]
        if at != stop or at >= size or data[at] & 0xC0 != 0x80:
            return spans
        start = at + 1
        while start < size and data[start] & 0xC0 == 0x80:
            start += 1
    return None


def _iterate_aligned(edge, rx, unanchored, data, start, end):
    """``_iterate`` for a pattern with `\\B` on a non-ASCII subject (see ``_aligned``)."""
    while start <= end:
        spans = _aligned(edge, rx, unanchored, data, start, end)
        if spans is None:
            return
        yield spans
        first, last = spans[0]
        start = last if last > first else last + 1 if last >= end else step(data, last)


def _window(string, pos, endpos):
    """``Pattern._open``'s result for the window ``pos..endpos`` of a subject that is not cached:
    only the window is encoded, with the ONE character before it, which is all RE2 and the edge
    runner ever look back at (`^`, `\\b`, a guard). ``endpos`` ends the buffer, as stdlib treats
    it as the end of the subject."""
    if pos > endpos:
        return b"", DIRECT, 1, 0, None, 0, string
    first = pos - 1 if pos else 0
    data, mode, part = window(string, first, endpos)
    start = pos - first
    if start and mode != DIRECT:
        start = step(data, 0)
    return data, mode, start, len(data), None, first, part


class _Narrow:
    """Stands in for the exact twin of a wide form that RE2 cannot hold: the pattern as written,
    with RE2's ASCII `\\b` (what matching has always used for a wide form over the budget)."""
    __slots__ = ("_anchors", "_edge", "_rx")

    def __init__(self, edge, rx, anchors):
        self._edge, self._rx, self._anchors = edge, rx, anchors

    def find(self, data, direct, pos, end, kind=0):
        if self._edge is not None:
            return self._edge.find(data, direct, pos, end, kind)
        spans = self._rx.Match(self._anchors[kind], data, pos, end)
        return None if spans[0][0] < 0 else spans


class Pattern:
    __slots__ = ("__weakref__", "_anchors", "_built", "_cost", "_edge", "_form", "_is_str", "_key", "_nonb",
                 "_ranks", "_rx", "_special", "_wide", "_wide_text", "flags", "groupindex", "groups", "pattern")
    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, pattern, flags):
        self.pattern, self.flags = pattern, RegexFlag(flags)
        is_str = self._is_str = isinstance(pattern, str)
        self._ranks = None
        self._built = False
        self._cost, self._key = 0, None
        if is_str:
            text, engine_flags = pattern, int(flags)
        else:  # bytes: matched byte-wise (Latin-1) with the ASCII classes
            text, engine_flags = pattern.decode("latin-1"), int(flags) | 256
        text = _comments.strip(text, pattern)[0]
        form = self._form = Form(pattern, text, engine_flags, not is_str, True, weakref.ref(self))  # raises ``error``
        self.groups, self.groupindex = form.groups, form.groupindex
        wide = None
        if is_str and "\\b" in text:
            wide = _edge_bound.rewrite(text, engine_flags)[0]
            if wide == text:
                wide = None
        self._wide = self._wide_text = wide  # ``_wide`` stays text until a non-ASCII subject
        self._nonb = is_str and "\\B" in text
        self._special = form.plan is not None or wide is not None or self._nonb
        _live.add(self)

    def _build(self) -> None:
        """Build the RE2 programs. Raises ``error`` when RE2 refuses one (the memory budget),
        ``ModuleNotFoundError`` when google-re2 cannot be imported."""
        if not _fork_hook and hasattr(os, "register_at_fork"):  # the first program of the process
            _fork_hook.append(os.register_at_fork(after_in_child=_after_fork))
        edge, rx = self._form.build()
        self._anchors = _engine.ANCHORS
        self._edge, self._rx = edge, rx
        self._built = True  # last: whoever sees it sees the programs

    def _grew(self, instructions: int) -> None:
        """One more RE2 program of ``instructions`` instructions exists for this pattern."""
        added = _engine.PROGRAM_BYTES + instructions * _engine.INSTRUCTION_BYTES
        self._cost += added
        if self._key is not None:
            _budget.grew(self._key, self, added)

    def _wide_form(self, text: str) -> Form:
        """The Form of the wide spelling. Should its exact twin ever be needed and not fit RE2's
        budget, matching goes back to the pattern as written."""
        owner = weakref.ref(self)

        def rescue():
            compiled = owner()  # alive: it is matching right now
            return _Narrow(compiled._edge, compiled._rx, compiled._anchors)
        return Form(self.pattern, text, int(self.flags), False, True, owner, rescue)

    def _force(self, wide: bool) -> None:
        """Build now, and prove that the exact twin of a cheaply spelled guard fits the budget;
        with ``wide`` also the wide (`\\b`) form, WITHOUT the quiet fall-back to RE2's ASCII
        `\\b` that matching uses when the wide form cannot be built."""
        if not self._built:
            self._build()
        self._form.verify()
        text = self._wide
        if wide and text.__class__ is str:
            form = self._wide_form(text)
            form.rescue = None
            built = form.build()
            form.verify()
            self._wide = built

    def _unbuild(self) -> list:
        """Forget the programs (they are built again on next use); returns them."""
        self._built = False
        dropped = [] if self._wide.__class__ is str else [self._wide]
        self._wide, self._cost = self._wide_text, 0
        for slot in (Pattern._edge, Pattern._rx):  # the slot itself: ``getattr`` would build an empty one
            try:
                dropped.append(slot.__get__(self))
                slot.__delete__(self)
            except AttributeError:
                continue
        return dropped

    def __getattr__(self, name):
        # Reached only for a slot that is not filled: a program slot of an unbuilt pattern.
        if name in _LAZY:
            self._build()
            return getattr(self, name)
        raise AttributeError(name)

    def __eq__(self, other):  # as stdlib: equal when pattern and flags are equal
        if other.__class__ is not Pattern:
            return NotImplemented
        return self.pattern == other.pattern and self.flags == other.flags and self._is_str == other._is_str

    def __hash__(self):
        return hash((self.pattern, int(self.flags)))

    def _for(self, mode):
        """``(Edge runner or None, RE2 program or None)`` for a subject: the wide (Unicode `\\b`)
        form for non-ASCII text, the pattern as written otherwise."""
        wide = self._wide
        if wide is None or mode == DIRECT:
            return self._edge, self._rx
        if wide.__class__ is str:
            try:
                wide = self._wide_form(wide).build()
            except error:  # not plannable, or too large for RE2 with the Unicode classes: `\\b` stays ASCII
                wide = (self._edge, self._rx)
            self._wide = wide
        return wide

    def __repr__(self):
        flags = "|".join(f"re.{f.name}" for f in (RegexFlag.I, RegexFlag.M, RegexFlag.S, RegexFlag.A)
                         if self.flags & f)
        return f"re.compile({self.pattern!r}{', ' + flags if flags else ''})"

    def __reduce__(self):
        # Unpickling goes through the compile cache, as in stdlib. The UNICODE bit that `checked`
        # adds to a str pattern is dropped so the key is the one `compile(pattern)` uses.
        flags = int(self.flags) & ~32 if self._is_str else int(self.flags)
        return cached, (self.pattern.__class__, self.pattern, flags)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def _open(self, string, pos, endpos):
        """``(buffer, mode, byte pos, byte endpos, offset index, shift, text source)`` or raise
        the TypeError stdlib raises. ``shift`` is the character index of the buffer's first
        character and ``text source`` the ``str`` / bytes the buffer stands for: the subject
        itself, or a window of it (``_window``).
        Every matching operation starts here, so this is where an unbuilt pattern builds itself."""
        if not self._built:
            self._build()
        if isinstance(string, str):
            if not self._is_str:
                raise TypeError("cannot use a bytes pattern on a string-like object")
        elif self._is_str:
            raise TypeError("cannot use a string pattern on a bytes-like object"
                            if isinstance(string, (bytes, bytearray, memoryview))
                            else f"expected string or bytes-like object, got {type(string).__name__!r}")
        size = len(string)
        pos = 0 if pos <= 0 else min(pos, size)
        endpos = size if endpos >= size else max(endpos, 0)
        if not self._is_str:
            data, mode, offsets = string, DIRECT, None
        elif (pos or endpos < size) and endpos - pos <= WINDOW and not held(string):
            return _window(string, pos, endpos)
        else:
            data, mode, offsets = subject(string)
        if mode != DIRECT:
            if offsets is None and (pos or endpos < size):
                offsets = Offsets(data)  # uncached subject (short, or over the cache limit)
            if pos:
                pos = offsets.byte(pos)
            endpos = len(data) if endpos >= size else offsets.byte(endpos)
        if endpos < len(data):  # stdlib treats endpos as the end of the subject for `$` and `\b`
            data = memoryview(data)[:endpos]
        return data, mode, pos, endpos, offsets, 0, string

    def _one(self, kind, string, pos, endpos):
        """One match attempt. ``kind``: 0 search, 1 match, 2 fullmatch."""
        data, mode, start, end, offsets, shift, _ = self._open(string, pos, endpos)
        if start > end:
            return None
        rx = self._rx
        if self._special:  # one flag test is all a pattern without guards or `\b` pays
            edge, rx = self._for(mode)
            if self._nonb and kind == 0 and mode != DIRECT:
                spans = _aligned(edge, rx, self._anchors[0], data, start, end)
                return None if spans is None else Match(self, string, data, mode, spans, pos, endpos, offsets, shift)
            if edge is not None:
                spans = edge.find(data, mode == DIRECT, start, end, kind)
                return None if spans is None else Match(self, string, data, mode, spans, pos, endpos, offsets, shift)
        spans = rx.Match(self._anchors[kind], data, start, end)
        if spans[0][0] < 0:
            return None
        return Match(self, string, data, mode, spans, pos, endpos, offsets, shift)

    def search(self, string, pos=0, endpos=sys.maxsize):
        return self._one(0, string, pos, endpos)

    def match(self, string, pos=0, endpos=sys.maxsize):
        return self._one(1, string, pos, endpos)

    def fullmatch(self, string, pos=0, endpos=sys.maxsize):
        return self._one(2, string, pos, endpos)

    def _spans(self, data, mode, start, end):
        """An iterator of successive match span lists, in the caller's group numbering."""
        rx = self._rx
        if self._special:
            edge, rx = self._for(mode)
            if self._nonb and mode != DIRECT:
                return _iterate_aligned(edge, rx, self._anchors[0], data, start, end)
            if edge is not None:
                return edge.iterate(data, mode == DIRECT, start, end)
        return _iterate(rx.Match, self._anchors[0], data, mode == DIRECT, start, end)

    def finditer(self, string, pos=0, endpos=sys.maxsize):
        data, mode, start, end, offsets, shift, _ = self._open(string, pos, endpos)  # argument errors raise now
        return (Match(self, string, data, mode, spans, pos, endpos, offsets, shift)
                for spans in self._spans(data, mode, start, end))

    def _texts(self, string, data, mode, offsets):
        """A function ``slice -> text`` over match offsets, and whether it yields UTF-8 bytes
        that must be decoded once at the end."""
        if mode == DIRECT:
            return string.__getitem__, False
        if mode == UTF8:
            return data.__getitem__, True
        to_char = (offsets or Offsets(data)).char

        def standin(span):
            return string[to_char(span.start):to_char(span.stop)]
        return standin, False

    def findall(self, string, pos=0, endpos=sys.maxsize):
        data, mode, start, end, offsets, _, source = self._open(string, pos, endpos)
        take, raw = self._texts(source, data, mode, offsets)
        empty = b"" if raw else string[:0]
        groups, out = self.groups, []
        for spans in self._spans(data, mode, start, end):
            if groups == 0:
                a, e = spans[0]
                out.append(take(slice(a, e)))
            elif groups == 1:
                a, e = spans[1]
                out.append(take(slice(a, e)) if a >= 0 else empty)
            else:
                out.append(tuple(take(slice(a, e)) if a >= 0 else empty for a, e in spans[1:]))
        if not raw:
            return out
        if groups < 2:  # str(buffer, "utf-8"): items are memoryview slices when endpos cut the subject
            return [str(item, "utf-8") for item in out]
        return [tuple(str(part, "utf-8") for part in item) for item in out]

    def split(self, string, maxsplit=0):
        data, mode, start, end, offsets, _, _ = self._open(string, 0, sys.maxsize)
        take, raw = self._texts(string, data, mode, offsets)
        out, last, done = [], 0, 0
        for spans in self._spans(data, mode, start, end):
            a, e = spans[0]
            out.append(take(slice(last, a)))
            out.extend(take(slice(ga, ge)) if ga >= 0 else None for ga, ge in spans[1:])
            last, done = e, done + 1
            if done == maxsplit:
                break
        out.append(take(slice(last, len(data))))
        return [None if item is None else item.decode("utf-8") for item in out] if raw else out

    def subn(self, repl, string, count=0):
        data, mode, start, end, offsets, _, _ = self._open(string, 0, sys.maxsize)
        take, raw = self._texts(string, data, mode, offsets)
        fill = repl if callable(repl) else _template(repl, self)
        out, last, done = [], 0, 0
        for spans in self._spans(data, mode, start, end):
            a, e = spans[0]
            out.append(take(slice(last, a)))
            if callable(fill):
                piece = fill(Match(self, string, data, mode, spans, 0, sys.maxsize, offsets))
            else:
                piece = fill
            out.append(piece.encode("utf-8") if raw else piece)
            last, done = e, done + 1
            if done == count:
                break
        out.append(take(slice(last, len(data))))
        if raw:
            return b"".join(out).decode("utf-8"), done
        return string[:0].join(out), done

    def sub(self, repl, string, count=0):
        return self.subn(repl, string, count)[0]


def cached(kind, pattern, flags) -> Pattern:
    """THE compile cache: one Pattern per (pattern type, pattern, flags), kept in ``_budget``
    (an LRU bounded by entry count and by program size). The flag check runs on a cache miss
    only, so a module-level call with a cached pattern pays one lookup."""
    key = (kind, pattern, flags)
    found = _budget.get(key)
    if found is None:
        found = Pattern(pattern, checked(kind, flags))
        found._key = key
        _budget.put(key, found)
    return found


def _after_fork() -> None:
    """In a forked child: no inherited RE2 program may be used (see the module docstring)."""
    for compiled in live():
        _parked.extend(compiled._unbuild())
    _budget.recount()
