"""Run a pattern whose edge guards were rewritten by ``_edge_plan``.

Every result is a list of spans in the USER's numbering: ``[(match start, match end), group 1,
...]``, in the subject buffer's offsets, so the facade's Match class needs no change.
No per-character Python loop: one RE2 call per attempt, two when the search begins at offset 0.

A plan may spell its guards on large Unicode classes cheaply (``_edge_guard``): exact for an ASCII
neighbour, accepting every non-ASCII one. Then each reported match is CHECKED: a guard character
that is not ASCII is put to the plan's tests. When they hold, the report is the exact answer.
When one fails:

  simple plan (one alternative of fixed width)  no match can begin where this one began, so the
        search goes on one character further. Each retry costs the width of the pattern, so the
        scan stays linear.
  any other plan  the exact answer may be another path from the same place, which only RE2 can
        find: the runner switches to ``exact()``, the same plan with the exact classes, for good.
"""
from __future__ import annotations


def back(data, i):
    """Offset of the UTF-8 character that ends at ``i``."""
    i -= 1
    while data[i] & 0xC0 == 0x80:
        i -= 1
    return i


def step(data, i):
    """Offset after the UTF-8 character that starts at ``i``."""
    lead = data[i]
    return i + (1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4)


class _Unsure(Exception):
    """A cheaply spelled guard accepted a neighbour its test refuses, in a plan that is not simple."""


class Edge:
    __slots__ = ("_anchors", "_build", "_exact", "_full", "_plan", "_reals", "_tests", "exact", "groups", "gspec",
                 "lead", "markers", "scan", "start")

    def __init__(self, plan, build, anchors, exact=None):
        """``build(source)`` compiles one Python-dialect source; ``anchors`` is the engine's
        ``(UNANCHORED, ANCHOR_START, ANCHOR_BOTH)``; ``exact()`` returns the runner to switch to
        when a cheap guard cannot be decided here (an object with this class's ``find``)."""
        self._plan, self._build, self._full, self._anchors = plan, build, None, anchors
        self._exact, self.exact = exact, None
        self._tests = (plan.lead_tests, plan.trail_tests, plan.simple) if plan.lead_tests or plan.trail_tests else None
        self.scan = build(plan.scan)
        self.start = build(plan.start) if plan.start is not None else None
        self.lead, self.markers, self.gspec, self.groups = plan.lead, plan.markers, plan.gspec, plan.groups
        # Fast path: no user group encloses a guard, so no span needs clamping.
        clamp = any(lead or trail for _, lead, trail in plan.gspec)
        self._reals = None if clamp else tuple(real for real, _, _ in plan.gspec)

    def _ending(self):
        """The (scan, start) programs that must end at the end of the subject: for fullmatch."""
        if self._full is None:
            scan, start = self._plan.full()
            self._full = (self._build(scan), self._build(start) if start is not None else None)
        return self._full

    def _user(self, spans, begin):
        """Spans in the user's numbering; ``begin`` is where the user's match starts."""
        end = spans[0][1]
        for marker in self.markers:
            crossed = spans[marker][1]
            if crossed >= 0:
                end = crossed
                break
        if self._reals is not None:
            return [(begin, end), *map(spans.__getitem__, self._reals)]
        out = [(begin, end)]
        for real, lead, trail in self.gspec:
            a, e = spans[real]
            if a >= 0:
                if lead and a < begin:  # the group encloses the consumed guard character
                    a = begin
                if trail and e > end:   # the group encloses the consumed suffix
                    e = end
            out.append((a, e))
        return out

    def find(self, data, direct, pos, end, kind=0):
        """The user-numbered spans of the first match at or after ``pos``, else ``None``.
        ``kind``: 0 search, 1 match (must start at ``pos``), 2 fullmatch (must also end at ``end``)."""
        exact = self.exact
        if exact is not None:
            return exact.find(data, direct, pos, end, kind)
        tests = None if direct else self._tests  # an ASCII subject has no neighbour to check
        try:
            spans = self._attempt(self.scan, self.start, data, direct, pos, end, kind, tests)
            if kind == 2 and spans is not None and spans[0][1] != end:
                # RE2 reached the end only by letting the suffix consume text. At the end of the
                # subject a guard must be decided for "nothing follows": the END flavour spells
                # that out. (When the user's match already ends at ``end`` the answer is the same
                # one the END flavour gives: its paths are a subset of these, in the same
                # priority order.)
                scan, start = self._ending()
                spans = self._attempt(scan, start, data, direct, pos, end, 1, tests)
        except _Unsure:
            exact = self.exact = self._exact()
            return exact.find(data, direct, pos, end, kind)
        return spans

    def _holds(self, tests, data, guard, raw, stop) -> bool:
        """Whether the guard characters of a reported match pass the plan's tests: the one at
        ``guard`` (-1: none was consumed) and the one at ``stop``, where the user's match ends,
        when the report runs on past it. An ASCII character needs no test."""
        if guard >= 0 and tests[0] and data[guard] >= 0x80:
            ch = str(data[guard:step(data, guard)], "utf-8")
            if not all(test(ch) for test in tests[0]):
                return False
        if tests[1] and stop < raw and data[stop] >= 0x80:
            ch = str(data[stop:step(data, stop)], "utf-8")
            return all(test(ch) for test in tests[1])
        return True

    def _refused(self, tests, kind) -> bool:
        """After a failed check: ``True`` when there is no match at all (an anchored attempt of
        a simple plan), ``False`` to search on; a plan that is not simple cannot go on here."""
        if not tests[2]:
            raise _Unsure
        return bool(kind)

    def _attempt(self, scan, start, data, direct, pos, end, kind, tests):
        anchor = self._anchors[kind]
        if not self.lead:
            while True:
                spans = scan.Match(anchor, data, pos, end)
                begin = spans[0][0]
                if begin < 0:
                    return None
                out = self._user(spans, begin)
                if tests is None or self._holds(tests, data, -1, spans[0][1], out[0][1]):
                    return out
                if self._refused(tests, kind):
                    return None
                pos = step(data, begin)
        if pos == 0:
            # Offset 0 has no character before it: only the alternatives whose leading guard
            # holds at the start of the text can match there, and a match there is leftmost.
            if start is not None:
                spans = start.Match(self._anchors[kind or 1], data, 0, end)
                if spans[0][0] >= 0:
                    out = self._user(spans, 0)
                    if tests is None or self._holds(tests, data, -1, spans[0][1], out[0][1]):
                        return out
                    self._refused(tests, kind)
            if kind:
                return None
            before = 0
        else:
            before = pos - 1 if direct else back(data, pos)
        while True:
            spans = scan.Match(anchor, data, before, end)
            guard = spans[0][0]
            if guard < 0:
                return None
            begin = guard + 1 if direct else step(data, guard)
            out = self._user(spans, begin)
            if tests is None or self._holds(tests, data, guard, spans[0][1], out[0][1]):
                return out
            if self._refused(tests, kind):
                return None
            before = begin
