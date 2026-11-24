"""One spelling of a pattern: translated when created, built into RE2 programs on demand.

A ``Form`` is everything ``compile()`` can know without the engine: whether the pattern needs the
edge runner, the RE2 source of each program, the number of capture groups and their names.
``build()`` turns it into ``(Edge runner or None, RE2 program or None)``.

A form is CHEAP by default: edge guards on large Unicode classes are spelled with a small class
and decided by the runner (``_edge_guard``). Its EXACT twin spells them out in full; the runner
asks for it (``_twin``) only when a cheap guard cannot be decided, and ``verify()`` builds it
once, for ``build()`` / ``build_all()``, to prove that it fits RE2's memory budget.
"""
from __future__ import annotations

from . import _edge_plan, _engine
from ._edge_run import Edge
from ._errors import error


class Form:
    __slots__ = ("cheap", "flags", "groupindex", "groups", "latin1", "original", "owner", "plan", "real",
                 "rescue", "sources", "text", "verified")

    def __init__(self, original, text: str, flags: int, latin1: bool, cheap: bool = True, owner=None, rescue=None):
        """``text``: the pattern ``original`` in the Python dialect, without comment groups.
        ``owner``: a weak reference to the Pattern that is charged for the programs.
        ``rescue()``: the runner to use when the exact twin cannot be built (else that raises)."""
        self.original, self.text, self.flags, self.latin1 = original, text, flags, latin1
        self.cheap, self.owner, self.rescue, self.verified = cheap, owner, rescue, False
        plan = self.plan = _edge_plan.plan(text, flags, cheap and not latin1) if _edge_plan.wanted(text) else None
        if plan is None:
            translated, self.groups, names = _engine.source(text, flags, latin1, original)
            self.sources, self.real, self.groupindex = {text: translated}, self.groups, dict(names)
            return
        translated, self.real, names = _engine.source(plan.scan, flags, latin1, original)
        self.sources = {plan.scan: translated}
        if plan.start is not None:
            self.sources[plan.start] = _engine.source(plan.start, flags, latin1, original)[0]
        # The scan source numbers its groups with the hidden markers in between; stdlib orders
        # groupindex (and so groupdict()) by group number.
        user = {real: number for number, (real, _, _) in enumerate(plan.gspec, 1)}
        self.groups, self.groupindex = plan.groups, {name: user[real] for name, real in names}

    def _program(self, text: str):
        """The RE2 program for one Python-dialect source of this form (the END flavours of a
        guarded pattern are translated here, when ``fullmatch`` first needs them). The owning
        Pattern is charged for it."""
        translated = self.sources.get(text)
        if translated is None:
            translated = _engine.source(text, self.flags, self.latin1, self.original)[0]
        compiled = _engine.program(translated, self.flags, self.latin1, self.original)
        owner = self.owner() if self.owner is not None else None
        if owner is not None:
            owner._grew(compiled.ProgramSize())
        return compiled

    def _needs_twin(self) -> bool:
        """Whether matching can ever ask for the exact twin: some guard is spelled cheaply and
        the plan is not one the runner settles by itself."""
        plan = self.plan
        return plan is not None and bool(plan.lead_tests or plan.trail_tests) and not plan.simple

    def _twin(self):
        """The runner of the exact twin (built now), or of ``rescue()`` when RE2 refuses it."""
        try:
            return Form(self.original, self.text, self.flags, self.latin1, False, self.owner).build()[0]
        except error:
            if self.rescue is None:
                raise
            return self.rescue()

    def verify(self) -> None:
        """Raise ``error`` now if the exact twin could not be built when matching asks for it.
        The twin built here is dropped again: only the verdict is kept."""
        if not self.verified and self._needs_twin():
            Form(self.original, self.text, self.flags, self.latin1, False).build()
        self.verified = True

    def build(self):
        """``(Edge runner or None, RE2 program or None)``. Raises ``error`` when RE2 refuses a
        program (the memory budget), ``ModuleNotFoundError`` when google-re2 cannot be imported."""
        _engine.load()
        if self.plan is None:
            edge, first = None, self._program(self.text)
            rx = first
        else:
            edge = Edge(self.plan, self._program, _engine.ANCHORS, self._twin if self._needs_twin() else None)
            rx, first = None, edge.scan
        if first.NumberOfCapturingGroups() != self.real:  # the validator and RE2 must agree
            raise error(f"internal: {self.real} capture groups expected, RE2 reports "
                        f"{first.NumberOfCapturingGroups()}", self.original)
        return edge, rx
