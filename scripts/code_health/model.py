"""Data shapes shared by the measurers and the comparison."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

MODULE_SCOPE = "<module>"


@dataclass
class Unit:
    """One function (Python def, TS function/method/arrow) and its metrics."""

    qualname: str
    line: int
    metrics: dict[str, int]
    body_hash: str
    parent: str | None = None  # nearest enclosing function (nested defs are their own units)
    # Last line of the unit, for comparing bodies of units that are not byte-identical.
    # None when the measurer does not report it (the comparison then estimates the span).
    end_line: int | None = None


@dataclass(frozen=True)
class Hit:
    """A pattern-rule fingerprint: never a line number, so edits above don't move it."""

    rule: str
    scope: str
    text: str


@dataclass
class FileMeasure:
    path: str
    metrics: dict[str, int] = field(default_factory=dict)
    units: dict[str, Unit] = field(default_factory=dict)
    hits: Counter[Hit] = field(default_factory=Counter)
    # Line numbers of each hit occurrence, for reporting only.
    hit_lines: dict[Hit, list[int]] = field(default_factory=dict)
    # 1-based line -> source text, for messages.
    lines: list[str] = field(default_factory=list)
    # 1-based line -> text of the real comment tokens on it (allow directives live only here,
    # so a string literal that happens to say "health: allow" waives nothing).
    comments: dict[int, str] = field(default_factory=dict)
    # Why this file could not be measured (unparseable, measurer failure): a measurement gap
    # must fail the check, never read as "no findings".
    error: str | None = None

    def code_line(self, line_no: int) -> str:
        """The line's code with its comment removed and whitespace normalised: the identity of
        a violation, so editing or dropping a comment (an allow included) never makes old debt
        look new."""
        line = self.source_line(line_no)
        comment = self.comments.get(line_no)
        if comment and comment in line:
            line = line[: line.rindex(comment)]
        return " ".join(line.split())

    def add_hit(self, rule: str, scope: str, line_no: int) -> None:
        hit = Hit(rule, scope, self.code_line(line_no))
        self.hits[hit] += 1
        self.hit_lines.setdefault(hit, []).append(line_no)

    def source_line(self, line_no: int) -> str:
        if 1 <= line_no <= len(self.lines):
            return self.lines[line_no - 1]
        return ""


@dataclass
class Finding:
    path: str
    rule: str
    scope: str
    line: int
    detail: str
    blocking: bool = True
    allowed_reason: str | None = None
