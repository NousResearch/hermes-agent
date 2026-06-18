"""Side-effect-free context-retention validation helpers.

These helpers let product-specific assistants (for example JARVIS) build an
executable "never lose important context" suite without mutating live memory
stores.  They intentionally separate the surfaces that can retain context:
working memory, durable curated memory, durable note/index recall, discarded
context, and unresolved conflicts that should trigger clarification instead of
silent overwrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

SURFACE_DURABLE_MEMORY = "durable_memory"
SURFACE_WORKING_MEMORY = "working_memory"
SURFACE_DURABLE_NOTES = "durable_notes"
SURFACE_DISCARDED_CONTEXT = "discarded_context"


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Normalize text for deterministic term matching."""
    return _WHITESPACE_RE.sub(" ", text.casefold()).strip()


def _terms_match(text: str, terms: Sequence[str]) -> bool:
    """Return true when every expected term is present in ``text``."""
    normalized = _normalize(text)
    return all(_normalize(term) in normalized for term in terms if term)


@dataclass(frozen=True)
class ContextHit:
    """A single expectation match on one retention surface."""

    key: str
    surface: str
    entry: str
    source: str = ""


@dataclass(frozen=True)
class ContextConflict:
    """Conflicting values for a durable context key.

    ``values`` are the candidate facts/answers that disagree.  Callers should
    ask the user for clarification before overwriting durable memory.
    """

    key: str
    values: tuple[str, ...]


@dataclass
class ContextValidationReport:
    """Retention report split by memory surface.

    ``missing`` records expected context that was not found on the required
    surface. ``unexpected_retained`` records filler/discardable context that
    leaked into a retained surface.
    """

    durable_memory: dict[str, tuple[ContextHit, ...]] = field(default_factory=dict)
    working_memory: dict[str, tuple[ContextHit, ...]] = field(default_factory=dict)
    durable_notes: dict[str, tuple[ContextHit, ...]] = field(default_factory=dict)
    discarded_context: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unresolved_conflicts: tuple[ContextConflict, ...] = ()
    missing: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unexpected_retained: dict[str, tuple[ContextHit, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether all expectations passed and no conflict needs clarification."""
        return (
            not self.missing
            and not self.unexpected_retained
            and not self.unresolved_conflicts
        )

    @property
    def requires_clarification(self) -> bool:
        """Whether conflicting durable context should stop silent overwrite."""
        return bool(self.unresolved_conflicts)

    def to_markdown(self) -> str:
        """Render a compact validation report for Linear/vault handoffs."""
        lines = ["# Context retention validation report", ""]
        lines.extend(_render_hit_section("Durable memory", self.durable_memory))
        lines.extend(_render_hit_section("Working memory", self.working_memory))
        lines.extend(_render_hit_section("Durable notes", self.durable_notes))

        lines.append("## Discarded context")
        if self.discarded_context:
            for key, terms in sorted(self.discarded_context.items()):
                lines.append(f"- `{key}` discarded as expected: {', '.join(terms)}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Unresolved conflicts")
        if self.unresolved_conflicts:
            for conflict in self.unresolved_conflicts:
                joined = " | ".join(conflict.values)
                lines.append(f"- `{conflict.key}` requires clarification: {joined}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Failures")
        if not self.missing and not self.unexpected_retained:
            lines.append("- none")
        for key, terms in sorted(self.missing.items()):
            lines.append(f"- `{key}` missing expected terms: {', '.join(terms)}")
        for key, hits in sorted(self.unexpected_retained.items()):
            sources = ", ".join(hit.surface for hit in hits)
            lines.append(f"- `{key}` unexpectedly retained on: {sources}")
        lines.append("")
        return "\n".join(lines)


def _render_hit_section(
    title: str,
    hits_by_key: Mapping[str, Sequence[ContextHit]],
) -> list[str]:
    lines = [f"## {title}"]
    if not hits_by_key:
        lines.append("- none")
        lines.append("")
        return lines

    for key, hits in sorted(hits_by_key.items()):
        rendered_sources = []
        for hit in hits:
            source = f" ({hit.source})" if hit.source else ""
            rendered_sources.append(f"{hit.surface}{source}")
        lines.append(f"- `{key}` found in {', '.join(rendered_sources)}")
    lines.append("")
    return lines


@dataclass(frozen=True)
class NoteIndexEntry:
    """One markdown note in a local durable-note index."""

    path: str
    text: str


@dataclass(frozen=True)
class LocalNoteIndex:
    """Minimal local markdown-note index for validation tests.

    The index is intentionally simple: it recursively reads ``*.md`` files and
    performs deterministic all-terms substring matching.  It is enough to prove
    that a validation suite can distinguish durable note recall from curated
    memory recall without requiring a vector database or live Obsidian plugin.
    """

    entries: tuple[NoteIndexEntry, ...]

    @classmethod
    def from_path(cls, root: Path | str) -> "LocalNoteIndex":
        base = Path(root)
        entries: list[NoteIndexEntry] = []
        if not base.exists():
            return cls(())

        for path in sorted(base.rglob("*.md")):
            if any(part.startswith(".") for part in path.relative_to(base).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(errors="replace")
            entries.append(NoteIndexEntry(str(path.relative_to(base)), text))
        return cls(tuple(entries))

    def search(self, key: str, terms: Sequence[str]) -> tuple[ContextHit, ...]:
        hits: list[ContextHit] = []
        for entry in self.entries:
            if _terms_match(entry.text, terms):
                hits.append(
                    ContextHit(
                        key=key,
                        surface=SURFACE_DURABLE_NOTES,
                        entry=entry.text,
                        source=entry.path,
                    )
                )
        return tuple(hits)


def build_context_validation_report(
    *,
    durable_memory_entries: Iterable[str] = (),
    working_memory_entries: Iterable[str] = (),
    note_index: LocalNoteIndex | None = None,
    durable_expectations: Mapping[str, Sequence[str]] | None = None,
    working_expectations: Mapping[str, Sequence[str]] | None = None,
    note_expectations: Mapping[str, Sequence[str]] | None = None,
    discarded_expectations: Mapping[str, Sequence[str]] | None = None,
    conflict_candidates: Mapping[str, Sequence[str]] | None = None,
) -> ContextValidationReport:
    """Validate context retention across working, durable, note, and discard surfaces.

    Expectations are ``key -> required terms``.  A durable expectation must
    match curated memory entries, a working expectation must match working
    context entries, and a note expectation must match the supplied local note
    index.  Discarded expectations must *not* match retained surfaces.
    """
    durable_entries = tuple(durable_memory_entries)
    working_entries = tuple(working_memory_entries)
    note_index = note_index or LocalNoteIndex(())

    report = ContextValidationReport(
        unresolved_conflicts=detect_context_conflicts(conflict_candidates or {}),
    )

    for key, terms in (durable_expectations or {}).items():
        hits = _hits_for_entries(key, SURFACE_DURABLE_MEMORY, durable_entries, terms)
        if hits:
            report.durable_memory[key] = hits
        else:
            report.missing[key] = tuple(terms)

    for key, terms in (working_expectations or {}).items():
        hits = _hits_for_entries(key, SURFACE_WORKING_MEMORY, working_entries, terms)
        if hits:
            report.working_memory[key] = hits
        else:
            report.missing[key] = tuple(terms)

    for key, terms in (note_expectations or {}).items():
        hits = note_index.search(key, terms)
        if hits:
            report.durable_notes[key] = hits
        else:
            report.missing[key] = tuple(terms)

    retained_surfaces = (
        (SURFACE_DURABLE_MEMORY, durable_entries),
        (SURFACE_WORKING_MEMORY, working_entries),
    )
    for key, terms in (discarded_expectations or {}).items():
        retained_hits: list[ContextHit] = []
        for surface, entries in retained_surfaces:
            retained_hits.extend(_hits_for_entries(key, surface, entries, terms))
        retained_hits.extend(note_index.search(key, terms))
        if retained_hits:
            report.unexpected_retained[key] = tuple(retained_hits)
        else:
            report.discarded_context[key] = tuple(terms)

    return report


def _hits_for_entries(
    key: str,
    surface: str,
    entries: Iterable[str],
    terms: Sequence[str],
) -> tuple[ContextHit, ...]:
    hits = [
        ContextHit(key=key, surface=surface, entry=entry)
        for entry in entries
        if _terms_match(entry, terms)
    ]
    return tuple(hits)


def detect_context_conflicts(
    conflict_candidates: Mapping[str, Sequence[str]],
) -> tuple[ContextConflict, ...]:
    """Return keys with more than one normalized candidate value."""
    conflicts: list[ContextConflict] = []
    for key, values in conflict_candidates.items():
        unique: dict[str, str] = {}
        for value in values:
            normalized = _normalize(value)
            if normalized:
                unique.setdefault(normalized, value)
        if len(unique) > 1:
            conflicts.append(ContextConflict(key=key, values=tuple(unique.values())))
    return tuple(conflicts)
