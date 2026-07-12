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


def _extend_id_lines(lines: list[str], label: str, ids: Sequence[str]) -> None:
    if ids:
        lines.append(f"- {label}: {', '.join(f'`{item}`' for item in ids)}")
    else:
        lines.append(f"- {label}: none")


@dataclass(frozen=True)
class NoteIndexEntry:
    """One markdown note in a local durable-note index."""

    path: str
    text: str


@dataclass(frozen=True)
class MemoryRecoveryWrite:
    """Side-effect-free snapshot of one durable-memory write.

    The snapshot intentionally carries only enough state to validate backup,
    sync, and recovery invariants.  Callers can feed real provider/Obsidian
    records into this type without mutating any live memory surface.
    """

    id: str
    content: str
    important: bool = False
    pinned: bool = False
    journaled: bool = False
    synced: bool = False
    local_indexed: bool = False
    durable_note_terms: tuple[str, ...] = ()
    conflict_key: str = ""

    @property
    def needs_durable_protection(self) -> bool:
        """Whether this write must survive GC, sync loss, and restarts."""
        return self.important or self.pinned


@dataclass
class MemoryBackupRecoveryReport:
    """Backup/sync/recovery status without exposing private memory content."""

    missing_journal_ids: tuple[str, ...] = ()
    missing_durable_note_ids: tuple[str, ...] = ()
    missing_local_index_ids: tuple[str, ...] = ()
    retryable_write_ids: tuple[str, ...] = ()
    recoverable_index_ids: tuple[str, ...] = ()
    unrecoverable_index_ids: tuple[str, ...] = ()
    recovery_sources_by_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    protected_from_gc_ids: tuple[str, ...] = ()
    unresolved_conflicts: tuple[ContextConflict, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether durable backup/sync/recovery checks are fully green."""
        return (
            not self.missing_journal_ids
            and not self.missing_durable_note_ids
            and not self.missing_local_index_ids
            and not self.unresolved_conflicts
        )

    def to_markdown(self) -> str:
        """Render redacted recovery diagnostics for operator handoffs.

        The report intentionally emits stable write ids, counters, timestamps,
        and conflict keys only.  It must not echo memory contents, note text, or
        conflicting fact values.
        """
        lines = ["# Memory backup recovery report", ""]
        lines.append("## Diagnostics")
        for key in (
            "recovery_status",
            "last_successful_sync_at",
            "last_garbage_collection_run_at",
            "queued_write_count",
            "conflict_count",
            "protected_memory_count",
        ):
            lines.append(f"- {key}: {self.diagnostics.get(key, 'unknown')}")
        checks = self.diagnostics.get("checks")
        if isinstance(checks, Mapping):
            for key, value in sorted(checks.items()):
                lines.append(f"- check.{key}: {value}")
        lines.append("")

        lines.append("## Write ids")
        _extend_id_lines(lines, "missing journal", self.missing_journal_ids)
        _extend_id_lines(lines, "missing durable note", self.missing_durable_note_ids)
        _extend_id_lines(lines, "missing local index", self.missing_local_index_ids)
        _extend_id_lines(lines, "retryable writes", self.retryable_write_ids)
        _extend_id_lines(lines, "recoverable local index", self.recoverable_index_ids)
        _extend_id_lines(lines, "unrecoverable local index", self.unrecoverable_index_ids)
        _extend_id_lines(lines, "protected from GC", self.protected_from_gc_ids)
        lines.append("")

        lines.append("## recovery sources")
        if self.recovery_sources_by_id:
            for write_id, sources in sorted(self.recovery_sources_by_id.items()):
                lines.append(f"- {write_id}: {', '.join(sources)}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Conflict keys")
        if self.unresolved_conflicts:
            for conflict in self.unresolved_conflicts:
                lines.append(f"- `{conflict.key}` requires clarification")
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)


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


def build_memory_backup_recovery_report(
    writes: Iterable[MemoryRecoveryWrite],
    *,
    note_index: LocalNoteIndex | None = None,
    last_successful_sync_at: str = "",
    last_gc_run_at: str = "",
) -> MemoryBackupRecoveryReport:
    """Validate durable-memory backup/sync/recovery invariants.

    This is a pure checker for scheduler/app tests: it never writes memory,
    Obsidian notes, journals, or local indexes.  The returned diagnostics carry
    counts and stable write ids only, not private memory text.
    """
    snapshots = tuple(writes)
    note_index = note_index or LocalNoteIndex(())

    missing_journal: list[str] = []
    missing_notes: list[str] = []
    missing_index: list[str] = []
    retryable: list[str] = []
    recoverable: list[str] = []
    unrecoverable: list[str] = []
    recovery_sources: dict[str, tuple[str, ...]] = {}
    protected: list[str] = []
    conflict_groups: dict[str, list[str]] = {}

    for write in snapshots:
        if write.needs_durable_protection:
            protected.append(write.id)
            if not write.journaled:
                missing_journal.append(write.id)

            note_terms = write.durable_note_terms or (write.content,)
            note_hits = note_index.search(write.id, note_terms)
            if not note_hits:
                missing_notes.append(write.id)

            if not write.local_indexed:
                missing_index.append(write.id)
                sources: list[str] = []
                if write.journaled:
                    sources.append("journal")
                if note_hits:
                    sources.append("durable_note")
                if write.journaled or note_hits:
                    recoverable.append(write.id)
                    recovery_sources[write.id] = tuple(sources)
                else:
                    unrecoverable.append(write.id)

        if write.journaled and not write.synced:
            retryable.append(write.id)

        if write.conflict_key:
            conflict_groups.setdefault(write.conflict_key, []).append(write.content)

    conflicts = detect_context_conflicts(conflict_groups)
    diagnostics = {
        "last_successful_sync_at": last_successful_sync_at or "unknown",
        "last_garbage_collection_run_at": last_gc_run_at or "unknown",
        "queued_write_count": len(retryable),
        "conflict_count": len(conflicts),
        "protected_memory_count": len(protected),
        "recovery_status": "ok"
        if not (missing_journal or missing_notes or missing_index or conflicts)
        else "needs_attention",
        "checks": {
            "missing_journal": len(missing_journal),
            "missing_durable_note": len(missing_notes),
            "missing_local_index": len(missing_index),
            "recoverable_index": len(recoverable),
            "unrecoverable_index": len(unrecoverable),
        },
    }

    return MemoryBackupRecoveryReport(
        missing_journal_ids=tuple(missing_journal),
        missing_durable_note_ids=tuple(missing_notes),
        missing_local_index_ids=tuple(missing_index),
        retryable_write_ids=tuple(retryable),
        recoverable_index_ids=tuple(recoverable),
        unrecoverable_index_ids=tuple(unrecoverable),
        recovery_sources_by_id=recovery_sources,
        protected_from_gc_ids=tuple(protected),
        unresolved_conflicts=conflicts,
        diagnostics=diagnostics,
    )


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
