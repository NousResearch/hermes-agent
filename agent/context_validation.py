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
import hashlib
import json
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
    recovery_warnings: tuple[str, ...] = ()
    sync_retry_attempts: int = 0
    next_sync_retry_at: str = ""
    last_sync_error_code: str = ""

    @property
    def needs_durable_protection(self) -> bool:
        """Whether this write must survive GC, sync loss, and restarts."""
        return self.important or self.pinned


def _memory_report_write_id(write: MemoryRecoveryWrite) -> str:
    """Return a diagnostics-safe write id for redacted memory reports.

    Callers should normally provide stable opaque ids.  If a caller accidentally
    passes the raw memory content as the id, the report must not echo it back in
    markdown or diagnostics; use a deterministic hash placeholder instead.
    """
    raw_id = str(write.id).strip() or "memory-write"
    normalized_id = _normalize(raw_id)
    normalized_content = _normalize(write.content)
    if normalized_content and normalized_content in normalized_id:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return f"memory-id-redacted-{digest[:16]}"
    return raw_id


@dataclass(frozen=True)
class _JournaledMemoryWrite:
    """Internal redaction-safe snapshot recovered directly from the WAL."""

    id: str
    target: str
    content: str
    important: bool
    pinned: bool
    synced: bool
    sync_status_known: bool = False
    sync_retry_attempts: int = 0
    next_sync_retry_at: str = ""
    last_sync_error_code: str = ""


_MEMORY_ENTRY_DELIMITER = "\n§\n"
_PINNED_MEMORY_PREFIXES = (
    "pinned memory:",
    "pinned:",
)


def build_memory_startup_recovery_writes(
    memory_dir: Path | str,
    *,
    journal_path: Path | str | None = None,
    local_index_ids: Sequence[str] = (),
    local_index_cache_path: Path | str | None = None,
) -> tuple[MemoryRecoveryWrite, ...]:
    """Load built-in memory files as redaction-safe startup recovery snapshots.

    This adapter is intentionally read-only: it parses the profile's ``MEMORY.md``
    and ``USER.md`` files, optionally correlates them with a write-ahead journal,
    and returns :class:`MemoryRecoveryWrite` records for the pure recovery checker.
    It never writes memory, rebuilds indexes, or touches provider state.

    Generated write ids are stable hashes of the target surface plus normalized
    content, so diagnostics can name recoverable gaps without printing memory
    text.  Journal records can match either by ``id`` / ``write_id`` or by the
    same ``target`` + ``content`` pair used by built-in memory writes.
    """
    base = Path(memory_dir)
    local_ids = set(local_index_ids)
    local_cache_ids, local_cache_warnings = _read_memory_local_index_cache(
        local_index_cache_path
    )
    local_ids.update(local_cache_ids)
    (
        journal_ids,
        journal_keys,
        journal_warnings,
        journaled_writes,
        journaled_by_id,
        journaled_by_key,
    ) = _read_memory_journal(
        journal_path,
        base,
    )
    recovery_warnings = tuple(dict.fromkeys((*local_cache_warnings, *journal_warnings)))

    writes: list[MemoryRecoveryWrite] = []
    seen_ids: set[str] = set()
    seen_contents: set[str] = set()
    for target, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
        for entry in _read_memory_entries(base / filename):
            write_id = _memory_snapshot_id(target, entry)
            normalized = _normalize(entry)
            journal_match = (
                journaled_by_id.get(write_id)
                or journaled_by_key.get((target, normalized))
                or journaled_by_key.get(("", normalized))
            )
            journaled = (
                journal_match is not None
                or write_id in journal_ids
                or (target, normalized) in journal_keys
                or ("", normalized) in journal_keys
            )
            seen_ids.add(write_id)
            seen_contents.add(normalized)
            writes.append(
                MemoryRecoveryWrite(
                    id=write_id,
                    content=entry,
                    important=True,
                    pinned=_looks_pinned_memory(entry),
                    journaled=journaled,
                    synced=journal_match.synced
                    if journal_match and journal_match.sync_status_known
                    else True,
                    local_indexed=write_id in local_ids,
                    recovery_warnings=recovery_warnings,
                    sync_retry_attempts=journal_match.sync_retry_attempts
                    if journal_match
                    else 0,
                    next_sync_retry_at=journal_match.next_sync_retry_at
                    if journal_match
                    else "",
                    last_sync_error_code=journal_match.last_sync_error_code
                    if journal_match
                    else "",
                )
            )
    for journaled_write in journaled_writes:
        normalized = _normalize(journaled_write.content)
        if journaled_write.id in seen_ids or normalized in seen_contents:
            continue
        writes.append(
            MemoryRecoveryWrite(
                id=journaled_write.id,
                content=journaled_write.content,
                important=journaled_write.important,
                pinned=journaled_write.pinned,
                journaled=True,
                synced=journaled_write.synced,
                local_indexed=journaled_write.id in local_ids,
                recovery_warnings=recovery_warnings,
                sync_retry_attempts=journaled_write.sync_retry_attempts,
                next_sync_retry_at=journaled_write.next_sync_retry_at,
                last_sync_error_code=journaled_write.last_sync_error_code,
            )
        )
    return tuple(writes)


def _memory_snapshot_id(target: str, content: str) -> str:
    digest = hashlib.sha256(f"{target}\0{_normalize(content)}".encode("utf-8")).hexdigest()
    return f"{target}-{digest[:16]}"


def _looks_pinned_memory(content: str) -> bool:
    normalized = _normalize(content)
    return normalized.startswith(_PINNED_MEMORY_PREFIXES) or " pinned memory" in normalized


def _journal_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _journal_int(value: object, *, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            return default
    return default


def _read_memory_entries(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(errors="replace")
    return tuple(entry.strip() for entry in raw.split(_MEMORY_ENTRY_DELIMITER) if entry.strip())


def _read_memory_journal(
    journal_path: Path | str | None,
    memory_dir: Path,
) -> tuple[
    set[str],
    set[tuple[str, str]],
    tuple[str, ...],
    tuple[_JournaledMemoryWrite, ...],
    dict[str, _JournaledMemoryWrite],
    dict[tuple[str, str], _JournaledMemoryWrite],
]:
    path = Path(journal_path) if journal_path is not None else memory_dir / "memory-wal.jsonl"
    if not path.exists():
        return set(), set(), (), (), {}, {}

    ids: set[str] = set()
    keys: set[tuple[str, str]] = set()
    warnings: list[str] = []
    journaled_writes: list[_JournaledMemoryWrite] = []
    journaled_by_id: dict[str, _JournaledMemoryWrite] = {}
    journaled_by_key: dict[tuple[str, str], _JournaledMemoryWrite] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return set(), set(), ("memory_journal_unreadable",), (), {}, {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            if "memory_journal_record_unreadable" not in warnings:
                warnings.append("memory_journal_record_unreadable")
            continue
        if not isinstance(record, Mapping):
            if "memory_journal_record_unrecognized" not in warnings:
                warnings.append("memory_journal_record_unrecognized")
            continue
        record_id = str(record.get("write_id") or record.get("id") or "").strip()
        if record_id:
            ids.add(record_id)
        content = record.get("content") or record.get("entry") or record.get("value")
        if isinstance(content, str) and content.strip():
            target = str(record.get("target") or "").strip()
            normalized_content = _normalize(content)
            keys.add((target, normalized_content))
            snapshot_id = record_id or _memory_snapshot_id(target or "memory", content)
            ids.add(snapshot_id)
            pinned = _journal_bool(record.get("pinned")) or _looks_pinned_memory(content)
            important = _journal_bool(record.get("important"), default=True) or pinned
            synced_value = (
                record["synced"]
                if "synced" in record
                else record.get("provider_synced")
            )
            sync_retry_attempts = _journal_int(
                record.get("sync_retry_attempts") or record.get("retry_attempts")
            )
            next_sync_retry_at = str(record.get("next_sync_retry_at") or "").strip()
            last_sync_error_code = str(record.get("last_sync_error_code") or "").strip()
            sync_status_known = (
                "synced" in record
                or "provider_synced" in record
                or bool(sync_retry_attempts or next_sync_retry_at or last_sync_error_code)
            )
            journaled_write = _JournaledMemoryWrite(
                id=snapshot_id,
                target=target,
                content=content,
                important=important,
                pinned=pinned,
                synced=_journal_bool(synced_value),
                sync_status_known=sync_status_known,
                sync_retry_attempts=sync_retry_attempts,
                next_sync_retry_at=next_sync_retry_at,
                last_sync_error_code=last_sync_error_code,
            )
            journaled_writes.append(journaled_write)
            journaled_by_id.setdefault(snapshot_id, journaled_write)
            if record_id:
                journaled_by_id.setdefault(record_id, journaled_write)
            journaled_by_key.setdefault((target, normalized_content), journaled_write)
    return (
        ids,
        keys,
        tuple(warnings),
        tuple(journaled_writes),
        journaled_by_id,
        journaled_by_key,
    )


def _read_memory_local_index_cache(
    cache_path: Path | str | None,
) -> tuple[set[str], tuple[str, ...]]:
    if cache_path is None:
        return set(), ()

    path = Path(cache_path)
    if not path.exists():
        return set(), ()

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set(), ("local_index_cache_unreadable",)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return set(), ("local_index_cache_unreadable",)

    candidates: object
    if isinstance(data, Mapping):
        candidates = (
            data.get("ids")
            or data.get("write_ids")
            or data.get("local_index_ids")
            or data.get("entries")
        )
        if candidates is None and ("id" in data or "write_id" in data):
            candidates = [data]
    else:
        candidates = data

    if isinstance(candidates, Mapping):
        candidates = [candidates]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return set(), ("local_index_cache_unrecognized",)

    ids: set[str] = set()
    for item in candidates:
        if isinstance(item, Mapping):
            value = item.get("write_id") or item.get("id")
        else:
            value = item
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return ids, ()


@dataclass(frozen=True)
class MemoryStartupRecoveryTask:
    """Redacted startup rebuild instruction for a missing local memory index."""

    write_id: str
    sources: tuple[str, ...]
    target_surface: str = "local_index"


@dataclass
class MemoryBackupRecoveryReport:
    """Backup/sync/recovery status without exposing private memory content."""

    missing_journal_ids: tuple[str, ...] = ()
    missing_durable_note_ids: tuple[str, ...] = ()
    missing_local_index_ids: tuple[str, ...] = ()
    missing_sync_checkpoint_ids: tuple[str, ...] = ()
    retryable_write_ids: tuple[str, ...] = ()
    sync_without_journal_ids: tuple[str, ...] = ()
    sync_retry_plan_by_id: dict[str, dict[str, object]] = field(default_factory=dict)
    recoverable_index_ids: tuple[str, ...] = ()
    unrecoverable_index_ids: tuple[str, ...] = ()
    recovery_sources_by_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    startup_recovery_tasks: tuple[MemoryStartupRecoveryTask, ...] = ()
    blocked_gc_delete_ids: tuple[str, ...] = ()
    approved_gc_delete_ids: tuple[str, ...] = ()
    gc_audit_by_id: dict[str, str] = field(default_factory=dict)
    gc_audit_log_id_by_write_id: dict[str, str] = field(default_factory=dict)
    protected_from_gc_ids: tuple[str, ...] = ()
    unresolved_conflicts: tuple[ContextConflict, ...] = ()
    conflict_write_ids_by_key: dict[str, tuple[str, ...]] = field(default_factory=dict)
    recovery_warnings_by_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    obsidian_conflict_ids: tuple[str, ...] = ()
    obsidian_conflict_sources_by_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    redacted_diagnostic_ids: tuple[str, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether durable backup/sync/recovery checks are fully green."""
        privacy_check = self.diagnostics.get("privacy_check")
        privacy_failed = (
            isinstance(privacy_check, Mapping)
            and privacy_check.get("passed") is False
        )
        return (
            not self.missing_journal_ids
            and not self.missing_durable_note_ids
            and not self.missing_local_index_ids
            and not self.missing_sync_checkpoint_ids
            and not self.retryable_write_ids
            and not self.sync_without_journal_ids
            and not self.blocked_gc_delete_ids
            and not self.unresolved_conflicts
            and not self.recovery_warnings_by_id
            and not self.obsidian_conflict_ids
            and not self.redacted_diagnostic_ids
            and not privacy_failed
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
            "last_refinement_run_at",
            "queued_write_count",
            "conflict_count",
            "protected_memory_count",
        ):
            lines.append(f"- {key}: {self.diagnostics.get(key, 'unknown')}")
        privacy_check = self.diagnostics.get("privacy_check")
        if isinstance(privacy_check, Mapping):
            for key in (
                "passed",
                "raw_content_match_count",
                "checked_write_count",
            ):
                lines.append(f"- privacy_check.{key}: {privacy_check.get(key, 'unknown')}")
        checks = self.diagnostics.get("checks")
        if isinstance(checks, Mapping):
            for key, value in sorted(checks.items()):
                lines.append(f"- check.{key}: {value}")
        lines.append("")

        lines.append("## Write ids")
        _extend_id_lines(lines, "missing journal", self.missing_journal_ids)
        _extend_id_lines(lines, "missing durable note", self.missing_durable_note_ids)
        _extend_id_lines(lines, "missing local index", self.missing_local_index_ids)
        _extend_id_lines(lines, "missing sync checkpoint", self.missing_sync_checkpoint_ids)
        _extend_id_lines(lines, "retryable writes", self.retryable_write_ids)
        _extend_id_lines(lines, "sync without journal", self.sync_without_journal_ids)
        _extend_id_lines(lines, "recoverable local index", self.recoverable_index_ids)
        _extend_id_lines(lines, "unrecoverable local index", self.unrecoverable_index_ids)
        _extend_id_lines(lines, "protected from GC", self.protected_from_gc_ids)
        _extend_id_lines(lines, "redacted diagnostic id", self.redacted_diagnostic_ids)
        lines.append("")

        lines.append("## Sync retry plan")
        if self.sync_retry_plan_by_id:
            for write_id, plan in sorted(self.sync_retry_plan_by_id.items()):
                attempts = plan.get("attempts", 0)
                next_retry_at = plan.get("next_retry_at") or "unknown"
                last_error_code = plan.get("last_error_code") or "unknown"
                lines.append(
                    f"- {write_id}: attempts={attempts} "
                    f"next_retry_at={next_retry_at} last_error_code={last_error_code}"
                )
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## recovery sources")
        if self.recovery_sources_by_id:
            for write_id, sources in sorted(self.recovery_sources_by_id.items()):
                lines.append(f"- {write_id}: {', '.join(sources)}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Recovery warnings")
        if self.recovery_warnings_by_id:
            for write_id, warnings in sorted(self.recovery_warnings_by_id.items()):
                lines.append(f"- {write_id}: {', '.join(warnings)}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Obsidian conflicts")
        if self.obsidian_conflict_sources_by_id:
            for write_id, sources in sorted(self.obsidian_conflict_sources_by_id.items()):
                lines.append(f"- {write_id}: {', '.join(sources)}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## startup recovery tasks")
        if self.startup_recovery_tasks:
            for task in self.startup_recovery_tasks:
                lines.append(
                    f"- {task.write_id} -> {task.target_surface} via {', '.join(task.sources)}"
                )
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## GC delete safety")
        if self.blocked_gc_delete_ids:
            lines.append(
                "- blocked protected deletes: "
                + ", ".join(f"`{item}`" for item in self.blocked_gc_delete_ids)
            )
        else:
            lines.append("- blocked protected deletes: none")
        if self.approved_gc_delete_ids:
            for write_id in self.approved_gc_delete_ids:
                rule = self.gc_audit_by_id.get(write_id, "unknown-rule")
                audit_id = self.gc_audit_log_id_by_write_id.get(write_id)
                audit_suffix = f" audit={audit_id}" if audit_id else ""
                lines.append(f"- {write_id} approved by {rule}{audit_suffix}")
        else:
            lines.append("- approved protected deletes: none")
        lines.append("")

        lines.append("## Conflict keys")
        if self.unresolved_conflicts:
            for conflict in self.unresolved_conflicts:
                lines.append(f"- `{conflict.key}` requires clarification")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Conflict write ids")
        if self.conflict_write_ids_by_key:
            for key, write_ids in sorted(self.conflict_write_ids_by_key.items()):
                lines.append(f"- {key}: {', '.join(write_ids)}")
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
    last_refinement_run_at: str = "",
    proposed_gc_delete_ids: Sequence[str] = (),
    explicit_gc_rules: Mapping[str, str] | None = None,
    gc_audit_log_ids: Sequence[str] = (),
    gc_audit_log_id_by_write_id: Mapping[str, str] | None = None,
) -> MemoryBackupRecoveryReport:
    """Validate durable-memory backup/sync/recovery invariants.

    This is a pure checker for scheduler/app tests: it never writes memory,
    Obsidian notes, journals, or local indexes.  The returned diagnostics carry
    counts and stable write ids only, not private memory text.
    """
    snapshots = tuple(writes)
    note_index = note_index or LocalNoteIndex(())
    proposed_gc_deletes = set(proposed_gc_delete_ids)
    gc_rules = explicit_gc_rules or {}
    audited_gc_deletes = set(gc_audit_log_ids)
    gc_audit_log_ids_by_write = {
        str(write_id): str(audit_id).strip()
        for write_id, audit_id in (gc_audit_log_id_by_write_id or {}).items()
        if str(write_id).strip() and str(audit_id).strip()
    }

    missing_journal: list[str] = []
    missing_notes: list[str] = []
    missing_index: list[str] = []
    missing_sync_checkpoint: list[str] = []
    retryable: list[str] = []
    sync_without_journal: list[str] = []
    sync_retry_plans: dict[str, dict[str, object]] = {}
    recoverable: list[str] = []
    unrecoverable: list[str] = []
    recovery_sources: dict[str, tuple[str, ...]] = {}
    startup_tasks: list[MemoryStartupRecoveryTask] = []
    blocked_gc_deletes: list[str] = []
    approved_gc_deletes: list[str] = []
    gc_audit_by_id: dict[str, str] = {}
    gc_approval_audit_log_ids: dict[str, str] = {}
    protected: list[str] = []
    conflict_groups: dict[str, list[str]] = {}
    conflict_id_groups: dict[str, list[str]] = {}
    recovery_warnings: dict[str, tuple[str, ...]] = {}
    obsidian_conflict_ids: list[str] = []
    obsidian_conflict_sources: dict[str, tuple[str, ...]] = {}
    redacted_diagnostic_ids: list[str] = []

    for write in snapshots:
        diagnostic_id = _memory_report_write_id(write)
        if diagnostic_id != (str(write.id).strip() or "memory-write"):
            redacted_diagnostic_ids.append(diagnostic_id)
        if write.synced and not write.journaled:
            sync_without_journal.append(diagnostic_id)
        if write.recovery_warnings:
            recovery_warnings[diagnostic_id] = write.recovery_warnings
        if write.needs_durable_protection:
            protected.append(diagnostic_id)
            if write.synced and not last_successful_sync_at:
                missing_sync_checkpoint.append(diagnostic_id)
            if write.id in proposed_gc_deletes:
                rule = str(gc_rules.get(write.id, "")).strip()
                audit_log_id = gc_audit_log_ids_by_write.get(write.id)
                if rule and (write.id in audited_gc_deletes or audit_log_id):
                    approved_gc_deletes.append(diagnostic_id)
                    gc_audit_by_id[diagnostic_id] = rule
                    if audit_log_id:
                        gc_approval_audit_log_ids[diagnostic_id] = audit_log_id
                else:
                    blocked_gc_deletes.append(diagnostic_id)
            if not write.journaled:
                missing_journal.append(diagnostic_id)

            note_terms = write.durable_note_terms or (write.content,)
            note_hits = note_index.search(diagnostic_id, note_terms)
            if len(note_hits) > 1:
                obsidian_conflict_ids.append(diagnostic_id)
                obsidian_conflict_sources[diagnostic_id] = tuple(hit.source for hit in note_hits)
            if not note_hits:
                missing_notes.append(diagnostic_id)

            if not write.local_indexed:
                missing_index.append(diagnostic_id)
                sources: list[str] = []
                if write.journaled:
                    sources.append("journal")
                if note_hits:
                    sources.append("durable_note")
                if write.journaled or note_hits:
                    recoverable.append(diagnostic_id)
                    recovery_sources[diagnostic_id] = tuple(sources)
                    startup_tasks.append(
                        MemoryStartupRecoveryTask(write_id=diagnostic_id, sources=tuple(sources))
                    )
                else:
                    unrecoverable.append(diagnostic_id)

        if write.journaled and not write.synced:
            retryable.append(diagnostic_id)
            if (
                write.sync_retry_attempts
                or write.next_sync_retry_at
                or write.last_sync_error_code
            ):
                sync_retry_plans[diagnostic_id] = {
                    "attempts": write.sync_retry_attempts,
                    "next_retry_at": write.next_sync_retry_at or "unknown",
                    "last_error_code": write.last_sync_error_code or "unknown",
                }

        if write.conflict_key:
            conflict_groups.setdefault(write.conflict_key, []).append(write.content)
            conflict_id_groups.setdefault(write.conflict_key, []).append(diagnostic_id)

    conflicts = detect_context_conflicts(conflict_groups)
    conflict_write_ids = {
        conflict.key: tuple(conflict_id_groups.get(conflict.key, ()))
        for conflict in conflicts
    }
    diagnostics = {
        "last_successful_sync_at": last_successful_sync_at or "unknown",
        "last_garbage_collection_run_at": last_gc_run_at or "unknown",
        "last_refinement_run_at": last_refinement_run_at or "unknown",
        "queued_write_count": len(retryable),
        "sync_retry_plan_count": len(sync_retry_plans),
        "startup_recovery_task_count": len(startup_tasks),
        "conflict_count": len(conflicts),
        "protected_memory_count": len(protected),
        "gc_delete_audit_count": len(gc_audit_by_id),
        "recovery_warning_count": len(recovery_warnings),
        "obsidian_conflict_count": len(obsidian_conflict_ids),
        "redacted_diagnostic_id_count": len(redacted_diagnostic_ids),
        "recovery_status": "ok"
        if not (
            missing_journal
            or missing_notes
            or missing_index
            or missing_sync_checkpoint
            or retryable
            or sync_without_journal
            or blocked_gc_deletes
            or conflicts
            or recovery_warnings
            or obsidian_conflict_ids
            or redacted_diagnostic_ids
        )
        else "needs_attention",
        "checks": {
            "missing_journal": len(missing_journal),
            "missing_durable_note": len(missing_notes),
            "missing_local_index": len(missing_index),
            "missing_sync_checkpoint": len(missing_sync_checkpoint),
            "sync_without_journal": len(sync_without_journal),
            "sync_retry_plan": len(sync_retry_plans),
            "recoverable_index": len(recoverable),
            "unrecoverable_index": len(unrecoverable),
            "blocked_gc_delete": len(blocked_gc_deletes),
            "approved_gc_delete": len(approved_gc_deletes),
            "conflict_keys": len(conflict_write_ids),
            "recovery_warnings": len(recovery_warnings),
            "obsidian_conflicts": len(obsidian_conflict_ids),
            "redacted_diagnostic_id": len(redacted_diagnostic_ids),
        },
    }
    privacy_payload = json.dumps(
        {
            "missing_journal_ids": missing_journal,
            "missing_durable_note_ids": missing_notes,
            "missing_local_index_ids": missing_index,
            "missing_sync_checkpoint_ids": missing_sync_checkpoint,
            "retryable_write_ids": retryable,
            "sync_without_journal_ids": sync_without_journal,
            "sync_retry_plan_by_id": sync_retry_plans,
            "recoverable_index_ids": recoverable,
            "unrecoverable_index_ids": unrecoverable,
            "recovery_sources_by_id": recovery_sources,
            "startup_recovery_tasks": [
                {
                    "write_id": task.write_id,
                    "sources": task.sources,
                    "target_surface": task.target_surface,
                }
                for task in startup_tasks
            ],
            "blocked_gc_delete_ids": blocked_gc_deletes,
            "approved_gc_delete_ids": approved_gc_deletes,
            "gc_audit_by_id": gc_audit_by_id,
            "gc_audit_log_id_by_write_id": gc_approval_audit_log_ids,
            "protected_from_gc_ids": protected,
            "conflict_keys": [conflict.key for conflict in conflicts],
            "conflict_write_ids_by_key": conflict_write_ids,
            "recovery_warnings_by_id": recovery_warnings,
            "obsidian_conflict_ids": obsidian_conflict_ids,
            "obsidian_conflict_sources_by_id": obsidian_conflict_sources,
            "redacted_diagnostic_ids": redacted_diagnostic_ids,
            "diagnostics": diagnostics,
        },
        sort_keys=True,
        default=str,
    )
    diagnostics["privacy_check"] = _memory_report_privacy_check(
        snapshots,
        privacy_payload,
    )

    return MemoryBackupRecoveryReport(
        missing_journal_ids=tuple(missing_journal),
        missing_durable_note_ids=tuple(missing_notes),
        missing_local_index_ids=tuple(missing_index),
        missing_sync_checkpoint_ids=tuple(missing_sync_checkpoint),
        retryable_write_ids=tuple(retryable),
        sync_without_journal_ids=tuple(sync_without_journal),
        sync_retry_plan_by_id=sync_retry_plans,
        recoverable_index_ids=tuple(recoverable),
        unrecoverable_index_ids=tuple(unrecoverable),
        recovery_sources_by_id=recovery_sources,
        startup_recovery_tasks=tuple(startup_tasks),
        blocked_gc_delete_ids=tuple(blocked_gc_deletes),
        approved_gc_delete_ids=tuple(approved_gc_deletes),
        gc_audit_by_id=gc_audit_by_id,
        gc_audit_log_id_by_write_id=gc_approval_audit_log_ids,
        protected_from_gc_ids=tuple(protected),
        unresolved_conflicts=conflicts,
        conflict_write_ids_by_key=conflict_write_ids,
        recovery_warnings_by_id=recovery_warnings,
        obsidian_conflict_ids=tuple(obsidian_conflict_ids),
        obsidian_conflict_sources_by_id=obsidian_conflict_sources,
        redacted_diagnostic_ids=tuple(redacted_diagnostic_ids),
        diagnostics=diagnostics,
    )


def _memory_report_privacy_check(
    writes: Sequence[MemoryRecoveryWrite],
    redacted_payload: str,
) -> dict[str, object]:
    """Return a compact proof that report diagnostics omit raw memory text."""
    normalized_payload = _normalize(redacted_payload)
    raw_content_match_count = 0
    checked_write_count = 0
    for write in writes:
        content = _normalize(write.content)
        if not content:
            continue
        checked_write_count += 1
        if content in normalized_payload:
            raw_content_match_count += 1
    return {
        "passed": raw_content_match_count == 0,
        "raw_content_match_count": raw_content_match_count,
        "checked_write_count": checked_write_count,
    }


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
