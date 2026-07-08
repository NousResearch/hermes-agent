"""Vault filesystem service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from llmwiki_hermes.constants import (
    CURRENT_SCHEMA_VERSION,
    INDEX_DB_NAME,
    INGEST_LOG_NAME,
    PRECOMPRESS_DIR,
    SESSION_LOG_DIR,
    STATE_FILE_NAME,
    VAULT_DIRS,
    VAULT_ROOT_NAME,
)
from llmwiki_hermes.errors import ConfigurationError, VaultNotInitializedError
from llmwiki_hermes.schemas.cli import CommandOutput
from llmwiki_hermes.schemas.diagnostics import DiagnosticIssue, DiagnosticReport, DiagnosticSeverity
from llmwiki_hermes.schemas.notes import NoteDocument, validate_note_frontmatter
from llmwiki_hermes.storage.frontmatter import load_note
from llmwiki_hermes.storage.links import extract_links
from llmwiki_hermes.utils.slug import slugify

logger = logging.getLogger(__name__)
NOTE_ID_PREFIXES = ("src-", "sem-", "epi-")


def _normalize_lookup_value(value: str) -> str:
    return slugify(value.replace("_", "-"))


def _strip_note_prefix(value: str) -> str:
    for prefix in NOTE_ID_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


class VaultService:
    """Encapsulates all vault path logic."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.sources_dir = self.root / "10_sources"
        self.semantic_dir = self.root / "20_semantic"
        self.episodic_dir = self.root / "30_episodic"
        self.system_dir = self.root / "90_system"
        self.hidden_dir = self.root / ".wiki"
        self.index_db_path = self.hidden_dir / INDEX_DB_NAME
        self.ingest_log_path = self.hidden_dir / INGEST_LOG_NAME
        self.state_path = self.hidden_dir / STATE_FILE_NAME

    @classmethod
    def from_user_path(cls, base_path: Path) -> "VaultService":
        """Create a service from the user-facing init path."""

        resolved = base_path.expanduser().resolve()
        root = resolved if resolved.name == VAULT_ROOT_NAME else resolved / VAULT_ROOT_NAME
        return cls(root)

    def initialize(self, force: bool = False) -> CommandOutput:
        """Create the vault directory structure."""

        if self.root.exists() and not force and any(self.root.iterdir()):
            raise ConfigurationError(f"Vault already exists: {self.root}")
        for name in VAULT_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.ingest_log_path.touch(exist_ok=True)
        if force or not self.state_path.exists():
            self.state_path.write_text(json.dumps({"version": "1.0.0"}, indent=2), encoding="utf-8")
        from llmwiki_hermes.storage.sqlite_index import IndexService

        IndexService(self).create_schema()
        return CommandOutput(
            message=f"Initialized vault at {self.root}",
            data={"vault_path": str(self.root), "index_path": str(self.index_db_path)},
        )

    def ensure_initialized(self) -> None:
        """Validate required directories exist."""

        if not self.root.exists():
            raise VaultNotInitializedError(f"Vault does not exist: {self.root}")
        missing = [name for name in VAULT_DIRS if not (self.root / name).exists()]
        if missing:
            raise VaultNotInitializedError(
                f"Vault is missing required paths: {', '.join(sorted(missing))}"
            )

    def iter_note_paths(self) -> Iterable[Path]:
        """Yield all managed note paths."""

        self.ensure_initialized()
        for directory in (self.sources_dir, self.semantic_dir, self.episodic_dir):
            yield from sorted(directory.glob("*.md"))

    def get_note(self, id_or_slug: str) -> NoteDocument:
        """Resolve a note by id, slug, or filename stem."""

        target = _normalize_lookup_value(id_or_slug)
        title_match: NoteDocument | None = None
        for path in self.iter_note_paths():
            path_key = _normalize_lookup_value(path.stem)
            if path_key == target or _strip_note_prefix(path_key) == target:
                return load_note(path)
            document = load_note(path)
            note_id = str(document.frontmatter.get("id", ""))
            note_id_key = _normalize_lookup_value(note_id)
            if (
                note_id == id_or_slug
                or note_id_key == target
                or _strip_note_prefix(note_id_key) == target
            ):
                return document
            title_key = _normalize_lookup_value(str(document.frontmatter.get("title", "")))
            if title_key == target and title_match is None:
                title_match = document
        if title_match is not None:
            return title_match
        raise FileNotFoundError(f"Note not found: {id_or_slug}")

    def doctor(self) -> CommandOutput:
        """Check directory structure and note readability."""

        issues: list[DiagnosticIssue] = []
        initialized = True
        try:
            self.ensure_initialized()
        except VaultNotInitializedError as exc:
            issues.append(
                DiagnosticIssue(
                    code="vault_not_initialized",
                    severity=DiagnosticSeverity.ERROR,
                    path=str(self.root),
                    message=str(exc),
                )
            )
            initialized = False
        note_paths = list(self.iter_note_paths()) if self.root.exists() and initialized else []
        valid_notes: list[tuple[Path, NoteDocument, Any]] = []
        legacy_schema_notes = 0
        notes_by_schema_version: dict[str, int] = {}

        for path in note_paths:
            try:
                document = load_note(path)
            except Exception as exc:
                logger.warning("Doctor failed to parse frontmatter for %s: %s", path, exc)
                issues.append(
                    DiagnosticIssue(
                        code="frontmatter_parse_error",
                        severity=DiagnosticSeverity.ERROR,
                        path=str(path),
                        message=str(exc),
                    )
                )
                continue

            raw_note_id = str(document.frontmatter.get("id", "")) or None
            raw_schema_version = document.frontmatter.get("schema_version")
            if raw_schema_version is None:
                legacy_schema_notes += 1
                notes_by_schema_version["legacy"] = notes_by_schema_version.get("legacy", 0) + 1
                issues.append(
                    DiagnosticIssue(
                        code="missing_schema_version",
                        severity=DiagnosticSeverity.WARNING,
                        path=str(path),
                        message="Note is missing schema_version and is treated as a legacy note.",
                        note_id=raw_note_id,
                    )
                )
            else:
                try:
                    parsed_schema_version = int(raw_schema_version)
                except (TypeError, ValueError):
                    parsed_schema_version = None
                if parsed_schema_version is not None:
                    key = str(parsed_schema_version)
                    notes_by_schema_version[key] = notes_by_schema_version.get(key, 0) + 1
                    if parsed_schema_version < CURRENT_SCHEMA_VERSION:
                        legacy_schema_notes += 1
                        issues.append(
                            DiagnosticIssue(
                                code="outdated_schema_version",
                                severity=DiagnosticSeverity.WARNING,
                                path=str(path),
                                message=(
                                    "Note uses outdated schema_version "
                                    f"{parsed_schema_version}; current is {CURRENT_SCHEMA_VERSION}."
                                ),
                                note_id=raw_note_id,
                            )
                        )
                    if parsed_schema_version > CURRENT_SCHEMA_VERSION:
                        issues.append(
                            DiagnosticIssue(
                                code="unsupported_schema_version",
                                severity=DiagnosticSeverity.ERROR,
                                path=str(path),
                                message=(
                                    "Note uses unsupported schema_version "
                                    f"{parsed_schema_version}; current is {CURRENT_SCHEMA_VERSION}."
                                ),
                                note_id=raw_note_id,
                            )
                        )
            for field in ("id", "kind", "title"):
                if field not in document.frontmatter:
                    issues.append(
                        DiagnosticIssue(
                            code="missing_required_field",
                            severity=DiagnosticSeverity.ERROR,
                            path=str(path),
                            message=f"Frontmatter is missing required field: {field}",
                            note_id=raw_note_id,
                        )
                    )
            if (
                document.frontmatter.get("kind") == "episodic"
                and "date" not in document.frontmatter
            ):
                issues.append(
                    DiagnosticIssue(
                        code="episodic_missing_date",
                        severity=DiagnosticSeverity.ERROR,
                        path=str(path),
                        message="Episodic note is missing required field: date",
                        note_id=raw_note_id,
                    )
                )

            try:
                frontmatter = validate_note_frontmatter(document.frontmatter)
            except Exception as exc:
                logger.warning("Doctor failed to validate frontmatter for %s: %s", path, exc)
                issues.append(
                    DiagnosticIssue(
                        code="frontmatter_validation_error",
                        severity=DiagnosticSeverity.ERROR,
                        path=str(path),
                        message=str(exc),
                        note_id=raw_note_id,
                    )
                )
                continue

            note_id = str(frontmatter.id)
            if frontmatter.kind.value == "semantic" and not frontmatter.source_refs:
                issues.append(
                    DiagnosticIssue(
                        code="semantic_missing_source_refs",
                        severity=DiagnosticSeverity.ERROR,
                        path=str(path),
                        message="Semantic note must retain at least one source_ref.",
                        note_id=note_id,
                    )
                )

            valid_notes.append((path, document, frontmatter))

        source_note_ids: set[str] = set()
        source_hash_groups: dict[str, list[tuple[str, str]]] = {}
        orphan_semantic_notes = 0
        orphan_episodic_notes = 0
        note_ids_by_path: dict[str, str] = {}

        for path, _document, frontmatter in valid_notes:
            note_id = str(frontmatter.id)
            note_ids_by_path[str(path)] = note_id
            if frontmatter.kind.value != "source":
                continue
            source_note_ids.add(note_id)
            content_hash = getattr(frontmatter, "content_hash", "")
            if content_hash:
                source_hash_groups.setdefault(str(content_hash), []).append((str(path), note_id))

        for path, _document, frontmatter in valid_notes:
            missing_source_refs = [
                source_ref
                for source_ref in frontmatter.source_refs
                if source_ref not in source_note_ids
            ]
            if frontmatter.kind.value == "semantic" and missing_source_refs:
                orphan_semantic_notes += 1
                issues.append(
                    DiagnosticIssue(
                        code="orphan_semantic_note",
                        severity=DiagnosticSeverity.WARNING,
                        path=str(path),
                        message=(
                            "Semantic note references missing source note(s): "
                            f"{', '.join(sorted(missing_source_refs))}"
                        ),
                        note_id=str(frontmatter.id),
                    )
                )
            if frontmatter.kind.value == "episodic" and (
                not frontmatter.source_refs or missing_source_refs
            ):
                orphan_episodic_notes += 1
                message = (
                    "Episodic note does not reference any source note."
                    if not frontmatter.source_refs
                    else (
                        "Episodic note references missing source note(s): "
                        f"{', '.join(sorted(missing_source_refs))}"
                    )
                )
                issues.append(
                    DiagnosticIssue(
                        code="orphan_episodic_note",
                        severity=DiagnosticSeverity.WARNING,
                        path=str(path),
                        message=message,
                        note_id=str(frontmatter.id),
                    )
                )

        duplicate_source_hash_groups = 0
        for content_hash, members in sorted(source_hash_groups.items()):
            if len(members) < 2:
                continue
            duplicate_source_hash_groups += 1
            member_ids = [note_id for _path, note_id in members]
            for member_path, note_id in members:
                duplicate_peers = [item for item in member_ids if item != note_id]
                issues.append(
                    DiagnosticIssue(
                        code="duplicate_source_content_hash",
                        severity=DiagnosticSeverity.WARNING,
                        path=member_path,
                        message=(
                            f"Source note shares content_hash {content_hash} with: "
                            f"{', '.join(sorted(duplicate_peers))}"
                        ),
                        note_id=note_id,
                    )
                )

        known_targets: set[str] = set()
        for path, _document, frontmatter in valid_notes:
            known_targets.update(
                {
                    path.stem,
                    slugify(path.stem.replace("_", "-")),
                    str(frontmatter.id),
                    slugify(str(frontmatter.id).replace("_", "-")),
                }
            )

        for path, document, frontmatter in valid_notes:
            for link in extract_links(document.body):
                normalized_link = slugify(link.replace("_", "-"))
                if link not in known_targets and normalized_link not in known_targets:
                    issues.append(
                        DiagnosticIssue(
                            code="broken_link",
                            severity=DiagnosticSeverity.WARNING,
                            path=str(path),
                            message=f"Broken wikilink target: {link}",
                            note_id=str(frontmatter.id),
                        )
                    )

        if not self.index_db_path.exists():
            issues.append(
                DiagnosticIssue(
                    code="missing_index",
                    severity=DiagnosticSeverity.ERROR,
                    path=str(self.index_db_path),
                    message=f"Missing SQLite index: {self.index_db_path}",
                )
            )
            indexed_notes = 0
            missing_index_entries = 0
            stale_index_entries = 0
        else:
            from llmwiki_hermes.storage.sqlite_index import IndexService

            index_service = IndexService(self)
            indexed_rows = index_service.note_rows()
            indexed_notes = len(indexed_rows)
            indexed_paths = {str(row["path"]): str(row["id"]) for row in indexed_rows}
            valid_paths = {str(path) for path, _document, _frontmatter in valid_notes}
            missing_paths = sorted(valid_paths - set(indexed_paths))
            stale_paths = sorted(set(indexed_paths) - valid_paths)
            missing_index_entries = len(missing_paths)
            stale_index_entries = len(stale_paths)
            if indexed_notes != len(note_paths):
                logger.warning(
                    "Doctor detected index/file count mismatch for %s: %s indexed vs %s files",
                    self.index_db_path,
                    indexed_notes,
                    len(note_paths),
                )
                issues.append(
                    DiagnosticIssue(
                        code="index_note_count_mismatch",
                        severity=DiagnosticSeverity.WARNING,
                        path=str(self.index_db_path),
                        message=(
                            "Index note count does not match the file tree count: "
                            f"{indexed_notes} indexed vs {len(note_paths)} files"
                        ),
                    )
                )
            for missing_path in missing_paths:
                issues.append(
                    DiagnosticIssue(
                        code="missing_index_entry",
                        severity=DiagnosticSeverity.WARNING,
                        path=missing_path,
                        message="Valid note is missing from the SQLite index.",
                        note_id=note_ids_by_path.get(missing_path),
                    )
                )
            for stale_path in stale_paths:
                issues.append(
                    DiagnosticIssue(
                        code="stale_index_entry",
                        severity=DiagnosticSeverity.WARNING,
                        path=stale_path,
                        message="SQLite index contains a note path that no longer exists.",
                        note_id=indexed_paths.get(stale_path),
                    )
                )

        report = DiagnosticReport(
            issues=issues,
            stats={
                "note_files": len(note_paths),
                "valid_notes": len(valid_notes),
                "invalid_notes": max(len(note_paths) - len(valid_notes), 0),
                "indexed_notes": indexed_notes,
                "orphan_semantic_notes": orphan_semantic_notes,
                "orphan_episodic_notes": orphan_episodic_notes,
                "duplicate_source_hash_groups": duplicate_source_hash_groups,
                "missing_index_entries": missing_index_entries,
                "stale_index_entries": stale_index_entries,
                "legacy_schema_notes": legacy_schema_notes,
                "notes_by_schema_version": notes_by_schema_version,
            },
            severity_counts={
                severity.value: sum(1 for issue in issues if issue.severity == severity)
                for severity in DiagnosticSeverity
                if any(issue.severity == severity for issue in issues)
            },
            recovery_workflow=["doctor", "reindex", "doctor"] if issues else [],
        )
        message = (
            "Vault passed checks."
            if not report.issues
            else f"Detected {len(report.issues)} issue(s)."
        )
        return CommandOutput(
            ok=not report.issues,
            message=message,
            data=report.model_dump(mode="json"),
        )

    def append_session_turn(self, session_id: str, payload: dict[str, Any]) -> None:
        """Append a JSONL session turn log."""

        session_dir = self.hidden_dir / SESSION_LOG_DIR
        session_dir.mkdir(parents=True, exist_ok=True)
        with (session_dir / f"{session_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_precompress_snapshot(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist a pre-compression snapshot."""

        snapshot_dir = self.hidden_dir / PRECOMPRESS_DIR
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / f"{session_id}.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_session_summary(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist a raw session summary for later episodic compilation."""

        summary_path = self.system_dir / f"session-{slugify(session_id)}.json"
        summary_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
