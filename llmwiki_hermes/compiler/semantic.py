"""Semantic note synthesis and maintenance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llmwiki_hermes.compiler.sections import (
    merge_line_blocks,
    merge_paragraph_blocks,
    normalize_heading,
    parse_markdown_sections,
    render_markdown_note,
    section_content,
)
from llmwiki_hermes.constants import CURRENT_SCHEMA_VERSION
from llmwiki_hermes.schemas.cli import CommandOutput
from llmwiki_hermes.schemas.notes import (
    EpisodicNoteFrontmatter,
    SemanticNoteFrontmatter,
    SourceNoteFrontmatter,
    is_unsupported_schema,
    validate_note_frontmatter,
)
from llmwiki_hermes.settings import WikiSettings
from llmwiki_hermes.storage.frontmatter import load_note, write_note
from llmwiki_hermes.storage.vault import VaultService
from llmwiki_hermes.templates import render_template
from llmwiki_hermes.utils.slug import build_note_id, slugify
from llmwiki_hermes.utils.time import today_utc

logger = logging.getLogger(__name__)

SEMANTIC_SECTION_ORDER = ("Definition", "Stable Facts", "Relations", "Source Notes")
EPISODIC_SECTION_ORDER = (
    "What Happened",
    "Decisions",
    "Open Questions",
    "Derived Semantic Updates",
)


def _bulletize(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line for line in lines if line.startswith(("-", "*"))]
    if bullets:
        return "\n".join(bullets[:5])
    if lines:
        return "\n".join(f"- {line.lstrip('-* ').strip()}" for line in lines[:5])
    return f"- {fallback}"


def extract_definition(content: str) -> str:
    """Pick a short definition paragraph."""

    explicit = section_content(content, "Definition")
    if explicit:
        return explicit.strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "No definition extracted yet."
    first = lines[0].lstrip("#").strip()
    return first if len(first) <= 240 else f"{first[:237]}..."


def extract_bullets(content: str) -> str:
    """Promote explicit stable facts or synthesize fallback bullets."""

    explicit = section_content(content, "Stable Facts")
    if explicit:
        return _bulletize(explicit, fallback="No stable facts extracted yet.")
    definition = extract_definition(content)
    return _bulletize(content, fallback=definition)


def extract_relations(content: str) -> str:
    """Extract relation lines from explicit sections or fallback heuristics."""

    explicit = section_content(content, "Relations")
    if explicit:
        return _bulletize(explicit, fallback="related_to: source-derived")
    lines = [
        line.strip()
        for line in content.splitlines()
        if "related_to" in line.lower() or "contrasted_with" in line.lower()
    ]
    if lines:
        return _bulletize("\n".join(lines), fallback="related_to: source-derived")
    return "- related_to: source-derived"


def render_semantic_note(
    title: str,
    content: str,
    source_note: SourceNoteFrontmatter,
    existing_path: Path | None = None,
) -> tuple[SemanticNoteFrontmatter, str, str]:
    """Create or update a semantic note."""

    note_id = build_note_id("sem", title)
    frontmatter = SemanticNoteFrontmatter(
        schema_version=CURRENT_SCHEMA_VERSION,
        id=note_id,
        title=title,
        aliases=[],
        entity_refs=[],
        source_refs=[source_note.id],
        updated_at=today_utc(),
        confidence="medium",
        tags=["concept"],
    )
    body = render_template(
        "semantic_note.md.j2",
        {
            "title": title,
            "definition": extract_definition(content),
            "stable_facts": extract_bullets(content),
            "relations": extract_relations(content),
            "source_notes": f"- [[{source_note.id}]]",
        },
    )
    filename = (
        existing_path.name
        if existing_path is not None
        else f"{slugify(note_id.replace('_', '-'))}.md"
    )
    return frontmatter, body, filename


def _semantic_identity_key(value: str) -> str:
    return slugify(value.replace("_", "-"))


def _semantic_identity_tokens(note: dict[str, Any]) -> set[str]:
    tokens = {_semantic_identity_key(note["title"])}
    tokens.update(_semantic_identity_key(alias) for alias in note.get("aliases", []) if alias)
    return {token for token in tokens if token}


def _resolve_existing_semantic_path(vault_service: VaultService, title: str) -> Path | None:
    target_id = build_note_id("sem", title)
    target_key = _semantic_identity_key(title)
    id_matches: list[Path] = []
    title_matches: list[Path] = []
    alias_matches: list[Path] = []

    for path in sorted(vault_service.semantic_dir.glob("*.md")):
        try:
            document = load_note(path)
            frontmatter = validate_note_frontmatter(document.frontmatter)
        except Exception:
            continue
        if not isinstance(frontmatter, SemanticNoteFrontmatter):
            continue
        if frontmatter.id == target_id:
            id_matches.append(path)
            continue
        if _semantic_identity_key(frontmatter.title) == target_key:
            title_matches.append(path)
            continue
        alias_keys = {_semantic_identity_key(alias) for alias in frontmatter.aliases}
        if target_key in alias_keys:
            alias_matches.append(path)

    for matches in (id_matches, title_matches, alias_matches):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def _merge_semantic_body(existing_body: str, new_body: str, title: str) -> str:
    existing = parse_markdown_sections(existing_body)
    incoming = parse_markdown_sections(new_body)

    existing_sections = {header: content for header, content in existing.sections}
    incoming_sections = {header: content for header, content in incoming.sections}
    managed = [
        (
            "Definition",
            merge_paragraph_blocks(
                existing_sections.get("Definition", ""),
                incoming_sections.get("Definition", ""),
            ),
        ),
        (
            "Stable Facts",
            merge_line_blocks(
                existing_sections.get("Stable Facts", ""),
                incoming_sections.get("Stable Facts", ""),
            ),
        ),
        (
            "Relations",
            merge_line_blocks(
                existing_sections.get("Relations", ""),
                incoming_sections.get("Relations", ""),
            ),
        ),
        (
            "Source Notes",
            merge_line_blocks(
                existing_sections.get("Source Notes", ""),
                incoming_sections.get("Source Notes", ""),
            ),
        ),
    ]
    extra_sections = [
        (header, content)
        for header, content in existing.sections
        if normalize_heading(header)
        not in {normalize_heading(item) for item in SEMANTIC_SECTION_ORDER}
    ]
    return render_markdown_note(title, existing.lead_lines, managed, extra_sections)


def _disambiguate_semantic_identity(
    frontmatter: SemanticNoteFrontmatter,
    source_note: SourceNoteFrontmatter,
) -> tuple[SemanticNoteFrontmatter, str]:
    note_id = build_note_id("sem", frontmatter.title, source_note.id)
    updated = frontmatter.model_copy(update={"id": note_id})
    filename = f"{slugify(note_id.replace('_', '-'))}.md"
    return updated, filename


def _canonical_note(notes: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(notes, key=lambda note: (note["id"], note["path"]))[0]


def _build_semantic_merge_preview(notes: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = _canonical_note(notes)
    return {
        "target_path": canonical["path"],
        "merged_source_refs": sorted(
            {source_ref for note in notes for source_ref in note.get("source_refs", [])}
        ),
        "merged_aliases": sorted(
            {
                alias
                for note in notes
                for alias in [note["title"], *note.get("aliases", [])]
                if _semantic_identity_key(alias) != _semantic_identity_key(canonical["title"])
            }
        ),
        "merged_tags": sorted({tag for note in notes for tag in note.get("tags", [])}),
    }


def _build_episodic_merge_preview(notes: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = _canonical_note(notes)
    return {
        "target_path": canonical["path"],
        "merged_source_refs": sorted(
            {source_ref for note in notes for source_ref in note.get("source_refs", [])}
        ),
        "merged_participants": sorted(
            {participant for note in notes for participant in note.get("participants", [])}
        ),
        "merged_entity_refs": sorted(
            {entity_ref for note in notes for entity_ref in note.get("entity_refs", [])}
        ),
        "project": canonical.get("project"),
    }


class SemanticMaintenanceService:
    """Reports potential duplicate semantic pages."""

    def __init__(self, vault_service: VaultService) -> None:
        self.vault_service = vault_service

    @classmethod
    def from_settings(cls, settings: WikiSettings) -> "SemanticMaintenanceService":
        return cls(VaultService(settings.vault_path))

    def compact_report(self) -> CommandOutput:
        semantic_notes = self._load_semantic_notes()
        episodic_notes = self._load_episodic_notes()

        duplicate_candidates = list(self._find_semantic_duplicates(semantic_notes).values())
        source_conflict_candidates = list(self._find_source_conflicts(semantic_notes).values())
        episodic_near_duplicates = list(
            self._find_episodic_near_duplicates(episodic_notes).values()
        )

        total_groups = (
            len(duplicate_candidates)
            + len(source_conflict_candidates)
            + len(episodic_near_duplicates)
        )
        summary = {
            "total_groups": total_groups,
            "semantic_duplicate_groups": len(duplicate_candidates),
            "semantic_source_conflict_groups": len(source_conflict_candidates),
            "episodic_near_duplicate_groups": len(episodic_near_duplicates),
        }
        return CommandOutput(
            message=(
                "No maintenance candidates found. Dry-run only."
                if total_groups == 0
                else (
                    f"Found {total_groups} candidate maintenance group(s): "
                    f"{summary['semantic_duplicate_groups']} semantic duplicate, "
                    f"{summary['semantic_source_conflict_groups']} semantic source conflict, "
                    f"{summary['episodic_near_duplicate_groups']} episodic near-duplicate. "
                    "Dry-run only."
                )
            ),
            data={
                "dry_run": True,
                "summary": summary,
                "semantic_duplicate_candidates": duplicate_candidates,
                "semantic_source_conflict_candidates": source_conflict_candidates,
                "episodic_near_duplicate_candidates": episodic_near_duplicates,
            },
        )

    def _load_semantic_notes(self) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        for path in sorted(self.vault_service.semantic_dir.glob("*.md")):
            try:
                document = load_note(path)
                frontmatter = validate_note_frontmatter(document.frontmatter)
                if not isinstance(frontmatter, SemanticNoteFrontmatter):
                    continue
                notes.append(
                    {
                        "id": frontmatter.id,
                        "title": frontmatter.title,
                        "path": str(path),
                        "aliases": frontmatter.aliases,
                        "tags": frontmatter.tags,
                        "source_refs": frontmatter.source_refs,
                        "entity_refs": frontmatter.entity_refs,
                        "schema_version": document.frontmatter.get("schema_version"),
                    }
                )
            except Exception:
                continue
        return notes

    def _load_episodic_notes(self) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        for path in sorted(self.vault_service.episodic_dir.glob("*.md")):
            try:
                document = load_note(path)
                frontmatter = validate_note_frontmatter(document.frontmatter)
                if not isinstance(frontmatter, EpisodicNoteFrontmatter):
                    continue
                notes.append(
                    {
                        "id": frontmatter.id,
                        "title": frontmatter.title,
                        "path": str(path),
                        "date": frontmatter.date.isoformat(),
                        "source_refs": frontmatter.source_refs,
                        "participants": frontmatter.participants,
                        "project": frontmatter.project,
                        "entity_refs": frontmatter.entity_refs,
                    }
                )
            except Exception:
                continue
        return notes

    def _find_semantic_duplicates(
        self, semantic_notes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for note in semantic_notes:
            for token in _semantic_identity_tokens(note):
                groups.setdefault(token, {"key": token, "notes": []})["notes"].append(note)

        deduped: dict[frozenset[str], tuple[str, dict[str, Any]]] = {}
        for token, group in sorted(groups.items()):
            if len(group["notes"]) < 2:
                continue
            note_set = frozenset(note["id"] for note in group["notes"])
            deduped.setdefault(note_set, (token, group))

        result: dict[str, dict[str, Any]] = {}
        for token, group in (item for item in deduped.values()):
            canonical = _canonical_note(group["notes"])
            result[token] = {
                "key": token,
                "canonical_note_id": canonical["id"],
                "match_reason": "shared_identity_token",
                "merge_preview": _build_semantic_merge_preview(group["notes"]),
                "requires_manual_review": True,
                "notes": group["notes"],
            }
        return result

    def _find_source_conflicts(
        self, semantic_notes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for note in semantic_notes:
            for source_ref in note["source_refs"]:
                groups.setdefault(source_ref, {"source_ref": source_ref, "notes": []})[
                    "notes"
                ].append(note)

        result: dict[str, dict[str, Any]] = {}
        for source_ref, group in groups.items():
            if len(group["notes"]) < 2:
                continue
            canonical = _canonical_note(group["notes"])
            result[source_ref] = {
                "source_ref": source_ref,
                "canonical_note_id": canonical["id"],
                "match_reason": "shared_source_ref",
                "merge_preview": _build_semantic_merge_preview(group["notes"]),
                "requires_manual_review": True,
                "notes": group["notes"],
            }
        return result

    def _find_episodic_near_duplicates(
        self, episodic_notes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for note in episodic_notes:
            title_key = self._near_duplicate_title_key(note["title"])
            key = f"{note['date']}::{title_key}"
            groups.setdefault(key, {"date": note["date"], "title_key": title_key, "notes": []})[
                "notes"
            ].append(note)

        result: dict[str, dict[str, Any]] = {}
        for key, group in groups.items():
            if len(group["notes"]) < 2:
                continue
            canonical = _canonical_note(group["notes"])
            result[key] = {
                "date": group["date"],
                "title_key": group["title_key"],
                "canonical_note_id": canonical["id"],
                "match_reason": "same_date_title_key",
                "merge_preview": _build_episodic_merge_preview(group["notes"]),
                "requires_manual_review": True,
                "notes": group["notes"],
            }
        return result

    def _near_duplicate_title_key(self, title: str) -> str:
        tokens = [token for token in slugify(title).split("-") if token]
        if not tokens:
            return "note"
        return "-".join(tokens[:2])


def upsert_semantic_note(
    vault_service: VaultService,
    title: str,
    content: str,
    source_note: SourceNoteFrontmatter,
    dry_run: bool = False,
) -> str | None:
    """Create or update a semantic note from source content."""

    existing_path = _resolve_existing_semantic_path(vault_service, title)
    frontmatter, body, filename = render_semantic_note(
        title,
        content,
        source_note,
        existing_path=existing_path,
    )
    path = existing_path or (vault_service.semantic_dir / filename)
    if existing_path is None and path.exists():
        frontmatter, filename = _disambiguate_semantic_identity(frontmatter, source_note)
        path = vault_service.semantic_dir / filename

    if path.exists():
        existing = load_note(path)
        if is_unsupported_schema(existing.frontmatter):
            logger.warning(
                (
                    "Skipping semantic auto-merge for %s because %s uses "
                    "unsupported schema_version %s."
                ),
                title,
                path,
                existing.frontmatter.get("schema_version"),
            )
            return None
        existing_frontmatter = SemanticNoteFrontmatter.model_validate(existing.frontmatter)
        merged_aliases = sorted(
            {
                *existing_frontmatter.aliases,
                *(
                    [title]
                    if _semantic_identity_key(title)
                    != _semantic_identity_key(existing_frontmatter.title)
                    else []
                ),
            }
        )
        merged_frontmatter = SemanticNoteFrontmatter.model_validate(
            {
                **existing.frontmatter,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "aliases": merged_aliases,
                "source_refs": sorted(set(existing_frontmatter.source_refs) | {source_note.id}),
                "tags": sorted(set(existing_frontmatter.tags) | set(frontmatter.tags)),
                "updated_at": str(today_utc()),
            }
        )
        frontmatter = merged_frontmatter
        body = _merge_semantic_body(existing.body, body, existing_frontmatter.title)
    if not dry_run:
        write_note(path, frontmatter.model_dump(mode="json"), body)
    return str(path)
