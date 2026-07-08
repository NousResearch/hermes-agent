"""Episodic note synthesis."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from llmwiki_hermes.compiler.classify import detect_date
from llmwiki_hermes.compiler.sections import (
    merge_line_blocks,
    normalize_heading,
    parse_markdown_sections,
    render_markdown_note,
    section_content,
)
from llmwiki_hermes.constants import CURRENT_SCHEMA_VERSION
from llmwiki_hermes.schemas.notes import (
    EpisodicNoteFrontmatter,
    SourceNoteFrontmatter,
    is_unsupported_schema,
)
from llmwiki_hermes.storage.frontmatter import load_note, write_note
from llmwiki_hermes.storage.vault import VaultService
from llmwiki_hermes.templates import render_template
from llmwiki_hermes.utils.slug import build_note_id, slugify
from llmwiki_hermes.utils.time import today_utc

logger = logging.getLogger(__name__)

PROJECT_PATTERN = re.compile(r"\bproject\s+([a-z0-9][a-z0-9_-]*)\b", re.IGNORECASE)
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


def extract_section_lines(
    content: str,
    section_headers: tuple[str, ...],
    markers: tuple[str, ...],
    fallback: str,
) -> str:
    """Extract bullets from explicit sections, then marker matches, then fallback."""

    explicit = section_content(content, *section_headers)
    if explicit:
        return _bulletize(explicit, fallback=fallback)

    picked = [
        line.strip()
        for line in content.splitlines()
        if any(marker in line.lower() for marker in markers)
    ]
    if picked:
        return "\n".join(f"- {line.lstrip('-* ').strip()}" for line in picked[:5])
    return f"- {fallback}"


def extract_project(title: str, content: str) -> str | None:
    """Extract a stable project key when one is explicitly present."""

    for candidate in (title, content):
        match = PROJECT_PATTERN.search(candidate)
        if match:
            return slugify(match.group(1))
    return None


def build_episodic_title_key(title: str) -> str:
    """Normalize an episodic title for strong-match dedupe."""

    return slugify(title.replace("_", "-"))


def render_episodic_note(
    title: str,
    content: str,
    source_note: SourceNoteFrontmatter,
) -> tuple[EpisodicNoteFrontmatter, str, str]:
    """Create episodic note frontmatter and body."""

    parsed_date = detect_date(content)
    event_date = date.fromisoformat(parsed_date) if parsed_date else today_utc()
    note_id = build_note_id("epi", title, event_date.isoformat())
    fallback_line = (
        content.splitlines()[0].strip() if content.splitlines() else "No summary extracted."
    )
    project = extract_project(title, content)
    frontmatter = EpisodicNoteFrontmatter(
        schema_version=CURRENT_SCHEMA_VERSION,
        id=note_id,
        title=title,
        date=event_date,
        participants=[],
        project=project,
        source_refs=[source_note.id],
        entity_refs=[],
        tags=["meeting", "project"] if project else ["event"],
    )
    body = render_template(
        "episodic_note.md.j2",
        {
            "title": title,
            "what_happened": extract_section_lines(
                content,
                section_headers=("What Happened",),
                markers=("happened", "discuss", "讨论", "meeting", "project"),
                fallback=fallback_line,
            ),
            "decisions": extract_section_lines(
                content,
                section_headers=("Decisions",),
                markers=("decision", "decide", "决定", "scope", "先做"),
                fallback="No explicit decisions extracted.",
            ),
            "open_questions": extract_section_lines(
                content,
                section_headers=("Open Questions",),
                markers=("question", "todo", "unknown", "待定", "open"),
                fallback="No open questions extracted.",
            ),
            "derived_semantic_updates": extract_section_lines(
                content,
                section_headers=("Derived Semantic Updates",),
                markers=("semantic", "concept", "derived"),
                fallback="No derived semantic updates extracted.",
            ),
        },
    )
    filename = f"{slugify(note_id.replace('_', '-'))}.md"
    return frontmatter, body, filename


def _resolve_existing_episodic_path(
    vault_service: VaultService,
    new_frontmatter: EpisodicNoteFrontmatter,
) -> Path | None:
    matches: list[Path] = []
    for path in sorted(vault_service.episodic_dir.glob("*.md")):
        try:
            document = load_note(path)
            frontmatter = EpisodicNoteFrontmatter.model_validate(document.frontmatter)
        except Exception:
            continue
        if frontmatter.date != new_frontmatter.date:
            continue
        if build_episodic_title_key(frontmatter.title) != build_episodic_title_key(
            new_frontmatter.title
        ):
            continue
        same_project = bool(
            frontmatter.project
            and new_frontmatter.project
            and slugify(frontmatter.project) == slugify(new_frontmatter.project)
        )
        shared_source = bool(set(frontmatter.source_refs) & set(new_frontmatter.source_refs))
        if same_project or shared_source:
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


def _merge_episodic_body(existing_body: str, new_body: str, title: str) -> str:
    existing = parse_markdown_sections(existing_body)
    incoming = parse_markdown_sections(new_body)

    existing_sections = {header: content for header, content in existing.sections}
    incoming_sections = {header: content for header, content in incoming.sections}
    managed = [
        (
            "What Happened",
            merge_line_blocks(
                existing_sections.get("What Happened", ""),
                incoming_sections.get("What Happened", ""),
            ),
        ),
        (
            "Decisions",
            merge_line_blocks(
                existing_sections.get("Decisions", ""),
                incoming_sections.get("Decisions", ""),
            ),
        ),
        (
            "Open Questions",
            merge_line_blocks(
                existing_sections.get("Open Questions", ""),
                incoming_sections.get("Open Questions", ""),
            ),
        ),
        (
            "Derived Semantic Updates",
            merge_line_blocks(
                existing_sections.get("Derived Semantic Updates", ""),
                incoming_sections.get("Derived Semantic Updates", ""),
            ),
        ),
    ]
    extra_sections = [
        (header, content)
        for header, content in existing.sections
        if normalize_heading(header)
        not in {normalize_heading(item) for item in EPISODIC_SECTION_ORDER}
    ]
    return render_markdown_note(title, existing.lead_lines, managed, extra_sections)


def _disambiguate_episodic_identity(
    frontmatter: EpisodicNoteFrontmatter,
    source_note: SourceNoteFrontmatter,
) -> tuple[EpisodicNoteFrontmatter, str]:
    note_id = build_note_id(
        "epi",
        frontmatter.title,
        f"{frontmatter.date.isoformat()}-{source_note.id}",
    )
    updated = frontmatter.model_copy(update={"id": note_id})
    filename = f"{slugify(note_id.replace('_', '-'))}.md"
    return updated, filename


def create_or_append_episodic_note(
    vault_service: VaultService,
    title: str,
    content: str,
    source_note: SourceNoteFrontmatter,
    dry_run: bool = False,
) -> str | None:
    """Create or append to an episodic note."""

    frontmatter, body, filename = render_episodic_note(title, content, source_note)
    existing_path = _resolve_existing_episodic_path(vault_service, frontmatter)
    path = existing_path or (vault_service.episodic_dir / filename)
    if existing_path is None and path.exists():
        frontmatter, filename = _disambiguate_episodic_identity(frontmatter, source_note)
        path = vault_service.episodic_dir / filename
    if path.exists():
        existing = load_note(path)
        if is_unsupported_schema(existing.frontmatter):
            logger.warning(
                (
                    "Skipping episodic auto-append for %s because %s uses "
                    "unsupported schema_version %s."
                ),
                title,
                path,
                existing.frontmatter.get("schema_version"),
            )
            return None
        existing_frontmatter = EpisodicNoteFrontmatter.model_validate(existing.frontmatter)
        merged_frontmatter = EpisodicNoteFrontmatter.model_validate(
            {
                **existing.frontmatter,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "project": existing_frontmatter.project or frontmatter.project,
                "source_refs": sorted(set(existing_frontmatter.source_refs) | {source_note.id}),
                "participants": sorted(
                    set(existing_frontmatter.participants) | set(frontmatter.participants)
                ),
                "entity_refs": sorted(
                    set(existing_frontmatter.entity_refs) | set(frontmatter.entity_refs)
                ),
                "tags": sorted(set(existing_frontmatter.tags) | set(frontmatter.tags)),
            }
        )
        frontmatter = merged_frontmatter
        body = _merge_episodic_body(existing.body, body, existing_frontmatter.title)
    if not dry_run:
        write_note(path, frontmatter.model_dump(mode="json"), body)
    return str(path)
