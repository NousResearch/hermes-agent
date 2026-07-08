"""Source note generation."""

from __future__ import annotations

from pathlib import Path

from llmwiki_hermes.constants import CURRENT_SCHEMA_VERSION
from llmwiki_hermes.schemas.notes import SourceNoteFrontmatter
from llmwiki_hermes.templates import render_template
from llmwiki_hermes.utils.hashing import sha256_text
from llmwiki_hermes.utils.slug import build_note_id, slugify
from llmwiki_hermes.utils.time import utc_now


def derive_title(path: Path | None, content: str) -> str:
    """Choose a readable title for imported content."""

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:80]
    if path is not None:
        return path.stem.replace("_", " ").replace("-", " ").title()
    return "Imported Source"


def render_source_note(
    content: str,
    path: Path | None,
    source_type: str,
    tags: list[str],
) -> tuple[SourceNoteFrontmatter, str, str]:
    """Create source note frontmatter and body."""

    title = derive_title(path, content)
    content_hash = sha256_text(content)
    stem = path.stem if path is not None else title
    note_id = build_note_id("src", stem, utc_now().date().isoformat())
    frontmatter = SourceNoteFrontmatter(
        schema_version=CURRENT_SCHEMA_VERSION,
        id=note_id,
        title=title,
        source_type=source_type,
        origin="stdin" if path is None else "user_upload",
        ingested_at=utc_now(),
        content_hash=content_hash,
        source_refs=[],
        tags=sorted(set(tags + ["source", source_type])),
    )
    body = render_template(
        "source_note.md.j2",
        {
            "title": frontmatter.title,
            "origin": frontmatter.origin,
            "source_type": source_type,
            "content_hash": content_hash,
            "extracted_content": content.strip() or "(empty input)",
        },
    )
    filename = f"{slugify(note_id.replace('_', '-'))}.md"
    return frontmatter, body, filename
