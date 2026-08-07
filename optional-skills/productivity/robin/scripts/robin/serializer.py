from __future__ import annotations

from datetime import date
from uuid import uuid4

from robin.models import Entry
from robin.parser import topic_slug


def generate_entry_id(date_added: str | None = None) -> str:
    stamp = (date_added or str(date.today())).replace("-", "")
    return f"{stamp}-{uuid4().hex[:6]}"


def build_text_entry(
    topic: str,
    content: str,
    description: str,
    source: str | None,
    note: str | None,
    tags: list[str],
    date_added: str,
    media_source: str | None = None,
    entry_id: str | None = None,
) -> Entry:
    body_parts: list[str] = []
    body_parts.append(content.strip())
    if note:
        body_parts.append("")
        body_parts.append(f"**Robin note:** {note.strip()}")

    return Entry(
        entry_id=entry_id or generate_entry_id(date_added),
        topic=topic_slug(topic),
        date_added=date_added,
        entry_type="text",
        media_source=(media_source or "").strip(),
        source=(source or "").strip(),
        description=description.strip(),
        tags=tags,
        body="\n".join(body_parts).strip(),
    )


def build_media_entry(
    topic: str,
    media_kind: str,
    media_source: str,
    description: str,
    creator: str,
    published_at: str,
    summary: str,
    content: str,
    source: str | None,
    note: str | None,
    tags: list[str],
    date_added: str,
    entry_id: str | None = None,
) -> Entry:
    body_parts: list[str] = []
    if content.strip():
        body_parts.append(content.strip())
    if note:
        if body_parts:
            body_parts.append("")
        body_parts.append(f"**Robin note:** {note.strip()}")

    return Entry(
        entry_id=entry_id or generate_entry_id(date_added),
        topic=topic_slug(topic),
        date_added=date_added,
        entry_type=media_kind,
        media_kind=media_kind,
        media_source=media_source.strip(),
        source=(source or "").strip(),
        description=description.strip(),
        creator=creator.strip(),
        published_at=published_at.strip(),
        summary=summary.strip(),
        tags=tags,
        body="\n".join(body_parts).strip(),
    )


def serialize_entry(entry: Entry) -> str:
    if not entry.description.strip():
        raise ValueError("Entry description is required.")
    lines = [f"id: {entry.entry_id}", f"date_added: {entry.date_added}"]
    optional_fields = [
        ("entry_type", entry.entry_type if entry.entry_type != "text" else ""),
        ("media_kind", entry.media_kind),
        ("media_source", entry.media_source),
        ("source", entry.source),
        ("description", entry.description),
        ("creator", entry.creator),
        ("published_at", entry.published_at),
        ("summary", entry.summary),
    ]
    for key, value in optional_fields:
        if value:
            lines.append(f"{key}: {value}")
    if entry.tags:
        lines.append(f"tags: [{', '.join(entry.tags)}]")
    # The blank line is required even when the body is empty; it terminates frontmatter.
    lines.extend(["", entry.body.strip()])
    return "\n".join(lines)
