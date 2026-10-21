from __future__ import annotations

import hashlib
import re
from pathlib import Path

from robin.config import topics_path
from robin.models import Entry

SEPARATOR = "\n***\n"


class RobinEntryParseError(ValueError):
    def __init__(self, filepath: Path, chunk_index: int, message: str):
        self.filepath = filepath
        self.chunk_index = chunk_index
        super().__init__(f"Failed to parse entry {chunk_index} in {filepath}: {message}")


def topic_to_filename(topic: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", topic.strip().lower())
    slug = normalized.strip("-") or "untitled"
    return f"{slug}.md"


def topic_slug(topic: str) -> str:
    return topic_to_filename(topic).removesuffix(".md")


def parse_tags(raw_value: str) -> list[str]:
    value = raw_value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def parse_frontmatter_and_body(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    frontmatter: dict[str, str | list[str]] = {}
    body_start = None

    for i, line in enumerate(lines):
        if not line.strip():
            body_start = i + 1
            break
        if ":" not in line:
            raise ValueError(f"Invalid frontmatter line {i + 1}: {line!r}")
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key == "tags":
            frontmatter["tags"] = parse_tags(normalized_value)
        else:
            frontmatter[normalized_key] = normalized_value

    if body_start is None:
        raise ValueError("Entry frontmatter must be followed by a blank line.")

    body = "\n".join(lines[body_start:]).strip()
    return frontmatter, body


def parse_entry(text: str, topic: str) -> Entry:
    frontmatter, body = parse_frontmatter_and_body(text)
    entry_id = str(frontmatter.get("id", "")).strip()
    if ":" in entry_id:
        raise ValueError("Entry id cannot contain colon characters.")
    date_added = str(frontmatter.get("date_added", "")).strip()
    if not date_added:
        raise ValueError("Entry is missing required date_added field.")
    entry_type = str(frontmatter.get("entry_type", "text")).strip() or "text"
    media_kind = str(frontmatter.get("media_kind", "")).strip()
    media_source = str(frontmatter.get("media_source", "")).strip()
    source = str(frontmatter.get("source", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    creator = str(frontmatter.get("creator", "")).strip()
    published_at = str(frontmatter.get("published_at", "")).strip()
    summary = str(frontmatter.get("summary", "")).strip()
    tags = list(frontmatter.get("tags", []))
    if not entry_id:
        fingerprint = hashlib.sha256(
            f"{topic}\n{date_added}\n{entry_type}\n{media_kind}\n{media_source}\n{source}\n{description}\n{creator}\n{published_at}\n{summary}\n{','.join(tags)}\n{body}".encode("utf-8")
        ).hexdigest()[:10]
        entry_id = f"legacy-{fingerprint}"

    return Entry(
        entry_id=entry_id,
        topic=topic,
        date_added=date_added,
        entry_type=entry_type,
        media_kind=media_kind,
        media_source=media_source,
        source=source,
        description=description,
        creator=creator,
        published_at=published_at,
        summary=summary,
        tags=tags,
        body=body,
    )


def load_topic_entries(filepath: Path) -> list[Entry]:
    content = filepath.read_text(encoding="utf-8")
    if not content.strip():
        return []

    entries: list[Entry] = []
    for chunk_index, chunk in enumerate(content.split(SEPARATOR), start=1):
        if not chunk.strip():
            continue
        try:
            entries.append(parse_entry(chunk.lstrip(), filepath.stem))
        except ValueError as exc:
            raise RobinEntryParseError(filepath, chunk_index, str(exc)) from exc
    return entries


def load_all_entries(config: dict, explicit_state_dir: str | None = None) -> list[Entry]:
    base = topics_path(config, explicit_state_dir)
    if not base.exists():
        return []

    entries: list[Entry] = []
    for filepath in sorted(base.glob("*.md")):
        entries.extend(load_topic_entries(filepath))
    return entries
