from __future__ import annotations

import os
import re
import uuid
import yaml
from datetime import date, datetime
from pathlib import Path
from typing import Any


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown text.

    Returns (frontmatter_dict, body_text).
    """
    text = text.strip()
    if not text.startswith("---\n"):
        return {}, text

    end_match = re.search(r"^---\s*$", text[4:], flags=re.MULTILINE)
    if not end_match:
        return {}, text

    end = 4 + end_match.start()
    yaml_str = text[4:end].strip()
    body = text[4 + end_match.end():].strip()

    try:
        fm = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        return {}, text

    return fm, body


def render_frontmatter(fm: dict[str, Any]) -> str:
    """Render a frontmatter dict to YAML string with --- delimiters."""
    cleaned = {}
    for k, v in fm.items():
        if isinstance(v, (date, datetime)):
            cleaned[k] = v.isoformat() if isinstance(v, datetime) else str(v)
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            cleaned[k] = v
        else:
            cleaned[k] = v

    yaml_str = yaml.dump(cleaned, default_flow_style=False, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{yaml_str}\n---"


def write_page(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    """Write a wiki page with frontmatter + body using atomic replacement."""
    fm_str = render_frontmatter(frontmatter)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"{fm_str}\n\n{body}\n"
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_page(path: Path) -> tuple[dict[str, Any], str]:
    """Read a wiki page, returning (frontmatter, body)."""
    if not path.exists():
        return {}, ""
    return parse_frontmatter(path.read_text(encoding="utf-8"))


def extract_wikilinks(text: str) -> list[str]:
    """Extract all [[wikilinks]] from text, resolving aliases.

    [[Page|display text]] resolves to "Page" (the target, not the display text).
    [[Page]] resolves to "Page".
    """
    raw = re.findall(r"\[\[([^\]]+)\]\]", text)
    return [link.split("|")[0].strip() for link in raw]


def slugify(title: str) -> str:
    """Convert a title to a filename slug."""
    title = title.split("|")[0].strip()
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def chunk_markdown(text: str, max_tokens: int = 500) -> list[str]:
    """Split markdown text into chunks, respecting heading structure.

    Tries to split on headings first, then paragraphs for oversized sections.
    Token count approximated as len(text) / 4.
    """
    sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    chunks = []

    for section in sections:
        trimmed = section.strip()
        if not trimmed:
            continue

        if len(trimmed) / 4 <= max_tokens:
            chunks.append(trimmed)
        else:
            heading_match = re.match(r"^(#{1,3}\s+.+)\n", trimmed)
            heading = heading_match.group(1) if heading_match else ""
            body = trimmed[len(heading):].strip() if heading else trimmed

            paragraphs = re.split(r"\n{2,}", body)
            buf = heading + "\n\n" if heading else ""

            for para in paragraphs:
                candidate = buf + para + "\n\n"
                if len(candidate) / 4 > max_tokens and buf.strip():
                    chunks.append(buf.strip())
                    buf = (heading + "\n\n" if heading else "") + para + "\n\n"
                else:
                    buf = candidate

            if buf.strip():
                chunks.append(buf.strip())

    return chunks if chunks else [text.strip()]
