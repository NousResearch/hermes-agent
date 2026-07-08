"""Minimal YAML frontmatter handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmwiki_hermes.errors import InvalidFrontmatterError
from llmwiki_hermes.schemas.notes import NoteDocument


def split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown document into frontmatter and body."""

    if not content.startswith("---\n"):
        raise InvalidFrontmatterError("Document is missing YAML frontmatter.")
    parts = content.split("\n---\n", 1)
    if len(parts) != 2:
        raise InvalidFrontmatterError("Document has malformed YAML frontmatter delimiters.")
    raw_frontmatter = parts[0][4:]
    body = parts[1].lstrip("\n")
    data = yaml.safe_load(raw_frontmatter) or {}
    if not isinstance(data, dict):
        raise InvalidFrontmatterError("YAML frontmatter must decode to an object.")
    return data, body


def dump_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """Serialize frontmatter and body into Markdown."""

    raw_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{raw_yaml}\n---\n\n{body.rstrip()}\n"


def load_note(path: Path) -> NoteDocument:
    """Read a note from disk."""

    frontmatter, body = split_frontmatter(path.read_text(encoding="utf-8"))
    return NoteDocument(frontmatter=frontmatter, body=body, path=str(path))


def write_note(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    """Persist a note document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_frontmatter(frontmatter, body), encoding="utf-8")
