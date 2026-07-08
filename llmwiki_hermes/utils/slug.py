"""Slug and identifier helpers."""

from __future__ import annotations

import re
import unicodedata


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""

    normalized = unicodedata.normalize("NFKD", value).strip().lower()
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    collapsed = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return collapsed or "note"


def build_note_id(prefix: str, title: str, suffix: str | None = None) -> str:
    """Create a stable note identifier."""

    parts = [prefix, slugify(title)]
    if suffix:
        parts.append(slugify(suffix))
    return "_".join(part for part in parts if part)
