"""Shared slug utility for the blog subpackage.

Consolidates the kebab-case slug logic so blog_generator and blog_assembler
use one definition. The article track has its own _slug_from_title that
predates this module; we do not create a cross-package dependency.
"""
from __future__ import annotations
import re


def slugify(title: str, max_words: int = 6) -> str:
    """Kebab-case slug, ASCII-only, for blog post filenames.

    Falls back to 'post' when the title has no word characters.
    """
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    if not words:
        return "post"
    return "-".join(words[:max_words])