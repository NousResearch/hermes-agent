"""Wikilink extraction."""

from __future__ import annotations

import re

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def extract_links(body: str) -> list[str]:
    """Extract simple wikilinks."""

    return [match.strip() for match in WIKILINK_PATTERN.findall(body) if match.strip()]
