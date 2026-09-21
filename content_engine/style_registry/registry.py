"""Search over the extended style-menu registry (self-contained).

This module searches an already-loaded registry dict (see
``style_registry/__init__.py``) by trait tags.  It deliberately has NO
filesystem/config dependency: it only operates on the registry data passed
in, so the live pipeline never touches the midlibrary-extractor source tree.

The registry surface is neutral: entries expose ``style_label`` (never the
protected source ``name``) and ``prompt_fragment`` (never SREF).
"""

from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    """Normalize a tag for matching: lowercase, hyphens -> spaces, collapse ws."""
    return re.sub(r"\s+", " ", s.strip().lower().replace("-", " "))


def search(registry: dict[str, Any], tags: list[str]) -> list[dict[str, Any]]:
    """Return matching entries (slug + one-line reason) for the given trait tags.

    Matching is case-insensitive substring over the entry's ``tags`` list, with
    hyphens and spaces treated as equivalent (so ``high contrast`` matches the
    ``high-contrast`` tag). An entry matches if ANY requested tag is present.
    Results are ordered by the number of matched tags (descending), then slug.
    """
    requested = [_norm(t) for t in tags if t.strip()]
    styles = registry.get("styles") or {}
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for slug, entry in styles.items():
        entry_tags = [_norm(t) for t in entry.get("tags") or []]
        hits = [t for t in requested if any(t in et for et in entry_tags)]
        if hits:
            scored.append((len(hits), slug, entry))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for n_hits, slug, entry in scored:
        out.append({
            "slug": slug,
            "style_label": entry["style_label"],
            "matched_tags": n_hits,
            "reason": f"{entry['style_label']} — {entry.get('medium')}",
        })
    return out
