from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

EXCLUSIONS_PATH = Path(__file__).resolve().parent.parent / "blog_topics" / "exclusions.json"


class ExcludedContentError(RuntimeError):
    """Raised when a generated item matches the blog exclusion list."""


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _title_variants(title: str) -> set[str]:
    raw = (title or "").strip()
    if not raw:
        return set()
    variants = {_norm(raw)}
    if ":" in raw:
        variants.add(_norm(raw.split(":", 1)[0]))
    return {v for v in variants if v}


def load_exclusions() -> list[dict]:
    if not EXCLUSIONS_PATH.exists():
        return []
    try:
        data = json.loads(EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    return [i for i in items if isinstance(i, dict)]


def match_exclusion(title: str = "", slug: str = "", topic_id: str = "") -> Optional[dict]:
    title_keys = _title_variants(title)
    slug_key = (slug or "").strip().lower()
    topic_key = (topic_id or "").strip().lower()
    for entry in load_exclusions():
        entry_title = _norm(entry.get("title", ""))
        entry_slug = (entry.get("slug", "") or "").strip().lower()
        entry_topic = (entry.get("topic_id", "") or "").strip().lower()
        if entry_title and entry_title in title_keys:
            return entry
        if entry_slug and slug_key and (slug_key == entry_slug or slug_key.startswith(f"{entry_slug}-")):
            return entry
        if entry_topic and topic_key and topic_key == entry_topic:
            return entry
    return None


def assert_allowed(title: str = "", slug: str = "", topic_id: str = "") -> None:
    entry = match_exclusion(title=title, slug=slug, topic_id=topic_id)
    if not entry:
        return
    reason = entry.get("reason", "excluded")
    replacement = entry.get("replacement_slug", "")
    suffix = f"; replacement_slug={replacement}" if replacement else ""
    label = title or slug or topic_id or "(unknown item)"
    raise ExcludedContentError(f"Excluded by policy: {label} ({reason}{suffix})")
