"""Local, read-only wiki-memory index over Hermes built-in memories.

The module turns ``MEMORY.md`` and ``USER.md`` entries into structured,
inspectable records. It deliberately does not summarize with an LLM, mutate
memory files, or inject context by itself; callers can use the exported index
for CLI/dashboard/cron surfaces and budgeted retrieval.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

ENTRY_SEPARATOR = "\n§\n"
_INDEX_VERSION = 1
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "only",
    "should",
    "that",
    "the",
    "their",
    "this",
    "use",
    "user",
    "uses",
    "with",
}
_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("decision", ("decision:", "decided", "choose", "chose", "use ", "switch ")),
    ("constraint", ("constraint:", "must ", "must not", "never ", "only ", "avoid", "forbid")),
    ("preference", ("prefer", "prefers", "likes", "wants", "expects")),
    ("identity", ("legal name", "user name", "role:", "mission:")),
    ("environment", ("host", "runs ", "path", "port", "repo", "project", "token", "credential")),
    ("procedural", ("run ", "command", "workflow", "procedure", "steps", "skill")),
)


def _memory_dir() -> Path:
    return get_hermes_home() / "memories"


def _split_entries(text: str) -> list[str]:
    return [entry.strip() for entry in text.split(ENTRY_SEPARATOR) if entry.strip()]


def _tokenize(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", text.lower()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _classify(text: str, source: str) -> str:
    lower = text.lower()
    for category, needles in _CATEGORY_PATTERNS:
        if any(needle in lower for needle in needles):
            return category
    if source == "user":
        return "preference" if "prefer" in lower else "profile"
    return "fact"


def _stable_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}\0{index}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:{index}:{digest}"


def _entry_title(text: str) -> str:
    first = text.splitlines()[0].strip().lstrip("# ").strip()
    return (first[:96] + "…") if len(first) > 96 else first


def _read_source(path: Path, source: str) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        timestamp = int(path.stat().st_mtime)
    except OSError:
        return []

    entries: list[dict[str, Any]] = []
    for idx, text in enumerate(_split_entries(raw)):
        entries.append(
            {
                "id": _stable_id(source, idx, text),
                "source": source,
                "index": idx,
                "category": _classify(text, source),
                "title": _entry_title(text),
                "text": text,
                "keywords": _tokenize(text)[:16],
                "timestamp": timestamp + idx,
                "provenance": {
                    "path": str(path),
                    "entry_index": idx,
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
            }
        )
    return entries


def build_memory_wiki_index() -> dict[str, Any]:
    """Return a structured, JSON-serializable index over built-in memory files."""

    base = _memory_dir()
    entries = _read_source(base / "MEMORY.md", "memory") + _read_source(base / "USER.md", "user")
    sources: dict[str, int] = {}
    categories: dict[str, int] = {}
    for entry in entries:
        sources[entry["source"]] = sources.get(entry["source"], 0) + 1
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1

    return {
        "version": _INDEX_VERSION,
        "entries": entries,
        "stats": {
            "entries": len(entries),
            "sources": sources,
        },
        "categories": categories,
    }


def _score_entry(query_tokens: set[str], entry: dict[str, Any]) -> int:
    keywords = set(entry.get("keywords") or [])
    title = str(entry.get("title") or "").lower()
    text = str(entry.get("text") or "").lower()
    score = 4 * len(query_tokens & keywords)
    score += sum(2 for token in query_tokens if token in title)
    score += sum(1 for token in query_tokens if token in text)
    category = entry.get("category")
    if category in {"decision", "constraint", "preference"}:
        score += 1
    return score


def select_memory_context(
    query: str,
    *,
    index: dict[str, Any] | None = None,
    max_chars: int = 1200,
) -> dict[str, Any]:
    """Select relevant memory-wiki entries that fit within ``max_chars``.

    This is retrieval-aware context assembly in miniature: relevance is scored
    first, then entries are admitted only while the caller's character budget
    allows. The returned string is fenced as data, not user input.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    index = index or build_memory_wiki_index()
    entries = list(index.get("entries") or [])
    if not entries:
        return {"entries": [], "context": "", "used_chars": 0}

    query_tokens = set(_tokenize(query))
    scored = [(_score_entry(query_tokens, entry), entry) for entry in entries]
    scored = [(score, entry) for score, entry in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))

    selected: list[dict[str, Any]] = []
    chunks: list[str] = []
    prefix = "<memory-wiki-context>\n"
    suffix = "\n</memory-wiki-context>"
    used = len(prefix) + len(suffix)
    for score, entry in scored:
        chunk = f"- [{entry['category']}/{entry['source']}] {entry['title']}: {entry['text']}"
        cost = len(chunk) + (1 if chunks else 0)
        if used + cost > max_chars:
            continue
        shaped = {k: entry[k] for k in ("id", "source", "category", "title", "provenance")}
        shaped["score"] = score
        selected.append(shaped)
        chunks.append(chunk)
        used += cost

    raw_context = "\n".join(chunks)
    if not raw_context:
        return {"entries": [], "context": "", "used_chars": 0}
    context = prefix + raw_context + suffix
    return {"entries": selected, "context": context, "used_chars": len(context)}
