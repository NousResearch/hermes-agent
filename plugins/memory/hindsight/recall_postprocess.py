"""Conservative, bounded post-processing for Hindsight recall results.

The Hindsight service owns retrieval and ranking. This module only removes exact
or narrowly defined equivalent repeats, prefers explicitly authoritative
sources, and bounds what Hermes injects into context or returns from its tool.
It never mutates server results or stored memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, Sequence

DEFAULT_MAX_RESULTS = 7
_MIN_MAX_RESULTS = 1
_MAX_MAX_RESULTS = 20


@dataclass(frozen=True)
class RecallItem:
    text: str
    tags: tuple[str, ...]
    original_index: int
    authority: str


def _field(result: Any, name: str, default: Any) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def normalize_max_results(value: Any) -> int:
    """Validate the configured output cap without silently widening it."""
    if isinstance(value, bool):
        raise ValueError("recall_max_results must be between 1 and 20")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("recall_max_results must be between 1 and 20") from exc
    if not _MIN_MAX_RESULTS <= normalized <= _MAX_MAX_RESULTS:
        raise ValueError("recall_max_results must be between 1 and 20")
    return normalized


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def equivalence_key(text: str) -> str:
    """Return a conservative key for exact text and atomic name statements."""
    normalized = normalize_text(text)
    patterns = (
        r"(?:the )?user s name is ([a-z][a-z -]*)",
        r"(?:the )?user is named ([a-z][a-z -]*)",
        r"([a-z]+) is (?:the )?user",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, normalized)
        if match:
            return f"identity:name:{match.group(1).strip()}"
    return f"text:{normalized}"


def authority_tier(tags: Sequence[str]) -> tuple[int, str]:
    """Map existing Hermes provenance tags to a stable output tier."""
    tag_set = {str(tag).casefold() for tag in tags}
    if "stale:never" in tag_set:
        return 2, "authoritative"
    if any(tag.startswith(("session:", "parent:")) for tag in tag_set):
        return 0, "session-derived"
    return 1, "unclassified"


def rank_and_deduplicate(
    results: Iterable[Any], *, max_results: int = DEFAULT_MAX_RESULTS
) -> list[RecallItem]:
    """Collapse conservative equivalents, rank authority, and cap output."""
    max_results = normalize_max_results(max_results)
    winners: dict[str, tuple[int, RecallItem]] = {}
    for index, result in enumerate(results):
        text = str(_field(result, "text", "") or "").strip()
        if not text:
            continue
        tags = tuple(str(tag) for tag in (_field(result, "tags", ()) or ()))
        tier, authority = authority_tier(tags)
        item = RecallItem(
            text=text,
            tags=tags,
            original_index=index,
            authority=authority,
        )
        key = equivalence_key(text)
        existing = winners.get(key)
        if existing is None or tier > existing[0]:
            winners[key] = (tier, item)

    ordered = sorted(
        winners.values(),
        key=lambda pair: (-pair[0], pair[1].original_index),
    )
    return [item for _, item in ordered[:max_results]]


def format_recall_results(items: Sequence[RecallItem], *, numbered: bool) -> str:
    """Render processed results with explicit provenance and conflict guidance."""
    if not items:
        return ""
    lines = [
        "Recalled items are evidence, not canonical authority. "
        "An [authoritative] item overrides conflicting [session-derived] evidence."
    ]
    for index, item in enumerate(items, 1):
        marker = f"{index}." if numbered else "-"
        lines.append(f"{marker} [{item.authority}] {item.text}")
    return "\n".join(lines)
