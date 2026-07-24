"""Conservative, bounded post-processing for Hindsight recall results.

The Hindsight service owns retrieval and ranking. This module only removes exact
or narrowly defined equivalent repeats, optionally prefers explicitly configured
authority tags, and bounds what Hermes injects into context or returns from its
tool. It never mutates server results or stored memory.
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
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("recall_max_results must be between 1 and 20")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("recall_max_results must be between 1 and 20") from exc
    if not _MIN_MAX_RESULTS <= normalized <= _MAX_MAX_RESULTS:
        raise ValueError("recall_max_results must be between 1 and 20")
    return normalized


def _display_text(value: Any) -> str:
    """Return one safe display line; recalled text cannot forge result markers."""
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    try:
        return tuple(str(tag).strip() for tag in value if str(tag).strip())
    except TypeError:
        stripped = str(value).strip()
        return (stripped,) if stripped else ()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def equivalence_key(text: str) -> str:
    """Return a conservative key for exact text and atomic name statements."""
    normalized = normalize_text(text)
    patterns = (
        r"(?:the )?user s name is ([a-z][a-z ]*)",
        r"(?:the )?user is named ([a-z][a-z ]*)",
        r"([a-z]+) is (?:the )?user",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, normalized)
        if match:
            return f"identity:name:{match.group(1).strip()}"
    return f"text:{normalized}"


def authority_tier(
    tags: Sequence[str], authority_tags: Sequence[str]
) -> tuple[int, str]:
    """Classify provenance; only explicitly configured tags can reorder results."""
    tag_set = {str(tag).casefold() for tag in tags}
    configured = {str(tag).casefold() for tag in authority_tags}
    if configured.intersection(tag_set):
        return 1, "authoritative"
    if any(tag.startswith(("session:", "parent:")) for tag in tag_set):
        return 0, "session-derived"
    return 0, "unclassified"


def rank_and_deduplicate(
    results: Iterable[Any],
    *,
    max_results: int | None = DEFAULT_MAX_RESULTS,
    authority_tags: Sequence[str] = (),
) -> list[RecallItem]:
    """Collapse conservative equivalents and preserve server order by default."""
    if max_results is not None:
        max_results = normalize_max_results(max_results)
    winners: dict[str, tuple[int, RecallItem]] = {}
    for index, result in enumerate(results):
        text = _display_text(_field(result, "text", None))
        if not text:
            continue
        tags = _tags(_field(result, "tags", ()))
        tier, authority = authority_tier(tags, authority_tags)
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
    items = [item for _, item in ordered]
    return items if max_results is None else items[:max_results]


def prepare_recall_results(
    results: Iterable[Any],
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    authority_tags: Sequence[str] = (),
) -> tuple[list[RecallItem], int]:
    """Return bounded unique results plus the number of distinct rows omitted."""
    cap = normalize_max_results(max_results)
    all_items = rank_and_deduplicate(
        results, max_results=None, authority_tags=authority_tags
    )
    return all_items[:cap], max(0, len(all_items) - cap)


def format_recall_results(
    items: Sequence[RecallItem], *, numbered: bool, omitted_count: int = 0
) -> str:
    """Render one line per result and disclose any distinct omitted results."""
    if not items:
        return ""
    lines = []
    for index, item in enumerate(items, 1):
        marker = f"{index}." if numbered else "-"
        lines.append(f"{marker} [{item.authority}] {item.text}")
    if omitted_count:
        noun = "result" if omitted_count == 1 else "results"
        lines.append(
            f"({omitted_count} more distinct {noun} not shown; refine the query to narrow recall.)"
        )
    return "\n".join(lines)
