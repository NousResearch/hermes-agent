"""Approved, local source lookup for the VIAJA CON CARLOS plugin.

The knowledge directory is deliberately a small, read-only corpus.  This
module indexes only entries named in ``source-map.json`` and never synthesizes
facts: an empty or unresolved lookup is an explicit confirmation request.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

_PLUGIN_DIR = Path(__file__).resolve().parent
_KNOWLEDGE_DIR = _PLUGIN_DIR / "knowledge"
_SOURCE_MAP_PATH = _KNOWLEDGE_DIR / "source-map.json"
_ENTRY_RE = re.compile(r"\*\*\[([A-Z]{3}-\d{3})\]\*\*\s*(.*?)\s+—\s+source:", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS = {"a", "al", "de", "del", "el", "en", "la", "las", "los", "para", "por", "que", "un", "una", "y"}


@dataclass(frozen=True)
class _Entry:
    source_id: str
    excerpt: str
    document_path: str
    source_lines: str
    status: str = "approved"
    conflict_group: str | None = None
    aliases: tuple[str, ...] = ()


def normalize_text(value: Any) -> str:
    """Case-fold, de-accent, and collapse whitespace for matching."""
    text = "" if value is None else str(value)
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_marks.casefold().split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(normalize_text(value)))


def _load_map() -> Mapping[str, Any]:
    try:
        with _SOURCE_MAP_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {"entries": {}, "aliases": {}, "conflicts": {}}
    return payload if isinstance(payload, dict) else {"entries": {}}


def _document_file(document_path: str) -> Path:
    """Resolve a manifest document path inside the bundled knowledge root."""
    path = Path(str(document_path or ""))
    # The source map intentionally uses repo-relative paths.  Only the final
    # filename is needed here, and keeping resolution inside knowledge prevents
    # a malformed map from escaping the plugin directory.
    return _KNOWLEDGE_DIR / path.name


def _iter_entries() -> Iterable[_Entry]:
    source_map = _load_map()
    mapped = source_map.get("entries")
    if not isinstance(mapped, dict):
        return
    aliases_by_filename = source_map.get("aliases")
    if not isinstance(aliases_by_filename, dict):
        aliases_by_filename = {}

    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for source_id, metadata in mapped.items():
        if not isinstance(metadata, dict) or not isinstance(source_id, str):
            continue
        doc = str(metadata.get("document_path") or "")
        grouped.setdefault(Path(doc).name, []).append((source_id.upper(), metadata))

    for filename, entries in grouped.items():
        document = _document_file(filename)
        try:
            lines = document.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        aliases = tuple(str(alias) for alias in aliases_by_filename.get(filename, []) if alias)
        for line in lines:
            match = _ENTRY_RE.search(line)
            if not match:
                continue
            source_id, excerpt = match.group(1).upper(), match.group(2).strip()
            metadata = next((item for item_id, item in entries if item_id == source_id), None)
            if metadata is None:
                continue
            yield _Entry(
                source_id=source_id,
                excerpt=excerpt,
                document_path=str(metadata.get("document_path") or f"knowledge/{filename}"),
                source_lines=str(metadata.get("source_lines") or ""),
                status=str(metadata.get("status") or "approved"),
                conflict_group=(
                    str(metadata["conflict_group"])
                    if metadata.get("conflict_group")
                    else None
                ),
                aliases=aliases,
            )


def _score(entry: _Entry, terms: tuple[str, ...], aliases: tuple[str, ...]) -> int:
    haystack = normalize_text(entry.excerpt)
    alias_text = " ".join(normalize_text(alias) for alias in (*entry.aliases, *aliases))
    score = sum(2 for term in terms if term in haystack)
    score += sum(4 for term in terms if term in alias_text)
    # A whole alias/property phrase is a strong signal, while a lone common
    # term (e.g. "price") remains useful but cannot surface every document.
    normalized_aliases = tuple(normalize_text(alias) for alias in (*entry.aliases, *aliases))
    score += sum(8 for alias in normalized_aliases if alias and alias in " ".join(terms))
    return score


def _has_relevant_fact(entry: _Entry, terms: tuple[str, ...], aliases: tuple[str, ...]) -> bool:
    """Reject alias-only matches when the query names an unanswered detail."""
    alias_tokens = set(_tokens(" ".join((*entry.aliases, *aliases))))
    detail_terms = tuple(term for term in terms if term not in alias_tokens and term not in _STOPWORDS)
    if not detail_terms:
        return True
    haystack = normalize_text(entry.excerpt)
    # A requested date/year or other number must be present in the approved
    # excerpt; otherwise the result would look like a quote for unknown dates.
    numeric_terms = tuple(term for term in detail_terms if term.isdigit())
    if numeric_terms and not all(term in haystack for term in numeric_terms):
        return False
    return any(term in haystack for term in detail_terms)


def lookup_sources(
    query: str,
    property_hint: str | None = None,
    topic_hint: str | None = None,
    *,
    max_results: int = 8,
) -> dict[str, Any]:
    """Return relevant, attributed knowledge excerpts without guessing.

    ``property_hint`` and ``topic_hint`` only improve retrieval.  They are not
    treated as facts.  ``confirmation_required`` is true when no approved
    excerpt answers the request or when a relevant excerpt belongs to an
    explicitly unresolved conflict group.
    """
    raw_query = " ".join(value for value in (query, property_hint, topic_hint) if value)
    normalized_query = normalize_text(raw_query)
    terms = _tokens(raw_query)
    explicit_aliases = tuple(value for value in (property_hint, topic_hint) if value)

    ranked: list[tuple[int, _Entry]] = []
    for entry in _iter_entries():
        if not _has_relevant_fact(entry, terms, explicit_aliases):
            continue
        score = _score(entry, terms, explicit_aliases)
        if score:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].source_id))

    selected = [entry for score, entry in ranked if score >= 2][: max(0, max_results)]
    excerpts = [
        {
            "source_id": entry.source_id,
            "excerpt": entry.excerpt,
            "document_path": entry.document_path,
            "source_lines": entry.source_lines,
            **({"status": entry.status} if entry.status != "approved" else {}),
            **(
                {"conflict_group": entry.conflict_group}
                if entry.conflict_group
                else {}
            ),
        }
        for entry in selected
    ]

    conflict = any(entry.status == "confirmation-required" or entry.conflict_group for entry in selected)
    if conflict:
        reason = "conflicting_facts"
    elif not selected:
        reason = "missing_fact"
    else:
        reason = None
    return {
        "query": query,
        "normalized_query": normalized_query,
        "excerpts": excerpts,
        "confirmation_required": bool(conflict or not selected),
        "confirmation_reason": reason,
    }


# Stable aliases make the public contract convenient for plugin callers while
# keeping one implementation and one registered tool.
source_lookup = lookup_sources


def source_lookup_tool(args: Mapping[str, Any] | None = None) -> str:
    """Registry handler: serialize a lookup result as the required JSON string."""
    payload = args if isinstance(args, Mapping) else {}
    result = lookup_sources(
        str(payload.get("query") or ""),
        property_hint=(str(payload["property_hint"]) if payload.get("property_hint") else None),
        topic_hint=(str(payload["topic_hint"]) if payload.get("topic_hint") else None),
    )
    return json.dumps(result, ensure_ascii=False)
