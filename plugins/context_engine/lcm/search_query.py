"""Helpers for search query handling across FTS and LIKE fallback paths."""

from __future__ import annotations

import re
from typing import List

_CJK_RE = re.compile(
    r"["
    r"\u3400-\u4dbf"
    r"\u4e00-\u9fff"
    r"\u3000-\u303f"
    r"\u3040-\u30ff"
    r"\uac00-\ud7af"
    r"\uff00-\uffef"
    r"]"
)
_EMOJI_RE = re.compile(
    r"["
    r"\u2600-\u27bf"
    r"\U0001F300-\U0001FAFF"
    r"]"
)
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
_BOOLEAN_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_RISKY_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9][\-:/][A-Za-z0-9]")
_SPLIT_PUNCT_RE = re.compile(r"[-:/]+")
_STRIP_EDGE_PUNCT = "\"'()[]{}.,;"
# Characters that are special in FTS5 query syntax
_FTS5_SPECIAL_CHARS = frozenset('"()*^-:{}.')


def _sanitize_unquoted_fts5_fragment(text: str) -> str:
    return "".join(" " if char in _FTS5_SPECIAL_CHARS else char for char in text)


def sanitize_fts5_query(query: str) -> str:
    """Strip FTS5 syntax operators while preserving balanced phrase quotes."""
    if not query:
        return ""

    result: list[str] = []
    quote_buffer: list[str] = []
    in_quote = False
    for char in query:
        if char == '"':
            if in_quote:
                result.append('"')
                result.extend(quote_buffer)
                result.append('"')
                quote_buffer = []
                in_quote = False
            else:
                if result and not result[-1].isspace():
                    result.append(" ")
                in_quote = True
                quote_buffer = []
            continue
        if in_quote:
            quote_buffer.append(char)
            continue
        result.append(" " if char in _FTS5_SPECIAL_CHARS else char)
    if in_quote and quote_buffer:
        result.extend(_sanitize_unquoted_fts5_fragment("".join(quote_buffer)))
    return "".join(result).strip()


_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


def contains_risky_fts_ascii(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.count('"') % 2:
        return True
    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", raw)
    return bool(_RISKY_FTS_TOKEN_RE.search(text_without_phrases))


def requires_like_fallback(query: str) -> bool:
    return contains_cjk(query) or contains_emoji(query) or contains_risky_fts_ascii(query)


def _token_variants(token: str) -> List[str]:
    cleaned = (token or "").strip().strip(_STRIP_EDGE_PUNCT)
    if not cleaned:
        return []
    if cleaned.upper() in _BOOLEAN_OPERATORS:
        return []

    variants = [cleaned]
    if _SPLIT_PUNCT_RE.search(cleaned):
        parts = [part for part in _SPLIT_PUNCT_RE.split(cleaned) if part]
        if len(parts) > 1:
            variants.extend(parts)

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def extract_search_terms(query: str) -> List[str]:
    text = (query or "").strip()
    if not text:
        return []

    terms: list[str] = []
    for phrase in _QUOTED_PHRASE_RE.findall(text):
        cleaned = phrase.strip()
        if cleaned:
            terms.append(cleaned)

    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", text)
    for token in text_without_phrases.split():
        terms.extend(_token_variants(token))

    if not terms:
        fallback_text = text.strip().strip(_STRIP_EDGE_PUNCT)
        if fallback_text:
            terms.append(fallback_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def extract_quoted_phrases(query: str) -> List[str]:
    return [phrase.strip() for phrase in _QUOTED_PHRASE_RE.findall(query or "") if phrase.strip()]


def escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def count_term_matches(text: str, term: str) -> int:
    haystack = (text or "")
    needle = (term or "")
    if not haystack or not needle:
        return 0
    return haystack.lower().count(needle.lower())


def compute_directness_score(text: str, terms: List[str], phrases: List[str] | None = None) -> float:
    content = text or ""
    if not content:
        return 0.0

    unique_hits = 0
    total_hits = 0
    non_phrase_unique_hits = 0
    non_phrase_total_hits = 0
    normalized_phrases = {(phrase or "").strip().lower() for phrase in (phrases or []) if (phrase or "").strip()}
    for term in terms:
        matches = count_term_matches(content, term)
        if matches > 0:
            unique_hits += 1
            total_hits += matches
            if term.strip().lower() not in normalized_phrases:
                non_phrase_unique_hits += 1
                non_phrase_total_hits += matches

    phrase_hits = 0
    lowered = content.lower()
    for phrase in phrases or []:
        if phrase and phrase.lower() in lowered:
            phrase_hits += 1

    repetition_penalty = max(0, total_hits - unique_hits)
    non_phrase_repetition_penalty = max(0, non_phrase_total_hits - non_phrase_unique_hits)
    score = float((unique_hits * 5) + (phrase_hits * 8))
    if not phrases:
        score -= min(repetition_penalty, 6)
    else:
        score -= min(non_phrase_repetition_penalty, 6)

    if phrases:
        for phrase in phrases:
            normalized_phrase = (phrase or "").strip().lower()
            if not normalized_phrase:
                continue
            phrase_occurrences = lowered.count(normalized_phrase)
            if phrase_occurrences <= 1:
                continue
            segments = re.split(re.escape(normalized_phrase), lowered)
            gap_unique_counts = []
            for segment in segments:
                segment_tokens = [
                    token.lower()
                    for token in _WORD_RE.findall(segment)
                    if any(char.isalpha() for char in token)
                ]
                gap_unique_counts.append(len(set(segment_tokens)))
            interior_gap_counts = gap_unique_counts[1:-1]
            tail_gap_count = gap_unique_counts[-1] if gap_unique_counts else 0
            extra_occurrences = phrase_occurrences - 1
            score -= extra_occurrences * 0.5
            score -= sum(1.5 for count in interior_gap_counts if 0 < count <= 4)
            if all(count == 0 for count in interior_gap_counts) and tail_gap_count <= 2:
                score -= min(extra_occurrences, 3) * 1.0

    return score


def _is_precise_query_shape(terms: List[str], phrases: List[str] | None = None) -> bool:
    if len(terms) == 1:
        return True
    return len(phrases or []) == 1 and len(terms) <= 2


def should_widen_candidate_fetch(terms: List[str], phrases: List[str] | None = None) -> bool:
    return _is_precise_query_shape(terms, phrases)


def should_apply_directness_rank_adjustment(terms: List[str], phrases: List[str] | None = None) -> bool:
    return _is_precise_query_shape(terms, phrases)


def compute_directness_rank_bonus_upper_bound(terms: List[str], phrases: List[str] | None = None) -> float:
    return float((len(terms) * 5) + (len(phrases or []) * 8))


def compute_search_fetch_limit(limit: int, terms: List[str], phrases: List[str] | None = None) -> int:
    base = max(limit * 5, limit, 20)
    if should_widen_candidate_fetch(terms, phrases):
        return max(base, limit * 10, 50)
    return base


def compute_like_fallback_fetch_limit(limit: int, terms: List[str], phrases: List[str] | None = None) -> int:
    """Bound LIKE fallback candidate rows before Python-side scoring/sorting."""
    return compute_search_fetch_limit(limit, terms, phrases)


def compute_search_candidate_cap(limit: int) -> int:
    """Return a hard upper bound on candidate rows inspected per search call."""
    return min(max(limit * 20, limit, 500), 5_000)


AGE_DECAY_RATE = 0.001


def normalize_search_sort(sort: str | None) -> str:
    """Normalize sort parameter to one of: recency, relevance, hybrid."""
    normalized = (sort or "recency").strip().lower()
    return normalized if normalized in {"recency", "relevance", "hybrid"} else "recency"


def build_snippet(text: str, terms: List[str], width: int = 80) -> str:
    content = (text or "")
    if not content:
        return ""
    lowered = content.lower()
    for term in terms:
        if not term:
            continue
        idx = lowered.find(term.lower())
        if idx >= 0:
            start = max(0, idx - width // 2)
            end = min(len(content), idx + len(term) + width // 2)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet
    return content[:width] + ("..." if len(content) > width else "")
