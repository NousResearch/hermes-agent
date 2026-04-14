"""Smart memory recall plugin (Phase D3).

Provides a lightweight two-layer memory recall strategy:
1. Rule-based pre-filter over typed memory entries.
2. Cheap algorithmic reranking with graceful fallback.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List

from tools.memory_tool import MemoryStore

_TYPE_BOOSTS = {
    "feedback": 2.0,
    "user": 1.0,
    "project": 1.0,
    "reference": 0.5,
    "uncategorized": 0.0,
}

_TOKEN_RE = re.compile(r"\b[a-z0-9_]+\b")
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}"
    r"(?:[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?"
    r"(?:z|[+-]\d{2}:\d{2})?\b",
    re.IGNORECASE,
)


class SmartRecall:
    """Recall relevant memory entries with graceful degradation."""

    def __init__(self, memory_store: MemoryStore | None = None):
        self.memory_store = memory_store or MemoryStore()

    def recall(
        self,
        query: str,
        top_k: int = 5,
        types: list[str] | None = None,
    ) -> list[dict]:
        """Recall relevant memories using a two-layer ranking strategy."""
        if top_k <= 0 or not query or not query.strip():
            return []

        try:
            self.memory_store.load_from_disk()
            entries = self._collect_entries(types=types)
            if not entries:
                return []

            candidates = self._prefilter(query=query, entries=entries, top_k=top_k)
            if not candidates:
                return []

            try:
                ranked = self._rank_candidates(query, candidates)
            except Exception:
                return [self._public_result(item) for item in candidates[:top_k]]

            return [self._public_result(item) for item in ranked[:top_k]]
        except Exception:
            return []

    def _collect_entries(self, types: list[str] | None = None) -> list[dict]:
        normalized_types = {
            memory_type.strip().lower()
            for memory_type in (types or [])
            if isinstance(memory_type, str) and memory_type.strip()
        }

        collected: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for target in ("user", "memory"):
            for item in self.memory_store._typed_entries(target):
                item_type = (item.get("type") or "uncategorized").lower()
                if normalized_types and item_type not in normalized_types:
                    continue

                content = (item.get("content") or "").strip()
                if not content:
                    continue

                key = (item_type, content)
                if key in seen:
                    continue
                seen.add(key)

                collected.append(
                    {
                        "type": item_type,
                        "title": self._entry_title(content),
                        "content": content,
                    }
                )

        return collected

    def _prefilter(self, query: str, entries: list[dict], top_k: int) -> list[dict]:
        scored: list[dict] = []

        for entry in entries:
            keyword_score = self._keyword_score(query, entry["content"])
            recency_score = self._recency_boost(entry["content"])
            type_score = _TYPE_BOOSTS.get(entry["type"], 0.0)
            prefilter_score = (keyword_score * 5.0) + recency_score + type_score

            scored.append(
                {
                    **entry,
                    "score": prefilter_score,
                    "_prefilter_score": prefilter_score,
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["title"]))
        return scored[: max(top_k * 3, top_k)]

    def _keyword_score(self, query: str, content: str) -> float:
        """Score a candidate by simple word overlap with the query."""
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return 0.0
        content_terms = set(self._tokenize(content))
        return len(query_terms & content_terms) / len(query_terms)

    def _rank_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rank candidates with a cheap TF-IDF-like relevance score."""
        if not candidates:
            return []

        query_terms = Counter(self._tokenize(query))
        if not query_terms:
            return sorted(candidates, key=lambda item: (-item["score"], item["title"]))

        tokenized_candidates: list[list[str]] = [
            self._tokenize(candidate["content"]) for candidate in candidates
        ]
        document_count = len(tokenized_candidates)
        document_frequency: Counter[str] = Counter()

        for tokens in tokenized_candidates:
            document_frequency.update(set(tokens))

        ranked: list[dict] = []
        for candidate, tokens in zip(candidates, tokenized_candidates):
            token_counts = Counter(tokens)
            total_terms = sum(token_counts.values()) or 1
            relevance_score = 0.0

            for term, query_weight in query_terms.items():
                if not token_counts.get(term):
                    continue
                tf = token_counts[term] / total_terms
                idf = math.log((document_count + 1) / (document_frequency[term] + 1)) + 1.0
                relevance_score += tf * idf * query_weight

            final_score = relevance_score + (candidate.get("_prefilter_score", 0.0) * 0.5)
            ranked.append({**candidate, "score": final_score})

        ranked.sort(key=lambda item: (-item["score"], item["title"]))
        return ranked

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall((text or "").lower())

    def _entry_title(self, content: str) -> str:
        try:
            return self.memory_store._entry_title(content)
        except Exception:
            first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
            title = first_line or content.strip()
            return title[:77].rstrip() + "..." if len(title) > 80 else title

    def _recency_boost(self, content: str) -> float:
        timestamp = self._extract_timestamp(content)
        if timestamp is None:
            return 0.0

        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_seconds = max((now - timestamp).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        return max(0.0, 1.0 - min(age_days, 365.0) / 365.0)

    @staticmethod
    def _extract_timestamp(content: str) -> datetime | None:
        match = _ISO_TS_RE.search(content or "")
        if not match:
            return None

        raw = match.group(0).replace(" ", "T")
        if raw.endswith(("z", "Z")):
            raw = raw[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(raw[:10])
            except ValueError:
                return None

        if parsed.tzinfo is None and "T" not in raw:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _public_result(item: Dict[str, object]) -> Dict[str, object]:
        return {
            "type": item["type"],
            "title": item["title"],
            "content": item["content"],
            "score": item["score"],
        }
