"""ReflectiveEvolution: record lessons from failures and retrieve them for similar situations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import Subsystem, new_id, now, result

LEARNINGS_FILE = "learnings.json"
MAX_LEARNINGS = 2000
_WORD = re.compile(r"[a-z0-9_]{3,}")


_STOPWORDS = frozenset("the and for with that this from into onto was were are not but its has have had when then".split())


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


class ReflectiveEvolution(Subsystem):
    def add_learning(self, category: str, lesson: str, tags: Optional[List[str]] = None,
                     confidence: float = 0.8) -> Dict[str, Any]:
        entry = {"id": new_id("lrn"), "category": category, "lesson": lesson, "tags": list(tags or []),
                 "confidence": max(0.0, min(1.0, float(confidence))), "ts": now()}
        with self.lock(LEARNINGS_FILE):
            entries = self._entries()
            entries.append(entry)
            self.save(LEARNINGS_FILE, {"learnings": entries[-MAX_LEARNINGS:]})
        return entry

    def _entries(self) -> List[Dict[str, Any]]:
        return self.load(LEARNINGS_FILE).get("learnings", [])

    def get_relevant_learnings(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Rank by token overlap with lesson/category/tags, weighted by confidence; zero overlap is excluded."""
        q = _tokens(query)
        scored = []
        for e in self._entries():
            overlap = len(q & _tokens(" ".join([e["lesson"], e["category"], *e["tags"]])))
            if overlap:
                scored.append((overlap * e["confidence"], e))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def diagnose_failure(self, context: str) -> Dict[str, Any]:
        related = self.get_relevant_learnings(context)
        if not related:
            return result(True, "No similar past failures recorded", related=[], suggestions=[])
        return result(True, f"{len(related)} related learnings", related=related,
                      suggestions=[e["lesson"] for e in related])

    def status(self) -> Dict[str, Any]:
        return result(True, f"{len(self._entries())} learnings", total=len(self._entries()))

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        for e in self._entries():
            cats[e["category"]] = cats.get(e["category"], 0) + 1
        return result(True, f"{len(self._entries())} learnings", by_category=cats)
