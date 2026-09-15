"""Query-specificity gates for destructive preprocess.

Stdlib only. No I/O. Do not import compress_context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .unicode_profile import matching_shadow, script_of, search_units

GENERIC_WORDS = frozenset(
    {
        "what",
        "how",
        "does",
        "do",
        "the",
        "is",
        "are",
        "was",
        "were",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "this",
        "that",
        "with",
        "into",
        "about",
        "for",
        "and",
        "or",
        "of",
        "to",
        "a",
        "an",
        "in",
        "on",
        "it",
        "be",
        "as",
        "by",
        "from",
        "at",
        "if",
        "not",
        "can",
        "you",
        "we",
        "they",
        "them",
        "their",
        "our",
        "your",
        "please",
        "tell",
        "give",
        "show",
        "explain",
        "summarize",
        "summary",
        "function",
        "return",
        "class",
        "def",
        "import",
        "hmm",
        "cert",
        "code",
        "codes",
        "part",
        "parts",
        "item",
        "items",
        "list",
        "every",
        "catalog",
        "name",
        "names",
        "value",
        "values",
        "data",
        "info",
        "keep",
        "track",
        "detail",
        "details",
        "weather",
        "paris",
        "capital",
        "special",
    }
)

LATIN_IDENT_RE = re.compile(
    r"[A-Za-z]*\d[A-Za-z0-9_-]*"
    r"|[A-Za-z0-9]+(?:[_./-][A-Za-z0-9]+)+"
    r"|[A-Z]{2,24}"
    r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*"
)

NON_LATIN_RE = re.compile(
    r"[\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0900-\u097F"
    r"\u0E00-\u0E7F\u3400-\u9FFF\uAC00-\uD7A3]"
)

_HYPHEN_RE = re.compile(r"\b\w+(?:-\w+)+\b")


def distinctive_query_terms(query: str) -> list[str]:
    """Return first-seen distinctive selectors; empty if the query is generic."""
    if not query or not str(query).strip():
        return []
    raw_text = str(query)
    text = matching_shadow(raw_text)
    seen: set[str] = set()
    out: list[str] = []

    def _add(raw: str) -> None:
        key = raw.casefold()
        if key in GENERIC_WORDS or len(key) < 2 or key in seen:
            return
        seen.add(key)
        out.append(key)

    for m in LATIN_IDENT_RE.findall(raw_text):
        _add(m)
    for m in _HYPHEN_RE.findall(text):
        if len(m) >= 3:
            _add(m)
    for unit in search_units(text)[:128]:
        representative = next((char for char in unit if char.isalnum()), unit[0])
        script = script_of(representative)
        if (
            script not in {"han", "kana", "hangul", "arabic", "thai", "latin"}
            and any(ord(char) > 127 for char in unit)
            and len(unit) >= 2
        ):
            _add(unit)
    nl = NON_LATIN_RE.search(text)
    if nl:
        sentinel = f"script:{nl.group(0)}"
        if sentinel not in seen:
            seen.add(sentinel)
            out.append(sentinel)
    return out


def query_has_distinctive_selectors(query: str) -> bool:
    """True when the query names something specific enough to crush on."""
    return bool(distinctive_query_terms(query))

@dataclass
class SemanticContractVerdict:
    passed: bool
    answer_recall: float = 1.0
    missing_evidence: list[str] = field(default_factory=list)
    forbidden_hits: tuple[str, ...] = field(default_factory=tuple)


def evaluate_semantic_contract(text: str, spec: dict) -> SemanticContractVerdict:
    answers = spec.get("answers", [])
    required_evidence = spec.get("required_evidence", [])
    forbidden_distractors = spec.get("forbidden_distractors", [])

    missing_evidence = [ev for ev in required_evidence if ev not in text]
    forbidden_hits = tuple(d for d in forbidden_distractors if d in text)

    answer_hits = sum(1 for a in answers if a in text)
    answer_recall = (answer_hits / len(answers)) if answers else 1.0

    passed = (
        len(missing_evidence) == 0
        and len(forbidden_hits) == 0
        and answer_recall >= 1.0
    )
    return SemanticContractVerdict(
        passed=passed,
        answer_recall=answer_recall,
        missing_evidence=missing_evidence,
        forbidden_hits=forbidden_hits,
    )
