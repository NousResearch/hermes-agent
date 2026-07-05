"""Local memory linting and retrieval benchmark helpers.

These helpers are deliberately LLM-free and read-only. They give Hermes a cheap
quality gate for prompt-injected MEMORY.md/USER.md entries before any heavier
memory optimization loop is considered.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STALE_RE = re.compile(r"\b(PR|issue|commit|phase|submitted|registered|fixed|done|completed|yesterday|today)\b", re.IGNORECASE)
_SKILL_LIKE_RE = re.compile(r"\b(run|execute|use|always|never|first|then|after)\b.+\b(pytest|git|commit|command|script|workflow|steps?)\b", re.IGNORECASE)


def _entries() -> list[dict[str, Any]]:
    mem_dir = get_hermes_home() / "memories"
    rows: list[dict[str, Any]] = []
    for filename, target in (("MEMORY.md", "memory"), ("USER.md", "user")):
        path = mem_dir / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for idx, part in enumerate(text.split("§")):
            entry = part.strip()
            if entry:
                rows.append({"target": target, "path": str(path), "index": idx, "text": entry})
    return rows


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {_normalize_token(t) for t in _TOKEN_RE.findall(str(text)) if len(t) > 1}


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def lint_memory_entries(*, duplicate_threshold: float = 0.6) -> list[dict[str, Any]]:
    """Return memory quality findings without mutating memory files."""

    rows = _entries()
    findings: list[dict[str, Any]] = []
    prior: list[dict[str, Any]] = []
    for row in rows:
        issues: list[str] = []
        text = str(row["text"])
        if len(text) > 700:
            issues.append("too_verbose")
        if _STALE_RE.search(text):
            issues.append("stale_task_progress")
        if _SKILL_LIKE_RE.search(text):
            issues.append("belongs_in_skill")
        for earlier in prior:
            if _jaccard(text, str(earlier["text"])) >= duplicate_threshold:
                issues.append("near_duplicate")
                break
        prior.append(row)
        if issues:
            finding = dict(row)
            finding["issues"] = sorted(set(issues))
            findings.append(finding)
    return findings


def _matches_expected(text: str, expected: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(str(e).lower() in lowered for e in expected)


def benchmark_memory_retrieval(cases: list[dict[str, Any]], *, k: int = 3) -> dict[str, Any]:
    """Compute simple Precision@K and Recall@K over built-in memory entries.

    Each case is ``{"query": str, "expected": [substring, ...]}``; an entry is
    relevant if it contains one expected substring. Retrieval uses token-overlap
    ranking, which mirrors Hermes' cheap memory-wiki behavior closely enough for
    regression testing without embeddings or network calls.
    """

    rows = _entries()
    k = max(1, int(k))
    case_reports: list[dict[str, Any]] = []
    precision_sum = 0.0
    recall_sum = 0.0
    for case in cases:
        query = str(case.get("query") or "")
        expected = [str(e) for e in case.get("expected") or []]
        qtok = _tokens(query)
        ranked = sorted(
            rows,
            key=lambda r: (len(qtok & _tokens(str(r["text"]))), _jaccard(query, str(r["text"]))),
            reverse=True,
        )[:k]

        relevant_retrieved = [r for r in ranked if _matches_expected(str(r["text"]), expected)]
        relevant_total = sum(1 for r in rows if _matches_expected(str(r["text"]), expected))
        precision = len(relevant_retrieved) / k
        recall = (len(relevant_retrieved) / relevant_total) if relevant_total else 0.0
        precision_sum += precision
        recall_sum += recall
        case_reports.append(
            {
                "query": query,
                "expected": expected,
                "retrieved": ranked,
                "precision_at_k": precision,
                "recall_at_k": recall,
            }
        )
    total = len(cases)
    return {
        "cases": total,
        "k": k,
        "precision_at_k": round(precision_sum / total, 4) if total else 0.0,
        "recall_at_k": round(recall_sum / total, 4) if total else 0.0,
        "case_reports": case_reports,
    }
