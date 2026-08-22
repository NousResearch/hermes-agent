"""Capability discovery P0/P1 — local only, no GBrain/Obsidian/NotebookLM.

P0: filesystem read of reports and codebase.
P1: read-only state_meta and hermes_memory.
Pure functions, no LLM, no provider calls, no network.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

from .types import (
    CapabilityCandidate,
    CapabilityDiscovery,
    NormalizedObjective,
    new_uuid,
    now_iso8601,
)

STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "is", "are", "be", "this", "that", "it", "as",
})

DEFAULT_REPORTS_DIR = Path.home() / ".hermes" / "reports"
PROJECT_ROOT_CANDIDATES = (
    Path("/home/jr-ubuntu/.hermes/hermes-agent/agent"),
    Path("/home/jr-ubuntu/.hermes/hermes-agent/hermes_cli"),
    Path("/home/jr-ubuntu/.hermes/hermes-agent/tools"),
    Path("/home/jr-ubuntu/.hermes/hermes-agent/optional-skills"),
)


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    return {t for t in tokens if t and t not in STOPWORDS and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _extract_keywords_from_normalized(n: NormalizedObjective) -> set[str]:
    keywords: set[str] = set()
    for c in n.constraints:
        keywords.update(_tokenize(c))
    for sc in n.success_criteria:
        keywords.update(_tokenize(sc))
    for kr in n.knowledge_requirements:
        if kr.startswith("kb:"):
            keywords.add(kr[3:])
    return keywords


def _read_top(path: Path, lines: int = 200) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[:lines])
    except (OSError, IOError):
        return ""


def _read_docstrings(path: Path) -> str:
    """Extract module/class/function docstrings from a Python file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError):
        return ""
    # Match triple-quoted strings ("""...""" or '''...''').
    matches = re.findall(r'(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', text, re.DOTALL)
    # Each match is a tuple (group1, group2) where one is empty.
    parts = [a or b for a, b in matches]
    return " ".join(parts)


def _discover_p0_reports(keywords: set[str], threshold: float = 0.3) -> list[CapabilityCandidate]:
    if not DEFAULT_REPORTS_DIR.is_dir():
        return []
    candidates: list[CapabilityCandidate] = []
    for report_path in DEFAULT_REPORTS_DIR.rglob("*.md"):
        text = _read_top(report_path, 200)
        text_tokens = _tokenize(text)
        score = _jaccard(keywords, text_tokens)
        if score >= threshold:
            candidates.append(
                CapabilityCandidate(
                    kind="report",
                    id=str(report_path),
                    name=report_path.name,
                    source_path=str(report_path),
                    description=text[:200],
                    keywords=tuple(sorted(text_tokens)),
                    match_score=score,
                    match_reasons=("p0:report_jaccard",),
                )
            )
    return candidates


def _discover_p0_codebase(
    keywords: set[str], threshold: float = 0.3
) -> list[CapabilityCandidate]:
    candidates: list[CapabilityCandidate] = []
    for root in PROJECT_ROOT_CANDIDATES:
        if not root.is_dir():
            continue
        for py_path in root.rglob("*.py"):
            text = _read_docstrings(py_path)
            text_tokens = _tokenize(text)
            score = _jaccard(keywords, text_tokens)
            if score >= threshold:
                kind = "tool" if "/tools/" in str(py_path) else "module"
                candidates.append(
                    CapabilityCandidate(
                        kind=kind,
                        id=str(py_path),
                        name=py_path.stem,
                        source_path=str(py_path),
                        description=text[:200],
                        keywords=tuple(sorted(text_tokens)),
                        match_score=score,
                        match_reasons=("p0:codebase_docstring_jaccard",),
                    )
                )
    return candidates


def _discover_p1_state_meta(
    keywords: set[str], threshold: float = 0.3
) -> list[CapabilityCandidate]:
    """Read-only P1 query via state_meta. No writes."""
    candidates: list[CapabilityCandidate] = []
    try:
        from hermes_state import get_session_db
        db = get_session_db()
    except Exception:
        return candidates
    try:
        try:
            raw_keys = db.list_meta_keys(prefix="objective:")
        except AttributeError:
            raw_keys = []
        for k in raw_keys or []:
            try:
                value = db.get_meta(k)
            except Exception:
                continue
            if not isinstance(value, str):
                continue
            value_tokens = _tokenize(value)
            score = _jaccard(keywords, value_tokens)
            if score >= threshold:
                candidates.append(
                    CapabilityCandidate(
                        kind="sessiondb_state",
                        id=k,
                        name=k,
                        source_path=f"state_meta:{k}",
                        description=value[:200],
                        keywords=tuple(sorted(value_tokens)),
                        match_score=score,
                        match_reasons=("p1:sessiondb_jaccard",),
                    )
                )
    finally:
        try:
            db.close()
        except Exception:
            pass
    return candidates


def _discover_p1_memory(
    normalized: NormalizedObjective,
    keywords: set[str],
    threshold: float = 0.3,
) -> list[CapabilityCandidate]:
    """Read-only P1 query via hermes_memory. No writes."""
    candidates: list[CapabilityCandidate] = []
    try:
        from agent.memory_manager import read_user_memory
    except Exception:
        return candidates
    try:
        snapshot = read_user_memory(normalized.created_by)
    except Exception:
        return candidates
    if not isinstance(snapshot, str):
        return candidates
    snap_tokens = _tokenize(snapshot)
    score = _jaccard(keywords, snap_tokens)
    if score >= threshold:
        candidates.append(
            CapabilityCandidate(
                kind="memory",
                id=f"memory:{normalized.created_by}",
                name=f"User memory: {normalized.created_by}",
                source_path="agent.memory_manager",
                description=snapshot[:200],
                keywords=tuple(sorted(snap_tokens)),
                match_score=score,
                match_reasons=("p1:memory_jaccard",),
            )
        )
    return candidates


def _decide(
    candidates: list[CapabilityCandidate],
) -> tuple[str, str, tuple[str, ...]]:
    if not candidates:
        return ("generate", "no candidates found", ("no_capability",))
    candidates_sorted = sorted(
        candidates, key=lambda c: c.match_score, reverse=True
    )
    best = candidates_sorted[0]
    if best.match_score >= 0.7:
        return ("reuse", f"best candidate {best.id} score={best.match_score:.2f}", ())
    if best.match_score >= 0.3:
        return (
            "hybrid",
            f"best {best.id} score={best.match_score:.2f} may need gap-filling",
            (),
        )
    return ("generate", "best below threshold", ("low_match_score",))


def discover_capabilities_p0_p1(
    normalized: NormalizedObjective,
    *,
    objective_id: str,
) -> CapabilityDiscovery:
    """Discover capabilities from P0 (reports, codebase) and P1
    (state_meta, memory). No GBrain, no Obsidian, no NotebookLM.
    """
    keywords = _extract_keywords_from_normalized(normalized)

    t0 = time.time()
    p0_candidates: list[CapabilityCandidate] = []
    p0_candidates.extend(_discover_p0_reports(keywords))
    p0_candidates.extend(_discover_p0_codebase(keywords))
    p0_ms = int((time.time() - t0) * 1000)

    t1 = time.time()
    p1_candidates: list[CapabilityCandidate] = []
    p1_candidates.extend(_discover_p1_state_meta(keywords))
    p1_candidates.extend(_discover_p1_memory(normalized, keywords))
    p1_ms = int((time.time() - t1) * 1000)

    all_candidates = p0_candidates + p1_candidates
    reuse_decision, rationale, gaps = _decide(all_candidates)

    return CapabilityDiscovery(
        objective_id=objective_id,
        discovered_at=now_iso8601(),
        candidates=tuple(all_candidates),
        reuse_decision=reuse_decision,
        rationale=rationale,
        gaps=gaps,
        p0_query_duration_ms=p0_ms,
        p1_query_duration_ms=p1_ms,
    )
