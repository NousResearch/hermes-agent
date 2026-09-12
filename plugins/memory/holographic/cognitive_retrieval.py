"""Cognitive retrieval: query planning, FTS-robust fallback, context firewall,
memory budget, and structured diagnostics. Zero LLM calls (stdlib only).
"""

from __future__ import annotations

import re
import time

from .cognition import contains_instruction, contains_secret
from .taxonomy import (
    ACTIVE, AGING, ARCHIVED, CONDITIONAL, CONSTRAINT, DECISION, EVIDENCE,
    HYPOTHESIS, INVARIANT, LESSON, PATTERN, PREFERENCE, PROJECT_STATE,
    QUARANTINE, QUARANTINED, SAFE, STALE, SUPERSEDED, SUPERSEDED_STATE,
)

# Firewall classification by (mem_type, lifecycle, trust).
_SAFE_TYPES = frozenset({CONSTRAINT, INVARIANT, DECISION, EVIDENCE, PROJECT_STATE})
_CONDITIONAL_TYPES = frozenset({LESSON, PREFERENCE, PATTERN, "fact", "general", "event"})
_QUARANTINE_TYPES = frozenset({HYPOTHESIS, SUPERSEDED})
_QUARANTINE_LIFECYCLES = frozenset({STALE, SUPERSEDED_STATE, ARCHIVED, QUARANTINE})

_TRUST_SAFE_MIN = 0.3
_TRUST_CONDITIONAL_MIN = 0.15


def firewall_class(fact: dict) -> str:
    """Classify one fact dict as safe / conditional / quarantine.

    Untrusted-by-construction inputs (secrets, injected instructions,
    contradictions, stale, hypotheses, low trust) never reach SAFE.
    """
    content = str(fact.get("content", ""))
    if contains_secret(content) or contains_instruction(content):
        return QUARANTINED
    lifecycle = str(fact.get("lifecycle", ACTIVE)).lower()
    if lifecycle in _QUARANTINE_LIFECYCLES:
        return QUARANTINED
    mem_type = str(fact.get("mem_type", "general")).lower()
    if mem_type in _QUARANTINE_TYPES:
        return QUARANTINED
    trust = float(fact.get("trust_score", fact.get("trust", 0.0)) or 0.0)
    if mem_type in _SAFE_TYPES and trust >= _TRUST_SAFE_MIN:
        return SAFE
    if mem_type in _CONDITIONAL_TYPES and trust >= _TRUST_CONDITIONAL_MIN:
        return CONDITIONAL
    if mem_type in _SAFE_TYPES:
        return CONDITIONAL  # trusted type but weak trust -> conditional
    return QUARANTINED


_TOKEN_RE = re.compile(r"[A-Za-zก-๙0-9_]{2,}")


def plan_query(query: str) -> dict:
    """Deterministic query planner: decompose into entity/subject/time/type
    signals plus a sanitized token list. No LLM, no FTS syntax passthrough."""
    text = query or ""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    lowered = text.lower()
    time_hint = None
    if re.search(r"(?i)\b(latest|current|now)\b", text) or "ปัจจุบัน" in text or "ล่าสุด" in text:
        time_hint = "current"
    elif re.search(r"(?i)\b(old|history|previous)\b", text) or "ก่อน" in text or "เก่า" in text or "ประวัติ" in text:
        time_hint = "historical"
    mem_type_hint = None
    if re.search(r"(?i)\b(decision|decided)\b", text) or "ตัดสินใจ" in text:
        mem_type_hint = DECISION
    elif re.search(r"(?i)\b(constraint|forbidden)\b", text) or "ห้าม" in text:
        mem_type_hint = CONSTRAINT
    elif re.search(r"(?i)\b(prefer|like)\b", text) or "ชอบ" in text or "ต้องการ" in text:
        mem_type_hint = PREFERENCE
    elif re.search(r"(?i)\b(lesson|failed)\b", text) or "ใช้ไม่ได้" in text:
        mem_type_hint = LESSON
    # Entity guess: quoted spans + capitalized words (cheap, deterministic).
    entities = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
    entities += re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    return {
        "raw": text,
        "tokens": tokens,
        "entities": list(dict.fromkeys(entities))[:5],
        "time_hint": time_hint,
        "mem_type_hint": mem_type_hint,
        "is_empty": not tokens,
    }


def sanitize_like_token(token: str) -> str:
    """Escape LIKE wildcards in a token."""
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def fallback_like_search(conn, tokens: list[str], category=None,
                         min_trust: float = 0.0, limit: int = 10) -> list[dict]:
    """Tokenized LIKE fallback when FTS5 MATCH fails or yields nothing.

    Never raises on malformed input; returns [] only when nothing matches.
    """
    if not tokens:
        return []
    clauses, params = [], []
    for tok in tokens[:6]:
        clauses.append("content LIKE ? ESCAPE '\\'")
        params.append(f"%{sanitize_like_token(tok)}%")
    where = "(" + " OR ".join(clauses) + ")"
    if category:
        where += " AND category = ?"
        params.append(category)
    params.extend([min_trust, limit])
    sql = ("SELECT fact_id, content, category, tags, trust_score, retrieval_count,"
           " helpful_count, created_at, updated_at FROM facts "
           f"WHERE {where} AND trust_score >= ? ORDER BY trust_score DESC LIMIT ?")
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def apply_firewall(facts: list[dict], allow_conditional: bool = True) -> tuple[list, list, list]:
    """Split facts into (safe, conditional, quarantined)."""
    safe, cond, quar = [], [], []
    for fact in facts:
        cls = firewall_class(fact)
        fact = dict(fact)
        fact["firewall"] = cls
        if cls == SAFE:
            safe.append(fact)
        elif cls == CONDITIONAL and allow_conditional:
            cond.append(fact)
        else:
            quar.append(fact)
    return safe, cond, quar


def apply_budget(facts: list[dict], max_memories: int = 5,
                 max_chars: int = 2000) -> tuple[list, int]:
    """Enforce hard injection budget; returns (selected, estimated_tokens)."""
    selected, used = [], 0
    for fact in facts:
        text = str(fact.get("content", ""))
        if len(selected) >= max_memories or used + len(text) > max_chars:
            break
        selected.append(fact)
        used += len(text)
    return selected, used // 4  # ~4 chars per token


def rank_candidates(plan: dict, facts: list[dict]) -> list[dict]:
    """Deterministic combined ranking: token overlap + trust + recency +
    authority. Current knowledge outranks historical knowledge."""
    tokens = set(plan.get("tokens", []))
    entities = [e.lower() for e in plan.get("entities", [])]
    hint = plan.get("mem_type_hint")
    scored = []
    for fact in facts:
        content = str(fact.get("content", "")).lower()
        overlap = sum(1 for t in tokens if t in content)
        overlap_score = overlap / max(1, len(tokens))
        entity_bonus = 0.2 if any(e in content for e in entities) else 0.0
        trust = float(fact.get("trust_score", 0.0) or 0.0)
        lifecycle = str(fact.get("lifecycle", ACTIVE)).lower()
        recency = 1.0 if lifecycle == ACTIVE else 0.7 if lifecycle == AGING else 0.3
        type_bonus = 0.1 if hint and str(fact.get("mem_type", "")) == hint else 0.0
        base = float(fact.get("score", 0.0) or 0.0)
        score = 0.35 * overlap_score + 0.25 * trust + 0.15 * recency + entity_bonus + type_bonus + 0.25 * min(1.0, base)
        row = dict(fact)
        row["cognitive_score"] = round(score, 4)
        row["rank_reasons"] = (
            f"overlap={overlap_score:.2f} trust={trust:.2f} "
            f"lifecycle={lifecycle} entity_bonus={entity_bonus:.1f}"
        )
        scored.append(row)
    scored.sort(key=lambda r: r["cognitive_score"], reverse=True)
    return scored


def diagnose(operation: str, started: float, candidates: int, selected: int,
             quarantined: int, conflicts: int, superseded: int,
             llm_calls: int, tokens: int) -> dict:
    """Structured diagnostics answering why memories were recalled/rejected."""
    return {
        "operation": operation,
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
        "candidate_count": candidates,
        "selected_count": selected,
        "rejected_count": max(0, candidates - selected - quarantined),
        "quarantined_count": quarantined,
        "conflicts": conflicts,
        "superseded": superseded,
        "llm_calls": llm_calls,
        "estimated_tokens": tokens,
    }


__all__ = [
    "firewall_class", "plan_query", "fallback_like_search", "apply_firewall",
    "apply_budget", "rank_candidates", "diagnose",
]
