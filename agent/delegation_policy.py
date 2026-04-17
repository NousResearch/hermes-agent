"""Delegation policy + subagent planner for routing v2.

Zero-token, deterministic. Additive to routing_v2.select_model output.

Public API:
    should_delegate(prompt, decision, *, cheap_model="gpt-5-mini", exceptions=None) -> (bool, reason)
    compute_delegation_rate(decisions) -> float (percent)
    assert_rate_ok(rate, floor=75.0) -> bool
    plan_subagents(prompt, decision, *, max_parallel=3) -> dict
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_GREETINGS = {"hola", "hi", "hello", "ok", "gracias", "thanks", "hey"}
_TRIVIAL_TIME_RE = re.compile(r"\b(time|hora|fecha|date|day|día|dia)\b", re.IGNORECASE)

_FILE_RE = re.compile(r"[\w./\\-]+\.(py|js|ts|go|rs|java|cpp|c|rb|php|sh)\b", re.IGNORECASE)
_VS_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+)$")
_DRAFTS_RE = re.compile(r"(\d+)\s+(drafts?|variantes?|versions?|versiones?)", re.IGNORECASE)


def _is_exception(prompt: str, exceptions: Optional[Iterable[str]] = None) -> Optional[str]:
    low = (prompt or "").strip().lower()
    if not low:
        return None
    if low in _GREETINGS:
        return "greeting"
    # trivial time/date queries, only when short
    if len(low.split()) <= 6 and _TRIVIAL_TIME_RE.search(low):
        return "time_query"
    if exceptions:
        for pat in exceptions:
            try:
                if re.search(pat, low, re.IGNORECASE):
                    return f"user_pattern:{pat}"
            except re.error:
                continue
    return None


def should_delegate(prompt: str,
                    decision: Dict[str, Any],
                    *,
                    cheap_model: str = "gpt-5-mini",
                    exceptions: Optional[Iterable[str]] = None) -> Tuple[bool, str]:
    exc = _is_exception(prompt, exceptions)
    if exc:
        return (False, f"exception:{exc}")
    tier = int(decision.get("tier", 1))
    model = decision.get("model") or ""
    if tier >= 3 or (model and model != cheap_model):
        return (True, "tier_or_non_cheap")
    return (False, "local_cheap")


def compute_delegation_rate(decisions: List[Dict[str, Any]]) -> float:
    if not decisions:
        return 0.0
    dele = sum(1 for d in decisions if bool(d.get("delegate")))
    return 100.0 * dele / len(decisions)


def assert_rate_ok(rate: float, floor: float = 75.0) -> bool:
    return float(rate) >= float(floor)


# ---------------------------------------------------------------------------
# Subagent planner
# ---------------------------------------------------------------------------

def _split_bullets(prompt: str) -> List[str]:
    return [m.group(1).strip() for m in _BULLET_RE.finditer(prompt or "")]


def _split_files(prompt: str) -> List[str]:
    found = []
    for m in _FILE_RE.finditer(prompt or ""):
        tok = m.group(0)
        if tok not in found:
            found.append(tok)
    return found


def _split_vs(prompt: str) -> List[str]:
    # Match "A vs B [vs C ...]" sequences; return list of terms.
    if not prompt or not _VS_RE.search(prompt):
        return []
    # Find a greedy contiguous run of "term vs term vs term"
    m = re.search(
        r"([A-Za-z0-9:._\-]+(?:\s+[A-Za-z0-9:._\-]+)*?)(?:\s+vs\.?\s+[A-Za-z0-9:._\-]+(?:\s+[A-Za-z0-9:._\-]+)*?)+",
        prompt,
        re.IGNORECASE,
    )
    if not m:
        return []
    # Expand greedily: split the entire matched segment
    parts = re.split(r"\s+vs\.?\s+", m.group(0), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _drafts_count(prompt: str) -> int:
    m = _DRAFTS_RE.search(prompt or "")
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def plan_subagents(prompt: str,
                   decision: Dict[str, Any],
                   *,
                   max_parallel: int = 3) -> Dict[str, Any]:
    category = (decision.get("category") or "").lower()
    tier = int(decision.get("tier", 1))
    prompt = prompt or ""

    subtasks: List[str] = []
    rationale = "single_task"

    # 1) Bullets / numbered list: always strong signal for split
    bullets = _split_bullets(prompt)
    if len(bullets) >= 2:
        subtasks = bullets
        rationale = "bullet_split"

    # 2) Code/debug with multiple files
    elif category in {"code", "debug"}:
        files = _split_files(prompt)
        if len(files) >= 2:
            subtasks = [f"handle {f}" for f in files]
            rationale = "file_split"

    # 3) Research comparisons "X vs Y vs Z"
    elif category == "research":
        terms = _split_vs(prompt)
        if len(terms) >= 2:
            subtasks = [f"research {t}" for t in terms]
            rationale = "vs_split"

    # 4) Writing/creative with explicit N drafts
    if not subtasks and category in {"writing", "creative"}:
        n = _drafts_count(prompt)
        if n >= 2:
            subtasks = [f"draft #{i+1}" for i in range(n)]
            rationale = "drafts_count"

    if not subtasks:
        subtasks = [prompt.strip() or "(primary task)"]
        rationale = rationale if rationale != "single_task" else "single_task"

    # cap
    if len(subtasks) > max_parallel:
        subtasks = subtasks[:max_parallel]
        rationale = f"{rationale}|capped_at_{max_parallel}"

    count = max(1, len(subtasks))

    # Quality floor: if tier>=4, ensure each subtask is annotated so caller
    # preserves heavy model; we do NOT shrink count on a high-tier task.
    if tier >= 4:
        rationale = f"{rationale}|quality_tier{tier}_inherit"

    return {"count": count, "rationale": rationale, "subtasks": subtasks}
