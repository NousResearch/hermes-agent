"""Cognitive memory taxonomy: semantic types, lifecycle, authority ordering.

Deterministic, dependency-free (stdlib only). Maps safely onto the existing
Holographic schema: ``category`` (user_pref/project/tool/general) is preserved;
``mem_type`` adds semantic precision without breaking compatibility.
"""

from __future__ import annotations

# Semantic memory types (stored in facts.mem_type; GENERAL = legacy default).
FACT = "fact"
PREFERENCE = "preference"
CONSTRAINT = "constraint"
DECISION = "decision"
INVARIANT = "invariant"
PROJECT_STATE = "project_state"
EVENT = "event"
LESSON = "lesson"
PATTERN = "pattern"
EVIDENCE = "evidence"
HYPOTHESIS = "hypothesis"
SUPERSEDED = "superseded"
GENERAL = "general"

MEM_TYPES = frozenset({
    FACT, PREFERENCE, CONSTRAINT, DECISION, INVARIANT, PROJECT_STATE,
    EVENT, LESSON, PATTERN, EVIDENCE, HYPOTHESIS, SUPERSEDED, GENERAL,
})

# Lifecycle states (stored in facts.lifecycle).
ACTIVE = "active"
AGING = "aging"
STALE = "stale"
SUPERSEDED_STATE = "superseded"
ARCHIVED = "archived"
QUARANTINE = "quarantine"

LIFECYCLES = frozenset({ACTIVE, AGING, STALE, SUPERSEDED_STATE, ARCHIVED, QUARANTINE})

# Salience tiers derived from salience score.
TIER_CANDIDATE = "candidate"
TIER_NORMAL = "normal"
TIER_IMPORTANT = "important"
TIER_DURABLE = "durable"
TIER_QUARANTINE = "quarantine"

# Firewall classes for context injection.
SAFE = "safe"
CONDITIONAL = "conditional"
QUARANTINED = "quarantine"

# Authority ordering, highest first (mission section MEMORY AUTHORITY).
# Lower index = higher authority. Historical/low-confidence memory never
# outranks current verified state; enforcement lives in ranking + firewall.
AUTHORITY_ORDER = (
    "repo_state",        # 0 current repository/source state
    "project_state",     # 1 current verified project state
    "constraint",        # 2 explicit user constraints/invariants
    "decision",          # 3 current architecture decisions
    "evidence",          # 4 verified evidence
    "lesson",            # 5 high-confidence experience/lessons
    "history",           # 6 historical memory
    "hypothesis",        # 7 hypotheses
    "unverified",        # 8 unverified/low-confidence memory
)


def authority_rank(kind: str) -> int:
    """Authority rank for a coarse kind string; unknown kinds are weakest."""
    try:
        return AUTHORITY_ORDER.index(kind)
    except ValueError:
        return len(AUTHORITY_ORDER)


def mem_type_authority(mem_type: str, trust: float = 0.5) -> str:
    """Map a memory type + trust to an authority kind."""
    if mem_type in (CONSTRAINT, INVARIANT):
        return "constraint"
    if mem_type == DECISION:
        return "decision"
    if mem_type == EVIDENCE:
        return "evidence"
    if mem_type in (LESSON, PATTERN):
        return "lesson" if trust >= 0.6 else "history"
    if mem_type == PROJECT_STATE:
        return "project_state"
    if mem_type == HYPOTHESIS:
        return "hypothesis"
    if mem_type in (SUPERSEDED, GENERAL, FACT, PREFERENCE, EVENT):
        if trust < 0.3:
            return "unverified"
        return "history"
    return "unverified"


def salience_tier(score: float, quarantined: bool = False) -> str:
    """Salience score [0,1] -> tier name."""
    if quarantined:
        return TIER_QUARANTINE
    if score >= 0.8:
        return TIER_DURABLE
    if score >= 0.6:
        return TIER_IMPORTANT
    if score >= 0.35:
        return TIER_NORMAL
    return TIER_CANDIDATE


def normalize_mem_type(value: str | None) -> str:
    """Coerce arbitrary input to a known mem_type; unknown -> GENERAL."""
    v = (value or "").strip().lower()
    return v if v in MEM_TYPES else GENERAL


def normalize_lifecycle(value: str | None) -> str:
    """Coerce arbitrary input to a known lifecycle; unknown -> ACTIVE."""
    v = (value or "").strip().lower()
    return v if v in LIFECYCLES else ACTIVE


# Legacy category -> default mem_type mapping (used only when mem_type unset).
CATEGORY_TO_MEM_TYPE = {
    "user_pref": PREFERENCE,
    "project": DECISION,
    "tool": FACT,
    "general": GENERAL,
}
