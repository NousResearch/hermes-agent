#!/usr/bin/env python3
"""
Confidence Scoring Engine — Generic Core

A four-dimensional scoring engine for memory entry prioritization.
Scores entries on [0,1] to determine TTL (time-to-live) and eviction priority.

Dimensions: source_weight × recency_decay + Σ(bonus_keywords × clamped_weight)

    preprocess(content, source, age_days) → dims
    assess(dims, profile=None) → {score, level, action, ttl, breakdown}
    assess_batch(items, profile=None) → list of results
    validate_profile(profile) → raises AssertionError if invalid

Reference: arxiv 2604.11364 (four-tier confidence model for agent memory)
"""

import re

# ═══════════════════════════════════════════════
# SAFELIST — patterns that ALWAYS bypass decay
# ═══════════════════════════════════════════════

# Add regex patterns for facts that bypass decay.
# Leave empty if you don't need safelist behavior — the engine
# works fine without it (relying on source_weight + recency + bonus).
# Example: CONSTANT_SAFELIST = [r'birthday', r'allerg']
CONSTANT_SAFELIST = []

# ═══════════════════════════════════════════════
# DEFAULT PROFILE (in-memory, YAML override via _default_profiles)
# ═══════════════════════════════════════════════

_PROFILES_CACHE = None


def _load_profiles():
    """Load from YAML if available, fall back to built-in defaults."""
    global _PROFILES_CACHE
    if _PROFILES_CACHE is not None:
        return _PROFILES_CACHE
    try:
        import yaml
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "config" / "confidence_profiles.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            if data and "profiles" in data:
                _PROFILES_CACHE = data["profiles"]
                return _PROFILES_CACHE
    except Exception:
        pass
    _PROFILES_CACHE = _default_profiles()
    return _PROFILES_CACHE


def _default_profiles():
    """Built-in default profile. Tune thresholds, buckets, and bonus weights here."""
    return {
        "memory_entry": {
            "base_weights": {"source_user_stated": 1.0, "source_llm_inferred": 0.4},
            "recency": {
                "enabled": True,
                "buckets": [
                    [0, 30, 1.0],
                    [30, 90, 0.8],
                    [90, 180, 0.6],
                    [180, 365, 0.4],
                    [365, float("inf"), 0.2],
                ],
            },
            "bonuses": {
                "permanence_marker": 0.3,
                "lesson_learned": 0.2,
                "personal_preference": 0.2,
                "work_decision": 0.1,
                "completed_task": -0.2,
                "one_time_info": -0.3,
                "uncertain_knowledge": -0.1,
            },
            "thresholds": {
                "permanent": 0.80,
                "long_term": 0.50,
                "medium_term": 0.30,
                "short_term": 0.10,
            },
            "ttl_map": {
                "permanent": 0,       # never expire
                "long_term": 365,     # days
                "medium_term": 180,
                "short_term": 90,
                "volatile": 30,
            },
        }
    }


def load_profile(subject):
    """Load a named profile. Raises ValueError if unknown."""
    profiles = _load_profiles()
    if subject not in profiles:
        raise ValueError(
            f"Unknown subject: {subject}. Available: {list(profiles.keys())}"
        )
    return profiles[subject]


# ═══════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════

def _match_safelist(content):
    """Check if content matches any CONSTANT_SAFELIST pattern."""
    if not content:
        return False
    for p in CONSTANT_SAFELIST:
        if not p:
            continue
        try:
            if re.search(p, content):
                return True
        except re.error:
            continue
    return False


def _apply_decay(age_days, buckets):
    """Map age in days to a decay multiplier via configured buckets.
    Uses half-open intervals [lo, hi) to avoid boundary ambiguity.
    Negative ages (future timestamps) are treated as 0-day: return 1.0."""
    if not buckets:
        return 1.0
    if age_days < 0:
        return 1.0
    for lo, hi, val in buckets:
        if lo <= age_days < hi:
            return val
    return buckets[-1][2]


def _map_thresholds(score, thresholds, ttl_map=None):
    """Map a [0,1] score to {level, action, ttl_days}."""
    if not thresholds:
        return "volatile", "discard", 30
    for level, th in sorted(thresholds.items(), key=lambda x: -x[1]):
        if score >= th:
            action = "keep" if level in ("permanent", "long_term", "medium_term") else "review"
            ttl = ttl_map.get(level, 30) if ttl_map else 30
            return level, action, ttl
    return "volatile", "discard", 30


# Bonus keyword patterns — regex matched against entry content.
# Each key maps to a weight in the profile's "bonuses" dict.
_BONUS_KEYWORDS = {
    "permanence_marker":   r'\b(permanent|always|lifetime|forever)\b',
    "lesson_learned":      r'\b(lesson|methodology|principle|rule|iron.?law)\b',
    "personal_preference": r'\b(prefers?|habit|usually|generally|most)\b',
    "work_decision":       r'\b(decided|chose|selected|designed|architected)\b',
    "completed_task":      r'\b(completed|deployed|fixed|resolved|finished)\b',
    "one_time_info":       r'\b(temporary|one.?off|one.?time|this.?time)\b',
    "uncertain_knowledge": r'\b(maybe|perhaps|possibly|uncertain|might)\b',
}


def _detect_bonus(content):
    """Detect bonus keyword presence in content. Returns dict of bools."""
    result = {}
    for key, pattern in _BONUS_KEYWORDS.items():
        result[key] = bool(re.search(pattern, content, re.IGNORECASE))
    return result


def preprocess(content, source="llm_inferred", age_days=0):
    """Extract dimensions from a memory entry. Returns dict for assess()."""
    dims = {"age_days": age_days, "_content": content}

    # fact_type classification (arxiv 2604.11364)
    dims["fact_type_constant"] = (
        _match_safelist(content)
        or (
            source == "user_stated"
            and not re.search(r'\b(now|currently|recently|this.?time|maybe|perhaps|possibly)\b', content, re.IGNORECASE)
            and not re.search(r'\b(prefers?|habit|usually|generally|most)\b', content, re.IGNORECASE)
            and len(content) > 20
        )
    )
    dims["fact_type_transient"] = bool(
        re.search(r'\b(temporary|one.?off|one.?time|this.?time)\b', content, re.IGNORECASE)
    )

    # source dimensions
    dims["source_user_stated"] = source == "user_stated"
    dims["source_llm_inferred"] = source == "llm_inferred"

    # bonus dimensions
    dims.update(_detect_bonus(content))

    return dims


def assess(dims, profile=None):
    """Score a memory entry. Returns {score, level, action, ttl, breakdown}."""
    if profile is None:
        profile = load_profile("memory_entry")

    validate_profile(profile)

    # 1. base
    base = 0.0
    for k, w in profile["base_weights"].items():
        if dims.get(k):
            base = max(base, w)

    # 2. recency (with safelist + fact_type bypass)
    content = dims.get("_content", "")
    recency = 1.0
    if _match_safelist(content):
        recency = 1.0
    elif dims.get("fact_type_constant"):
        recency = 1.0
    elif profile["recency"]["enabled"] and "age_days" in dims:
        age = dims["age_days"]
        buckets = profile["recency"]["buckets"]
        if dims.get("fact_type_transient"):
            recency = _apply_decay(age, buckets) * 0.5
        else:
            recency = _apply_decay(age, buckets)

    # 3. bonus (clamp ±0.3)
    bonus = 0.0
    breakdown = {"base": base, "recency": recency}
    if "bonuses" in profile:
        for key, val in profile["bonuses"].items():
            if dims.get(key):
                clamped = max(-0.3, min(0.3, val))
                bonus += clamped
                breakdown[f"bonus_{key}"] = clamped

    # 4. formula: score = source_weight × recency_decay + Σbonus
    score = base * recency + bonus
    score = max(0.0, min(1.0, score))
    score = round(score, 4)

    # 5. threshold mapping
    level, action, ttl = _map_thresholds(
        score, profile["thresholds"], profile.get("ttl_map")
    )

    return {
        "score": score,
        "level": level,
        "action": action,
        "ttl": ttl,
        "breakdown": breakdown,
    }


def assess_batch(items, profile=None):
    """Batch-assess multiple entries. Returns list of results."""
    results = []
    for item in items:
        dims = preprocess(
            item.get("content", ""),
            item.get("source", "llm_inferred"),
            item.get("age_days", 0),
        )
        results.append(assess(dims, profile))
    return results


# ═══════════════════════════════════════════════
# PROFILE VALIDATION
# ═══════════════════════════════════════════════

def validate_profile(profile):
    """Validate profile structure. Raises AssertionError on failure."""
    required = ["base_weights", "recency", "thresholds"]
    for key in required:
        assert key in profile, f"Missing required key: {key}"
    assert "enabled" in profile["recency"], "Missing recency.enabled"
    assert "buckets" in profile["recency"], "Missing recency.buckets"
    if "bonuses" in profile:
        for k, v in profile["bonuses"].items():
            assert -0.3 <= v <= 0.3, f"Bonus {k}={v} out of [-0.3, 0.3]"
        # Verify bonus keys match keyword pattern keys (prevent silent drift)
        extra = set(profile["bonuses"]) - set(_BONUS_KEYWORDS)
        if extra:
            raise AssertionError(f"Bonuses without matching keywords: {extra}")
    thresholds = profile["thresholds"]
    assert len(set(thresholds.values())) == len(thresholds), "Duplicate threshold values"
