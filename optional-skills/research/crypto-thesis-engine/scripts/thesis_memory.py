#!/usr/bin/env python3
"""
Crypto Thesis Engine — Decision Engine Memory System
====================================================
Self-improving memory with bias detection, adaptive confidence,
risk override warnings, and outcome tracking.

Uses only Python standard library — zero external dependencies.

Usage:
    # Record an analysis
    python3 thesis_memory.py record --token solana --price 105.50 \
        --momentum MIXED_BEARISH --sentiment cautiously_bullish \
        --categories layer-1 --rank 7 --confidence 65

    # Record outcome (learn)
    python3 thesis_memory.py learn --token solana --outcome correct
    python3 thesis_memory.py learn --token arbitrum --outcome wrong

    # Get confidence score for a token (with adaptive adjustments)
    python3 thesis_memory.py confidence --token solana --categories layer-1

    # Find similar past analyses
    python3 thesis_memory.py similar --token solana --categories layer-1 --rank 7

    # Diagnose system biases and blind spots
    python3 thesis_memory.py diagnose

    # Check risk overrides for a specific analysis setup
    python3 thesis_memory.py risk-check --token near --categories layer-1 \
        --sentiment bullish --momentum MIXED_BEARISH

    # View full history / stats / export
    python3 thesis_memory.py history --token solana
    python3 thesis_memory.py stats
    python3 thesis_memory.py export

Author: Crypto Thesis Engine Skill
License: MIT
"""

import argparse
import json
import os
import sys
import copy
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MEMORY_DIR = Path.home() / ".hermes" / "thesis-memory"
MEMORY_FILE = MEMORY_DIR / "memory.json"
DEFAULT_CONFIDENCE = 50  # Starting confidence when no history exists
MIN_CONFIDENCE = 20
MAX_CONFIDENCE = 95

# Confidence weights — how much each factor contributes
WEIGHTS = {
    "token_accuracy": 0.30,      # How accurate we've been on THIS specific token
    "category_accuracy": 0.25,   # How accurate we've been on this CATEGORY
    "global_accuracy": 0.10,     # Overall accuracy across all tokens
    "data_richness": 0.10,       # How much data was available for analysis
    "recency_bonus": 0.10,       # Bonus for recent successful analyses
    "bias_penalty": 0.15,        # Penalty/boost from bias detection
}

# Bias detection thresholds
BIAS_THRESHOLDS = {
    "min_samples": 3,              # Minimum outcomes before bias analysis kicks in
    "sentiment_skew_threshold": 0.70,  # If >70% analyses are same sentiment → bias
    "category_fail_threshold": 0.50,   # If <50% accuracy in category → weak spot
    "category_strong_threshold": 0.75, # If >75% accuracy → strength
    "override_pattern_threshold": 0.60, # If >60% similar patterns failed → warning
}

# Sentiment values ordered from bearish to bullish
SENTIMENT_ORDER = [
    "bearish", "cautiously_bearish", "neutral",
    "cautiously_bullish", "bullish"
]

# Market cap tier boundaries for similarity matching
MCAP_TIERS = {
    "mega": (50_000_000_000, float("inf")),     # >$50B
    "large": (10_000_000_000, 50_000_000_000),  # $10B-$50B
    "mid": (1_000_000_000, 10_000_000_000),     # $1B-$10B
    "small": (100_000_000, 1_000_000_000),       # $100M-$1B
    "micro": (10_000_000, 100_000_000),          # $10M-$100M
    "nano": (0, 10_000_000),                      # <$10M
}


# ─────────────────────────────────────────────────────────────────────────────
# Memory Store
# ─────────────────────────────────────────────────────────────────────────────

def _empty_memory() -> dict:
    """Return the skeleton structure for a fresh memory file."""
    return {
        "version": "2.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "analyses": {},       # token_id -> list of analysis records
        "outcomes": {},       # token_id -> {correct, wrong, total, accuracy}
        "category_stats": {}, # category -> {correct, wrong, total, accuracy}
        "global_stats": {
            "total_analyses": 0,
            "total_outcomes": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": DEFAULT_CONFIDENCE,
        },
    }


def load_memory() -> dict:
    """Load memory from disk, creating if it doesn't exist."""
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
            # Migration: ensure all keys exist
            base = _empty_memory()
            for key in base:
                if key not in data:
                    data[key] = base[key]
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Memory file corrupted, starting fresh: {e}", file=sys.stderr)
            return _empty_memory()
    return _empty_memory()


def save_memory(memory: dict):
    """Persist memory to disk."""
    memory["updated_at"] = datetime.now(timezone.utc).isoformat()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # Write atomically: write to temp, then rename
    tmp_file = MEMORY_FILE.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    tmp_file.rename(MEMORY_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Record Analysis
# ─────────────────────────────────────────────────────────────────────────────

def record_analysis(
    token_id: str,
    price: float = None,
    momentum: str = None,
    sentiment: str = None,
    categories: list = None,
    rank: int = None,
    confidence: float = None,
    market_cap: float = None,
    key_metrics: dict = None,
) -> dict:
    """
    Record a new analysis in memory.
    Called after generating a thesis report.
    """
    memory = load_memory()

    record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "price_at_analysis": price,
        "market_cap_at_analysis": market_cap,
        "momentum": momentum,
        "sentiment": sentiment,  # bullish, cautiously_bullish, neutral, cautiously_bearish, bearish
        "categories": categories or [],
        "market_cap_rank": rank,
        "confidence": confidence or DEFAULT_CONFIDENCE,
        "key_metrics": key_metrics or {},
        "outcome": None,         # Filled by learn command
        "outcome_date": None,
        "outcome_note": None,
    }

    if token_id not in memory["analyses"]:
        memory["analyses"][token_id] = []

    memory["analyses"][token_id].append(record)
    memory["global_stats"]["total_analyses"] += 1

    save_memory(memory)

    return {
        "status": "recorded",
        "token": token_id,
        "analysis_number": len(memory["analyses"][token_id]),
        "confidence": record["confidence"],
        "date": record["date"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Learn from Outcomes
# ─────────────────────────────────────────────────────────────────────────────

def learn_outcome(token_id: str, outcome: str, note: str = None) -> dict:
    """
    Record whether a past analysis was correct or wrong.
    Updates token-level, category-level, and global accuracy stats.
    """
    if outcome not in ("correct", "wrong"):
        return {"error": f"Outcome must be 'correct' or 'wrong', got '{outcome}'"}

    memory = load_memory()

    # Check if token has any analyses
    if token_id not in memory["analyses"] or not memory["analyses"][token_id]:
        return {
            "error": f"No analyses found for '{token_id}'. Run an analysis first.",
            "suggestion": f"Try: /crypto-thesis-engine analyze {token_id}",
        }

    # Find the most recent analysis without an outcome
    target_idx = None
    for i in range(len(memory["analyses"][token_id]) - 1, -1, -1):
        if memory["analyses"][token_id][i]["outcome"] is None:
            target_idx = i
            break

    if target_idx is None:
        # All analyses already have outcomes, apply to the most recent one (overwrite)
        target_idx = len(memory["analyses"][token_id]) - 1
        was_overwrite = True
        old_outcome = memory["analyses"][token_id][target_idx]["outcome"]
    else:
        was_overwrite = False
        old_outcome = None

    # Update the analysis record
    analysis = memory["analyses"][token_id][target_idx]
    analysis["outcome"] = outcome
    analysis["outcome_date"] = datetime.now(timezone.utc).isoformat()
    analysis["outcome_note"] = note

    # ── Update Token Stats ───────────────────────────────────────────────
    if token_id not in memory["outcomes"]:
        memory["outcomes"][token_id] = {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0}

    token_stats = memory["outcomes"][token_id]

    if was_overwrite and old_outcome:
        # Undo old outcome before applying new
        token_stats[old_outcome] -= 1
        token_stats["total"] -= 1
        memory["global_stats"][old_outcome] -= 1
        memory["global_stats"]["total_outcomes"] -= 1

    token_stats[outcome] += 1
    token_stats["total"] += 1
    token_stats["accuracy"] = round(
        token_stats["correct"] / token_stats["total"] * 100, 1
    ) if token_stats["total"] > 0 else 0

    # ── Update Category Stats ────────────────────────────────────────────
    categories = analysis.get("categories", [])
    for cat in categories:
        if cat not in memory["category_stats"]:
            memory["category_stats"][cat] = {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0}
        cat_stats = memory["category_stats"][cat]

        if was_overwrite and old_outcome:
            cat_stats[old_outcome] -= 1
            cat_stats["total"] -= 1

        cat_stats[outcome] += 1
        cat_stats["total"] += 1
        cat_stats["accuracy"] = round(
            cat_stats["correct"] / cat_stats["total"] * 100, 1
        ) if cat_stats["total"] > 0 else 0

    # ── Update Global Stats ──────────────────────────────────────────────
    gs = memory["global_stats"]
    gs[outcome] += 1
    gs["total_outcomes"] += 1
    gs["accuracy"] = round(
        gs["correct"] / gs["total_outcomes"] * 100, 1
    ) if gs["total_outcomes"] > 0 else DEFAULT_CONFIDENCE

    save_memory(memory)

    return {
        "status": "learned",
        "token": token_id,
        "outcome": outcome,
        "note": note,
        "token_accuracy": f"{token_stats['accuracy']}% ({token_stats['correct']}/{token_stats['total']})",
        "global_accuracy": f"{gs['accuracy']}% ({gs['correct']}/{gs['total_outcomes']})",
        "analysis_date": analysis["date"],
        "was_overwrite": was_overwrite,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _get_mcap_tier(market_cap: float) -> str:
    """Determine market cap tier for similarity matching."""
    if market_cap is None:
        return "unknown"
    for tier, (low, high) in MCAP_TIERS.items():
        if low <= market_cap < high:
            return tier
    return "unknown"


def compute_confidence(
    token_id: str,
    categories: list = None,
    data_completeness: float = None,
) -> dict:
    """
    Compute a confidence score for an upcoming analysis based on
    historical accuracy across multiple dimensions.

    Returns:
        {
            "score": 72,
            "grade": "MODERATE",
            "breakdown": {...},
            "reasoning": "...",
        }
    """
    memory = load_memory()
    breakdown = {}
    reasoning_parts = []

    # ── Factor 1: Token-specific accuracy ────────────────────────────────
    token_stats = memory["outcomes"].get(token_id, {})
    if token_stats.get("total", 0) >= 1:
        token_acc = token_stats["accuracy"]
        breakdown["token_accuracy"] = {
            "value": token_acc,
            "weight": WEIGHTS["token_accuracy"],
            "detail": f"{token_stats['correct']}/{token_stats['total']} correct",
        }
        reasoning_parts.append(
            f"Token-specific track record: {token_stats['correct']}/{token_stats['total']} "
            f"({token_acc}%) — {'strong' if token_acc >= 70 else 'weak' if token_acc < 50 else 'moderate'}"
        )
    else:
        # No history for this token — use a neutral default
        token_acc = DEFAULT_CONFIDENCE
        breakdown["token_accuracy"] = {
            "value": DEFAULT_CONFIDENCE,
            "weight": WEIGHTS["token_accuracy"],
            "detail": "No prior analyses for this token",
        }
        reasoning_parts.append("No prior analyses for this token — using baseline confidence")

    # ── Factor 2: Category accuracy ──────────────────────────────────────
    cat_scores = []
    categories = categories or []
    for cat in categories:
        cat_stats = memory["category_stats"].get(cat, {})
        if cat_stats.get("total", 0) >= 1:
            cat_scores.append(cat_stats["accuracy"])

    if cat_scores:
        cat_avg = sum(cat_scores) / len(cat_scores)
        breakdown["category_accuracy"] = {
            "value": cat_avg,
            "weight": WEIGHTS["category_accuracy"],
            "detail": f"Avg across {len(cat_scores)} matching categories",
        }
        reasoning_parts.append(
            f"Category track record: {cat_avg:.1f}% avg across {', '.join(categories)}"
        )
    else:
        cat_avg = DEFAULT_CONFIDENCE
        breakdown["category_accuracy"] = {
            "value": DEFAULT_CONFIDENCE,
            "weight": WEIGHTS["category_accuracy"],
            "detail": "No category history",
        }
        reasoning_parts.append("No category-level history yet")

    # ── Factor 3: Global accuracy ────────────────────────────────────────
    gs = memory["global_stats"]
    if gs["total_outcomes"] >= 1:
        global_acc = gs["accuracy"]
        breakdown["global_accuracy"] = {
            "value": global_acc,
            "weight": WEIGHTS["global_accuracy"],
            "detail": f"{gs['correct']}/{gs['total_outcomes']} correct overall",
        }
        reasoning_parts.append(f"Overall track record: {gs['correct']}/{gs['total_outcomes']} ({global_acc}%)")
    else:
        global_acc = DEFAULT_CONFIDENCE
        breakdown["global_accuracy"] = {
            "value": DEFAULT_CONFIDENCE,
            "weight": WEIGHTS["global_accuracy"],
            "detail": "No outcomes recorded yet",
        }
        reasoning_parts.append("No outcomes recorded yet — system is learning")

    # ── Factor 4: Data richness ──────────────────────────────────────────
    data_score = (data_completeness or 0.7) * 100  # Default 70% if not provided
    breakdown["data_richness"] = {
        "value": data_score,
        "weight": WEIGHTS["data_richness"],
        "detail": f"{data_score:.0f}% of expected data fields present",
    }

    # ── Factor 5: Recency bonus ──────────────────────────────────────────
    recency_score = DEFAULT_CONFIDENCE
    token_analyses = memory["analyses"].get(token_id, [])
    recent_outcomes = []

    # Look at last 3 analyses with outcomes (across all tokens in same category)
    all_analyses = []
    for tid, analyses in memory["analyses"].items():
        for a in analyses:
            if a.get("outcome") and any(c in categories for c in a.get("categories", [])):
                all_analyses.append(a)

    all_analyses.sort(key=lambda x: x["date"], reverse=True)
    recent = all_analyses[:5]

    if recent:
        recent_correct = sum(1 for a in recent if a["outcome"] == "correct")
        recency_score = recent_correct / len(recent) * 100
        breakdown["recency_bonus"] = {
            "value": recency_score,
            "weight": WEIGHTS["recency_bonus"],
            "detail": f"{recent_correct}/{len(recent)} recent related analyses correct",
        }
        reasoning_parts.append(f"Recent performance: {recent_correct}/{len(recent)} related analyses correct")
    else:
        breakdown["recency_bonus"] = {
            "value": DEFAULT_CONFIDENCE,
            "weight": WEIGHTS["recency_bonus"],
            "detail": "No recent related analyses",
        }

    # ── Factor 6: Bias penalty / boost (ADAPTIVE) ────────────────────────
    bias_adjustment = _compute_bias_adjustment(memory, token_id, categories)
    breakdown["bias_penalty"] = {
        "value": bias_adjustment["score"],
        "weight": WEIGHTS["bias_penalty"],
        "detail": bias_adjustment["detail"],
    }
    if bias_adjustment["warnings"]:
        reasoning_parts.append(f"Bias adjustment: {bias_adjustment['detail']}")

    # ── Compute weighted score ───────────────────────────────────────────
    weighted_score = 0
    for factor_key, factor_data in breakdown.items():
        weighted_score += factor_data["value"] * factor_data["weight"]

    # Clamp to range
    final_score = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, round(weighted_score)))

    # Determine grade
    if final_score >= 80:
        grade = "HIGH"
    elif final_score >= 60:
        grade = "MODERATE"
    elif final_score >= 40:
        grade = "LOW"
    else:
        grade = "VERY_LOW"

    # Build final reasoning
    if gs["total_outcomes"] == 0:
        reasoning = (
            "This is a new system with no outcome history yet. "
            "Confidence starts at baseline and will improve as you provide feedback via the "
            "'learn' command. Current score reflects data availability only."
        )
    else:
        reasoning = " | ".join(reasoning_parts)

    return {
        "score": final_score,
        "grade": grade,
        "breakdown": breakdown,
        "reasoning": reasoning,
        "bias_warnings": bias_adjustment["warnings"],
        "total_analyses": gs["total_analyses"],
        "total_outcomes": gs["total_outcomes"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Similar Analysis Finder (Pattern Awareness)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_analyses(
    token_id: str,
    categories: list = None,
    market_cap: float = None,
    rank: int = None,
    limit: int = 5,
) -> dict:
    """
    Find past analyses similar to the current token based on:
    - Same categories
    - Similar market cap tier
    - Similar rank range

    Returns analyses with their outcomes for pattern awareness.
    """
    memory = load_memory()
    candidates = []
    target_tier = _get_mcap_tier(market_cap)
    categories = set(categories or [])

    for tid, analyses in memory["analyses"].items():
        if tid == token_id:
            continue  # Skip the token itself

        for analysis in analyses:
            if analysis.get("outcome") is None:
                continue  # Only include analyses with known outcomes

            score = 0
            match_reasons = []

            # Category overlap
            analysis_cats = set(analysis.get("categories", []))
            overlap = categories & analysis_cats
            if overlap:
                score += len(overlap) * 30
                match_reasons.append(f"same category: {', '.join(overlap)}")

            # Market cap tier similarity
            analysis_mcap = analysis.get("market_cap_at_analysis")
            if analysis_mcap and market_cap:
                analysis_tier = _get_mcap_tier(analysis_mcap)
                if analysis_tier == target_tier:
                    score += 25
                    match_reasons.append(f"same mcap tier: {target_tier}")
                elif abs(list(MCAP_TIERS.keys()).index(analysis_tier) -
                         list(MCAP_TIERS.keys()).index(target_tier)) <= 1:
                    score += 10
                    match_reasons.append(f"adjacent mcap tier: {analysis_tier}")

            # Rank proximity
            analysis_rank = analysis.get("market_cap_rank")
            if analysis_rank and rank:
                rank_diff = abs(analysis_rank - rank)
                if rank_diff <= 10:
                    score += 20
                    match_reasons.append(f"similar rank (#{analysis_rank})")
                elif rank_diff <= 30:
                    score += 10
                    match_reasons.append(f"nearby rank (#{analysis_rank})")

            # Momentum similarity
            if analysis.get("momentum"):
                score += 5  # Small bonus for having momentum data

            if score > 0:
                candidates.append({
                    "token_id": tid,
                    "similarity_score": score,
                    "match_reasons": match_reasons,
                    "analysis_date": analysis["date"],
                    "outcome": analysis["outcome"],
                    "sentiment": analysis.get("sentiment"),
                    "momentum": analysis.get("momentum"),
                    "price_at_analysis": analysis.get("price_at_analysis"),
                    "confidence_at_time": analysis.get("confidence"),
                    "categories": analysis.get("categories", []),
                })

    # Sort by similarity score descending
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    top = candidates[:limit]

    # Compute pattern summary
    if top:
        correct_count = sum(1 for c in top if c["outcome"] == "correct")
        wrong_count = sum(1 for c in top if c["outcome"] == "wrong")
        pattern_rate = correct_count / len(top) * 100

        pattern_summary = (
            f"Found {len(top)} similar past analyses. "
            f"Pattern accuracy: {correct_count}/{len(top)} ({pattern_rate:.0f}%) correct."
        )
        if pattern_rate >= 70:
            pattern_sentiment = "Similar tokens have been well-predicted historically."
        elif pattern_rate >= 50:
            pattern_sentiment = "Mixed results with similar tokens — moderate caution advised."
        else:
            pattern_sentiment = "Similar tokens have been poorly predicted — extra caution warranted."
    else:
        pattern_summary = "No similar past analyses found. This is novel territory."
        pattern_sentiment = "No historical patterns to reference."

    return {
        "similar_analyses": top,
        "total_found": len(candidates),
        "showing": len(top),
        "pattern_summary": pattern_summary,
        "pattern_sentiment": pattern_sentiment,
    }


# ─────────────────────────────────────────────────────────────────────────────
# History & Stats
# ─────────────────────────────────────────────────────────────────────────────

def get_token_history(token_id: str) -> dict:
    """Get full analysis history for a token."""
    memory = load_memory()

    analyses = memory["analyses"].get(token_id, [])
    outcomes = memory["outcomes"].get(token_id, {})

    return {
        "token_id": token_id,
        "total_analyses": len(analyses),
        "outcomes": outcomes,
        "analyses": analyses,
        "has_history": len(analyses) > 0,
    }


def get_global_stats() -> dict:
    """Get overall system performance stats."""
    memory = load_memory()

    # Top performing categories
    cat_stats = memory["category_stats"]
    sorted_cats = sorted(
        [(k, v) for k, v in cat_stats.items() if v.get("total", 0) >= 2],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    # Top performing tokens
    token_stats = memory["outcomes"]
    sorted_tokens = sorted(
        [(k, v) for k, v in token_stats.items() if v.get("total", 0) >= 2],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    # Most analyzed tokens
    most_analyzed = sorted(
        [(k, len(v)) for k, v in memory["analyses"].items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return {
        "global": memory["global_stats"],
        "top_categories": [{"category": k, **v} for k, v in sorted_cats[:5]],
        "top_tokens": [{"token": k, **v} for k, v in sorted_tokens[:5]],
        "worst_tokens": [{"token": k, **v} for k, v in sorted_tokens[-3:]] if len(sorted_tokens) >= 3 else [],
        "most_analyzed": [{"token": k, "count": c} for k, c in most_analyzed],
        "memory_file": str(MEMORY_FILE),
        "memory_exists": MEMORY_FILE.exists(),
        "created_at": memory.get("created_at"),
        "updated_at": memory.get("updated_at"),
    }


def export_memory() -> dict:
    """Export the full memory for backup or inspection."""
    memory = load_memory()
    return memory


# ─────────────────────────────────────────────────────────────────────────────
# Bias Detection & Adaptive Confidence
# ─────────────────────────────────────────────────────────────────────────────

def _compute_bias_adjustment(memory: dict, token_id: str, categories: list) -> dict:
    """
    Internal: compute a bias-based adjustment score for confidence.
    Returns a score 0-100 (50 = neutral, <50 = penalty, >50 = boost)
    along with warnings.
    """
    warnings = []
    adjustments = []  # list of (value, weight) tuples
    categories = list(categories or [])
    gs = memory["global_stats"]

    if gs.get("total_outcomes", 0) < BIAS_THRESHOLDS["min_samples"]:
        return {"score": DEFAULT_CONFIDENCE, "detail": "Insufficient data for bias analysis", "warnings": []}

    # ── 1. Sentiment skew detection ──────────────────────────────────────
    all_analyses_with_outcome = []
    for tid, analyses in memory["analyses"].items():
        for a in analyses:
            if a.get("outcome") is not None and a.get("sentiment"):
                all_analyses_with_outcome.append(a)

    if len(all_analyses_with_outcome) >= BIAS_THRESHOLDS["min_samples"]:
        sentiment_counts = {}
        for a in all_analyses_with_outcome:
            s = a["sentiment"]
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

        total = len(all_analyses_with_outcome)
        bullish_count = sum(sentiment_counts.get(s, 0) for s in ["bullish", "cautiously_bullish"])
        bearish_count = sum(sentiment_counts.get(s, 0) for s in ["bearish", "cautiously_bearish"])
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total

        # Check if bullish analyses are actually wrong more often
        bullish_wrong = sum(
            1 for a in all_analyses_with_outcome
            if a["sentiment"] in ("bullish", "cautiously_bullish") and a["outcome"] == "wrong"
        )
        bearish_wrong = sum(
            1 for a in all_analyses_with_outcome
            if a["sentiment"] in ("bearish", "cautiously_bearish") and a["outcome"] == "wrong"
        )

        if bullish_ratio >= BIAS_THRESHOLDS["sentiment_skew_threshold"]:
            bullish_accuracy = (bullish_count - bullish_wrong) / bullish_count * 100 if bullish_count > 0 else 50
            warnings.append({
                "type": "BULLISH_BIAS",
                "severity": "HIGH" if bullish_accuracy < 50 else "MEDIUM",
                "message": f"System leans bullish ({bullish_ratio:.0%} of analyses). "
                           f"Bullish accuracy: {bullish_accuracy:.0f}%",
                "bullish_ratio": round(bullish_ratio * 100, 1),
                "bullish_accuracy": round(bullish_accuracy, 1),
            })
            # Penalize if bullish bias AND low bullish accuracy
            if bullish_accuracy < 50:
                adjustments.append((25, 0.5))  # Strong penalty
            elif bullish_accuracy < 65:
                adjustments.append((40, 0.3))  # Moderate penalty

        elif bearish_ratio >= BIAS_THRESHOLDS["sentiment_skew_threshold"]:
            bearish_accuracy = (bearish_count - bearish_wrong) / bearish_count * 100 if bearish_count > 0 else 50
            warnings.append({
                "type": "BEARISH_BIAS",
                "severity": "HIGH" if bearish_accuracy < 50 else "MEDIUM",
                "message": f"System leans bearish ({bearish_ratio:.0%} of analyses). "
                           f"Bearish accuracy: {bearish_accuracy:.0f}%",
                "bearish_ratio": round(bearish_ratio * 100, 1),
                "bearish_accuracy": round(bearish_accuracy, 1),
            })
            if bearish_accuracy < 50:
                adjustments.append((25, 0.5))
            elif bearish_accuracy < 65:
                adjustments.append((40, 0.3))

    # ── 2. Category weakness / strength detection ────────────────────────
    for cat in categories:
        cat_stats = memory["category_stats"].get(cat, {})
        if cat_stats.get("total", 0) >= BIAS_THRESHOLDS["min_samples"]:
            cat_acc = cat_stats["accuracy"]

            if cat_acc < BIAS_THRESHOLDS["category_fail_threshold"] * 100:
                warnings.append({
                    "type": "CATEGORY_WEAKNESS",
                    "severity": "HIGH",
                    "message": f"{cat} analyses have low accuracy: "
                               f"{cat_stats['correct']}/{cat_stats['total']} ({cat_acc}%)",
                    "category": cat,
                    "accuracy": cat_acc,
                })
                adjustments.append((20, 0.4))  # Significant penalty

            elif cat_acc >= BIAS_THRESHOLDS["category_strong_threshold"] * 100:
                warnings.append({
                    "type": "CATEGORY_STRENGTH",
                    "severity": "INFO",
                    "message": f"{cat} analyses are strong: "
                               f"{cat_stats['correct']}/{cat_stats['total']} ({cat_acc}%)",
                    "category": cat,
                    "accuracy": cat_acc,
                })
                adjustments.append((80, 0.3))  # Boost

    # ── 3. Momentum-sentiment mismatch detection ─────────────────────────
    # If past analyses with conflicting momentum/sentiment failed, flag it
    conflicting = []
    for tid, analyses in memory["analyses"].items():
        for a in analyses:
            if a.get("outcome") is None:
                continue
            momentum = a.get("momentum", "")
            sentiment = a.get("sentiment", "")
            is_bullish_sentiment = sentiment in ("bullish", "cautiously_bullish")
            is_bearish_momentum = momentum in ("ACCELERATING_DOWN", "STEADY_DOWN", "MIXED_BEARISH")
            is_bearish_sentiment = sentiment in ("bearish", "cautiously_bearish")
            is_bullish_momentum = momentum in ("ACCELERATING_UP", "STEADY_UP", "MIXED_BULLISH")

            if (is_bullish_sentiment and is_bearish_momentum) or \
               (is_bearish_sentiment and is_bullish_momentum):
                conflicting.append(a)

    if len(conflicting) >= 2:
        conflict_wrong = sum(1 for a in conflicting if a["outcome"] == "wrong")
        conflict_rate = conflict_wrong / len(conflicting)
        if conflict_rate > 0.5:
            warnings.append({
                "type": "MOMENTUM_SENTIMENT_CONFLICT",
                "severity": "MEDIUM",
                "message": f"Analyses disagreeing with momentum trend have {conflict_rate:.0%} failure rate "
                           f"({conflict_wrong}/{len(conflicting)})",
                "failure_rate": round(conflict_rate * 100, 1),
            })
            adjustments.append((35, 0.2))

    # ── Combine adjustments ──────────────────────────────────────────────
    if adjustments:
        total_weight = sum(w for _, w in adjustments)
        weighted_sum = sum(v * w for v, w in adjustments)
        bias_score = weighted_sum / total_weight
    else:
        bias_score = DEFAULT_CONFIDENCE  # Neutral — no bias detected

    # Build detail string
    warning_types = [w["type"] for w in warnings if w["severity"] in ("HIGH", "MEDIUM")]
    if warning_types:
        detail = f"Active biases: {', '.join(warning_types)}"
    else:
        detail = "No significant biases detected"

    return {
        "score": round(bias_score, 1),
        "detail": detail,
        "warnings": warnings,
    }


def diagnose_biases() -> dict:
    """
    Full system self-diagnosis. Analyzes all memory to detect:
    - Sentiment bias (too bullish / too bearish)
    - Category blind spots (low accuracy areas)
    - Momentum-sentiment conflicts
    - Overconfidence / underconfidence patterns
    - Recommendations for improvement
    """
    memory = load_memory()
    gs = memory["global_stats"]
    diagnosis = {
        "system_health": "HEALTHY",
        "total_analyses": gs["total_analyses"],
        "total_outcomes": gs["total_outcomes"],
        "overall_accuracy": gs["accuracy"],
        "biases": [],
        "blind_spots": [],
        "strengths": [],
        "recommendations": [],
        "confidence_calibration": {},
    }

    if gs["total_outcomes"] < BIAS_THRESHOLDS["min_samples"]:
        diagnosis["system_health"] = "INSUFFICIENT_DATA"
        diagnosis["recommendations"].append(
            f"Need at least {BIAS_THRESHOLDS['min_samples']} outcomes for meaningful diagnosis. "
            f"Currently have {gs['total_outcomes']}. Use 'learn' command after analyses."
        )
        return diagnosis

    # ── 1. Sentiment Distribution Analysis ───────────────────────────────
    all_with_outcome = []
    for tid, analyses in memory["analyses"].items():
        for a in analyses:
            if a.get("outcome") is not None:
                all_with_outcome.append(a)

    sentiment_stats = {}
    for s in SENTIMENT_ORDER:
        matching = [a for a in all_with_outcome if a.get("sentiment") == s]
        if matching:
            correct = sum(1 for a in matching if a["outcome"] == "correct")
            sentiment_stats[s] = {
                "count": len(matching),
                "correct": correct,
                "wrong": len(matching) - correct,
                "accuracy": round(correct / len(matching) * 100, 1),
                "share": round(len(matching) / len(all_with_outcome) * 100, 1),
            }

    diagnosis["sentiment_distribution"] = sentiment_stats

    # Detect directional bias
    total_outcomes = len(all_with_outcome)
    bullish_total = sum(
        sentiment_stats.get(s, {}).get("count", 0)
        for s in ["bullish", "cautiously_bullish"]
    )
    bearish_total = sum(
        sentiment_stats.get(s, {}).get("count", 0)
        for s in ["bearish", "cautiously_bearish"]
    )

    if total_outcomes > 0:
        bullish_share = bullish_total / total_outcomes
        bearish_share = bearish_total / total_outcomes

        if bullish_share >= BIAS_THRESHOLDS["sentiment_skew_threshold"]:
            # Check if the bullish bias is actually hurting accuracy
            bullish_analyses = [a for a in all_with_outcome if a.get("sentiment") in ("bullish", "cautiously_bullish")]
            bullish_correct = sum(1 for a in bullish_analyses if a["outcome"] == "correct")
            bullish_acc = bullish_correct / len(bullish_analyses) * 100 if bullish_analyses else 0

            bias_entry = {
                "type": "BULLISH_BIAS",
                "severity": "HIGH" if bullish_acc < 50 else "MEDIUM",
                "share": round(bullish_share * 100, 1),
                "accuracy_when_bullish": round(bullish_acc, 1),
                "description": f"{bullish_share:.0%} of analyses are bullish/cautiously_bullish. "
                               f"Accuracy when bullish: {bullish_acc:.0f}%.",
            }
            diagnosis["biases"].append(bias_entry)

            if bullish_acc < 50:
                diagnosis["recommendations"].append(
                    "⚠️ CRITICAL: Bullish predictions are wrong more than half the time. "
                    "Consider: (1) Weighting bear case arguments more heavily, "
                    "(2) Requiring stronger data support for bullish theses, "
                    "(3) Defaulting to 'cautiously_bullish' instead of 'bullish'."
                )
            elif bullish_acc < 65:
                diagnosis["recommendations"].append(
                    "Moderate bullish bias detected. Consider being more selective "
                    "about when to issue bullish verdicts."
                )

        elif bearish_share >= BIAS_THRESHOLDS["sentiment_skew_threshold"]:
            bearish_analyses = [a for a in all_with_outcome if a.get("sentiment") in ("bearish", "cautiously_bearish")]
            bearish_correct = sum(1 for a in bearish_analyses if a["outcome"] == "correct")
            bearish_acc = bearish_correct / len(bearish_analyses) * 100 if bearish_analyses else 0

            diagnosis["biases"].append({
                "type": "BEARISH_BIAS",
                "severity": "HIGH" if bearish_acc < 50 else "MEDIUM",
                "share": round(bearish_share * 100, 1),
                "accuracy_when_bearish": round(bearish_acc, 1),
                "description": f"{bearish_share:.0%} of analyses are bearish/cautiously_bearish. "
                               f"Accuracy when bearish: {bearish_acc:.0f}%.",
            })

            if bearish_acc < 50:
                diagnosis["recommendations"].append(
                    "⚠️ CRITICAL: Bearish predictions are wrong more than half the time. "
                    "Consider being more open to bullish scenarios."
                )

    # ── 2. Category Analysis ─────────────────────────────────────────────
    for cat, stats in memory["category_stats"].items():
        if stats.get("total", 0) < 2:
            continue

        acc = stats["accuracy"]
        entry = {
            "category": cat,
            "accuracy": acc,
            "record": f"{stats['correct']}/{stats['total']}",
        }

        if acc < BIAS_THRESHOLDS["category_fail_threshold"] * 100:
            entry["status"] = "WEAK"
            diagnosis["blind_spots"].append(entry)
            diagnosis["recommendations"].append(
                f"📉 {cat}: Only {acc}% accuracy ({stats['correct']}/{stats['total']}). "
                f"Consider: (1) Lowering confidence for {cat} analyses, "
                f"(2) Being more conservative in this category, "
                f"(3) Re-examining what went wrong with past {cat} analyses."
            )
        elif acc >= BIAS_THRESHOLDS["category_strong_threshold"] * 100:
            entry["status"] = "STRONG"
            diagnosis["strengths"].append(entry)

    # ── 3. Confidence Calibration ────────────────────────────────────────
    # Are high-confidence predictions actually more accurate?
    high_conf = [a for a in all_with_outcome if (a.get("confidence") or 50) >= 70]
    low_conf = [a for a in all_with_outcome if (a.get("confidence") or 50) < 50]
    mid_conf = [a for a in all_with_outcome if 50 <= (a.get("confidence") or 50) < 70]

    def _acc(lst):
        if not lst:
            return None
        return round(sum(1 for a in lst if a["outcome"] == "correct") / len(lst) * 100, 1)

    diagnosis["confidence_calibration"] = {
        "high_confidence": {"count": len(high_conf), "actual_accuracy": _acc(high_conf)},
        "mid_confidence": {"count": len(mid_conf), "actual_accuracy": _acc(mid_conf)},
        "low_confidence": {"count": len(low_conf), "actual_accuracy": _acc(low_conf)},
    }

    high_acc = _acc(high_conf)
    low_acc = _acc(low_conf)
    if high_acc is not None and low_acc is not None:
        if high_acc <= low_acc:
            diagnosis["biases"].append({
                "type": "OVERCONFIDENCE",
                "severity": "HIGH",
                "description": f"High-confidence analyses ({high_acc}% accurate) perform WORSE "
                               f"than low-confidence ones ({low_acc}%). System is overconfident.",
            })
            diagnosis["recommendations"].append(
                "⚠️ Overconfidence detected. High-confidence predictions are no better than "
                "low-confidence ones. Recalibrate by being more skeptical of strong convictions."
            )
        elif high_acc is not None and high_acc < 60:
            diagnosis["biases"].append({
                "type": "OVERCONFIDENCE",
                "severity": "MEDIUM",
                "description": f"High-confidence analyses are only {high_acc}% accurate. "
                               f"Confidence should be more conservative.",
            })

    # ── 4. Overall Health ────────────────────────────────────────────────
    severe_issues = sum(1 for b in diagnosis["biases"] if b.get("severity") == "HIGH")
    severe_issues += sum(1 for s in diagnosis["blind_spots"])

    if severe_issues >= 3:
        diagnosis["system_health"] = "CRITICAL"
    elif severe_issues >= 1:
        diagnosis["system_health"] = "NEEDS_ATTENTION"
    elif gs["accuracy"] >= 65:
        diagnosis["system_health"] = "HEALTHY"
    else:
        diagnosis["system_health"] = "DEVELOPING"

    if not diagnosis["recommendations"]:
        diagnosis["recommendations"].append(
            "✅ No significant issues detected. Continue collecting outcomes to maintain accuracy."
        )

    return diagnosis


def check_risk_overrides(
    token_id: str,
    categories: list = None,
    sentiment: str = None,
    momentum: str = None,
    market_cap: float = None,
    rank: int = None,
) -> dict:
    """
    Check if the current analysis setup matches historically
    underperforming patterns. Returns warnings if so.

    This is the "pre-flight check" before generating a thesis.
    """
    memory = load_memory()
    categories = set(categories or [])
    overrides = []
    risk_level = "CLEAR"  # CLEAR, CAUTION, WARNING, DANGER

    gs = memory["global_stats"]
    if gs.get("total_outcomes", 0) < BIAS_THRESHOLDS["min_samples"]:
        return {
            "risk_level": "CLEAR",
            "overrides": [],
            "message": "Insufficient history for risk override analysis.",
        }

    # ── 1. Category failure pattern ──────────────────────────────────────
    for cat in categories:
        cat_stats = memory["category_stats"].get(cat, {})
        if cat_stats.get("total", 0) >= BIAS_THRESHOLDS["min_samples"]:
            if cat_stats["accuracy"] < BIAS_THRESHOLDS["category_fail_threshold"] * 100:
                overrides.append({
                    "type": "CATEGORY_UNDERPERFORMANCE",
                    "severity": "HIGH",
                    "message": f"⚠️ {cat} analyses have historically underperformed: "
                               f"{cat_stats['correct']}/{cat_stats['total']} ({cat_stats['accuracy']}% accuracy)",
                    "recommendation": f"Be extra conservative with {cat} predictions. "
                                      f"Consider defaulting to neutral sentiment.",
                })
                risk_level = "WARNING"

    # ── 2. Sentiment track record in context ─────────────────────────────
    if sentiment:
        # How does this specific sentiment perform in these categories?
        matching_analyses = []
        for tid, analyses in memory["analyses"].items():
            for a in analyses:
                if a.get("outcome") is None:
                    continue
                if a.get("sentiment") == sentiment:
                    a_cats = set(a.get("categories", []))
                    if categories & a_cats:  # Overlapping categories
                        matching_analyses.append(a)

        if len(matching_analyses) >= 2:
            correct = sum(1 for a in matching_analyses if a["outcome"] == "correct")
            acc = correct / len(matching_analyses) * 100
            if acc < 50:
                overrides.append({
                    "type": "SENTIMENT_CATEGORY_MISMATCH",
                    "severity": "HIGH",
                    "message": f"⚠️ '{sentiment}' verdicts in {', '.join(categories)} have "
                               f"{acc:.0f}% accuracy ({correct}/{len(matching_analyses)})",
                    "recommendation": f"This sentiment has been unreliable for this category. "
                                      f"Consider a more conservative stance.",
                })
                if risk_level != "DANGER":
                    risk_level = "WARNING"

    # ── 3. Momentum-sentiment conflict ───────────────────────────────────
    if sentiment and momentum:
        is_bullish = sentiment in ("bullish", "cautiously_bullish")
        is_bearish_momentum = momentum in ("ACCELERATING_DOWN", "STEADY_DOWN", "MIXED_BEARISH")
        is_bearish = sentiment in ("bearish", "cautiously_bearish")
        is_bullish_momentum = momentum in ("ACCELERATING_UP", "STEADY_UP", "MIXED_BULLISH")

        has_conflict = (is_bullish and is_bearish_momentum) or (is_bearish and is_bullish_momentum)

        if has_conflict:
            # Check how conflicts performed historically
            conflict_analyses = []
            for tid, analyses in memory["analyses"].items():
                for a in analyses:
                    if a.get("outcome") is None:
                        continue
                    a_sent = a.get("sentiment", "")
                    a_mom = a.get("momentum", "")
                    a_bull = a_sent in ("bullish", "cautiously_bullish")
                    a_bear_mom = a_mom in ("ACCELERATING_DOWN", "STEADY_DOWN", "MIXED_BEARISH")
                    a_bear = a_sent in ("bearish", "cautiously_bearish")
                    a_bull_mom = a_mom in ("ACCELERATING_UP", "STEADY_UP", "MIXED_BULLISH")
                    if (a_bull and a_bear_mom) or (a_bear and a_bull_mom):
                        conflict_analyses.append(a)

            if conflict_analyses:
                conflict_wrong = sum(1 for a in conflict_analyses if a["outcome"] == "wrong")
                conflict_rate = conflict_wrong / len(conflict_analyses) * 100
                if conflict_rate > 50:
                    overrides.append({
                        "type": "MOMENTUM_CONFLICT",
                        "severity": "MEDIUM",
                        "message": f"⚠️ Sentiment '{sentiment}' conflicts with momentum '{momentum}'. "
                                   f"Such conflicts have {conflict_rate:.0f}% failure rate historically.",
                        "recommendation": "When sentiment disagrees with momentum, momentum tends to win. "
                                          "Consider aligning verdict with the momentum signal.",
                    })
                    if risk_level == "CLEAR":
                        risk_level = "CAUTION"
            else:
                overrides.append({
                    "type": "MOMENTUM_CONFLICT",
                    "severity": "LOW",
                    "message": f"Sentiment '{sentiment}' conflicts with momentum '{momentum}'. "
                               f"No historical data on this conflict pattern yet.",
                    "recommendation": "Be aware that your verdict opposes the current momentum trend.",
                })
                if risk_level == "CLEAR":
                    risk_level = "CAUTION"

    # ── 4. Similar pattern failure rate ──────────────────────────────────
    similar = find_similar_analyses(
        token_id=token_id,
        categories=list(categories),
        market_cap=market_cap,
        rank=rank,
        limit=5,
    )
    similar_analyses = similar.get("similar_analyses", [])
    if len(similar_analyses) >= 2:
        similar_wrong = sum(1 for a in similar_analyses if a["outcome"] == "wrong")
        similar_fail_rate = similar_wrong / len(similar_analyses)
        if similar_fail_rate >= BIAS_THRESHOLDS["override_pattern_threshold"]:
            overrides.append({
                "type": "SIMILAR_PATTERN_FAILURE",
                "severity": "HIGH",
                "message": f"⚠️ WARNING: Similar setups have historically underperformed. "
                           f"{similar_wrong}/{len(similar_analyses)} ({similar_fail_rate:.0%}) failed.",
                "failed_examples": [
                    {"token": a["token_id"], "outcome": a["outcome"], "date": a["analysis_date"]}
                    for a in similar_analyses if a["outcome"] == "wrong"
                ],
                "recommendation": "This setup has historically underperformed. "
                                  "Increase skepticism and widen risk margins.",
            })
            risk_level = "DANGER" if similar_fail_rate >= 0.75 else "WARNING"

    # ── Build message ────────────────────────────────────────────────────
    if not overrides:
        message = "✅ No historical risk patterns detected. Proceed with standard analysis."
    else:
        high_count = sum(1 for o in overrides if o["severity"] == "HIGH")
        message = f"{len(overrides)} risk override(s) detected ({high_count} high severity). Review before proceeding."

    return {
        "risk_level": risk_level,
        "overrides": overrides,
        "override_count": len(overrides),
        "message": message,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Generation (Pattern Mining)
# ─────────────────────────────────────────────────────────────────────────────

def _mine_patterns(analyses: list, min_samples: int = 2) -> list:
    """
    Mine patterns from a list of analyses by cross-referencing dimensions.
    Returns a list of pattern dicts sorted by accuracy.
    """
    buckets = {}  # pattern_key -> {correct, wrong, total, analyses}

    for a in analyses:
        if a.get("outcome") is None:
            continue

        momentum = a.get("momentum", "UNKNOWN")
        sentiment = a.get("sentiment", "unknown")
        categories = a.get("categories", [])
        mcap = a.get("market_cap_at_analysis")
        mcap_tier = _get_mcap_tier(mcap) if mcap else "unknown"
        outcome = a["outcome"]

        # ── Single-dimension patterns ─────────────────────────────────
        single_dims = [
            ("momentum", momentum),
            ("sentiment", sentiment),
            ("mcap_tier", mcap_tier),
        ]
        for cat in categories:
            single_dims.append(("category", cat))

        for dim_name, dim_val in single_dims:
            if dim_val == "unknown" or dim_val == "UNKNOWN":
                continue
            key = f"{dim_name}={dim_val}"
            if key not in buckets:
                buckets[key] = {"correct": 0, "wrong": 0, "total": 0, "dimensions": {dim_name: dim_val}}
            buckets[key][outcome] += 1
            buckets[key]["total"] += 1

        # ── Two-dimension combos ──────────────────────────────────────
        combos_2d = []

        # category × momentum
        for cat in categories:
            combos_2d.append(({"category": cat, "momentum": momentum}, f"category={cat}+momentum={momentum}"))

        # category × sentiment
        for cat in categories:
            combos_2d.append(({"category": cat, "sentiment": sentiment}, f"category={cat}+sentiment={sentiment}"))

        # momentum × sentiment
        combos_2d.append(({"momentum": momentum, "sentiment": sentiment}, f"momentum={momentum}+sentiment={sentiment}"))

        # mcap_tier × momentum
        if mcap_tier != "unknown":
            combos_2d.append(({"mcap_tier": mcap_tier, "momentum": momentum}, f"mcap_tier={mcap_tier}+momentum={momentum}"))

        # mcap_tier × sentiment
        if mcap_tier != "unknown":
            combos_2d.append(({"mcap_tier": mcap_tier, "sentiment": sentiment}, f"mcap_tier={mcap_tier}+sentiment={sentiment}"))

        for dims, key in combos_2d:
            if any(v in ("unknown", "UNKNOWN") for v in dims.values()):
                continue
            if key not in buckets:
                buckets[key] = {"correct": 0, "wrong": 0, "total": 0, "dimensions": dims}
            buckets[key][outcome] += 1
            buckets[key]["total"] += 1

        # ── Three-dimension combos (the interesting ones) ─────────────
        for cat in categories:
            if mcap_tier != "unknown" and momentum not in ("UNKNOWN", "NEUTRAL"):
                key3 = f"category={cat}+mcap_tier={mcap_tier}+momentum={momentum}"
                dims3 = {"category": cat, "mcap_tier": mcap_tier, "momentum": momentum}
                if key3 not in buckets:
                    buckets[key3] = {"correct": 0, "wrong": 0, "total": 0, "dimensions": dims3}
                buckets[key3][outcome] += 1
                buckets[key3]["total"] += 1

            if sentiment not in ("unknown",) and momentum not in ("UNKNOWN", "NEUTRAL"):
                key3 = f"category={cat}+sentiment={sentiment}+momentum={momentum}"
                dims3 = {"category": cat, "sentiment": sentiment, "momentum": momentum}
                if key3 not in buckets:
                    buckets[key3] = {"correct": 0, "wrong": 0, "total": 0, "dimensions": dims3}
                buckets[key3][outcome] += 1
                buckets[key3]["total"] += 1

    # ── Convert to list, compute accuracy, filter ─────────────────────
    patterns = []
    for key, data in buckets.items():
        if data["total"] < min_samples:
            continue
        accuracy = round(data["correct"] / data["total"] * 100, 1)
        dim_count = len(data["dimensions"])
        patterns.append({
            "pattern_key": key,
            "dimensions": data["dimensions"],
            "dimension_count": dim_count,
            "correct": data["correct"],
            "wrong": data["wrong"],
            "total": data["total"],
            "accuracy": accuracy,
        })

    return patterns


def _pattern_to_readable(p: dict) -> str:
    """Convert a pattern dict to a human-readable description."""
    dim_labels = {
        "category": lambda v: f"{v} tokens",
        "momentum": lambda v: v.lower().replace("_", " "),
        "sentiment": lambda v: f"{v} verdict",
        "mcap_tier": lambda v: f"{v}-cap",
    }
    parts = []
    for dim, val in p["dimensions"].items():
        formatter = dim_labels.get(dim, lambda v: f"{dim}={v}")
        parts.append(formatter(val))
    return " + ".join(parts)


def generate_strategy() -> dict:
    """
    Analyze the system's own research performance to identify where
    its analyses have been most and least accurate.

    NOTE: This is NOT investment advice. It's a self-assessment of the
    system's analytical accuracy across different conditions. The output
    helps the system (and user) understand where its research is reliable
    vs. unreliable — not what to buy or sell.

    Returns patterns of high/low accuracy and research quality insights.
    """
    memory = load_memory()
    gs = memory["global_stats"]

    # Collect all analyses with outcomes
    all_analyses = []
    for tid, analyses in memory["analyses"].items():
        for a in analyses:
            if a.get("outcome") is not None:
                a_copy = dict(a)
                a_copy["_token_id"] = tid
                all_analyses.append(a_copy)

    if len(all_analyses) < 3:
        return {
            "status": "INSUFFICIENT_DATA",
            "total_outcomes": len(all_analyses),
            "minimum_required": 3,
            "message": "Need at least 3 outcomes to generate meaningful patterns. "
                       "Keep analyzing and using 'learn' to build history.",
            "best_patterns": [],
            "worst_patterns": [],
            "rules": [],
        }

    # Mine patterns
    all_patterns = _mine_patterns(all_analyses, min_samples=2)

    # Separate best and worst
    best_patterns = sorted(
        [p for p in all_patterns if p["accuracy"] >= 70],
        key=lambda p: (-p["accuracy"], -p["total"], -p["dimension_count"]),
    )
    worst_patterns = sorted(
        [p for p in all_patterns if p["accuracy"] <= 40],
        key=lambda p: (p["accuracy"], -p["total"], -p["dimension_count"]),
    )

    # Prefer multi-dimensional patterns (they're more specific/useful)
    # but keep single-dim ones as fallback
    def _deduplicate_top(patterns: list, limit: int = 10) -> list:
        """Pick top patterns, preferring higher dimension count."""
        multi = [p for p in patterns if p["dimension_count"] >= 2]
        single = [p for p in patterns if p["dimension_count"] == 1]
        result = multi[:limit]
        remaining = limit - len(result)
        if remaining > 0:
            result.extend(single[:remaining])
        return result

    best_top = _deduplicate_top(best_patterns, 10)
    worst_top = _deduplicate_top(worst_patterns, 10)

    # ── Generate research quality insights ────────────────────────────
    rules = []
    high_acc_insights = []
    low_acc_insights = []

    for p in best_top:
        desc = _pattern_to_readable(p)
        high_acc_insights.append({
            "type": "HIGH_ACCURACY",
            "condition": desc,
            "accuracy": p["accuracy"],
            "sample_size": p["total"],
            "insight": f"Research accuracy is {p['accuracy']}% ({p['correct']}/{p['total']}) in this setup. "
                       f"Analyses in this category have been historically reliable.",
        })

    for p in worst_top:
        desc = _pattern_to_readable(p)
        low_acc_insights.append({
            "type": "LOW_ACCURACY",
            "condition": desc,
            "accuracy": p["accuracy"],
            "sample_size": p["total"],
            "insight": f"Research accuracy is only {p['accuracy']}% ({p['correct']}/{p['total']}) in this setup. "
                       f"Analyses here have been unreliable — extra skepticism warranted.",
        })

    rules = high_acc_insights + low_acc_insights

    # ── Meta-strategy summary ─────────────────────────────────────────
    # Determine overall stance recommendation
    total_correct = gs.get("correct", 0)
    total_outcomes = gs.get("total_outcomes", 0)
    overall_acc = gs.get("accuracy", 50)

    # Find strongest and weakest categories
    cat_stats = memory.get("category_stats", {})
    strong_cats = [
        (k, v["accuracy"], f"{v['correct']}/{v['total']}")
        for k, v in cat_stats.items()
        if v.get("total", 0) >= 2 and v["accuracy"] >= 65
    ]
    weak_cats = [
        (k, v["accuracy"], f"{v['correct']}/{v['total']}")
        for k, v in cat_stats.items()
        if v.get("total", 0) >= 2 and v["accuracy"] < 50
    ]
    strong_cats.sort(key=lambda x: -x[1])
    weak_cats.sort(key=lambda x: x[1])

    # Find best momentum signal
    momentum_perf = {}
    for a in all_analyses:
        m = a.get("momentum", "UNKNOWN")
        if m in ("UNKNOWN", "NEUTRAL"):
            continue
        if m not in momentum_perf:
            momentum_perf[m] = {"correct": 0, "total": 0}
        momentum_perf[m]["total"] += 1
        if a["outcome"] == "correct":
            momentum_perf[m]["correct"] += 1

    best_momentum = None
    worst_momentum = None
    for m, stats in momentum_perf.items():
        if stats["total"] >= 2:
            acc = stats["correct"] / stats["total"] * 100
            if best_momentum is None or acc > best_momentum[1]:
                best_momentum = (m, acc, stats["total"])
            if worst_momentum is None or acc < worst_momentum[1]:
                worst_momentum = (m, acc, stats["total"])

    # Build executive summary
    exec_summary_parts = []

    if strong_cats:
        names = ", ".join(c[0] for c in strong_cats[:3])
        exec_summary_parts.append(f"Strongest research areas: {names} (historically accurate analyses)")
    if weak_cats:
        names = ", ".join(c[0] for c in weak_cats[:3])
        exec_summary_parts.append(f"Weakest research areas: {names} (historically inaccurate — treat with skepticism)")
    if best_momentum:
        exec_summary_parts.append(
            f"Best momentum signal: {best_momentum[0].lower().replace('_', ' ')} "
            f"({best_momentum[1]:.0f}% accuracy, n={best_momentum[2]})"
        )
    if worst_momentum and worst_momentum[0] != (best_momentum[0] if best_momentum else None):
        exec_summary_parts.append(
            f"Worst momentum signal: {worst_momentum[0].lower().replace('_', ' ')} "
            f"({worst_momentum[1]:.0f}% accuracy, n={worst_momentum[2]})"
        )
    if overall_acc < 50:
        exec_summary_parts.append(
            "Overall research accuracy is below 50%. Analyses from this system should be "
            "treated as low-confidence research, not reliable signals."
        )

    # Format best/worst for output
    best_formatted = []
    for p in best_top:
        best_formatted.append({
            "pattern": _pattern_to_readable(p),
            "accuracy": p["accuracy"],
            "record": f"{p['correct']}/{p['total']}",
            "dimensions": p["dimensions"],
            "dimension_count": p["dimension_count"],
        })

    worst_formatted = []
    for p in worst_top:
        worst_formatted.append({
            "pattern": _pattern_to_readable(p),
            "accuracy": p["accuracy"],
            "record": f"{p['correct']}/{p['total']}",
            "dimensions": p["dimensions"],
            "dimension_count": p["dimension_count"],
        })

    return {
        "status": "OK",
        "disclaimer": "This is a research performance self-assessment, NOT investment advice. "
                      "Patterns reflect where this system's analyses have been accurate or inaccurate. "
                      "Past analytical accuracy does not predict future results. Always DYOR.",
        "total_outcomes": len(all_analyses),
        "overall_accuracy": overall_acc,
        "patterns_mined": len(all_patterns),
        "best_patterns": best_formatted,
        "worst_patterns": worst_formatted,
        "insights": rules,
        "executive_summary": exec_summary_parts,
        "category_accuracy": {
            "strong": [{"category": c[0], "accuracy": c[1], "record": c[2]} for c in strong_cats],
            "weak": [{"category": c[0], "accuracy": c[1], "record": c[2]} for c in weak_cats],
        },
        "momentum_accuracy": {
            m: {"accuracy": round(s["correct"] / s["total"] * 100, 1), "total": s["total"]}
            for m, s in momentum_perf.items() if s["total"] >= 2
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Thesis Engine — Decision Engine Memory System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s record --token solana --price 105.50 --momentum MIXED_BEARISH --sentiment cautiously_bullish --categories layer-1 --rank 7
  %(prog)s learn --token solana --outcome correct
  %(prog)s learn --token arbitrum --outcome wrong --note "L2 narrative faded faster than expected"
  %(prog)s confidence --token ethereum --categories layer-1,smart-contracts
  %(prog)s similar --token near --categories layer-1 --rank 30 --mcap 3000000000
  %(prog)s diagnose
  %(prog)s risk-check --token near --categories layer-1 --sentiment bullish --momentum MIXED_BEARISH
  %(prog)s strategy
  %(prog)s history --token solana
  %(prog)s stats
  %(prog)s export
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── record ───────────────────────────────────────────────────────────
    p_record = subparsers.add_parser("record", help="Record a new analysis")
    p_record.add_argument("--token", required=True, help="Token ID")
    p_record.add_argument("--price", type=float, help="Price at time of analysis")
    p_record.add_argument("--mcap", type=float, help="Market cap at time of analysis")
    p_record.add_argument("--momentum", help="Momentum signal (e.g., ACCELERATING_UP)")
    p_record.add_argument("--sentiment", help="Analysis sentiment (bullish/neutral/bearish)")
    p_record.add_argument("--categories", help="Comma-separated categories")
    p_record.add_argument("--rank", type=int, help="Market cap rank")
    p_record.add_argument("--confidence", type=float, help="Confidence score")

    # ── learn ────────────────────────────────────────────────────────────
    p_learn = subparsers.add_parser("learn", help="Record an outcome (correct/wrong)")
    p_learn.add_argument("--token", required=True, help="Token ID")
    p_learn.add_argument("--outcome", required=True, choices=["correct", "wrong"],
                         help="Was the analysis correct or wrong?")
    p_learn.add_argument("--note", help="Optional note about why")

    # ── confidence ───────────────────────────────────────────────────────
    p_conf = subparsers.add_parser("confidence", help="Compute confidence score")
    p_conf.add_argument("--token", required=True, help="Token ID")
    p_conf.add_argument("--categories", help="Comma-separated categories")
    p_conf.add_argument("--data-completeness", type=float, default=0.7,
                        help="Fraction of data fields available (0-1)")

    # ── similar ──────────────────────────────────────────────────────────
    p_sim = subparsers.add_parser("similar", help="Find similar past analyses")
    p_sim.add_argument("--token", required=True, help="Token ID")
    p_sim.add_argument("--categories", help="Comma-separated categories")
    p_sim.add_argument("--mcap", type=float, help="Market cap for tier matching")
    p_sim.add_argument("--rank", type=int, help="Market cap rank")
    p_sim.add_argument("--limit", type=int, default=5, help="Max results")

    # ── history ──────────────────────────────────────────────────────────
    p_hist = subparsers.add_parser("history", help="View token analysis history")
    p_hist.add_argument("--token", required=True, help="Token ID")

    # ── stats ────────────────────────────────────────────────────────────
    subparsers.add_parser("stats", help="View global performance stats")

    # ── diagnose ─────────────────────────────────────────────────────────
    subparsers.add_parser("diagnose", help="Diagnose system biases and blind spots")

    # ── risk-check ───────────────────────────────────────────────────────
    p_risk = subparsers.add_parser("risk-check", help="Check risk overrides for a setup")
    p_risk.add_argument("--token", required=True, help="Token ID")
    p_risk.add_argument("--categories", help="Comma-separated categories")
    p_risk.add_argument("--sentiment", help="Planned sentiment (bullish/neutral/bearish)")
    p_risk.add_argument("--momentum", help="Current momentum signal")
    p_risk.add_argument("--mcap", type=float, help="Market cap")
    p_risk.add_argument("--rank", type=int, help="Market cap rank")

    # ── strategy ──────────────────────────────────────────────────────────
    subparsers.add_parser("strategy", help="Generate strategy from historical patterns")

    # ── export ───────────────────────────────────────────────────────────
    subparsers.add_parser("export", help="Export full memory as JSON")

    args = parser.parse_args()

    try:
        if args.command == "record":
            categories = [c.strip() for c in args.categories.split(",")] if args.categories else []
            result = record_analysis(
                token_id=args.token,
                price=args.price,
                market_cap=args.mcap,
                momentum=args.momentum,
                sentiment=args.sentiment,
                categories=categories,
                rank=args.rank,
                confidence=args.confidence,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "learn":
            result = learn_outcome(args.token, args.outcome, args.note)
            print(json.dumps(result, indent=2))

        elif args.command == "confidence":
            categories = [c.strip() for c in args.categories.split(",")] if args.categories else []
            result = compute_confidence(
                token_id=args.token,
                categories=categories,
                data_completeness=args.data_completeness,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "similar":
            categories = [c.strip() for c in args.categories.split(",")] if args.categories else []
            result = find_similar_analyses(
                token_id=args.token,
                categories=categories,
                market_cap=args.mcap,
                rank=args.rank,
                limit=args.limit,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "history":
            result = get_token_history(args.token)
            print(json.dumps(result, indent=2))

        elif args.command == "stats":
            result = get_global_stats()
            print(json.dumps(result, indent=2))

        elif args.command == "diagnose":
            result = diagnose_biases()
            print(json.dumps(result, indent=2))

        elif args.command == "risk-check":
            categories = [c.strip() for c in args.categories.split(",")] if args.categories else []
            result = check_risk_overrides(
                token_id=args.token,
                categories=categories,
                sentiment=args.sentiment,
                momentum=args.momentum,
                market_cap=args.mcap,
                rank=args.rank,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "strategy":
            result = generate_strategy()
            print(json.dumps(result, indent=2))

        elif args.command == "export":
            result = export_memory()
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
