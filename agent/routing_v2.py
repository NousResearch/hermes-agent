"""Minimal routing_v2 implementation to satisfy TDD adversarial suite.

This is intentionally small and self-contained: pure-Python logic used by
tests in tests/routing/v2. It implements select_model, escalate and
maybe_downscale according to the test contract.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

_CONTINUATION_MARKERS = {"continúa", "continua", "sigue", "resume", "dale", "ok", "mismo tema", "dale"}

_CODE_KEYWORDS = {"refactor", "pytest", "stacktrace", "debug", "implement", "bug", "patch", "traceback"}
_RESEARCH_KEYWORDS = {"investiga", "investigar", "noticias", "fuente", "fuentes", "resume", "resumen", "investigate", "research"}
_ANALYSIS_KEYWORDS = {"analyze", "analysis", "analiza", "analizar", "analysis", "investigate"}
_VISION_MARKERS = {"image", ".png", ".jpg", "imagen", "describe this image", "describe image"}
_SIMPLE_MARKERS = {"hola", "hi", "hey", "hello"}


def _detect_category(prompt: str) -> str:
    if not prompt:
        return "simple"
    low = prompt.lower()
    for m in _VISION_MARKERS:
        if m in low:
            return "vision"
    for k in _CODE_KEYWORDS:
        if k in low:
            return "code"
    for k in _RESEARCH_KEYWORDS:
        if k in low:
            return "research"
    for k in _ANALYSIS_KEYWORDS:
        if k in low:
            return "analysis"
    for k in _SIMPLE_MARKERS:
        if low.strip() == k:
            return "simple"
    # default heuristic: short prompts -> simple, otherwise analysis
    if len(low.split()) < 4:
        return "simple"
    return "analysis"


def _best_model_for_category(benchmarks: Dict[str, Dict[str, float]], category: str) -> Optional[str]:
    models = benchmarks.get(category) or {}
    if not models:
        return None
    # pick model with highest score
    best = max(models.items(), key=lambda kv: kv[1])
    return best[0]


def _score_for_model(benchmarks: Dict[str, Dict[str, float]], category: str, model: str) -> float:
    return float(benchmarks.get(category, {}).get(model, 0.0))


def _find_tier_for_model(tiers: List[List[str]], model: str) -> Optional[int]:
    for idx, group in enumerate(tiers, start=1):
        if model in group:
            return idx
    return None


def select_model(prompt: str, benchmarks: Dict[str, Dict[str, float]], tiers: List[List[str]], task_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Select model based on category + benchmarks + task_state stickiness.

    Returns keys used by tests: category, model, tier, benchmark_score, reason
    """
    # Continuation / silence handling
    if task_state and task_state.get("active_task"):
        last_tier = int(task_state.get("last_tier", 0))
        last_model = task_state.get("last_model")
        # if continuation marker or empty prompt -> preserve
        if (not prompt) or any(m for m in _CONTINUATION_MARKERS if m in (prompt or "").lower()):
            return {
                "category": task_state.get("last_category", "unknown"),
                "model": last_model,
                "tier": last_tier,
                "benchmark_score": _score_for_model(benchmarks, task_state.get("last_category", ""), last_model) if last_model else 0.0,
                "reason": "continuation",
            }

    category = _detect_category(prompt)
    # best model according to benchmarks
    best = _best_model_for_category(benchmarks, category)
    if not best:
        # fallback: choose first model from highest tier available
        for group in reversed(tiers):
            if group:
                best = group[-1]
                break
    tier = _find_tier_for_model(tiers, best) or len(tiers)
    return {
        "category": category,
        "model": best,
        "tier": tier,
        "benchmark_score": _score_for_model(benchmarks, category, best),
        "reason": "benchmark_best",
    }


def escalate(current_model: str, tiers: List[List[str]], reason: str = "") -> Dict[str, Any]:
    """Move exactly one tier up. Cap at top.

    Returns: {tier, model, capped}
    """
    cur_tier = _find_tier_for_model(tiers, current_model) or 1
    if cur_tier >= len(tiers):
        # already at top
        return {"tier": len(tiers), "model": current_model, "capped": True}
    next_tier = cur_tier + 1
    # pick first model in next tier (deterministic)
    next_models = tiers[next_tier - 1]
    next_model = next_models[0] if next_models else current_model
    return {"tier": next_tier, "model": next_model, "capped": False}


def maybe_downscale(state: Dict[str, Any], tiers: List[List[str]]) -> Dict[str, Any]:
    """Consider downscaling one tier after easy_streak >=2 and not active_task.

    state: expects last_tier and easy_streak and active_task
    """
    last_tier = int(state.get("last_tier", 1))
    easy_streak = int(state.get("easy_streak", 0))
    active = bool(state.get("active_task", False))
    if active:
        return {"tier": last_tier}
    if easy_streak >= 2 and last_tier > 1:
        return {"tier": last_tier - 1}
    return {"tier": last_tier}
