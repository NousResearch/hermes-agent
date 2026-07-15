"""
Thompson-sampling bandit for cost-optimal model routing.

Routes autonomous tasks (cron jobs, subagents, curator) to the cheapest model
that reliably succeeds for that task complexity. Never overrides the user's
interactive session model.

State is persisted to ~/.hermes/bandit_state.json and survives restarts.
Learning is per-complexity-bucket (simple/moderate/complex) so failures
on hard tasks don't suppress cheap models for easy tasks.

Enable via config.yaml:
    bandit_router:
      enabled: true
      candidates:
        - model: "claude-haiku-3.5"
          provider: "anthropic"
          cost_per_mtok: 0.25
        - model: "claude-sonnet-4"
          provider: "anthropic"
          cost_per_mtok: 3.0
        - model: "claude-opus-4"
          provider: "anthropic"
          cost_per_mtok: 15.0
      quality_floor: 0.6
"""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Complexity buckets
SIMPLE = "simple"
MODERATE = "moderate"
COMPLEX = "complex"

STATE_FILE = Path("~/.hermes/bandit_state.json").expanduser()
MAX_OUTCOMES_HISTORY = 200


@dataclass
class ModelCandidate:
    model: str
    provider: str
    cost_per_mtok: float


@dataclass
class RouteDecision:
    model: str
    provider: str
    bucket: str
    sampled_theta: float
    reason: str


# ═══════════════════════════════════════════════════════════════════
# Complexity Classification (pure heuristics, ~0.1ms)
# ═══════════════════════════════════════════════════════════════════

COMPLEX_KEYWORDS = [
    "debug", "review", "security", "architecture", "refactor",
    "research", "analyze", "exploit", "reverse", "pentest",
    "implement", "design", "optimize", "migrate", "vulnerability",
    "cve", "audit", "investigate",
]

MODERATE_KEYWORDS = [
    "search", "find", "compare", "evaluate", "summarize",
    "gather", "collect", "scan", "enumerate", "discover",
    "report", "extract", "parse", "fetch", "query",
    "write", "generate", "create", "build", "draft", "brief",
]

SIMPLE_KEYWORDS = [
    "format", "check", "alert", "notify", "list",
    "status", "ping", "monitor", "count", "log",
]


def classify_complexity(ctx: Dict[str, Any]) -> str:
    """Classify task complexity from context. Pure heuristics, no LLM call."""
    prompt = ctx.get("prompt", "")
    toolsets = ctx.get("toolsets") or []
    skills = ctx.get("skills") or []
    has_script = ctx.get("has_script", False)
    no_agent = ctx.get("no_agent", False)

    # Script-only / no_agent jobs are always simple
    if has_script or no_agent:
        return SIMPLE

    score = 0.0

    # Prompt length signal
    plen = len(prompt)
    if plen > 2000:
        score += 0.3
    elif plen > 800:
        score += 0.15

    # Tool complexity (strong signal for autonomous tasks)
    score += min(len(toolsets) * 0.12, 0.35)

    # Skills = structured methodology
    score += min(len(skills) * 0.15, 0.3)

    # Keyword signals
    prompt_lower = prompt.lower()
    for kw in COMPLEX_KEYWORDS:
        if kw in prompt_lower:
            score += 0.10
    for kw in MODERATE_KEYWORDS:
        if kw in prompt_lower:
            score += 0.08
    for kw in SIMPLE_KEYWORDS:
        if kw in prompt_lower:
            score -= 0.08

    score = max(0.0, min(1.0, score))

    if score < 0.25:
        return SIMPLE
    elif score < 0.55:
        return MODERATE
    else:
        return COMPLEX


# ═══════════════════════════════════════════════════════════════════
# State Management
# ═══════════════════════════════════════════════════════════════════

def _load_state() -> Dict[str, Any]:
    """Load bandit state from disk. Returns fresh state if missing/corrupt."""
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            if data.get("version") == 1:
                return data
    except Exception as e:
        logger.warning("Bandit state load failed, using fresh state: %s", e)
    return {"version": 1, "priors": {}, "outcomes": [], "updated": None}


def _save_state(state: Dict[str, Any]) -> None:
    """Persist bandit state to disk."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning("Bandit state save failed: %s", e)


def _get_prior(state: Dict, bucket: str, model: str) -> Dict[str, float]:
    """Get Beta(alpha, beta) prior for a model in a bucket."""
    return (
        state.get("priors", {})
        .get(bucket, {})
        .get(model, {"alpha": 1.0, "beta": 1.0})
    )


# ═══════════════════════════════════════════════════════════════════
# Core Algorithm: Thompson Sampling
# ═══════════════════════════════════════════════════════════════════

def select_model(
    task_context: Dict[str, Any],
    candidates: List[ModelCandidate],
    quality_floor: float = 0.6,
) -> RouteDecision:
    """Thompson sample → pick cheapest viable model.

    Args:
        task_context: dict with prompt, toolsets, skills, has_script, no_agent
        candidates: list of ModelCandidate (ordered doesn't matter)
        quality_floor: minimum sampled θ to consider viable

    Returns:
        RouteDecision with chosen model, bucket, sample value, and reason
    """
    bucket = classify_complexity(task_context)
    state = _load_state()

    # Sample θ ~ Beta(α, β) for each candidate
    samples: Dict[str, float] = {}
    for c in candidates:
        prior = _get_prior(state, bucket, c.model)
        theta = random.betavariate(prior["alpha"], prior["beta"])
        samples[c.model] = theta

    # Cost-optimal: among models with θ >= floor, pick cheapest
    sorted_by_cost = sorted(candidates, key=lambda c: c.cost_per_mtok)
    for c in sorted_by_cost:
        if samples[c.model] >= quality_floor:
            return RouteDecision(
                model=c.model,
                provider=c.provider,
                bucket=bucket,
                sampled_theta=samples[c.model],
                reason=f"cheapest above floor (θ={samples[c.model]:.3f} >= {quality_floor})",
            )

    # Nothing above floor → pick highest θ (best bet regardless of cost)
    best = max(candidates, key=lambda c: samples[c.model])
    return RouteDecision(
        model=best.model,
        provider=best.provider,
        bucket=bucket,
        sampled_theta=samples[best.model],
        reason=f"highest sample (θ={samples[best.model]:.3f}), all below floor",
    )


def record_outcome(
    model: str,
    bucket: str,
    success: bool,
    duration_s: Optional[float] = None,
    tokens_used: Optional[int] = None,
) -> None:
    """Update Beta prior after task completes.

    Args:
        model: model ID that was used
        bucket: complexity bucket (simple/moderate/complex)
        success: True if task succeeded without error
        duration_s: optional task duration
        tokens_used: optional total tokens consumed
    """
    state = _load_state()

    # Ensure nested structure
    if bucket not in state.get("priors", {}):
        state.setdefault("priors", {})[bucket] = {}
    if model not in state["priors"][bucket]:
        state["priors"][bucket][model] = {"alpha": 1.0, "beta": 1.0}

    prior = state["priors"][bucket][model]
    if success:
        prior["alpha"] += 1.0
    else:
        prior["beta"] += 1.0

    # Record outcome for debugging/audit
    outcome = {
        "model": model,
        "bucket": bucket,
        "success": success,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if duration_s is not None:
        outcome["duration_s"] = round(duration_s, 2)
    if tokens_used is not None:
        outcome["tokens_used"] = tokens_used

    outcomes = state.setdefault("outcomes", [])
    outcomes.append(outcome)
    if len(outcomes) > MAX_OUTCOMES_HISTORY:
        state["outcomes"] = outcomes[-MAX_OUTCOMES_HISTORY:]

    _save_state(state)


# ═══════════════════════════════════════════════════════════════════
# Config Integration
# ═══════════════════════════════════════════════════════════════════

def get_candidates_from_config(cfg: Dict[str, Any]) -> List[ModelCandidate]:
    """Parse bandit_router.candidates from Hermes config.yaml."""
    br = cfg.get("bandit_router") or {}
    raw = br.get("candidates") or []
    candidates = []
    for entry in raw:
        if isinstance(entry, dict) and entry.get("model"):
            candidates.append(ModelCandidate(
                model=entry["model"],
                provider=entry.get("provider", ""),
                cost_per_mtok=float(entry.get("cost_per_mtok", 1.0)),
            ))
    return candidates


def is_enabled(cfg: Dict[str, Any]) -> bool:
    """Check if bandit router is enabled in config."""
    br = cfg.get("bandit_router") or {}
    return bool(br.get("enabled", False))


def get_quality_floor(cfg: Dict[str, Any]) -> float:
    """Get quality floor from config (default 0.6)."""
    br = cfg.get("bandit_router") or {}
    return float(br.get("quality_floor", 0.6))


# ═══════════════════════════════════════════════════════════════════
# Status / CLI helpers
# ═══════════════════════════════════════════════════════════════════

def get_status() -> Dict[str, Any]:
    """Return current bandit state summary for display."""
    state = _load_state()
    priors = state.get("priors", {})
    outcomes = state.get("outcomes", [])

    summary = {
        "updated": state.get("updated"),
        "total_outcomes": len(outcomes),
        "buckets": {},
    }

    for bucket in [SIMPLE, MODERATE, COMPLEX]:
        bucket_priors = priors.get(bucket, {})
        models = {}
        for model, prior in bucket_priors.items():
            alpha = prior.get("alpha", 1.0)
            beta = prior.get("beta", 1.0)
            total = alpha + beta - 2  # subtract initial 1,1
            models[model] = {
                "alpha": alpha,
                "beta": beta,
                "mean": round(alpha / (alpha + beta), 3),
                "total_tasks": int(total),
            }
        summary["buckets"][bucket] = models

    return summary


def reset_state() -> None:
    """Reset all priors to uniform Beta(1,1). Irreversible."""
    state = {"version": 1, "priors": {}, "outcomes": [], "updated": None}
    _save_state(state)
    logger.info("Bandit state reset to uniform priors")
