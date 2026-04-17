"""Zero-token benchmark evaluator.

Design requirement: benchmarks must NOT burn tokens by default.
This module provides a deterministic, heuristic evaluator that scores
model/prompt pairs using cached priors + cheap string features. Results
feed benchmark_runner so routing_v2 gets a realistic ranking without
calling any LLM provider.

If you want real-LLM scoring later, swap in a different evaluator — the
runner contract is a simple callable (category, model, prompt) -> dict.

Usage:
    from agent.benchmark_harness import heuristic_evaluator, DEFAULT_SUITE
    from agent import benchmark_runner
    report = benchmark_runner.run_benchmarks(DEFAULT_SUITE, heuristic_evaluator)
    benchmark_runner.save_report(report, "~/.hermes/router/benchmarks.json")
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

# Static priors per (category, model). Editable without retraining.
# Values are rough stand-ins; they are perturbed deterministically per prompt
# so each model has a reproducible distribution, not a flat score.
MODEL_PRIORS: Dict[str, Dict[str, float]] = {
    "code": {
        "qwen3-coder-next":      0.92,
        "kimi-k2.5":             0.80,
        "deepseek-v3.2":         0.74,
        "qwen3.5:397b":          0.78,
        "gpt-5-mini":            0.60,
        "glm-5.1":               0.52,
    },
    "debug": {
        "qwen3-coder-next":      0.91,
        "kimi-k2.5":             0.79,
        "qwen3.5:397b":          0.75,
        "gpt-5-mini":            0.58,
        "glm-5.1":               0.50,
    },
    "analysis": {
        "qwen3.5:397b":          0.90,
        "kimi-k2.5":             0.80,
        "deepseek-v3.2":         0.74,
        "gpt-5-mini":            0.62,
        "glm-5.1":               0.56,
    },
    "research": {
        "deepseek-v3.2":         0.88,
        "kimi-k2.5":             0.74,
        "qwen3.5:397b":          0.78,
        "glm-5.1":               0.54,
    },
    "writing": {
        "mistral-large-3:675b":  0.89,
        "kimi-k2.5":             0.76,
        "gpt-5-mini":            0.62,
        "glm-5.1":               0.55,
    },
    "creative": {
        "minimax-m2.7":          0.85,
        "kimi-k2.5":             0.72,
        "mistral-large-3:675b":  0.80,
        "glm-5.1":               0.52,
    },
    "vision": {
        "qwen3-vl:235b-instruct": 0.91,
    },
    "simple": {
        "glm-5.1":               0.82,
        "gpt-5-mini":            0.80,
        "kimi-k2.5":             0.74,
    },
}


# Small gold-prompt suite per category. Kept short on purpose: benchmark is
# about ranking, not exhaustive eval.
DEFAULT_SUITE: Dict[str, Dict[str, List[str]]] = {}

_GOLD_PROMPTS: Dict[str, List[str]] = {
    "code":     ["refactor bubble sort in place",
                 "fix off-by-one in slicing",
                 "write pytest for regex util",
                 "optimize nested loop O(n^2)->O(n)",
                 "explain stacktrace KeyError"],
    "debug":    ["diagnose flaky CI job",
                 "find race condition in thread pool",
                 "why does import order break module?",
                 "trace high memory in pandas pipeline",
                 "fix pickling error for dataclass"],
    "analysis": ["compare two architecture diagrams",
                 "estimate cost tradeoffs of plan A vs B",
                 "summarize 3 pros and cons of choice X",
                 "identify single point of failure in system",
                 "evaluate latency budget for 5 services"],
    "research": ["latest on nuclear fusion breakthroughs",
                 "compare Postgres vs MySQL replication 2026",
                 "summarize FDA drug approvals Q1",
                 "find top 3 papers on retrieval augmented gen",
                 "recent changes to GDPR AI regulation"],
    "writing":  ["draft concise release notes",
                 "write a friendly onboarding email",
                 "expand bullet list into paragraph",
                 "rewrite in formal register",
                 "shorten to 80 words without losing content"],
    "creative": ["tagline for a coffee brand",
                 "short sci-fi hook about time loops",
                 "3 metaphors for rapid learning",
                 "name a cozy puzzle game",
                 "lyric for a lo-fi track"],
    "vision":   ["describe this diagram /tmp/x.png",
                 "extract table from screenshot",
                 "identify UI element in mockup",
                 "caption photo with mood",
                 "count objects in image"],
    "simple":   ["hola",
                 "ok",
                 "thanks",
                 "what time is it",
                 "cual es la capital de francia"],
}

for _cat, _models in MODEL_PRIORS.items():
    DEFAULT_SUITE[_cat] = {m: list(_GOLD_PROMPTS.get(_cat, ["probe"])) for m in _models}


def _seed_offset(category: str, model: str, prompt: str) -> float:
    """Deterministic offset in [-0.05, +0.05] derived from the triple."""
    h = hashlib.sha256(f"{category}|{model}|{prompt}".encode()).digest()
    v = int.from_bytes(h[:2], "big") / 0xFFFF  # [0, 1]
    return (v - 0.5) * 0.10


def heuristic_evaluator(category: str, model: str, prompt: str) -> Dict[str, float]:
    """Token-free evaluator.

    - score: prior[category][model] + deterministic noise, clamped [0, 1].
    - latency_ms: synthetic, penalizes very large models slightly.
    - cost_usd: synthetic, larger-model penalty, small absolute values.
    """
    priors = MODEL_PRIORS.get(category, {})
    base = float(priors.get(model, 0.50))
    score = max(0.0, min(1.0, base + _seed_offset(category, model, prompt)))
    # rough synthetic latency proxy: bigger name -> slower
    size_penalty = 1.0 + 0.15 * model.count(":") + 0.05 * len(model)
    latency_ms = 200.0 * size_penalty
    cost_usd = 0.0005 * size_penalty
    return {"score": score, "latency_ms": latency_ms, "cost_usd": cost_usd}
