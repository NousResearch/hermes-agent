"""Advanced benchmark runner for routing v2.

Design goals:
- Deterministic in test/mock mode so the double-TDD suite can assert precise
  rankings and deltas without hitting real providers.
- Per-category, per-model scoring via pluggable rubric callables.
- Latency + cost capture alongside quality score (weighted aggregate).
- Bootstrap confidence interval (simple percentile, stdlib only).
- JSON persistence to ~/.hermes/router/benchmarks.json with schema_version
  so older consumers fail fast instead of silently drifting.
- Delta detection between runs (regression alarm per model/category).

Public API:
    run_benchmarks(suite, evaluator, *, seed=0) -> BenchmarkReport
    rank_models(report, category)             -> list[(model, score)]
    detect_regressions(prev, curr, threshold) -> list[dict]
    save_report(report, path)                 -> None
    load_report(path)                         -> BenchmarkReport

A "suite" is:
    {category: {model: [prompt, ...]}}
An "evaluator" is a callable:
    evaluator(category, model, prompt) -> {"score": float, "latency_ms": float, "cost_usd": float}
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA_VERSION = 2


@dataclass
class ModelResult:
    model: str
    category: str
    n: int
    score_mean: float
    score_ci_low: float
    score_ci_high: float
    latency_p95_ms: float
    cost_mean_usd: float
    raw_scores: List[float] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    schema_version: int
    seed: int
    categories: Dict[str, Dict[str, ModelResult]]  # category -> model -> result
    generated_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "generated_at": self.generated_at,
            "categories": {
                cat: {m: asdict(r) for m, r in models.items()}
                for cat, models in self.categories.items()
            },
        }


# ---------------------------------------------------------------------------
# Weighted aggregate score (quality dominant, latency/cost as tie-breakers)
# ---------------------------------------------------------------------------

def weighted_score(quality: float, latency_ms: float, cost_usd: float,
                   *, w_q: float = 1.0, w_l: float = 0.05, w_c: float = 0.05,
                   latency_norm: float = 2000.0, cost_norm: float = 0.02) -> float:
    """Higher is better. Penalizes latency and cost relative to norms."""
    q = max(0.0, min(1.0, quality))
    l_pen = min(1.0, latency_ms / latency_norm) if latency_norm > 0 else 0.0
    c_pen = min(1.0, cost_usd / cost_norm) if cost_norm > 0 else 0.0
    raw = w_q * q - w_l * l_pen - w_c * c_pen
    # clamp into [0, 1] for easier downstream consumption
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval (percentile)
# ---------------------------------------------------------------------------

def bootstrap_ci(samples: List[float], *, iters: int = 500, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    if not samples:
        return (0.0, 0.0)
    if len(samples) == 1:
        v = float(samples[0])
        return (v, v)
    rng = random.Random(seed)
    means: List[float] = []
    n = len(samples)
    for _ in range(iters):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int(math.floor(iters * (alpha / 2)))]
    hi = means[int(math.ceil(iters * (1 - alpha / 2))) - 1]
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_benchmarks(suite: Dict[str, Dict[str, List[str]]],
                   evaluator: Callable[[str, str, str], Dict[str, float]],
                   *, seed: int = 0) -> BenchmarkReport:
    """Execute the suite deterministically and return a BenchmarkReport."""
    random.seed(seed)
    categories: Dict[str, Dict[str, ModelResult]] = {}
    for category, models in suite.items():
        categories[category] = {}
        for model, prompts in models.items():
            scores: List[float] = []
            latencies: List[float] = []
            costs: List[float] = []
            for prompt in prompts:
                out = evaluator(category, model, prompt) or {}
                q = float(out.get("score", 0.0))
                lat = float(out.get("latency_ms", 0.0))
                cost = float(out.get("cost_usd", 0.0))
                scores.append(weighted_score(q, lat, cost))
                latencies.append(lat)
                costs.append(cost)
            if not scores:
                continue
            ci_low, ci_high = bootstrap_ci(scores, seed=seed)
            p95 = _percentile(latencies, 95.0) if latencies else 0.0
            categories[category][model] = ModelResult(
                model=model,
                category=category,
                n=len(scores),
                score_mean=sum(scores) / len(scores),
                score_ci_low=ci_low,
                score_ci_high=ci_high,
                latency_p95_ms=p95,
                cost_mean_usd=sum(costs) / len(costs) if costs else 0.0,
                raw_scores=list(scores),
            )
    return BenchmarkReport(
        schema_version=SCHEMA_VERSION,
        seed=seed,
        categories=categories,
        generated_at=time.time(),
    )


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


# ---------------------------------------------------------------------------
# Ranking and regression detection
# ---------------------------------------------------------------------------

def rank_models(report: BenchmarkReport, category: str) -> List[Tuple[str, float]]:
    models = report.categories.get(category, {})
    ranked = sorted(
        ((m, r.score_mean) for m, r in models.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked


def detect_regressions(prev: BenchmarkReport, curr: BenchmarkReport,
                       *, threshold: float = 0.05) -> List[Dict[str, Any]]:
    """Return entries where curr.score_mean < prev.score_mean - threshold."""
    regressions: List[Dict[str, Any]] = []
    for cat, models in curr.categories.items():
        prev_models = prev.categories.get(cat, {})
        for model, r_curr in models.items():
            r_prev = prev_models.get(model)
            if not r_prev:
                continue
            delta = r_curr.score_mean - r_prev.score_mean
            if delta < -abs(threshold):
                regressions.append({
                    "category": cat,
                    "model": model,
                    "prev_score": r_prev.score_mean,
                    "curr_score": r_curr.score_mean,
                    "delta": delta,
                })
    return regressions


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(report: BenchmarkReport, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_report(path: str | os.PathLike) -> BenchmarkReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            f"benchmark schema mismatch: expected {SCHEMA_VERSION}, got {data.get('schema_version')}"
        )
    categories: Dict[str, Dict[str, ModelResult]] = {}
    for cat, models in (data.get("categories") or {}).items():
        categories[cat] = {m: ModelResult(**r) for m, r in models.items()}
    return BenchmarkReport(
        schema_version=int(data["schema_version"]),
        seed=int(data.get("seed", 0)),
        categories=categories,
        generated_at=float(data.get("generated_at", 0.0)),
    )


# ---------------------------------------------------------------------------
# Convenience: extract the flat {category: {model: score}} structure the
# routing_v2.select_model() function consumes.
# ---------------------------------------------------------------------------

def report_to_benchmarks(report: BenchmarkReport) -> Dict[str, Dict[str, float]]:
    return {
        cat: {m: r.score_mean for m, r in models.items()}
        for cat, models in report.categories.items()
    }
