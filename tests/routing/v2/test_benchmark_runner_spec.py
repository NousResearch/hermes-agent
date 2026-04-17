"""Spec-level TDD for benchmark_runner (suite A of double TDD).

Focus: behavioral contract — given a suite + evaluator, produce a well-formed
report, correct rankings, regression detection, and schema-versioned I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import benchmark_runner as br


@pytest.fixture
def suite():
    return {
        "code": {
            "qwen3-coder-next": ["fix bug 1", "refactor 2", "write pytest 3"],
            "glm-5.1":          ["fix bug 1", "refactor 2", "write pytest 3"],
        },
        "simple": {
            "glm-5.1":    ["hola", "hi", "hello"],
            "gpt-5-mini": ["hola", "hi", "hello"],
        },
    }


@pytest.fixture
def evaluator_factory():
    def make(scores):
        def _ev(category, model, prompt):
            return {
                "score": scores[(category, model)],
                "latency_ms": 500.0,
                "cost_usd": 0.001,
            }
        return _ev
    return make


def test_report_schema_is_versioned(suite, evaluator_factory):
    ev = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.9, ("code", "glm-5.1"): 0.5,
        ("simple", "glm-5.1"): 0.8, ("simple", "gpt-5-mini"): 0.75,
    })
    rep = br.run_benchmarks(suite, ev, seed=42)
    assert rep.schema_version == br.SCHEMA_VERSION
    assert rep.seed == 42


def test_rank_models_sorted_by_score(suite, evaluator_factory):
    ev = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.92, ("code", "glm-5.1"): 0.40,
        ("simple", "glm-5.1"): 0.80, ("simple", "gpt-5-mini"): 0.70,
    })
    rep = br.run_benchmarks(suite, ev, seed=0)
    ranked = br.rank_models(rep, "code")
    assert ranked[0][0] == "qwen3-coder-next"
    assert ranked[0][1] > ranked[1][1]


def test_confidence_interval_bounds(suite, evaluator_factory):
    ev = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.9, ("code", "glm-5.1"): 0.5,
        ("simple", "glm-5.1"): 0.8, ("simple", "gpt-5-mini"): 0.75,
    })
    rep = br.run_benchmarks(suite, ev, seed=1)
    r = rep.categories["code"]["qwen3-coder-next"]
    assert r.score_ci_low <= r.score_mean <= r.score_ci_high


def test_regression_detection(suite, evaluator_factory):
    ev_prev = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.9, ("code", "glm-5.1"): 0.6,
        ("simple", "glm-5.1"): 0.8, ("simple", "gpt-5-mini"): 0.75,
    })
    ev_curr = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.7,  # regressed
        ("code", "glm-5.1"): 0.6,
        ("simple", "glm-5.1"): 0.82, ("simple", "gpt-5-mini"): 0.76,
    })
    prev = br.run_benchmarks(suite, ev_prev, seed=0)
    curr = br.run_benchmarks(suite, ev_curr, seed=0)
    regs = br.detect_regressions(prev, curr, threshold=0.05)
    assert any(r["model"] == "qwen3-coder-next" and r["category"] == "code" for r in regs)


def test_roundtrip_save_load(tmp_path: Path, suite, evaluator_factory):
    ev = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.9, ("code", "glm-5.1"): 0.5,
        ("simple", "glm-5.1"): 0.8, ("simple", "gpt-5-mini"): 0.75,
    })
    rep = br.run_benchmarks(suite, ev, seed=7)
    out = tmp_path / "bench.json"
    br.save_report(rep, out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == br.SCHEMA_VERSION
    back = br.load_report(out)
    assert back.seed == 7
    assert back.categories["code"]["qwen3-coder-next"].n == 3


def test_report_to_benchmarks_flattens_for_router(suite, evaluator_factory):
    ev = evaluator_factory({
        ("code", "qwen3-coder-next"): 0.9, ("code", "glm-5.1"): 0.5,
        ("simple", "glm-5.1"): 0.8, ("simple", "gpt-5-mini"): 0.75,
    })
    rep = br.run_benchmarks(suite, ev, seed=0)
    flat = br.report_to_benchmarks(rep)
    assert "code" in flat and "qwen3-coder-next" in flat["code"]
    assert 0.0 <= flat["code"]["qwen3-coder-next"] <= 1.0
