"""End-to-end TDD for routing v2: benchmark + router + task_state together.

Simulates a full session:
1. Run the token-free benchmark harness.
2. Feed the report into routing_v2.select_model.
3. Drive a multi-turn conversation through task_state to prove that
   continuation markers ("dale", "hazlo", "sigue", silence) keep the
   heavy model engaged instead of dropping to the cheap tier.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent import benchmark_runner as br
from agent import benchmark_harness as bh
from agent import routing_v2 as rv2
from agent import task_state as ts


@pytest.fixture(scope="module")
def report():
    return br.run_benchmarks(bh.DEFAULT_SUITE, bh.heuristic_evaluator, seed=0)


@pytest.fixture(scope="module")
def benches(report):
    return br.report_to_benchmarks(report)


def test_benchmark_report_has_all_categories(report):
    expected = {"code", "debug", "analysis", "research", "writing", "creative", "vision", "simple"}
    assert expected.issubset(set(report.categories.keys()))


def test_code_category_ranks_qwen_coder_first(benches):
    decision = rv2.select_model(
        "refactor pipeline and fix failing pytest",
        benches, rv2.DEFAULT_TIERS, task_state=None,
    )
    assert decision["category"] == "code"
    assert decision["model"] == "qwen3-coder-next"


def test_full_session_continuation_never_downgrades(benches, tmp_path: Path):
    state = ts.default_state()
    # Turn 1: heavy code task arrives.
    d1 = rv2.select_model("refactor and add pytest coverage for tools/foo.py",
                          benches, rv2.DEFAULT_TIERS, task_state=state)
    assert d1["category"] == "code"
    assert d1["model"] == "qwen3-coder-next"
    state = ts.start_task(state, tier=d1["tier"], model=d1["model"], category=d1["category"])

    heavy_tier = d1["tier"]
    heavy_model = d1["model"]

    # Turns 2..6: tiny continuation inputs. Router MUST keep heavy tier.
    for msg in ["dale", "sigue", "hazlo", "", "ok"]:
        d = rv2.select_model(msg, benches, rv2.DEFAULT_TIERS, task_state=state)
        assert d["tier"] == heavy_tier, f"downgrade on {msg!r}: tier={d['tier']}"
        assert d["model"] == heavy_model
        state = ts.record_turn(state, msg, was_easy=False)

    # Turn 7: explicit new unrelated easy topic — router can pivot away.
    d_new = rv2.select_model("hola",
                             benches, rv2.DEFAULT_TIERS, task_state=None)
    assert d_new["tier"] <= 2


def test_persisted_task_state_round_trip(tmp_path: Path, benches):
    path = tmp_path / "task_state.json"
    s = ts.start_task(ts.default_state(), tier=4, model="qwen3-coder-next", category="code")
    ts.save(path, s)

    loaded = ts.load(path)
    # Use loaded state directly
    d = rv2.select_model("dale", benches, rv2.DEFAULT_TIERS, task_state=loaded)
    assert d["tier"] >= 4
    assert d["model"] == "qwen3-coder-next"
    assert d["reason"] == "continuation"
