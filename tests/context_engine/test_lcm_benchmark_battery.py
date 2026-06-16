from __future__ import annotations

from pathlib import Path

import pytest

from scripts import bench_lcm_context_engine as bench


def _stats(arm: str, *, correct: int, total: int = 100, observed_tokens: int = 50_000) -> bench.ArmStats:
    return bench.ArmStats(
        arm=arm,
        correct=correct,
        total=total,
        estimated_prompt_tokens=observed_tokens,
        observed_prompt_tokens=observed_tokens,
        estimated_completion_tokens=1_000,
        observed_completion_tokens=1_000,
        estimated_cost_usd=0.01,
        observed_cost_usd=0.01,
    )


def test_dry_run_report_compares_raw_compressor_and_lcm(tmp_path: Path) -> None:
    out_path = tmp_path / "lcm-benchmark.md"

    run = bench.run_benchmark_battery(
        dry_run=True,
        out_path=out_path,
        budget=bench.BudgetPolicy(max_usd=10.0),
    )

    assert run.gate.verdict == "GO", "\n".join(run.gate.failures)
    assert set(run.arm_stats) == {"raw", "compressor", "lcm"}
    assert run.arm_stats["raw"].correctness == pytest.approx(1.0)
    assert run.arm_stats["lcm"].correctness == pytest.approx(run.arm_stats["raw"].correctness)
    assert run.arm_stats["compressor"].correctness < run.arm_stats["lcm"].correctness
    assert run.arm_stats["lcm"].observed_tokens < run.arm_stats["raw"].observed_tokens
    assert run.spend_events[0].phase == "preflight"
    assert [event.arm for event in run.spend_events if event.phase == "post-arm"] == ["raw", "compressor", "lcm"]

    report = out_path.read_text(encoding="utf-8")
    assert "# PRD-3 LCM Context Engine Benchmark Battery" in report
    assert "correctness(lcm)" in report
    assert "correctness(compressor)" in report
    assert "raw baseline" in report
    assert "GO/NARROW-GO/NO-GO verdict: GO" in report
    assert "provider-observed tokens" in report
    assert "estimated before run" in report
    assert "observed after arm" in report


def test_adversarial_fixtures_show_lcm_recall_expand_where_compressor_drops_facts(tmp_path: Path) -> None:
    run = bench.run_benchmark_battery(
        dry_run=True,
        out_path=tmp_path / "lcm-benchmark.md",
        budget=bench.BudgetPolicy(max_usd=10.0),
    )
    records = {(record.arm, record.fixture_id): record for record in run.records}

    expected_kinds = {"dense-negations", "near-duplicate-numeric-outliers", "names", "long-tool-dump"}
    assert {fixture.kind for fixture in run.fixtures} == expected_kinds

    compressor_drops = set()
    for fixture in run.fixtures:
        compressor_record = records[("compressor", fixture.fixture_id)]
        lcm_record = records[("lcm", fixture.fixture_id)]
        assert lcm_record.correct is True
        assert {call["name"] for call in lcm_record.tool_calls} >= {"lcm_grep", "lcm_expand"}
        if not compressor_record.correct:
            compressor_drops.add(fixture.kind)

    assert compressor_drops == expected_kinds

    report = (tmp_path / "lcm-benchmark.md").read_text(encoding="utf-8")
    assert "LCM recall/expand evidence" in report
    assert "compressor dropped" in report
    assert "dense-negations" in report
    assert "near-duplicate-numeric-outliers" in report
    assert "long-tool-dump" in report


def test_gate_decisions_cover_go_narrow_go_no_go_and_budget_abort() -> None:
    policy = bench.GatePolicy(
        go_correctness_vs_raw_min=0.98,
        narrow_correctness_vs_raw_min=0.90,
        min_observed_savings_vs_raw=0.20,
    )

    go = bench.decide_gate(
        raw=_stats("raw", correct=100, observed_tokens=100_000),
        compressor=_stats("compressor", correct=70, observed_tokens=35_000),
        lcm=_stats("lcm", correct=100, observed_tokens=40_000),
        policy=policy,
    )
    assert go.verdict == "GO"

    narrow = bench.decide_gate(
        raw=_stats("raw", correct=100, observed_tokens=100_000),
        compressor=_stats("compressor", correct=75, observed_tokens=35_000),
        lcm=_stats("lcm", correct=93, observed_tokens=40_000),
        policy=policy,
    )
    assert narrow.verdict == "NARROW-GO"
    assert any("below GO correctness" in failure for failure in narrow.failures)

    no_go = bench.decide_gate(
        raw=_stats("raw", correct=100, observed_tokens=100_000),
        compressor=_stats("compressor", correct=80, observed_tokens=35_000),
        lcm=_stats("lcm", correct=70, observed_tokens=40_000),
        policy=policy,
    )
    assert no_go.verdict == "NO-GO"
    assert any("below compressor" in failure for failure in no_go.failures)

    budget_abort = bench.decide_gate(
        raw=_stats("raw", correct=0, observed_tokens=0),
        compressor=_stats("compressor", correct=0, observed_tokens=0),
        lcm=_stats("lcm", correct=0, observed_tokens=0),
        policy=policy,
        aborted_reason="budget cap preflight abort: estimated spend exceeds PRD I-11 cap",
    )
    assert budget_abort.verdict == "NO-GO"
    assert any("budget cap" in failure for failure in budget_abort.failures)


def test_budget_preflight_aborts_before_any_arm_runs(tmp_path: Path) -> None:
    out_path = tmp_path / "budget-abort.md"

    run = bench.run_benchmark_battery(
        dry_run=True,
        out_path=out_path,
        budget=bench.BudgetPolicy(
            max_usd=0.000001,
            estimated_input_usd_per_mtok=5_000.0,
            estimated_output_usd_per_mtok=5_000.0,
            max_completion_tokens=4096,
        ),
    )

    assert run.gate.verdict == "NO-GO"
    assert run.records == []
    assert run.aborted_reason is not None
    assert "budget cap preflight abort" in run.aborted_reason
    assert run.spend_events[0].phase == "preflight"
    assert run.spend_events[0].estimated_cost_usd > run.budget.max_usd

    report = out_path.read_text(encoding="utf-8")
    assert "budget cap preflight abort" in report
    assert "GO/NARROW-GO/NO-GO verdict: NO-GO" in report
