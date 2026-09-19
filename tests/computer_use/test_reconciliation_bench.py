"""Reconciliation overhead bench (#112734 §D / Phase 1A exit gate).

The exit gate demands reconciliation overhead materially below the work it could
eventually replace. The bench times the shadow path per capture, splits the
shared build substrate from reconciliation's added cost (diff + delta
serialize), aggregates p50/p95, judges the gate against the real-rig capture
round-trip baseline, and persists the report as JSON so cost trends accumulate
across runs.
"""

from tools.computer_use.reconciliation_bench import (
    aggregate as bench_aggregate,
    bench_all_fixtures, load_report, save_report, scaled_list_fixture,
    time_transition,
)
from tools.computer_use.semantic_fixtures import all_fixtures, form_fixture


def test_reconciliation_overhead_materially_below_replaced_work():
    # Phase 1A exit gate: reconciliation's added cost (diff + delta serialize)
    # p95 must stay under half the replaced-work baseline — the real-rig
    # fresh-capture round-trip (~28ms). Fixtures measure fractions of a ms.
    report = bench_all_fixtures(all_fixtures(), repeats=3)
    assert report.samples == 8 * 3  # 7 fixture transitions + the 200-element stress tree
    assert report.replaced_work_baseline_ms == 28.0
    assert report.extra_p95_ms < 0.5 * report.replaced_work_baseline_ms
    assert report.gate_passes, report.render()


def test_bench_aggregates_per_transition_percentiles():
    samples = []
    for fx in all_fixtures():
        for t in fx.transitions:
            samples.extend(time_transition(fx, t, repeats=2))
    report = bench_aggregate(samples)
    assert report.samples == 7 * 2
    assert report.extra_p50_ms > 0
    assert len(report.per_transition) == 7
    for key, summary in report.per_transition.items():
        assert summary["extra_p50_ms"] > 0
        assert summary["reconcile_p50_ms"] > 0


def test_scaled_list_fixture_stresses_overhead_at_size():
    fx = scaled_list_fixture(200)
    samples = time_transition(fx, fx.transitions[0], repeats=3)
    report = bench_aggregate(samples)
    assert report.gate_passes, report.render()  # overhead stays flat as trees grow


def test_reconciliation_bench_persists_and_reloads(tmp_path):
    report = bench_all_fixtures([form_fixture()], repeats=2, include_scaled=False)
    path = save_report(report, tmp_path / "bench" / "report.json")
    assert path.exists()
    reloaded = load_report(path)
    assert reloaded.samples == report.samples
    assert reloaded.extra_p95_ms == report.extra_p95_ms
    assert reloaded.gate_passes == report.gate_passes


def test_bench_render_carries_exit_gate_verdict():
    report = bench_all_fixtures([form_fixture()], repeats=2, include_scaled=False)
    text = report.render()
    assert "replaced-work baseline" in text
    assert "gate:" in text and "PASS" in text
