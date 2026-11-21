"""Hermetic behavior tests for volume helpers and the disposable catalog detector."""

import csv
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


@pytest.fixture
def fixture_app(tmp_path, monkeypatch):
    root = Path(__file__).parent / "fixtures" / "volume_test_app"
    monkeypatch.syspath_prepend(str(root))
    import catalog
    import data
    import run_demo

    app = catalog.Catalog(tmp_path / "catalog.sqlite")

    def call(path, payload=None):
        if payload is not None:
            result = app.prepare(payload["run"], payload["rows"]) if path == "/prepare" else app.cleanup(payload["run"])
        else:
            parsed = urlparse(path)
            args = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            run = args.pop("run")
            if parsed.path == "/counts":
                result = app.counts(run)
            else:
                result = app.query(run, page=int(args.pop("page", 1)), page_size=int(args.pop("page_size", 50)), export=parsed.path == "/export", **args)
                if parsed.path == "/export":
                    return catalog.csv_text(result["rows"])
        return json.dumps(result)

    plan = {"run": "volume-test", "seed": 42, "volumes": [10, 51], "samples": 10,
            "duration_seconds": 30, "max_records": 10000, "max_errors": 50}
    return app, data, run_demo, call, plan


def test_reproducible_prefix_and_absolute_targets(fixture_app, tmp_path):
    app, data, runner, call, plan = fixture_app
    assert [data.record(42, i) for i in range(10)] == [data.record(42, i) for i in range(51)][:10]
    assert data.record(41, 9) != data.record(42, 9)
    manifest = {}
    runner.prepare(call, plan["run"], 10, 42, runner.Budget(30), manifest, lambda: None, batch_size=3)
    runner.prepare(call, plan["run"], 51, 42, runner.Budget(30), manifest, lambda: None, batch_size=7)
    runner.prepare(call, plan["run"], 51, 42, runner.Budget(30), manifest, lambda: None)
    assert app.counts(plan["run"]) == {"products": 51, "categories": 8, "orphaned": 0, "sentinel_intact": True}
    all_rows = data.reference([data.record(42, i) for i in range(51)])
    assert runner.check_pages(call, plan["run"], all_rows, {}, runner.Budget(30))["correct"]
    app.cleanup(plan["run"])
    assert app.counts(plan["run"])["sentinel_intact"]
    with pytest.raises(ValueError):
        app.cleanup("sentinel")


def test_ambiguous_partial_batch_reconciles_then_resumes(fixture_app):
    app, data, runner, call, plan = fixture_app
    manifest = {}

    def partial(path, payload=None):
        if path == "/prepare":
            app.prepare(plan["run"], payload["rows"][:2])
            raise TimeoutError("Simulated ambiguous write")
        return call(path, payload)

    with pytest.raises(TimeoutError):
        runner.prepare(partial, plan["run"], 10, 42, runner.Budget(30), manifest, lambda: None)
    assert manifest["reconciled_ids"] == [0, 1]
    assert manifest["confirmed_products"] == 2
    runner.prepare(call, plan["run"], 10, 42, runner.Budget(30), manifest, lambda: None)
    assert app.counts(plan["run"])["products"] == 10
    conflicting = data.record(42, 0)
    conflicting["price"] += 1
    with pytest.raises(ValueError):
        app.prepare(plan["run"], [conflicting])


def test_metrics_include_timeouts_and_failed_samples(fixture_app):
    _, data, runner, _, plan = fixture_app
    operation = next(runner.operations([data.record(42, i) for i in range(10)]))

    def timeout(*_args):
        raise TimeoutError("Request expired")

    sample = runner.measure(timeout, plan["run"], 10, operation, "warm")
    assert sample["timeout"] and sample["error"] and sample["duration_ms"] >= 0
    metrics = data.SUPPORT.summarize([sample, {"duration_ms": 4, "correct": False}, {"duration_ms": 8, "correct": True}])
    assert metrics["samples"] == 3 and metrics["timed_samples"] == 3
    assert metrics["errors"] == metrics["timeouts"] == metrics["incorrect"] == 1
    assert data.SUPPORT.summarize([])["median_ms"] is None
    with pytest.raises(ValueError):
        data.SUPPORT.summarize([{"duration_ms": float("nan")}])


def test_oracle_detects_missing_duplicate_order_and_content(fixture_app):
    _, data, _, _, _ = fixture_app
    expected = data.reference([data.record(42, i) for i in range(10)])
    for actual in (expected[:-1], expected + expected[:1], list(reversed(expected))):
        assert not data.SUPPORT.compare_rows(actual, expected, data.FIELDS)["correct"]
    changed = [dict(row) for row in expected]
    changed[0]["price"] += 1
    assert not data.SUPPORT.compare_rows(changed, expected, data.FIELDS)["correct"]


def test_controlled_export_fault_is_detected_without_red_suite(fixture_app):
    app, data, runner, call, plan = fixture_app
    runner.prepare(call, plan["run"], 1000, 42, runner.Budget(30), {}, lambda: None)
    rows = [data.record(42, i) for i in range(1000)]
    operation = next(op for op in runner.operations(rows) if op[0] == "export")
    assert runner.measure(call, plan["run"], 1000, operation, "warm")["correct"]
    app.injected_export_bug = True
    detected = runner.measure(call, plan["run"], 1000, operation, "warm")
    assert not detected["correct"] and detected["expected_count"] == 1000 and detected["actual_count"] == 50
    assert detected["missing_count"] == 950


def test_complete_small_run_has_raw_samples_and_verified_cleanup(fixture_app, tmp_path):
    app, _, runner, call, plan = fixture_app
    result = runner.execute(call, plan, tmp_path / "output")
    assert result["status"] == "completed" and result["cleanup"]["verified"]
    assert [s["products"] for s in result["stages"]] == [10, 51]
    assert len(result["samples"]) == 2 * 7 * 12
    assert all(s["correct"] for s in result["samples"])
    assert json.loads((tmp_path / "output" / "manifest.json").read_text())["confirmed_products"] == 51
    assert app.counts(plan["run"])["products"] == 0


def test_interruption_preserves_partial_report_and_cleans(fixture_app, tmp_path, monkeypatch):
    app, _, runner, call, plan = fixture_app
    original = runner.measure
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(*args)

    monkeypatch.setattr(runner, "measure", interrupted)
    result = runner.execute(call, plan, tmp_path / "output")
    assert result["status"] == "interrupted" and len(result["samples"]) == 1
    assert result["cleanup"]["verified"] and app.counts(plan["run"])["sentinel_intact"]
    report = (tmp_path / "output" / "report.md").read_text()
    assert "unrun" in report and "KeyboardInterrupt" in report


def test_duration_limit_and_cleanup_failure_are_explicit(fixture_app, tmp_path):
    app, _, runner, call, plan = fixture_app

    class Expired:
        def check(self):
            raise TimeoutError("Duration limit")

    result = runner.execute(call, plan, tmp_path / "expired", budget=Expired())
    assert not result["samples"] and not result["stages"] and result["cleanup"]["verified"]

    def failed_cleanup(path, payload=None):
        if path == "/cleanup":
            raise OSError("Cleanup unavailable")
        return call(path, payload)

    app.prepare(plan["run"], [{"id": 0, "name": "Synthetic residual", "price": 1, "category": 0}])
    result = runner.execute(failed_cleanup, plan, tmp_path / "cleanup", budget=Expired())
    assert not result["cleanup"]["verified"] and "Cleanup unavailable" in result["cleanup"]["error"]
    assert result["cleanup"]["residuals"]["products"] == 1
    assert app.counts(plan["run"])["sentinel_intact"]


def test_filtered_csv_matches_independent_content(fixture_app):
    _, data, runner, call, plan = fixture_app
    runner.prepare(call, plan["run"], 101, 42, runner.Budget(30), {}, lambda: None)
    exported = list(csv.DictReader(io.StringIO(call("/export?run=volume-test&category=0&sort=price"))))
    expected = data.reference([data.record(42, i) for i in range(101)], category="0", sort="price")
    assert data.SUPPORT.compare_rows(exported, expected, data.FIELDS)["correct"]
