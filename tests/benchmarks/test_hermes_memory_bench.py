import json
import subprocess
import sys

from benchmarks.hermes_memory_bench.core import DIMENSIONS, POLICY, run_benchmark


REQUIRED_TOP_LEVEL_KEYS = {
    "benchmark_type",
    "generated_at",
    "suite",
    "scores",
    "cases",
    "aggregate",
    "policy",
}


def test_smoke_benchmark_schema_and_policy():
    report = run_benchmark("smoke")

    assert REQUIRED_TOP_LEVEL_KEYS <= report.keys()
    assert report["benchmark_type"] == "hermes_memory_bench_v0.1"
    assert report["suite"] == "smoke"
    assert set(DIMENSIONS) <= report["scores"].keys()
    assert report["aggregate"]["overall_score"] == 1.0
    assert report["aggregate"]["case_count"] >= 6
    assert report["policy"] == POLICY

    for case in report["cases"]:
        assert {"id", "dimension", "query", "expected_answer", "actual_answer", "score", "latency_ms", "passed", "evidence"} <= case.keys()
        assert case["dimension"] in DIMENSIONS


def test_smoke_benchmark_cli_writes_report(tmp_path):
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.hermes_memory_bench.run",
            "--suite",
            "smoke",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate"]["overall_score"] == 1.0
    assert report["policy"] == POLICY


def test_benchmark_does_not_write_graph_or_operation_ledger(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    report = run_benchmark("smoke")

    assert report["policy"]["read_only"] is True
    assert report["policy"]["would_write_memory"] is False
    assert report["policy"]["would_modify_config"] is False
    assert report["policy"]["would_write_graph"] is False
    assert report["policy"]["does_not_create_operation_events"] is True
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()
