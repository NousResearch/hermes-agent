"""Offline contracts for swarm benchmark aggregation."""
import importlib.util
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2] / "optional-skills/autonomous-ai-agents/swarm-benchmark-analysis"
SPEC = importlib.util.spec_from_file_location("swarm_benchmark_analysis", SKILL / "scripts/summarize_runs.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SwarmBenchmarkAnalysisTests(unittest.TestCase):
    def test_example_reports_summary_and_paired_delta(self):
        report = MODULE.summarize(MODULE.read_runs(SKILL / "examples/runs.jsonl"), baseline="flat")
        self.assertEqual(report["benchmark_id"], "graph-synthetic-v1")
        self.assertEqual(report["variants"]["flat"]["acceptance_rate"], 0.5)
        self.assertEqual(report["variants"]["flat"]["cost_microusd_total"], 800)
        self.assertEqual(report["paired_comparisons"]["hierarchical"]["paired_cases"], 2)
        self.assertEqual(report["paired_comparisons"]["hierarchical"]["acceptance_delta"], 0.5)

    def test_unknown_cost_is_not_reported_as_zero(self):
        rows = [{"benchmark_id": "b", "case_id": "c", "variant": "v",
                 "accepted": True, "tokens": 10, "latency_ms": 12}]
        report = MODULE.summarize(rows)
        self.assertIsNone(report["variants"]["v"]["cost_microusd_total"])
        self.assertIsNone(report["variants"]["v"]["cost_microusd_per_accepted"])

    def test_bad_cost_provenance_and_duplicate_pairs_fail_closed(self):
        row = {"benchmark_id": "b", "case_id": "c", "variant": "v",
               "accepted": True, "tokens": 10, "latency_ms": 12,
               "cost_microusd": 2}
        with self.assertRaisesRegex(MODULE.BenchmarkDataError, "cost_source"):
            MODULE.summarize([row])
        valid = {"benchmark_id": "b", "case_id": "c", "variant": "v",
                 "accepted": True, "tokens": 10, "latency_ms": 12}
        with self.assertRaisesRegex(MODULE.BenchmarkDataError, "duplicate"):
            MODULE.summarize([valid, dict(valid)])


if __name__ == "__main__":
    unittest.main()
