"""Tests for eval storage and reporting."""

import time
import pytest

from agent.evals.types import CaseResult, CaseStatus, RunSummary
from agent.evals.storage import EvalStore
from agent.evals.reporting import (
    format_run_summary,
    format_recent_runs,
    format_run_detail,
    format_run_row,
)


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_eval.db"
    s = EvalStore(db_path=db_path)
    yield s
    s.close()


def _make_summary(run_id="run-001", suite="smoke", passed=4, failed=2) -> RunSummary:
    results = []
    for i in range(passed):
        results.append(CaseResult(
            run_id=run_id,
            case_id=f"case-pass-{i}",
            category="file_workspace",
            status=CaseStatus.PASSED,
            deterministic_score=1.0,
            total_score=1.0,
            duration_ms=100 + i,
        ))
    for i in range(failed):
        results.append(CaseResult(
            run_id=run_id,
            case_id=f"case-fail-{i}",
            category="reliability",
            status=CaseStatus.FAILED,
            deterministic_score=0.0,
            total_score=0.0,
            duration_ms=200 + i,
            failure_summary=f"check failed for case {i}",
        ))
    total = passed + failed
    avg = passed / total if total else 0.0
    return RunSummary(
        run_id=run_id,
        suite_name=suite,
        label="test-label",
        case_count=total,
        passed_count=passed,
        failed_count=failed,
        avg_score=round(avg, 4),
        case_results=results,
    )


class TestEvalStore:
    """Tests for storage persistence and retrieval."""

    def test_save_and_list(self, store):
        summary = _make_summary()
        store.save_run(summary)
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == "run-001"
        assert runs[0]["suite_name"] == "smoke"
        assert runs[0]["passed_count"] == 4
        assert runs[0]["failed_count"] == 2

    def test_get_run(self, store):
        store.save_run(_make_summary())
        run = store.get_run("run-001")
        assert run is not None
        assert run["id"] == "run-001"

    def test_get_run_missing(self, store):
        assert store.get_run("nonexistent") is None

    def test_get_case_results(self, store):
        store.save_run(_make_summary())
        results = store.get_case_results("run-001")
        assert len(results) == 6
        passed = [r for r in results if r["status"] == "passed"]
        failed = [r for r in results if r["status"] == "failed"]
        assert len(passed) == 4
        assert len(failed) == 2

    def test_get_run_with_results(self, store):
        store.save_run(_make_summary())
        full = store.get_run_with_results("run-001")
        assert full is not None
        assert "case_results" in full
        assert len(full["case_results"]) == 6

    def test_multiple_runs_ordering(self, store):
        s1 = _make_summary("run-a", passed=3, failed=0)
        s1.created_at = time.time() - 100
        store.save_run(s1)

        s2 = _make_summary("run-b", passed=2, failed=1)
        s2.created_at = time.time()
        store.save_run(s2)

        runs = store.list_runs()
        assert len(runs) == 2
        assert runs[0]["id"] == "run-b"  # newest first
        assert runs[1]["id"] == "run-a"

    def test_list_runs_limit(self, store):
        for i in range(5):
            s = _make_summary(f"run-{i}", passed=1, failed=0)
            s.created_at = time.time() + i
            store.save_run(s)
        runs = store.list_runs(limit=3)
        assert len(runs) == 3


class TestReporting:
    """Tests for reporting formatters."""

    def test_format_run_summary(self):
        summary = _make_summary()
        text = format_run_summary(summary)
        assert "run-001" in text
        assert "smoke" in text
        assert "passed: 4" in text
        assert "failed: 2" in text
        assert "✓" in text
        assert "✗" in text

    def test_format_run_row(self):
        run = {
            "id": "abc123",
            "created_at": time.time(),
            "suite_name": "smoke",
            "passed_count": 5,
            "case_count": 6,
            "avg_score": 0.83,
            "label": "v1",
        }
        text = format_run_row(run)
        assert "abc123" in text
        assert "smoke" in text
        assert "5/6" in text
        assert "[v1]" in text

    def test_format_recent_runs_empty(self):
        text = format_recent_runs([])
        assert "No eval runs" in text

    def test_format_recent_runs(self):
        runs = [
            {"id": "r1", "created_at": time.time(), "suite_name": "smoke",
             "passed_count": 6, "case_count": 6, "avg_score": 1.0, "label": ""},
        ]
        text = format_recent_runs(runs)
        assert "r1" in text
        assert "6/6" in text

    def test_format_run_detail(self):
        run = {
            "id": "r1",
            "created_at": time.time(),
            "suite_name": "smoke",
            "case_count": 2,
            "passed_count": 1,
            "failed_count": 1,
            "avg_score": 0.5,
            "label": "test",
            "case_results": [
                {"case_id": "c1", "status": "passed", "total_score": 1.0,
                 "duration_ms": 50, "failure_summary": ""},
                {"case_id": "c2", "status": "failed", "total_score": 0.0,
                 "duration_ms": 80, "failure_summary": "file not found"},
            ],
        }
        text = format_run_detail(run)
        assert "r1" in text
        assert "✓ c1" in text
        assert "✗ c2" in text
        assert "file not found" in text
