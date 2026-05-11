"""Tests for failure storage, aggregation, and eval ingestion."""

from __future__ import annotations

import time
import pytest

from agent.failure_analysis.types import NormalizedFailure
from agent.failure_analysis.storage import FailureStore
from agent.failure_analysis.aggregator import (
    ingest_eval_failures, get_top_failures, get_failure_summary,
)
from agent.failure_analysis.reporting import (
    format_recent_failures, format_top_failures,
    format_fingerprint_detail, format_failure_summary,
)


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_failures.db"
    s = FailureStore(db_path=db_path)
    yield s
    s.close()


def _make_failure(
    id: str = "f-001",
    failure_type: str = "eval",
    failure_subtype: str = "failed_check",
    severity: str = "medium",
    fingerprint: str = "fp-aaa",
    summary: str = "check failed",
    source_surface: str = "eval",
    eval_run_id: str | None = "run-001",
    **kwargs,
) -> NormalizedFailure:
    return NormalizedFailure(
        id=id,
        failure_type=failure_type,
        failure_subtype=failure_subtype,
        severity=severity,
        fingerprint=fingerprint,
        summary=summary,
        source_surface=source_surface,
        eval_run_id=eval_run_id,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestFailureStore:
    def test_insert_and_list(self, store):
        store.insert(_make_failure())
        rows = store.list_recent()
        assert len(rows) == 1
        assert rows[0]["id"] == "f-001"
        assert rows[0]["failure_type"] == "eval"

    def test_insert_many(self, store):
        failures = [
            _make_failure(id="f-001"),
            _make_failure(id="f-002"),
            _make_failure(id="f-003"),
        ]
        count = store.insert_many(failures)
        assert count == 3
        assert len(store.list_recent()) == 3

    def test_insert_duplicate_ignored(self, store):
        store.insert(_make_failure(id="f-dup"))
        store.insert(_make_failure(id="f-dup"))  # OR IGNORE
        assert len(store.list_recent()) == 1

    def test_list_recent_ordering(self, store):
        f1 = _make_failure(id="f-old")
        f1.created_at = time.time() - 100
        store.insert(f1)

        f2 = _make_failure(id="f-new")
        f2.created_at = time.time()
        store.insert(f2)

        rows = store.list_recent()
        assert rows[0]["id"] == "f-new"

    def test_get_by_fingerprint(self, store):
        store.insert(_make_failure(id="f-a", fingerprint="fp-xyz"))
        store.insert(_make_failure(id="f-b", fingerprint="fp-xyz"))
        store.insert(_make_failure(id="f-c", fingerprint="fp-other"))

        rows = store.get_by_fingerprint("fp-xyz")
        assert len(rows) == 2

    def test_get_by_eval_run(self, store):
        store.insert(_make_failure(id="f-a", eval_run_id="run-A"))
        store.insert(_make_failure(id="f-b", eval_run_id="run-A"))
        store.insert(_make_failure(id="f-c", eval_run_id="run-B"))

        rows = store.get_by_eval_run("run-A")
        assert len(rows) == 2

    def test_top_fingerprints(self, store):
        for i in range(5):
            store.insert(_make_failure(id=f"f-a{i}", fingerprint="fp-frequent"))
        for i in range(2):
            store.insert(_make_failure(id=f"f-b{i}", fingerprint="fp-rare"))

        top = store.top_fingerprints(limit=5)
        assert len(top) == 2
        assert top[0]["fingerprint"] == "fp-frequent"
        assert top[0]["count"] == 5

    def test_top_fingerprints_respects_window(self, store):
        old = _make_failure(id="f-old", fingerprint="fp-old")
        old.created_at = time.time() - 86400 * 30  # 30 days ago
        store.insert(old)

        store.insert(_make_failure(id="f-new", fingerprint="fp-new"))

        top = store.top_fingerprints(window_seconds=86400 * 7, limit=10)
        fps = [t["fingerprint"] for t in top]
        assert "fp-new" in fps
        assert "fp-old" not in fps

    def test_count_total(self, store):
        assert store.count_total() == 0
        store.insert(_make_failure(id="f-1"))
        store.insert(_make_failure(id="f-2"))
        assert store.count_total() == 2

    def test_count_total_windowed(self, store):
        old = _make_failure(id="f-old")
        old.created_at = time.time() - 86400 * 30
        store.insert(old)
        store.insert(_make_failure(id="f-new"))
        assert store.count_total(window_seconds=86400) == 1


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestIngestEvalFailures:
    def test_ingest_skips_passed(self, store):
        cases = [
            {"case_id": "c1", "status": "passed", "failure_summary": ""},
            {"case_id": "c2", "status": "failed", "failure_summary": "check failed"},
        ]
        failures = ingest_eval_failures("run-x", cases, store=store)
        assert len(failures) == 1
        assert failures[0].case_id == "c2"
        assert len(store.list_recent()) == 1

    def test_ingest_detects_regression(self, store):
        cases = [
            {"case_id": "c1", "status": "failed", "failure_summary": "score dropped"},
        ]
        prior = {"c1": "passed"}
        failures = ingest_eval_failures("run-y", cases, store=store,
                                        prior_results=prior)
        assert len(failures) == 1
        assert failures[0].failure_subtype == "regression"

    def test_ingest_without_store(self):
        cases = [
            {"case_id": "c1", "status": "error", "failure_summary": "timeout occurred"},
        ]
        failures = ingest_eval_failures("run-z", cases, store=None)
        assert len(failures) == 1
        # No store — shouldn't crash


# ---------------------------------------------------------------------------
# Aggregation helper tests
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_get_top_failures(self, store):
        for i in range(3):
            store.insert(_make_failure(id=f"f-{i}", fingerprint="fp-top"))
        top = get_top_failures(store, window_days=7, limit=5)
        assert len(top) == 1
        assert top[0]["count"] == 3

    def test_get_failure_summary(self, store):
        store.insert(_make_failure(id="f-1"))
        summary = get_failure_summary(store)
        assert summary["total_all"] == 1
        assert isinstance(summary["top_patterns"], list)


# ---------------------------------------------------------------------------
# Reporting formatter tests
# ---------------------------------------------------------------------------

class TestReporting:
    def test_format_recent_empty(self):
        assert "No failures" in format_recent_failures([])

    def test_format_recent(self):
        rows = [
            {"severity": "high", "created_at": time.time(),
             "failure_type": "eval", "failure_subtype": "regression",
             "summary": "score dropped", "fingerprint": "abcdef1234567890"},
        ]
        text = format_recent_failures(rows)
        assert "eval.regression" in text
        assert "abcdef12" in text

    def test_format_top_empty(self):
        assert "No failure patterns" in format_top_failures([])

    def test_format_top(self):
        patterns = [
            {"count": 5, "failure_type": "tool", "failure_subtype": "timeout",
             "fingerprint": "fp-aaa", "summary": "timed out",
             "latest_at": time.time(), "tool_name": "terminal"},
        ]
        text = format_top_failures(patterns)
        assert "tool.timeout" in text
        assert "5x" in text

    def test_format_fingerprint_detail_empty(self):
        text = format_fingerprint_detail("fp-missing", [])
        assert "No failures found" in text

    def test_format_fingerprint_detail(self):
        occs = [
            {"failure_type": "eval", "failure_subtype": "failed_check",
             "severity": "medium", "created_at": time.time(),
             "summary": "check failed", "tool_name": None, "model": None,
             "source_surface": "eval", "eval_run_id": "run-001",
             "session_id": None},
        ]
        text = format_fingerprint_detail("fp-abc", occs)
        assert "fp-abc" in text
        assert "Occurrences: 1" in text

    def test_format_failure_summary(self):
        summary = {
            "total_24h": 3,
            "total_7d": 10,
            "total_all": 25,
            "top_patterns": [
                {"count": 5, "failure_type": "eval",
                 "failure_subtype": "regression", "summary": "score dropped"},
            ],
        }
        text = format_failure_summary(summary)
        assert "3 failures" in text
        assert "10 failures" in text
        assert "eval.regression" in text
