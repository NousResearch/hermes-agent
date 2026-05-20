"""Tests for the synthetic tool result compaction benchmark."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark_tool_result_compaction import run_benchmark


def test_benchmark_reports_expected_compaction_metrics():
    result = run_benchmark(
        tool_results=3,
        result_chars=20_000,
        threshold_tokens=1_000,
        preview_chars=200,
        max_disk_mb=500,
        keep_dir=False,
    )

    assert result["tool_results"] == 3
    assert result["result_chars_each"] == 20_000
    assert result["compacted_count"] == 3
    assert result["raw_file_count"] == 3
    assert result["before_tokens_estimate"] > result["after_tokens_estimate"]
    assert result["saved_percent_estimate"] > 0
    assert result["sample_raw_path"] is not None
    assert not Path(result["raw_result_dir"]).exists()


def test_benchmark_keep_dir_preserves_raw_result_dir():
    result = run_benchmark(
        tool_results=1,
        result_chars=20_000,
        threshold_tokens=1_000,
        preview_chars=200,
        max_disk_mb=500,
        keep_dir=True,
    )

    raw_result_dir = Path(result["raw_result_dir"])
    try:
        assert raw_result_dir.exists()
        assert result["raw_file_count"] == 1
        assert result["sample_raw_path"] is not None
        assert Path(result["sample_raw_path"]).exists()
    finally:
        for path in sorted(raw_result_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        raw_result_dir.rmdir()
