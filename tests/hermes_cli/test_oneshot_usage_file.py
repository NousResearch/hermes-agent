"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import json

import pytest

from hermes_cli.oneshot import _write_usage_file, run_oneshot


def _result(**overrides):
    base = {
        "estimated_cost_usd": 0.1234,
        "cost_status": "estimated",
        "cost_source": "pricing-table",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 800,
        "cache_write_tokens": 0,
        "reasoning_tokens": 50,
        "total_tokens": 1250,
        "api_calls": 3,
        "model": "openai/gpt-5.5",
        "provider": "openrouter",
        "session_id": "abc123",
        "completed": True,
        "failed": False,
    }
    base.update(overrides)
    return base


class TestWriteUsageFile:
    def test_writes_report_with_cost_and_tokens(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result())
        report = json.loads(path.read_text())
        assert report["estimated_cost_usd"] == 0.1234
        assert report["input_tokens"] == 1000
        assert report["output_tokens"] == 200
        assert report["model"] == "openai/gpt-5.5"
        assert report["api_calls"] == 3
        assert report["failed"] is False
        assert "failure" not in report

    def test_none_path_is_noop(self, tmp_path):
        # Must not raise and must not create a report file.
        _write_usage_file(None, _result())
        assert not (tmp_path / "usage.json").exists()

    def test_failure_marks_failed_and_records_message(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), {}, failure="boom")
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure"] == "boom"
        # Missing result fields serialize as null, not KeyError.
        assert report["estimated_cost_usd"] is None


@pytest.mark.parametrize(
    "result",
    [
        {"final_response": "incomplete", "completed": False},
        {"final_response": "partial", "partial": True},
        {"final_response": "interrupted", "interrupted": True},
    ],
)
def test_oneshot_returns_usage_error_for_non_terminal_success_results(monkeypatch, result):
    """-z must fail even when an incomplete turn produced visible text."""
    monkeypatch.setattr("hermes_cli.oneshot._run_agent", lambda *_args, **_kwargs: (result["final_response"], result))

    assert run_oneshot("hello") == 2

