"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import io
import json
import sys

import pytest

import hermes_cli.oneshot as oneshot
from hermes_cli.oneshot import _write_usage_file


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

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "usage.json"
        _write_usage_file(str(path), _result())
        assert json.loads(path.read_text())["total_tokens"] == 1250

    def test_unwritable_path_never_raises(self):
        # Root-owned path — the write must be swallowed, not raised.
        _write_usage_file("/proc/definitely/not/writable/usage.json", _result())

    def test_result_failed_flag_carries_through(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result(failed=True))
        assert json.loads(path.read_text())["failed"] is True


@pytest.mark.parametrize(
    ("run_failure", "expected"),
    [
        (None, 0),
        (RuntimeError("private run detail"), 1),
        (KeyboardInterrupt(), KeyboardInterrupt),
        (SystemExit(23), SystemExit),
    ],
)
def test_oneshot_usage_is_written_for_normal_success_error_and_interruption(
    monkeypatch,
    tmp_path,
    run_failure,
    expected,
):
    usage_path = tmp_path / "usage.json"
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    result = _result()

    def run(*_args, delivery, **_kwargs):
        if run_failure is None:
            delivery.deliver("final")
            return "final", result
        raise run_failure

    monkeypatch.setattr(oneshot, "_run_agent", run)
    monkeypatch.setattr(oneshot._OneshotResources, "close", lambda *_a, **_kw: None)
    if isinstance(expected, type) and issubclass(expected, BaseException):
        with pytest.raises(expected) as raised:
            oneshot.run_oneshot("private prompt", usage_file=str(usage_path))
        assert raised.value is run_failure
    else:
        assert oneshot.run_oneshot("private prompt", usage_file=str(usage_path)) == expected

    report = json.loads(usage_path.read_text())
    if run_failure is None:
        assert report["failed"] is False
        assert report["estimated_cost_usd"] == result["estimated_cost_usd"]
        assert report["total_tokens"] == result["total_tokens"]
    else:
        assert report["failed"] is True


def test_cleanup_failure_does_not_overwrite_primary_usage_result(monkeypatch, tmp_path):
    usage_path = tmp_path / "usage.json"
    result = _result(estimated_cost_usd=9.75, total_tokens=9876)

    class CleanupFailureResources:
        def __init__(self):
            self.cleanup_failed = False

        def mark_final_delivered(self):
            return None

        def close(self, primary_failure=None):
            self.cleanup_failed = True

    monkeypatch.setattr(oneshot, "_OneshotResources", CleanupFailureResources)
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, delivery, **_kwargs: (
            delivery.deliver("final") and "final",
            result,
        ),
    )

    assert oneshot.run_oneshot("private prompt", usage_file=str(usage_path)) == 1
    report = json.loads(usage_path.read_text())
    assert report["failed"] is True
    assert report["estimated_cost_usd"] == 9.75
    assert report["total_tokens"] == 9876
    assert report["session_id"] == result["session_id"]
