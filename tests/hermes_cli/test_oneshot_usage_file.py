"""Unit tests for structured failure facts in the one-shot usage report.

Contract: ``hermes -z --usage-file`` preserves the
structured terminal-failure facts the runtime already produced
(``failure_reason`` / ``failure_retryable``) plus narrow, provider-neutral
discriminators (``failure_status_code``, ``failure_errno``,
``failure_provider_code``) so a downstream caller can tell the observed terminal
provider-failure classes apart without parsing message text. Several distinct
quota walls collapse onto the same ``billing`` + 429 tuple upstream, so the
provider's sanitized error-code token (e.g. ``usage_limit_reached`` vs
``insufficient_quota``) is part of the contract. No message text, provider body,
or prompt content is written to the report.
"""

import json

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


class TestUsageFileFailureFacts:
    """Structured terminal-failure facts survive into the report."""

    def test_failure_reason_and_retryable_are_preserved(self, tmp_path):
        path = tmp_path / "usage.json"
        result = _result(failed=True, completed=False, failure_reason="overloaded", failure_retryable=True)
        _write_usage_file(str(path), result)
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure_reason"] == "overloaded"
        assert report["failure_retryable"] is True

    def test_success_run_has_null_failure_facts_and_no_discriminators(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result())
        report = json.loads(path.read_text())
        assert report["failure_reason"] is None
        assert report["failure_retryable"] is None
        # Discriminators are omitted entirely on a clean run: they exist only
        # to distinguish terminal failures.
        assert "failure_status_code" not in report
        assert "failure_errno" not in report
        assert "failure_provider_code" not in report

    def test_errno_and_status_discriminators_land_only_when_present(self, tmp_path):
        path = tmp_path / "usage.json"
        result = _result(
            failed=True, completed=False,
            failure_reason="timeout", failure_retryable=True,
            failure_status_code=None, failure_errno=32,
        )
        _write_usage_file(str(path), result)
        report = json.loads(path.read_text())
        assert report["failure_errno"] == 32
        # A null status is not a fact; the key is dropped rather than written as null.
        assert "failure_status_code" not in report

    def test_status_discriminator_lands_when_present(self, tmp_path):
        path = tmp_path / "usage.json"
        result = _result(
            failed=True, completed=False,
            failure_reason="billing", failure_retryable=False,
            failure_status_code=429, failure_errno=None,
        )
        _write_usage_file(str(path), result)
        report = json.loads(path.read_text())
        assert report["failure_status_code"] == 429
        assert "failure_errno" not in report

    def test_provider_code_discriminator_lands_when_present(self, tmp_path):
        """The observed usage_limit_reached case vs any other billing+429 quota
        wall: the sanitized code token is what preserves the distinction."""
        path = tmp_path / "usage.json"
        result = _result(
            failed=True, completed=False,
            failure_reason="billing", failure_retryable=False,
            failure_status_code=429, failure_provider_code="usage_limit_reached",
        )
        _write_usage_file(str(path), result)
        report = json.loads(path.read_text())
        assert report["failure_provider_code"] == "usage_limit_reached"
        assert report["failure_status_code"] == 429

    def test_distinct_quota_walls_differ_only_by_provider_code(self, tmp_path):
        """OpenAI insufficient_quota and the observed Anthropic usage_limit_reached
        produce the same reason+status tuple; the report must not collapse them."""
        path = tmp_path / "usage.json"
        observed = _result(
            failed=True, completed=False, failure_reason="billing", failure_retryable=False,
            failure_status_code=429, failure_provider_code="usage_limit_reached",
        )
        other = _result(
            failed=True, completed=False, failure_reason="billing", failure_retryable=False,
            failure_status_code=429, failure_provider_code="insufficient_quota",
        )
        _write_usage_file(str(path), observed)
        report_observed = json.loads(path.read_text())
        path2 = tmp_path / "usage2.json"
        _write_usage_file(str(path2), other)
        report_other = json.loads(path2.read_text())
        assert report_observed["failure_provider_code"] != report_other["failure_provider_code"]

    def test_no_message_or_body_text_leaks_into_failure_fields(self, tmp_path):
        path = tmp_path / "usage.json"
        result = _result(
            failed=True, completed=False,
            failure_reason="billing", failure_retryable=False,
            failure_status_code=429,
            failure_provider_code="usage_limit_reached",
            error="Your account has reached its usage limit. Your plan includes 10000 tokens per day.",
        )
        _write_usage_file(str(path), result)
        report = json.loads(path.read_text())
        serialized = json.dumps(report)
        # The report carries the classifier's reason and the code token, never the
        # provider's message text or body. The ``error`` key of the run result is
        # not copied.
        assert "your plan includes" not in serialized.lower()
        assert "error" not in report
        assert report["failure_provider_code"] == "usage_limit_reached"
