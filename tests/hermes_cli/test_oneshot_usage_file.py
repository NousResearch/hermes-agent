"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import json

import pytest

from hermes_cli.oneshot import _run_agent, _write_usage_file


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


@pytest.mark.parametrize(
    ("global_value", "platform_value", "expected"),
    [(False, True, True), (True, False, False)],
)
def test_oneshot_agent_honors_cli_platform_activity_override(
    monkeypatch, global_value, platform_value, expected
):
    config = {
        "model": {"default": "test-model", "provider": "test"},
        "display": {
            "tool_reasons": global_value,
            "tool_result_summaries": global_value,
            "platforms": {
                "cli": {
                    "tool_reasons": platform_value,
                    "tool_result_summaries": platform_value,
                }
            },
        },
    }
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = []

        def run_conversation(self, _prompt):
            return {"final_response": "done"}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "provider": "test",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.oneshot._create_session_db_for_oneshot", lambda: None)

    response, _result_data = _run_agent("hello", use_config_toolsets=False)

    assert response == "done"
    assert captured["tool_reasons_enabled"] is expected
    assert captured["tool_result_summaries_enabled"] is expected


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
