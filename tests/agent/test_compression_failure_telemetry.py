"""Compression telemetry keeps enough safe detail to diagnose a failed summary call."""

import json
import logging
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import _emit_compression_attempt_telemetry


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "session-telemetry-test"


def _extract_telemetry(caplog):
    records = [
        record.getMessage()
        for record in caplog.records
        if "context compression attempt telemetry:" in record.getMessage()
    ]
    assert len(records) == 1
    return json.loads(records[0].split("context compression attempt telemetry: ", 1)[1])


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        return ContextCompressor(
            model="gpt-6.1-sol",
            provider="openai-codex",
            threshold_percent=0.85,
            quiet_mode=True,
            config_context_length=272_000,
        )


def test_successful_attempt_records_result_estimate(caplog):
    compressor = _compressor()
    compressor._begin_compression_telemetry(current_tokens=231_200, session_id="session-success")

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        _emit_compression_attempt_telemetry(
            _Agent(compressor),
            started_at=0.0,
            commit_status="committed",
            split_status="not_applicable",
            result_estimated_tokens=18_432,
        )

    payload = _extract_telemetry(caplog)
    assert payload["main_model"] == "gpt-6.1-sol"
    assert payload["main_context_limit"] == 272_000
    assert payload["effective_threshold"] == 231_200
    assert payload["result_estimated_tokens"] == 18_432


def test_failed_attempt_records_redacted_auxiliary_diagnostics(caplog):
    compressor = _compressor()
    compressor._begin_compression_telemetry(current_tokens=231_200, session_id="session-failure")
    compressor._active_compression_telemetry["aux_model"] = "broken-aux-model"
    compressor._active_compression_telemetry["aux_provider"] = "openai-codex"
    error = Exception(
        "HTTP 504 provider rejected OPENAI_API_KEY=sk-proj-" + "X" * 40
        + " https://localhost/callback?access_token=opaque-token"
    )
    error.status_code = 504
    # Model this as the second failed call after the existing aux→main retry. The
    # retry guard keeps the test on the real failure-classification path without
    # making a provider call.
    compressor._summary_model_fallen_back = True
    compressor._on_summary_failure(error, [], None, "")

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        _emit_compression_attempt_telemetry(
            _Agent(compressor),
            started_at=0.0,
            commit_status="aborted",
            split_status="aborted",
            failure_class="summary_timeout",
        )

    payload = _extract_telemetry(caplog)
    assert payload["failed_aux_provider"] == "openai-codex"
    assert payload["failed_aux_model"] == "broken-aux-model"
    assert payload["failure_error_type"] == "Exception"
    assert payload["failure_status_code"] == 504
    assert payload["failure_reason"] == "timed out"
    assert "sk-proj-" not in payload["failure_reason"]
    assert "access_token=opaque-token" not in payload["failure_reason"]
    assert "session-failure" in json.dumps(payload)
    assert "opaque-token" not in json.dumps(payload)


def test_main_route_does_not_masquerade_as_auxiliary_failure():
    compressor = _compressor()
    compressor._begin_compression_telemetry(current_tokens=231_200, session_id="session-main-failure")
    compressor._active_compression_telemetry["aux_model"] = compressor.model
    compressor._active_compression_telemetry["aux_provider"] = compressor.provider
    compressor._record_summary_failure_telemetry(Exception("provider failed"))

    assert compressor._active_compression_telemetry["failed_aux_provider"] is None
    assert compressor._active_compression_telemetry["failed_aux_model"] is None
    assert compressor._active_compression_telemetry["failure_reason"] == "failed"
