"""Compression telemetry keeps enough safe detail to diagnose a failed summary call."""

import json
import logging
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.conversation_compression_telemetry import _emit_compression_attempt_telemetry


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "session-telemetry-test"
        self._compression_attempt_id = compressor._active_compression_telemetry["attempt_id"]


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
    error = Exception("provider failed")
    error.status_code = 400
    compressor._record_summary_failure_telemetry(error)
    # A later auxiliary failure must not fill the first main failure's empty identity.
    compressor._active_compression_telemetry.update(aux_model="later-aux-model", aux_provider="later-aux-provider")
    retry_kind = compressor._record_summary_failure_telemetry(TimeoutError("timed out"))

    assert compressor._active_compression_telemetry["failed_aux_provider"] is None
    assert compressor._active_compression_telemetry["failed_aux_model"] is None
    assert compressor._active_compression_telemetry["failure_reason"] == "failed"
    assert compressor._active_compression_telemetry["failure_error_type"] == "Exception"
    assert compressor._active_compression_telemetry["failure_status_code"] == 400
    assert retry_kind.timeout is True


def test_auxiliary_timeout_does_not_borrow_the_main_retry_status():
    compressor = _compressor()
    telemetry = compressor._begin_compression_telemetry(current_tokens=231_200, session_id="session-retry-failure")
    telemetry.update(aux_model="first-aux-model", aux_provider="first-aux-provider")
    compressor._record_summary_failure_telemetry(TimeoutError("timed out"))

    telemetry.update(aux_model=compressor.model, aux_provider=compressor.provider)
    retry_error = Exception("request rejected")
    retry_error.status_code = 400
    retry_kind = compressor._record_summary_failure_telemetry(retry_error)

    assert telemetry["failed_aux_provider"] == "first-aux-provider"
    assert telemetry["failed_aux_model"] == "first-aux-model"
    assert telemetry["failure_error_type"] == "TimeoutError"
    assert telemetry["failure_status_code"] is None
    assert retry_kind.timeout is False
