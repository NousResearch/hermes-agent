"""Provider errors may contain credentials and must not enter stream warnings."""

import logging
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("dropped_tool", FailoverReason.unknown),
        ("context_overflow", FailoverReason.context_overflow),
        ("text_only", FailoverReason.unknown),
    ],
)
def test_partial_stream_warning_omits_provider_error(kind, reason, caplog, monkeypatch):
    from agent import chat_completion_helpers as helpers
    import agent.error_classifier as classifier

    marker = "synthetic-sensitive-marker"
    monkeypatch.setattr(
        classifier, "classify_api_error",
        lambda error, **kwargs: SimpleNamespace(reason=reason),
    )
    call = object.__new__(helpers._StreamingCall)
    call.agent = SimpleNamespace(
        _current_streamed_assistant_text="already visible",
        _warning_presentation_enabled=lambda: False,
        provider="synthetic-provider",
        model="synthetic-model",
        api_mode="chat_completions",
    )
    call.result = {
        "error": RuntimeError(f"provider failed: password={marker}"),
        "partial_tool_names": ["read_file"] if kind == "dropped_tool" else [],
    }

    with caplog.at_level(logging.WARNING, logger=helpers.__name__):
        response = call._partial_stream_stub()

    assert response._overflow_terminal is (kind == "context_overflow")
    assert caplog.records
    assert all(marker not in record.getMessage() for record in caplog.records)
