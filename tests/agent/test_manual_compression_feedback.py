"""Behavioral coverage for manual compression status messages."""

from types import SimpleNamespace

import pytest

from agent.manual_compression_feedback import (
    summarize_manual_compression,
)


def _messages(count: int) -> list[dict[str, str]]:
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(count)
    ]


def test_failure_reason_redaction_is_forced_at_ui_boundary(monkeypatch):
    messages = _messages(12)
    fake_secret = "sk-proj-" + "X" * 40
    state = SimpleNamespace(
        _last_compress_aborted=True,
        _last_summary_fallback_used=False,
        _last_summary_error=f"provider rejected OPENAI_API_KEY={fake_secret}",
    )
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)

    feedback = summarize_manual_compression(
        messages,
        list(messages),
        120_000,
        120_000,
        compression_state=state,
    )

    assert fake_secret not in feedback["note"]
    assert "OPENAI_API_KEY=" in feedback["note"]


def test_fallback_compression_reports_dropped_message_count():
    before = _messages(12)
    after = before[:2] + before[-2:]
    state = SimpleNamespace(
        _last_compress_aborted=False,
        _last_summary_fallback_used=True,
        _last_summary_dropped_count=8,
        _last_summary_error="summary provider returned an invalid response",
    )

    feedback = summarize_manual_compression(
        before,
        after,
        120_000,
        40_000,
        compression_state=state,
    )

    assert feedback["aborted"] is False
    assert feedback["fallback_used"] is True
    assert "12" in feedback["headline"] and "4" in feedback["headline"]
    assert "8" in feedback["note"]
    assert "invalid response" in feedback["note"]


@pytest.mark.parametrize("surface", ["shared", "gateway"])
def test_manual_receipt_renders_effective_route_across_surfaces(surface):
    from agent.context_compressor_telemetry import CompressionTelemetryMixin
    from agent.conversation_compression_manual import CompressRequest, CompressResult, render_compress_result
    from gateway.slash_commands_session import _manual_compression_reply_lines

    before = _messages(12)
    after = before[:2] + before[-2:]
    state = SimpleNamespace(
        _last_compress_aborted=False, _last_summary_fallback_used=False,
        _last_summary_error=None, _last_aux_model_failure_model=None,
        _active_compression_telemetry={"effective_aux_context": None},
    )
    # Exercise the real telemetry recorder, including accumulated call time.
    for duration in (1_000, 234):
        CompressionTelemetryMixin._record_aux_compression_call(
            state, prompt_messages=[], max_tokens=100, duration_ms=duration,
            aux_provider="custom", aux_model="effective-summary-model",
        )
    state._last_compression_telemetry = state._active_compression_telemetry
    summary = summarize_manual_compression(before, after, 120_000, 40_000, compression_state=state)
    expected = "Summary route: custom / effective-summary-model · 1.2s"
    if surface == "shared":
        result = CompressResult("compressed", before, after, 120_000, 40_000, CompressRequest(), summary=summary)
        lines = render_compress_result(result)
    else:
        lines = _manual_compression_reply_lines(summary, state, None)
    assert expected in lines


@pytest.mark.parametrize("telemetry", [
    None, {}, {"aux_provider": "custom", "aux_model": "model"},
    {"aux_provider": "", "aux_model": "model", "aux_call_duration_ms": 1},
    {"aux_provider": "custom", "aux_model": "model", "aux_call_duration_ms": -1},
    {"aux_provider": "custom", "aux_model": "model", "aux_call_duration_ms": True},
])
def test_manual_receipt_is_omitted_without_valid_telemetry(telemetry):
    messages = _messages(12)
    state = SimpleNamespace(_last_compression_telemetry=telemetry)
    feedback = summarize_manual_compression(
        messages, messages[:2] + messages[-2:], 120_000, 40_000, compression_state=state,
    )
    assert feedback.get("receipt_line") is None


@pytest.mark.parametrize("flag", ["_last_compress_aborted", "_last_compress_refused_would_grow", "_last_summary_fallback_used"])
def test_failed_or_refused_compression_never_claims_a_successful_summary_route(flag):
    messages = _messages(12)
    state = SimpleNamespace(**{flag: True}, _last_compression_telemetry={
        "aux_provider": "custom", "aux_model": "failed-summary-model", "aux_call_duration_ms": 1234,
    })
    feedback = summarize_manual_compression(
        messages, messages[:2] + messages[-2:], 120_000, 40_000, compression_state=state,
    )
    assert feedback.get("receipt_line") is None


