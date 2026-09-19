"""Behavioral coverage for manual compression status messages."""

from types import SimpleNamespace

from agent.manual_compression_feedback import (
    describe_compression_lock_skip,
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
    assert feedback["headline"] == "Compressed with fallback: 12 → 4 messages"
    assert "removed 8 message(s)" in feedback["note"]
    assert "invalid response" in feedback["note"]


def test_manual_receipt_reports_the_effective_aux_route_and_elapsed_time():
    messages = _messages(12)
    state = SimpleNamespace(
        _last_compress_aborted=False,
        _last_compress_refused_would_grow=False,
        _last_summary_fallback_used=False,
        _last_summary_error=None,
        _last_compression_telemetry={
            "aux_provider": "custom",
            "aux_model": "gpt-5.6-luna",
            "aux_call_duration_ms": 1_234,
        },
    )

    feedback = summarize_manual_compression(
        messages,
        messages[:2] + messages[-2:],
        120_000,
        40_000,
        compression_state=state,
    )

    assert feedback["receipt_line"] == "Summary route: custom / gpt-5.6-luna · 1.2s"


def test_manual_receipt_is_omitted_without_complete_telemetry():
    messages = _messages(12)
    state = SimpleNamespace(
        _last_compress_aborted=False,
        _last_compress_refused_would_grow=False,
        _last_summary_fallback_used=False,
        _last_summary_error=None,
        _last_compression_telemetry={"aux_provider": "custom", "aux_model": "gpt-5.6-luna"},
    )

    feedback = summarize_manual_compression(messages, messages[:2] + messages[-2:], 120_000, 40_000,
                                             compression_state=state)

    assert feedback["receipt_line"] is None




