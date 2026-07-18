"""Behavioral coverage for manual compression status messages."""

from types import SimpleNamespace

from agent.manual_compression_feedback import summarize_manual_compression


def _messages(count: int) -> list[dict[str, str]]:
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(count)
    ]


def test_aborted_compression_reports_preserved_messages_and_reason():
    messages = _messages(12)
    state = SimpleNamespace(
        _last_compress_aborted=True,
        _last_summary_fallback_used=False,
        _last_summary_error=(
            "Provider 'opencode-zen' is set in config.yaml but no API key was found."
        ),
    )

    feedback = summarize_manual_compression(
        messages,
        list(messages),
        120_000,
        120_000,
        compression_state=state,
    )

    assert feedback["aborted"] is True
    assert feedback["fallback_used"] is False
    assert feedback["headline"] == "Compression aborted: 12 messages preserved"
    assert "no messages were removed" in feedback["note"]
    assert "no API key was found" in feedback["note"]


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
