"""Facade invariants for gateway.run strangler seams."""

from gateway import run
from gateway.config import Platform
from gateway.runtime.delivery_redaction import redact_gateway_user_facing_secrets
from gateway.runtime.auto_resume_freshness import (
    auto_continue_freshness_window,
    coerce_gateway_timestamp,
    is_fresh_gateway_interruption,
    last_transcript_timestamp,
    startup_auto_resume_max,
)
import gateway.run as gateway_run


def test_gateway_run_redaction_facade_preserves_public_private_import_surface():
    raw = "token sk-testsecret12345 and Bearer abcdefghijklmnopqrstuvwxyz"
    assert run._redact_gateway_user_facing_secrets(raw) == redact_gateway_user_facing_secrets(raw)
    assert "[REDACTED]" in run._redact_gateway_user_facing_secrets(raw)
    assert "sk-testsecret12345" not in run._redact_gateway_user_facing_secrets(raw)


def test_gateway_provider_error_reply_still_uses_redacted_text():
    raw = "Provider authentication failed: sk-tes...3456"
    sanitized = run._sanitize_gateway_final_response("telegram", raw)
    assert "Provider authentication failed" in sanitized
    assert "sk-tes...3456" not in sanitized


def test_status_message_filter_facade_delegates_without_changing_contract(monkeypatch):
    """gateway.run keeps its private import surface while delegating to the seam."""
    calls = []

    def fake_prepare(platform, event_type, message):
        calls.append((platform, event_type, message))
        return "prepared-status"

    monkeypatch.setattr(gateway_run, "prepare_gateway_status_message", fake_prepare)

    assert (
        gateway_run._prepare_gateway_status_message(
            Platform.TELEGRAM,
            "lifecycle",
            "raw status",
        )
        == "prepared-status"
    )
    assert calls == [(Platform.TELEGRAM, "lifecycle", "raw status")]


def test_status_message_filter_facade_preserves_existing_behavior():
    """The old gateway.run helper remains behavior-compatible for callers/tests."""
    noisy = "⏳ Retrying in 4.2s (attempt 1/3)..."

    assert gateway_run._prepare_gateway_status_message(Platform.TELEGRAM, "warn", noisy) is None
    assert gateway_run._prepare_gateway_status_message(Platform.DISCORD, "warn", noisy) == noisy


def test_auto_resume_freshness_facade_preserves_private_import_surface():
    history = [{"role": "user", "timestamp": 100.0}]

    assert gateway_run._coerce_gateway_timestamp("1700000000") == coerce_gateway_timestamp(
        "1700000000"
    )
    assert gateway_run._is_fresh_gateway_interruption(
        90.0, now=100.0, window_secs=10.0
    ) == is_fresh_gateway_interruption(90.0, now=100.0, window_secs=10.0)
    assert gateway_run._last_transcript_timestamp(history) == last_transcript_timestamp(history)
    assert gateway_run._auto_continue_freshness_window() == auto_continue_freshness_window()
    assert gateway_run._startup_auto_resume_max() == startup_auto_resume_max()
