"""Behavior-preserving tests for the extracted gateway status-message filter seam."""

from gateway.config import Platform
from gateway.runtime.status_message_filter import prepare_gateway_status_message


def test_status_filter_suppresses_telegram_auxiliary_and_retry_noise():
    """Telegram keeps transient agent retry/auxiliary chatter out of chat."""
    noisy_messages = [
        "⚠ Auxiliary title generation failed: HTTP 400: Operation contains cybersecurity risk",
        "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
        "ℹ Configured compression model 'small-model' failed (timeout). Recovered using main model — check auxiliary.compression.model in config.yaml.",
        "⏳ Retrying in 4.2s (attempt 1/3)...",
        "⏱️ Rate limited. Waiting 30.0s (attempt 2/3)...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
        "⚠ Stream drop mid tool-call recovered on retry 1/3",
    ]

    for message in noisy_messages:
        assert prepare_gateway_status_message(Platform.TELEGRAM, "warn", message) is None


def test_status_filter_keeps_non_telegram_messages_unchanged():
    """The extracted seam must not change Discord/local diagnostics."""
    message = "  ⏳ Retrying in 4.2s (attempt 1/3)...  "

    assert prepare_gateway_status_message(Platform.DISCORD, "lifecycle", message) == message.strip()
    assert prepare_gateway_status_message("local", "lifecycle", message) == message.strip()


def test_status_filter_redacts_and_maps_telegram_provider_errors():
    """Telegram status callbacks get the same safe provider-error mapping as before."""
    raw = (
        "❌ API failed after 3 retries — HTTP 400: request blocked because "
        "Operation contains cybersecurity risk. request_id=req_123 sk-live-secret"
    )

    sanitized = prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", raw)

    assert sanitized is not None
    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_123" not in sanitized
    assert "sk-live-secret" not in sanitized


def test_status_filter_returns_none_for_blank_messages():
    assert prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", "  \n\t  ") is None
    assert prepare_gateway_status_message(Platform.DISCORD, "lifecycle", None) is None
