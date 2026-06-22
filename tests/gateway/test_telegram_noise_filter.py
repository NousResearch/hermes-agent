"""Telegram-specific gateway filtering for noisy status/error output."""

from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)


def test_telegram_status_suppresses_auxiliary_and_retry_noise():
    """Auxiliary failures and retry backoff chatter should not hit Telegram."""
    noisy_messages = [
        "⚠ Auxiliary title generation failed: HTTP 400: Operation contains cybersecurity risk",
        "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "ℹ Configured compression model 'small-model' failed (timeout). Recovered using main model — check auxiliary.compression.model in config.yaml.",
        "⏳ Retrying in 4.2s (attempt 1/3)...",
        "⏱️ Rate limited. Waiting 30.0s (attempt 2/3)...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
    ]

    for message in noisy_messages:
        assert _prepare_gateway_status_message(Platform.TELEGRAM, "warn", message) is None


def test_non_telegram_status_is_unchanged():
    """The Telegram quieting policy must not hide CLI/Discord diagnostics."""
    message = "⏳ Retrying in 4.2s (attempt 1/3)..."

    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", message) == message
    assert _prepare_gateway_status_message("local", "lifecycle", message) == message


def test_telegram_status_sanitizes_raw_provider_security_errors():
    """Provider policy/security bodies should be replaced before chat delivery."""
    raw = (
        "❌ API failed after 3 retries — HTTP 400: request blocked because "
        "Operation contains cybersecurity risk. request_id=req_123"
    )

    sanitized = _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", raw)

    assert sanitized is not None
    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_123" not in sanitized


def test_telegram_final_response_sanitizes_raw_provider_errors():
    """Final Telegram replies should not expose raw provider/security details."""
    raw = (
        "API call failed after 3 retries: HTTP 400: This request was blocked "
        "under the provider cybersecurity risk policy. request_id=req_abc"
    )

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_abc" not in sanitized


def test_telegram_final_response_redacts_auth_secrets():
    """Authentication errors should be useful without leaking key material."""
    raw = (
        "⚠️ Provider authentication failed: Incorrect API key provided: "
        "sk-live_abcdefghijklmnopqrstuvwxyz1234567890"
    )

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "authentication failed" in sanitized.lower()
    assert "check the configured credentials" in sanitized.lower()
    assert "sk-live" not in sanitized


def test_telegram_final_response_keeps_normal_answers():
    """Normal assistant content should not be rewritten."""
    answer = "Here is the clean summary you asked for."

    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer


# --- Low-tier mobile inboxes (Photon/iMessage, BlueBubbles, Signal, ...) ---
# These are permanent-message mobile inboxes, so they get the same quieting as
# Telegram: internal compaction/auxiliary/retry banners never leak into chat,
# raw provider error bodies are replaced with a safe category, but normal final
# replies still pass through untouched. Discord/CLI keep full diagnostics.


def test_low_tier_status_suppresses_internal_banners():
    """Compaction/auxiliary/retry status noise must not reach low-tier inboxes."""
    noisy_messages = [
        "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "⏳ Retrying in 4.2s (attempt 1/3)...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
    ]
    for platform in ("photon", Platform.BLUEBUBBLES, Platform.SIGNAL):
        for message in noisy_messages:
            assert (
                _prepare_gateway_status_message(platform, "warn", message) is None
            ), (platform, message)


def test_low_tier_status_sanitizes_raw_provider_errors():
    """Real provider errors still reach low-tier inboxes, as a safe category."""
    raw = (
        "❌ API failed after 3 retries — HTTP 400: request blocked because "
        "Operation contains cybersecurity risk. request_id=req_123"
    )

    sanitized = _prepare_gateway_status_message("photon", "lifecycle", raw)

    assert sanitized is not None
    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_123" not in sanitized


def test_low_tier_final_response_redacts_secrets_and_raw_errors():
    """Final low-tier replies redact secrets and raw provider error bodies."""
    raw = (
        "⚠️ Provider authentication failed: Incorrect API key provided: "
        "sk-live_abcdefghijklmnopqrstuvwxyz1234567890"
    )

    sanitized = _sanitize_gateway_final_response("photon", raw)

    assert "authentication failed" in sanitized.lower()
    assert "sk-live" not in sanitized


def test_low_tier_final_response_keeps_normal_answers():
    """Normal assistant content is never rewritten for low-tier platforms."""
    answer = "Here is the clean summary you asked for."

    assert _sanitize_gateway_final_response("photon", answer) == answer


def test_high_tier_and_cli_still_see_internal_banners():
    """Regression guard: Discord/CLI keep full diagnostics; banners NOT suppressed."""
    banner = "⚠ Compression summary failed: upstream error. Inserted a fallback context marker."

    assert _prepare_gateway_status_message(Platform.DISCORD, "warn", banner) == banner
    assert _prepare_gateway_status_message("local", "warn", banner) == banner
