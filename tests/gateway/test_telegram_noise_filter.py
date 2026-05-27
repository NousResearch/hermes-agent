"""Telegram-specific gateway filtering for noisy status/error output."""

from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
    _should_send_still_working_notice,
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


def test_telegram_closeout_status_is_rewritten_to_exact_packet():
    """Completion/status closeouts need the canonical packet before Telegram send."""
    raw = "Status: fixed\nProof: pytest passed"

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "Status / Proof / Changed / Next" in sanitized
    assert "Status:" in sanitized
    assert "Proof:" in sanitized
    assert "Changed:" in sanitized
    assert "Next:" in sanitized


def test_telegram_blocker_is_rewritten_to_exact_blocker_packet():
    """Blocker-shaped closeouts need the canonical blocker packet before send."""
    raw = "Blocked: need Anthony to approve the gateway restart."

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "BLOCKER PACKET" in sanitized
    assert "Blocked on:" in sanitized
    assert "Needed from Anthony:" in sanitized
    assert "Risk if skipped:" in sanitized
    assert "Next safe action:" in sanitized


def test_telegram_status_callback_gets_closeout_linter():
    """Status callbacks are linted too, not only final responses."""
    raw = "Completion: deployed\nProof: smoke test passed"

    sanitized = _prepare_gateway_status_message(Platform.TELEGRAM, "status", raw)

    assert sanitized is not None
    assert "Status / Proof / Changed / Next" in sanitized



def test_still_working_notice_suppresses_after_two_repeats():
    """Repeated still-working notices should go quiet unless detail changes."""
    state = {}

    assert _should_send_still_working_notice(3, " — iteration 1/90, running: terminal", state)
    assert _should_send_still_working_notice(6, " — iteration 1/90, running: terminal", state)
    assert not _should_send_still_working_notice(9, " — iteration 1/90, running: terminal", state)
    assert _should_send_still_working_notice(12, " — iteration 2/90, running: pytest", state)


def test_still_working_notice_suppresses_after_twenty_minutes_without_new_artifact():
    state = {}

    assert _should_send_still_working_notice(3, " — iteration 1/90, running: terminal", state)
    assert not _should_send_still_working_notice(21, " — iteration 1/90, running: terminal", state)
    assert _should_send_still_working_notice(24, " — blocker packet ready; user approval needed", state)
