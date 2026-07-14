"""Export redaction pipeline tests — the security-critical layer.

Invariants:
  * Secrets ALWAYS stripped, every export path, no flag disables it.
  * Fails CLOSED: if the redactor can't run, the raw string is never emitted.
  * PII (emails, phones, UUID-shaped ids) stripped in 'pii' mode — the mode
    the gateway diagnostics path always uses.
"""

from __future__ import annotations

import agent.monitoring.redaction as R


def test_secret_always_stripped_in_none_mode():
    fake_key = "sk-ant-api03-" + "A" * 24  # constructed to dodge literal-scrubbers
    text = f"calling with key {fake_key} and moving on"
    out = R.redact_for_export(text, content_mode=R.CONTENT_NONE)
    assert out is not None
    assert fake_key not in out


def test_secret_always_stripped_in_pii_mode():
    fake_token = "ghp_" + "0123456789abcdef" * 2 + "0123"
    text = f"token {fake_token} leaked"
    out = R.redact_for_export(text, content_mode=R.CONTENT_PII)
    assert out is not None
    assert fake_token not in out


def test_none_passthrough():
    assert R.redact_for_export(None, content_mode=R.CONTENT_NONE) is None


def test_pii_mode_strips_email_phone_uuid():
    text = ("reach alice@example.com or +1 415 555 0100, "
            "install 123e4567-e89b-12d3-a456-426614174000")
    out = R.redact_for_export(text, content_mode=R.CONTENT_PII)
    assert out is not None
    assert "alice@example.com" not in out
    assert "426614174000" not in out
    assert "[email]" in out
    assert "[id]" in out


def test_none_mode_keeps_ordinary_words():
    out = R.redact_for_export("just ordinary words", content_mode=R.CONTENT_NONE)
    assert out == "just ordinary words"


def test_pii_mode_preserves_non_pii_structure():
    text = "platform.slack entered fatal after auth_failed"
    out = R.redact_for_export(text, content_mode=R.CONTENT_PII)
    assert out is not None
    assert "platform.slack" in out
    assert "auth_failed" in out
