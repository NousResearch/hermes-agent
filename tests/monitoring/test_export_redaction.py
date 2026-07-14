"""Export redaction tests — the security-critical layer.

Invariants:
  * One unconditional scrub: secrets AND PII, no modes, no knobs.
  * Fails CLOSED: if the redactor can't run, the raw string is never emitted.
  * Structure (subsystem names, error codes) survives; free-text PII does not.
"""

from __future__ import annotations

from unittest import mock

import agent.monitoring.redaction as R


def test_secret_key_always_stripped():
    fake_key = "sk-ant-api03-" + "A" * 24  # constructed to dodge literal-scrubbers
    out = R.redact_for_export(f"calling with key {fake_key} and moving on")
    assert out is not None
    assert fake_key not in out


def test_token_shapes_stripped():
    ghp = "ghp_" + "0123456789abcdef" * 2 + "0123"
    slack = "xoxb-" + "123456789012-abcdefABCDEF"
    out = R.redact_for_export(f"token {ghp} and {slack} leaked")
    assert out is not None
    assert ghp not in out
    assert slack not in out
    assert "[redacted]" in out


def test_bearer_header_stripped():
    out = R.redact_for_export("Authorization: Bearer abc.def-ghi_jkl")
    assert out is not None
    assert "abc.def-ghi_jkl" not in out


def test_none_passthrough():
    assert R.redact_for_export(None) is None


def test_pii_always_stripped():
    text = ("reach alice@example.com or +1 415 555 0100, "
            "install 123e4567-e89b-12d3-a456-426614174000")
    out = R.redact_for_export(text)
    assert out is not None
    assert "alice@example.com" not in out
    assert "426614174000" not in out
    assert "[email]" in out
    assert "[id]" in out
    assert "[phone]" in out


def test_ordinary_words_survive():
    assert R.redact_for_export("just ordinary words") == "just ordinary words"


def test_structure_preserved():
    out = R.redact_for_export("platform.slack entered fatal after auth_failed")
    assert out is not None
    assert "platform.slack" in out
    assert "auth_failed" in out


def test_fails_closed_when_redactor_unavailable():
    with mock.patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError):
        out = R.redact_for_export("secret sauce sk-live-key")
    assert out == "[redaction-unavailable]"
