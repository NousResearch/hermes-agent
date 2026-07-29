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




def test_bearer_header_stripped():
    out = R.redact_for_export("Authorization: Bearer abc.def-ghi_jkl")
    assert out is not None
    assert "abc.def-ghi_jkl" not in out


def test_none_passthrough():
    assert R.redact_for_export(None) is None




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
