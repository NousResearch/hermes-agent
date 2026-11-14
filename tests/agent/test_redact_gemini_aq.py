"""Bare ``AQ.``-prefix Gemini authorization keys are masked by the known-prefix pass (#66920)."""

import pytest

from agent.redact import redact_sensitive_text

KEY = "AQ.Ab8RN6" + "xY3kQ9zLm2" * 5  # placeholder, not a real credential


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize(
    "text",
    [
        KEY,
        "AQ." + "B" * 48,
        f"400 INVALID_ARGUMENT: API key {KEY} not valid",
        f"https://generativelanguage.googleapis.com/v1beta/models?key={KEY}",
        f'{{"key": "{KEY}"}}',
    ],
)
def test_bare_and_embedded_aq_key_is_masked(text):
    key = KEY if KEY in text else "AQ." + "B" * 48
    assert key not in redact_sensitive_text(text, force=True)


def test_labelled_aq_key_is_masked():
    assert KEY not in redact_sensitive_text("GEMINI_API_KEY=" + KEY, force=True)


@pytest.mark.parametrize(
    "text",
    [
        "AQ." + "B" * 10,
        "See the AQ. section for the full project spec.",
        "AQ. is a placeholder variable we never log.",
    ],
)
def test_short_or_prose_aq_is_unchanged(text):
    assert redact_sensitive_text(text, force=True) == text
