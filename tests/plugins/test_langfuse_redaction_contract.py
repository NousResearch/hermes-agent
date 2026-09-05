"""Regression tests for Langfuse capture-mode secret handling."""
from __future__ import annotations

from plugins.observability import langfuse as mod


def test_sanitized_mode_redacts_nested_secrets(monkeypatch):
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "sanitized")
    api_key = "sk-" + ("A" * 32)
    bearer = "ghp_" + ("B" * 36)
    result = mod._capture_content(
        {
            "api_key": api_key,
            "nested": [{"authorization": f"Bearer {bearer}"}],
        }
    )

    assert api_key not in str(result)
    assert bearer not in str(result)


def test_sanitized_mode_redacts_before_truncating_long_secret(monkeypatch):
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "sanitized")
    monkeypatch.setenv("HERMES_LANGFUSE_MAX_CHARS", "8")
    secret = "sk-" + ("A" * 64)
    result = mod._capture_content(secret)

    # Truncating first would leave too few token characters for the redactor
    # to recognize and would export this raw credential prefix. Assert the
    # redactor's masking marker too so an empty/dropped payload cannot pass.
    assert result
    assert "..." in result
    assert secret[:8] not in result


def test_full_mode_preserves_secret_shaped_text_when_truncating(monkeypatch):
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "full")
    monkeypatch.setenv("HERMES_LANGFUSE_MAX_CHARS", "8")
    secret = "sk-" + ("A" * 64)
    result = mod._capture_content(secret)

    assert result.startswith(secret[:8])
    assert secret not in result
    assert "truncated" in result
