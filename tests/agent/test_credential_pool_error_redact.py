"""Credential-pool auth error messages must be safe to persist."""

from __future__ import annotations


def test_redact_exception_message_masks_refresh_token_value():
    from agent.credential_pool import _redact_exception_message

    exc = RuntimeError("refresh failed: refresh_token=rt-secret-value")

    message = _redact_exception_message(exc)

    assert "rt-secret-value" not in message
    assert "refresh_token=***" in message


def test_redact_exception_message_fails_closed(monkeypatch):
    from agent import credential_pool

    def broken_redactor(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr(
        "agent.redact.redact_sensitive_text",
        broken_redactor,
    )

    message = credential_pool._redact_exception_message(
        RuntimeError("refresh failed: access_token=at-secret-value")
    )

    assert message == "credential refresh failed"
    assert "at-secret-value" not in message
