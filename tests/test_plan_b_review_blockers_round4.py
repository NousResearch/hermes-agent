"""Round-four regressions for final API egress redaction."""

from gateway.platforms.api_server_runs import _safe_api_status, _safe_api_text


def test_api_status_serializer_redacts_persisted_free_text():
    raw = "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token"
    sanitized = _safe_api_status(
        {
            "output": raw,
            "error": raw,
            "pending_steer": raw,
            "approval": {"command": raw, "description": raw, "title": raw},
        }
    )
    serialized = str(sanitized)
    assert "fixture-user:fixture-pass@" not in serialized
    assert "fixture-redacted-token" not in serialized


def test_api_text_redaction_fails_closed_on_unusable_value(monkeypatch):
    monkeypatch.setattr(
        "agent.redact.redact_sensitive_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture")),
    )
    assert _safe_api_text("fixture-sensitive") == "[REDACTED]"
