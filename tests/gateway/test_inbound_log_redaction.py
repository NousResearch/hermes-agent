from agent.redact import redact_sensitive_text


def test_inbound_log_preview_redacts_secret_like_content(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    preview = redact_sensitive_text(
        "Use OPENAI_API_KEY=example-secret-1234 for this",
        force=True,
    )

    assert "example-secret-1234" not in preview
    assert "OPENAI_API_KEY=" in preview
    assert "exampl...1234" in preview
