from __future__ import annotations


def test_active_openrouter_without_credentials_is_blocking_issue(monkeypatch):
    from hermes_cli import doctor

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    issue = doctor._active_openrouter_credentials_issue("openrouter")

    assert issue is not None
    assert "OpenRouter" in issue
    assert "OPENROUTER_API_KEY" in issue
    assert "OPENAI_API_KEY" in issue


def test_active_openrouter_accepts_openai_key_fallback(monkeypatch):
    from hermes_cli import doctor

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openrouter-fallback-key")

    assert doctor._active_openrouter_credentials_issue("openrouter") is None


def test_non_openrouter_provider_does_not_emit_openrouter_issue(monkeypatch):
    from hermes_cli import doctor

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert doctor._active_openrouter_credentials_issue("anthropic") is None
