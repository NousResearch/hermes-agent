"""Tests for native OpenAI API provider auto-detection."""

from hermes_cli.auth import resolve_provider


def test_auto_detects_openai_api_key_as_openai(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})

    assert resolve_provider("auto") == "openai"


def test_openrouter_key_still_wins_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})

    assert resolve_provider("auto") == "openrouter"
