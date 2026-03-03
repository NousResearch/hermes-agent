"""Regression tests for CLI API key selection across providers."""

from __future__ import annotations

from cli import HermesCLI


def _clear_provider_env(monkeypatch) -> None:
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_API_KEY",
        "HERMES_INFERENCE_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


def test_cli_prefers_openrouter_key_for_openrouter_base(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")

    cli = HermesCLI()

    assert cli.base_url == "https://openrouter.ai/api/v1"
    assert cli.api_key == "sk-or-test"


def test_cli_prefers_openai_key_for_custom_openai_base(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    cli = HermesCLI()

    assert cli.base_url == "https://api.openai.com/v1"
    assert cli.api_key == "sk-proj-test"
