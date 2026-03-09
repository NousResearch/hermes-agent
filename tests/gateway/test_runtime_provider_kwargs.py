import pytest


def test_resolve_runtime_agent_kwargs_includes_max_tokens(monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key",
            "base_url": "https://example.com/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "max_tokens": 32768,
        },
    )

    kwargs = gateway_run._resolve_runtime_agent_kwargs()

    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://example.com/v1"
    assert kwargs["provider"] == "openrouter"
    assert kwargs["api_mode"] == "chat_completions"
    assert kwargs["max_tokens"] == 32768


def test_resolve_runtime_agent_kwargs_wraps_errors(monkeypatch):
    from gateway import run as gateway_run

    def _raise(**kwargs):
        raise RuntimeError("raw")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _raise)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error",
        lambda exc: "formatted-runtime-provider-error",
    )

    with pytest.raises(RuntimeError, match="formatted-runtime-provider-error"):
        gateway_run._resolve_runtime_agent_kwargs()
