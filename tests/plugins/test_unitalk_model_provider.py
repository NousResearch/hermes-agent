"""The bundled Unitalk provider resolves its own endpoint and ordered keys."""

import pytest


@pytest.mark.parametrize(
    ("unitalk_key", "openai_key", "expected"),
    [("unitalk-secret", "openai-secret", "unitalk-secret"),
     (None, "openai-secret", "openai-secret"),
     ("unitalk-secret", None, "unitalk-secret")],
)
def test_bundled_provider_runtime(monkeypatch, unitalk_key, openai_key, expected):
    from providers import get_provider_profile, provider_source
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.delenv("UNITALK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if unitalk_key:
        monkeypatch.setenv("UNITALK_API_KEY", unitalk_key)
    if openai_key:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)

    assert get_provider_profile("unitalk") is not None
    assert provider_source("unitalk") == "bundled"
    runtime = resolve_runtime_provider(requested="unitalk")
    assert runtime["provider"] == "unitalk"
    assert runtime["base_url"] == "https://llmgateway.unitalk.ai/v1"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["api_key"] == expected


def test_unitalk_key_replaces_previous_openai_fallback(monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.delenv("UNITALK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert resolve_runtime_provider(requested="unitalk")["api_key"] == "openai-secret"
    monkeypatch.setenv("UNITALK_API_KEY", "unitalk-secret")
    assert resolve_runtime_provider(requested="unitalk")["api_key"] == "unitalk-secret"
