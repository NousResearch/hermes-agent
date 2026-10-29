"""An explicitly requested local alias must not inherit another provider's model.base_url.

ECCO-0 H4 reproduction: --provider ollama inherited a stale llamacpp URL and
continued retrying the wrong local endpoint instead of refusing unknown routing.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp


@pytest.mark.parametrize(
    ("requested", "configured"),
    [("ollama", "llamacpp"), ("vllm", "ollama"), ("ollama", "custom")],
)
def test_local_alias_refuses_another_providers_saved_endpoint(
    monkeypatch, requested, configured
):
    url = "http://127.0.0.1:59041/v1"
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {"model": {"provider": configured, "base_url": url}},
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda _provider: SimpleNamespace(has_credentials=lambda: False),
    )
    for name in ("CUSTOM_BASE_URL", "OPENROUTER_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        rp.AuthError, match=rf"provider '{requested}' has no endpoint"
    ) as error:
        rp.resolve_runtime_provider(requested=requested)
    assert (error.value.provider, error.value.code) == (requested, "missing_base_url")


def test_matching_local_alias_keeps_saved_endpoint(monkeypatch):
    url = "http://127.0.0.1:11434/v1"
    monkeypatch.setattr(
        rp, "load_config", lambda: {"model": {"provider": "ollama", "base_url": url}}
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda _provider: SimpleNamespace(has_credentials=lambda: False),
    )
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    runtime = rp.resolve_runtime_provider(requested="ollama")
    assert (runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
        "custom",
        url,
        "no-key-required",
    )


def test_explicit_local_endpoint_overrides_different_saved_provider(monkeypatch):
    stale_url = "http://127.0.0.1:59041/v1"
    correct_url = "http://127.0.0.1:11434/v1"
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {"model": {"provider": "llamacpp", "base_url": stale_url}},
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda _provider: SimpleNamespace(has_credentials=lambda: False),
    )
    runtime = rp.resolve_runtime_provider(
        requested="ollama", explicit_base_url=correct_url
    )
    assert runtime["base_url"] == correct_url
    assert runtime["provider"] == "custom"
