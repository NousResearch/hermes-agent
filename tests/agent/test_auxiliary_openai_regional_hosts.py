"""The auxiliary client must treat OpenAI's regional hosts like the canonical one.

``providers.is_official_openai_host()`` is the shared predicate: the canonical
``api.openai.com`` plus OpenAI's documented data-residency hosts
(``us.``/``eu.api.openai.com``), hostname-parsed so lookalikes stay rejected.
``host_mandated_api_mode()`` already routes through it.

Two auxiliary-client decisions still compared hostnames for exact equality, so
on a residency host the auxiliary client picked the wrong transport and the
wrong token parameter — both of which the endpoint rejects.
"""

from __future__ import annotations

import pytest

import agent.auxiliary_client as aux

CANONICAL = "https://api.openai.com/v1"
REGIONAL = ["https://us.api.openai.com/v1", "https://eu.api.openai.com/v1"]
LOOKALIKE = "https://api.openai.com.attacker.test/v1"


@pytest.fixture
def fake_openai(monkeypatch):
    """Stand in for the OpenAI SDK client so no network/credentials are needed."""

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key", "")
            self.base_url = kwargs.get("base_url", "")

    monkeypatch.setattr(aux, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def _resolve(base_url: str, model: str = "gpt-5-codex"):
    return aux.resolve_provider_client(
        "custom",
        model,
        explicit_base_url=base_url,
        explicit_api_key="sk-test",
    )[0]


@pytest.mark.parametrize("base_url", REGIONAL)
def test_regional_host_wraps_codex_model_for_responses_api(fake_openai, base_url):
    """The regression: a residency host mandates codex_responses, so wrap it."""
    from hermes_cli.providers import host_mandated_api_mode

    assert host_mandated_api_mode(base_url) == "codex_responses"
    assert isinstance(_resolve(base_url), aux.CodexAuxiliaryClient)


def test_canonical_host_still_wraps(fake_openai):
    assert isinstance(_resolve(CANONICAL), aux.CodexAuxiliaryClient)


def test_lookalike_host_is_not_treated_as_openai(fake_openai):
    """Widening to the host family must not widen to spoofed hostnames."""
    assert not isinstance(_resolve(LOOKALIKE), aux.CodexAuxiliaryClient)


def test_unrelated_host_with_codex_named_model_is_not_wrapped(fake_openai):
    """A codex-shaped model name alone must not force the Responses API."""
    assert not isinstance(
        _resolve("https://llm.internal.test/v1"), aux.CodexAuxiliaryClient
    )


@pytest.fixture
def direct_openai_endpoint(monkeypatch):
    """Isolate auxiliary_max_tokens_param from pool/OpenRouter short-circuits."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(aux, "_read_nous_auth", lambda: None)

    def _point_at(base_url: str):
        monkeypatch.setattr(aux, "_current_custom_base_url", lambda: base_url)

    return _point_at


@pytest.mark.parametrize("base_url", REGIONAL + [CANONICAL])
def test_official_openai_hosts_use_max_completion_tokens(direct_openai_endpoint, base_url):
    """Newer OpenAI models reject ``max_tokens``; residency hosts serve them too."""
    direct_openai_endpoint(base_url)
    assert aux.auxiliary_max_tokens_param(256) == {"max_completion_tokens": 256}


def test_lookalike_host_keeps_plain_max_tokens(direct_openai_endpoint):
    direct_openai_endpoint(LOOKALIKE)
    assert aux.auxiliary_max_tokens_param(256) == {"max_tokens": 256}
