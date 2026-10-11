"""Contract tests for the native Google Gemini provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def gemini_profile():
    import model_tools
    import providers

    profile = providers.get_provider_profile("gemini")
    assert profile is not None, "gemini provider profile must be registered"
    return profile


def test_native_gemini_auxiliary_default_is_in_curated_catalog(gemini_profile):
    """The profile's default_aux_model must stay in lockstep with the curated
    model picker catalog — whatever model the default points at has to be
    one the picker can actually offer. Deliberately durable against future
    model-generation bumps: it does not pin either side to a frozen
    model-name string, only to the invariant that they never drift apart.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    assert gemini_profile.default_aux_model in _PROVIDER_MODELS["gemini"]


def test_create_client_is_native_only_on_the_native_surface(gemini_profile):
    from agent.gemini_native_adapter import GeminiNativeClient

    assert gemini_profile.create_client(api_key="k", base_url="https://example.invalid/v1") is None
    assert gemini_profile.create_client(
        api_key="k", base_url="https://generativelanguage.googleapis.com/v1beta/openai") is None
    client = gemini_profile.create_client(
        api_key="k", base_url="https://generativelanguage.googleapis.com/v1beta", command="ignored")
    try:
        assert isinstance(client, GeminiNativeClient)
        assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"
    finally:
        client.close()


def test_create_client_carries_the_tls_decision_only_when_given(gemini_profile, monkeypatch):
    import agent.process_bootstrap as bootstrap

    seen = []
    real = bootstrap.build_keepalive_http_client

    def _spy(base_url="", **kwargs):
        seen.append(kwargs.get("verify"))
        return real(base_url, **kwargs)

    monkeypatch.setattr(bootstrap, "build_keepalive_http_client", _spy)
    url = "https://generativelanguage.googleapis.com/v1beta"
    gemini_profile.create_client(api_key="k", base_url=url).close()
    assert seen == []  # auxiliary callers: the client's own transport
    gemini_profile.create_client(api_key="k", base_url=url, httpx_verify=False).close()
    assert seen == [False]
    supplied = object()
    client = gemini_profile.create_client(api_key="k", base_url=url, httpx_verify=False, http_client=supplied)
    assert client._http is supplied and seen == [False]
