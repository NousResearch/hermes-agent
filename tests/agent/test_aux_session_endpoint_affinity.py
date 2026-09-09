"""Auxiliary routing sticks to the session's configured OpenAI endpoint; a rejection elsewhere is not a dead key.

Proxy users (`OPENAI_BASE_URL` / `providers.openai` pointing at a corporate gateway) saw compression
hop to api.openai.com, 401 with the proxy-issued key, and then have that key quarantined.
"""
from types import SimpleNamespace

from agent import auxiliary_client as aux


def test_openai_alias_prefers_configured_endpoint_over_public_default(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm-proxy.corp.example/v1")
    provider, base = aux._expand_direct_api_alias("openai", None)
    assert provider == "custom"
    assert base == "https://llm-proxy.corp.example/v1"
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert aux._expand_direct_api_alias("openai", None) == ("custom", "https://api.openai.com/v1")


def test_rejection_at_foreign_host_does_not_name_the_session_pool():
    runtime = {"provider": "openai-api", "model": "gpt-5.4",
               "base_url": "https://llm-proxy.corp.example/v1", "api_key": "sk-proxy"}
    foreign = SimpleNamespace(base_url="https://api.openai.com/v1/", api_key="sk-proxy")
    same = SimpleNamespace(base_url="https://llm-proxy.corp.example/v1/", api_key="sk-proxy")
    assert aux._recoverable_pool_provider("openai-api", foreign, main_runtime=runtime) is None
    assert aux._recoverable_pool_provider("openai-api", same, main_runtime=runtime) == "openai-api"
