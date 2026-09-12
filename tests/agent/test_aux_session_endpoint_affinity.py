"""Auxiliary routing sticks to the session's configured OpenAI endpoint; a rejection elsewhere is not a dead key.

Proxy users (`OPENAI_BASE_URL` / `providers.openai` pointing at a corporate gateway) saw compression
hop to api.openai.com, 401 with the proxy-issued key, and then have that key quarantined.
"""
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux

SESSION = {"provider": "openai-api", "model": "gpt-5.4",
           "base_url": "https://proxy.example:8443/v1", "api_key": "sk-session"}


def test_openai_alias_prefers_configured_endpoint_over_public_default(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm-proxy.corp.example/v1")
    provider, base = aux._expand_direct_api_alias("openai", None)
    assert provider == "custom"
    assert base == "https://llm-proxy.corp.example/v1"
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert aux._expand_direct_api_alias("openai", None) == ("custom", "https://api.openai.com/v1")


@pytest.mark.parametrize("rejecting_base", [
    "https://api.openai.com/v1/",          # different host
    "https://proxy.example:9443/v1/",      # same host, different port
    "http://proxy.example:8443/v1/",       # same host+port, scheme downgrade
])
def test_session_key_rejected_at_foreign_origin_is_not_rotated(rejecting_base):
    client = SimpleNamespace(base_url=rejecting_base, api_key="sk-session")
    assert aux._recoverable_pool_provider("openai-api", client, main_runtime=SESSION) is None


def test_rotation_survives_at_session_origin_and_for_independent_aux_pool():
    same_origin = SimpleNamespace(base_url="https://proxy.example:8443/v1/", api_key="sk-session")
    assert aux._recoverable_pool_provider("openai-api", same_origin, main_runtime=SESSION) == "openai-api"
    # A separately owned aux credential at its own configured endpoint must keep rotating.
    independent = SimpleNamespace(base_url="https://api.openai.com/v1/", api_key="sk-aux-pool-1")
    assert aux._recoverable_pool_provider("openai-api", independent, main_runtime=SESSION) == "openai-api"


def test_named_provider_defaults_compose_under_task_overrides(monkeypatch, tmp_path):
    """URL-only / key-only task overrides win field-by-field; the named entry fills only the blanks."""
    monkeypatch.setenv("NAMED_KEY", "named-key")
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openai\n  default: gpt-5.4\n"
        "providers:\n  openai:\n    api: https://named.example/v1\n    key_env: NAMED_KEY\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def route(**overrides):
        aux._client_cache.clear()
        client, _ = aux.resolve_provider_client("openai", "gpt-5.4-mini", **overrides)
        assert client is not None
        return str(client.base_url).rstrip("/"), str(client.api_key)

    assert route() == ("https://named.example/v1", "named-key")
    assert route(explicit_base_url="https://aux-explicit.example/v1") == ("https://aux-explicit.example/v1", "named-key")
    assert route(explicit_api_key="task-key") == ("https://named.example/v1", "task-key")
    assert route(explicit_base_url="https://aux-explicit.example/v1", explicit_api_key="task-key") == (
        "https://aux-explicit.example/v1", "task-key")
