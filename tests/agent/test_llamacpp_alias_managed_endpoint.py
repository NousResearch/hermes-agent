"""The llama.cpp alias must reach the managed local server on the auxiliary/fallback path.

Regression: ``resolve_provider_client("llamacpp", ...)`` with no explicit base_url fell through
``_resolve_custom_branch`` to ``_try_custom_endpoint`` / ``_resolve_api_key_provider``, so the
fallback path (``try_activate_fallback`` resolves through this function) returned whichever CLOUD
provider held credentials and posted a local model slug there. Observed in the wild: a GGUF slug sent
to ``generativelanguage.googleapis.com``, HTTP 400 "unexpected model name format", which the agent
surfaced to the user as "the model server rejected this request as too large" against a context
window the local server was never asked about.

The main ladder (``hermes_cli/runtime_provider.py``) already resolved this through
``_resolve_llamacpp_runtime``; only the auxiliary path was missing the rung.
"""

from unittest.mock import MagicMock

import pytest

from agent import auxiliary_client as aux

MODEL = "Qwen3.8-27B-UD-Q4_K_M"
MANAGED_BASE = "http://127.0.0.1:59999/v1"
MANAGED_KEY = "managed-key-0123456789abcdef"


@pytest.fixture
def managed_route(monkeypatch):
    """Pretend the supervised local server is up; return the list of lookups made."""
    lookups = []

    def _fake_endpoint(*, wait_for_boot_s=8.0, config=None):
        lookups.append(wait_for_boot_s)
        return {"base_url": MANAGED_BASE, "api_key": MANAGED_KEY}

    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint", _fake_endpoint)
    return lookups


def test_alias_without_base_url_resolves_to_managed_endpoint(managed_route):
    """The alias means "my own server": it must resolve there, not to a cloud provider."""
    client, model = aux.resolve_provider_client(
        "llamacpp", model=MODEL, raw_codex=True, explicit_base_url=None,
        explicit_api_key=None, api_mode="chat_completions",
    )

    assert managed_route, "the managed endpoint was never consulted"
    assert client is not None, "alias with no base_url must still resolve to the managed server"
    assert str(client.base_url).startswith(MANAGED_BASE), str(client.base_url)
    assert str(client.api_key) == MANAGED_KEY
    assert model == MODEL


def test_managed_key_wins_over_ambient_openai_key(managed_route, monkeypatch):
    """A local alias never borrows OPENAI_API_KEY — that secret is not for this host."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-must-not-be-used")

    client, _ = aux.resolve_provider_client(
        "llamacpp", model=MODEL, raw_codex=True, explicit_base_url=None,
        explicit_api_key=None, api_mode="chat_completions",
    )

    assert str(client.api_key) == MANAGED_KEY


def test_explicit_base_url_wins_over_managed_endpoint(managed_route):
    """Pointing at a specific server means that server; the managed route must not be consulted."""
    client, _ = aux.resolve_provider_client(
        "llamacpp", model=MODEL, raw_codex=True, explicit_base_url="http://127.0.0.1:1234/v1",
        explicit_api_key="explicit-key", api_mode="chat_completions",
    )

    assert str(client.base_url).startswith("http://127.0.0.1:1234/v1")
    assert managed_route == [], "an explicit base_url must not trigger a managed lookup"


def test_no_managed_endpoint_delegates_to_the_normal_fallthrough(monkeypatch):
    """Server off: delegate instead of inventing an endpoint (nothing is running to route to)."""
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint",
                        lambda **kw: None)
    calls = []

    def _fake_custom_endpoint():
        calls.append("custom_endpoint")
        return MagicMock(name="fallthrough-client"), "fallthrough-model"

    monkeypatch.setattr(aux, "_try_custom_endpoint", _fake_custom_endpoint)
    monkeypatch.setattr(aux, "_resolve_api_key_provider", lambda: (None, None))

    _, model = aux.resolve_provider_client(
        "llamacpp", model=MODEL, raw_codex=True, explicit_base_url=None,
        explicit_api_key=None, api_mode="chat_completions",
    )

    assert calls == ["custom_endpoint"], "the normal fall-through must still run"
    assert model == MODEL, "the caller's model must survive delegation untouched"


def test_non_alias_custom_endpoint_unaffected(managed_route):
    """Bare `custom` keeps its OPENAI_BASE_URL semantics: no managed lookup, no rewriting."""
    client, _ = aux.resolve_provider_client(
        "custom", model="some-model", raw_codex=True, explicit_base_url="http://127.0.0.1:11434/v1",
        explicit_api_key="k", api_mode="chat_completions",
    )

    assert str(client.base_url).startswith("http://127.0.0.1:11434/v1")
    assert managed_route == []
