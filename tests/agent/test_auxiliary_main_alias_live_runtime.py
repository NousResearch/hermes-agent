"""Regression coverage for the ``provider: main`` alias under a live runtime.

Production sequence (context-overflow crash, 2026-07-24): the main agent runs
on a Codex Responses custom endpoint bound via ``set_runtime_main(
provider='custom', requested_provider='custom:codex-lb',
api_mode='codex_responses')``.  ``auxiliary.compression.provider: main`` then
resolves through the anonymous-custom arm, which adopted the runtime's
base_url + api_key but dropped its api_mode — building a plain Chat
Completions client against a Responses-only endpoint.  The compression
summary call then 400s (``Unknown parameter: 'reasoning.enabled'``) and the
session dies with "Context length exceeded ... Cannot compress further."

Separately, the client cache keyed ``main`` entries without any live-runtime
identity, so a client built under one runtime shape (endpoint/api_mode/key)
survived a mid-session runtime switch and was reused for a different one.
"""

from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_client import CodexAuxiliaryClient


@pytest.fixture(autouse=True)
def _clean_aux_state():
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    yield
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()


def _codex_runtime(model: str = "gpt-5.6-sol") -> dict:
    return {
        "provider": "custom",
        "requested_provider": "custom:codex-lb",
        "model": model,
        "base_url": "http://codex-lb.test:2455/v1",
        "api_key": "test-key-not-a-secret",
        "api_mode": "codex_responses",
        "auth_mode": "api_key",
    }


def _chat_runtime(model: str = "gpt-5.6-sol") -> dict:
    return {
        "provider": "custom",
        "requested_provider": "custom:chat-lb",
        "model": model,
        "base_url": "http://chat-lb.test:8000/v1",
        "api_key": "other-key-not-a-secret",
        "api_mode": "chat_completions",
        "auth_mode": "api_key",
    }


def test_main_alias_adopts_live_runtime_codex_api_mode():
    """The production sequence: compression on ``main`` over a live Codex runtime.

    A ``provider: main`` auxiliary client must speak the same wire as the
    live main runtime it inherits the endpoint from — a codex_responses
    endpoint requires the Codex Responses adapter, not plain chat.completions.
    """
    aux.set_runtime_main(**_codex_runtime())

    client, model = aux._get_cached_client("main", "gpt-5.6-sol")

    assert client is not None
    assert model == "gpt-5.6-sol"
    assert isinstance(client, CodexAuxiliaryClient), (
        "provider=main with a live codex_responses runtime must build the "
        f"Codex Responses adapter, got {type(client).__name__}"
    )
    assert "codex-lb.test" in str(client._real_client.base_url)


def test_main_alias_client_not_reused_across_runtime_shape_change():
    """A ``main`` client cached under one live runtime dies with that runtime."""
    aux.set_runtime_main(**_chat_runtime())
    first_client, _ = aux._get_cached_client("main", "gpt-5.6-sol")
    assert first_client is not None
    assert not isinstance(first_client, CodexAuxiliaryClient)
    assert "chat-lb.test" in str(first_client.base_url)

    aux.set_runtime_main(**_codex_runtime())
    second_client, _ = aux._get_cached_client("main", "gpt-5.6-sol")

    assert second_client is not first_client, (
        "a provider=main client built for one endpoint/api_mode must not be "
        "reused after the live main runtime changed"
    )
    assert isinstance(second_client, CodexAuxiliaryClient)
    assert "codex-lb.test" in str(second_client._real_client.base_url)


def test_main_alias_client_reused_when_runtime_unchanged():
    """Unchanged live runtime keeps hitting the same cached ``main`` client."""
    aux.set_runtime_main(**_codex_runtime())
    first_client, _ = aux._get_cached_client("main", "gpt-5.6-sol")
    second_client, _ = aux._get_cached_client("main", "gpt-5.6-sol")

    assert first_client is not None
    assert second_client is first_client


def test_main_alias_cache_key_covers_live_runtime_surface():
    """Endpoint/credential/wire/identity changes each re-key ``main`` clients."""
    base = _codex_runtime()
    variants = [
        {**base, "provider": "openrouter"},
        {**base, "base_url": "https://other.test/v1"},
        {**base, "api_key": "rotated-key"},
        {**base, "api_mode": "chat_completions"},
        {**base, "model": "gpt-5.7"},
    ]

    aux.set_runtime_main(**base)
    baseline = aux._client_cache_key("main", async_mode=False, model="gpt-5.6-sol")
    repeat = aux._client_cache_key("main", async_mode=False, model="gpt-5.6-sol")
    assert repeat == baseline

    keys = []
    for variant in variants:
        aux.set_runtime_main(**variant)
        keys.append(aux._client_cache_key("main", async_mode=False, model="gpt-5.6-sol"))

    assert all(key != baseline for key in keys)


def test_main_alias_cache_key_never_carries_raw_secret():
    """Runtime credentials discriminate the key without living in it."""
    secret = "super-secret-main-runtime-key"
    aux.set_runtime_main(**{**_codex_runtime(), "api_key": secret})

    key = aux._client_cache_key("main", async_mode=False, model="gpt-5.6-sol")

    assert secret not in repr(key)


def test_explicit_api_mode_override_wins_over_runtime_api_mode():
    """A caller-pinned api_mode is honoured even when the runtime differs."""
    aux.set_runtime_main(**_codex_runtime())

    client, _ = aux._get_cached_client(
        "main", "gpt-5.6-sol", api_mode="chat_completions"
    )

    assert client is not None
    assert not isinstance(client, CodexAuxiliaryClient)
