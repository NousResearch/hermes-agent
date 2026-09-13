"""Contract of agent.api_credential and the client rebuild paths that depend on it.

A credential is a string or a token provider (``key_cmd``, Azure Entra ID). These tests
pin the behaviour call sites rely on: a provider is never stringified, never dropped when a
client is rebuilt, and handed to ``AsyncOpenAI`` in the awaitable shape it requires.
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI, OpenAI

from agent.api_credential import (
    async_credential,
    bearer_headers,
    client_credential,
    has_credential,
    is_token_provider,
    materialize_credential,
    normalize_credential,
    static_credential,
)
from agent.command_token_source import CommandTokenSource, materialize_probe_api_key


class _Provider:
    """A sync provider that counts how often it is asked to mint."""

    def __init__(self, token="sk-minted"):
        self.token = token
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.token


def _boom():
    raise RuntimeError("helper printed sk-secret-in-message")


# --- shape and presence --------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("sk-abc", False), ("", False), (None, False), (_Provider(), True), (lambda: "t", True),
])
def test_is_token_provider(value, expected):
    assert is_token_provider(value) is expected


def test_normalize_strips_strings_passes_providers_and_drops_the_rest():
    p = _Provider()
    assert normalize_credential("  sk-abc \n") == "sk-abc"
    assert normalize_credential(p) is p
    assert p.calls == 0, "normalizing must not mint"
    assert normalize_credential(None) == ""
    assert normalize_credential(object()) == ""


def test_has_credential_does_not_mint():
    p = _Provider()
    assert has_credential(p) and p.calls == 0
    assert has_credential("sk-abc")
    assert not has_credential("   ") and not has_credential(None) and not has_credential(42)


# --- turning a credential into a string ------------------------------------------------------

def test_materialize_mints_from_a_provider():
    assert materialize_credential(_Provider("  sk-x ")) == "sk-x"
    assert materialize_credential(" sk-y ") == "sk-y"


@pytest.mark.parametrize("value", [_boom, _Provider(None), _Provider(123), object(), None])
def test_materialize_degrades_to_empty_never_to_a_repr(value):
    assert materialize_credential(value) == ""


def test_materialize_does_not_log_the_provider_error_message(caplog):
    caplog.set_level("DEBUG", logger="agent.api_credential")
    materialize_credential(_boom)
    assert "sk-secret" not in caplog.text


def test_materialize_refuses_an_async_provider_instead_of_leaking_a_coroutine():
    async def provider():
        return "sk-async"
    assert materialize_credential(provider) == ""


def test_static_credential_never_mints():
    p = _Provider()
    assert static_credential(p) == "" and p.calls == 0
    assert static_credential(" sk-abc ") == "sk-abc"


def test_bearer_headers_for_both_shapes_and_the_empty_case():
    assert bearer_headers("sk-abc") == {"Authorization": "Bearer sk-abc"}
    assert bearer_headers(_Provider("sk-p")) == {"Authorization": "Bearer sk-p"}
    assert bearer_headers(_boom) == {}
    assert bearer_headers("") == {}
    assert bearer_headers("k", header="api-key", scheme="") == {"api-key": "k"}


def test_a_key_cmd_source_is_never_sent_as_its_repr():
    source = CommandTokenSource("printf sk-from-cmd", "custom")
    headers = bearer_headers(source)
    assert headers == {"Authorization": "Bearer sk-from-cmd"}
    assert "object at" not in headers["Authorization"]


def test_legacy_probe_name_is_the_same_function():
    assert materialize_probe_api_key is materialize_credential


# --- reading the credential back off an SDK client -----------------------------------------

def test_client_credential_recovers_the_provider_the_sdk_hid():
    p = _Provider()
    client = OpenAI(api_key=p, base_url="http://127.0.0.1:9/v1")
    assert client.api_key == "", "precondition: the SDK stores providers out of sight"
    assert client_credential(client) is p


def test_client_credential_reads_a_static_key():
    client = OpenAI(api_key="sk-static", base_url="http://127.0.0.1:9/v1")
    assert client_credential(client) == "sk-static"


def test_client_credential_does_not_mistake_a_mock_for_a_provider():
    client = MagicMock()
    client.api_key = "sk-mock"
    assert client_credential(client) == "sk-mock"


# --- the async shape ----------------------------------------------------------------------

def test_async_credential_wraps_a_sync_provider_off_the_event_loop():
    loop_thread = []
    minted_in = []

    def provider():
        minted_in.append(threading.get_ident())
        return "sk-async-ok"

    async def run():
        loop_thread.append(threading.get_ident())
        return await async_credential(provider)()

    assert asyncio.run(run()) == "sk-async-ok"
    assert minted_in and minted_in[0] != loop_thread[0], "minting must not block the event loop"


def test_async_credential_passes_strings_and_async_providers_through():
    async def provider():
        return "sk-a"
    assert async_credential(" sk-s ") == "sk-s"
    assert async_credential(provider) is provider


def test_async_openai_accepts_the_wrapped_provider():
    client = AsyncOpenAI(api_key=async_credential(_Provider("sk-for-async")), base_url="http://127.0.0.1:9/v1")
    asyncio.run(client._refresh_api_key())
    assert client.api_key == "sk-for-async"


# --- regression: client rebuild paths keep the provider --------------------------------------

def test_async_auxiliary_client_keeps_a_key_cmd_provider():
    """_to_async_client copied ``sync_client.api_key`` — ``""`` for a provider client —
    so every async auxiliary call (vision above all) went out without a credential."""
    from agent.auxiliary_client import _to_async_client

    sync_client = OpenAI(api_key=_Provider("sk-aux"), base_url="http://127.0.0.1:9/v1")
    async_client, _model = _to_async_client(sync_client, "some-model")
    asyncio.run(async_client._refresh_api_key())
    assert async_client.api_key == "sk-aux"


def test_routed_client_kwargs_keep_a_key_cmd_provider():
    from agent.agent_init import _client_kwargs_from_routed

    p = _Provider("sk-routed")
    kwargs = _client_kwargs_from_routed(OpenAI(api_key=p, base_url="http://127.0.0.1:9/v1"), None)
    assert kwargs["api_key"] is p


def test_codex_auxiliary_wrapper_exposes_the_real_credential():
    from agent.auxiliary_client import CodexAuxiliaryClient

    p = _Provider()
    wrapper = CodexAuxiliaryClient(OpenAI(api_key=p, base_url="http://127.0.0.1:9/v1"), "gpt-x")
    assert wrapper.api_key is p


# --- regression: auxiliary resolution passes an explicit provider through uncalled -----------

def _spy_client_factory(monkeypatch):
    import agent.auxiliary_client as ac
    from types import SimpleNamespace

    seen = {}

    def _spy(*, api_key, base_url, **kw):
        seen["api_key"] = api_key
        return SimpleNamespace(api_key=api_key, base_url=base_url)

    monkeypatch.setattr(ac, "_create_openai_client", _spy)
    return ac, seen


def test_explicit_custom_endpoint_keeps_a_provider(monkeypatch):
    """``(req.explicit_api_key or "").strip()`` raised AttributeError on a provider."""
    ac, seen = _spy_client_factory(monkeypatch)
    p = _Provider()
    ac.resolve_provider_client("custom", model="m1", explicit_base_url="https://example.invalid/v1",
                               explicit_api_key=p, api_mode="chat_completions")
    assert seen.get("api_key") is p and p.calls == 0


def test_named_custom_provider_keeps_an_explicit_provider(monkeypatch):
    from hermes_cli import runtime_provider as rp

    ac, seen = _spy_client_factory(monkeypatch)
    monkeypatch.setattr(rp, "_get_named_custom_provider",
                        lambda name: {"name": "dbx", "base_url": "https://example.invalid/v1", "model": "m1"}
                        if name == "dbx" else None)
    p = _Provider()
    ac.resolve_provider_client("dbx", explicit_api_key=p)
    assert seen.get("api_key") is p and p.calls == 0


def test_main_runtime_provider_is_reused_not_stringified(monkeypatch):
    """``str(main_runtime.get("api_key"))`` turned the provider into its repr."""
    ac, seen = _spy_client_factory(monkeypatch)
    p = _Provider()
    ac.resolve_provider_client("custom", model="m1", api_mode="chat_completions",
                               main_runtime={"base_url": "https://example.invalid/v1", "api_key": p})
    key = seen.get("api_key")
    assert key is p, f"expected the provider itself, got {key!r}"
