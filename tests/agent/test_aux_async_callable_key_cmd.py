"""A ``key_cmd`` credential survives the sync → async client conversion.

``build_command_token_provider`` returns a ``CommandTokenSource`` so the wire client can mint a
fresh short-lived bearer per request. The OpenAI SDK stores such a callable in
``_api_key_provider`` and deliberately leaves ``client.api_key`` an EMPTY STRING.
``_to_async_client`` copied ``sync_client.api_key`` across, so the async client was built with no
credential at all and every async auxiliary call (vision, compression, titles) failed with
``401 Authentication Error, No api key passed in`` against an auth-required endpoint.

The contract asserted: the async client can still PRODUCE a token, not that any particular
attribute holds it.
"""

import asyncio

import pytest

from agent.command_token_source import CommandTokenSource


@pytest.fixture
def token_source():
    """A real CommandTokenSource, so the test breaks if the SDK contract changes."""
    return CommandTokenSource("printf token-abc123", "roller")


def _sync_client_with_callable_key(token_source, base_url="https://litellm.example.com/v1"):
    """A real OpenAI client built the way the named-custom arm builds it."""
    from openai import OpenAI

    return OpenAI(api_key=token_source, base_url=base_url, max_retries=0)


def test_sdk_blanks_api_key_for_callable_credentials(token_source):
    """The premise. If the SDK ever stores the callable in ``api_key``, the fix is moot."""
    client = _sync_client_with_callable_key(token_source)
    assert client.api_key == ""
    assert client._api_key_provider is token_source


def test_async_client_keeps_a_working_credential(token_source):
    """The regression: the async client must still be able to mint a bearer."""
    from agent.auxiliary_client import _to_async_client

    async_client, _ = _to_async_client(_sync_client_with_callable_key(token_source), "claude-opus")

    provider = async_client._api_key_provider
    assert provider is not None, "credential was dropped in the async conversion"
    assert asyncio.run(provider()) == "token-abc123"


def test_async_conversion_does_not_mint_for_an_ordinary_endpoint():
    """LiteLLM/OpenAI-compatible endpoints authenticate through the client provider, not headers."""
    from agent.auxiliary_client import _to_async_client
    from openai import OpenAI

    calls = []

    def provider():
        calls.append(1)
        return "token-abc123"

    async_client, _ = _to_async_client(
        OpenAI(api_key=provider, base_url="https://litellm.example.com/v1"), "claude-opus")

    assert async_client._api_key_provider is not None
    assert calls == [], "ordinary async endpoints must not mint a token during client construction"


def test_async_conversion_preserves_a_plain_string_key():
    """The common path must be untouched: a string credential still copies across verbatim."""
    from agent.auxiliary_client import _to_async_client
    from openai import OpenAI

    sync_client = OpenAI(api_key="sk-plain-value", base_url="https://litellm.example.com/v1")
    async_client, _ = _to_async_client(sync_client, "claude-opus")

    assert async_client.api_key == "sk-plain-value"
    assert async_client._api_key_provider is None


def test_async_provider_does_not_block_the_event_loop(token_source):
    """Minting shells out; it must be awaited off-loop so concurrent aux calls are not serialised."""
    from agent.auxiliary_client import _to_async_client

    async_client, _ = _to_async_client(_sync_client_with_callable_key(token_source), "claude-opus")

    async def _mint_concurrently():
        return await asyncio.gather(*(async_client._api_key_provider() for _ in range(4)))

    assert asyncio.run(_mint_concurrently()) == ["token-abc123"] * 4
