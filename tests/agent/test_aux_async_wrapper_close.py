"""Cached async auxiliary wrappers must release their underlying client.

``_to_async_client`` builds ``AsyncCodexAuxiliaryClient`` /
``AsyncAnthropicAuxiliaryClient`` / ``AsyncBedrockAuxiliaryClient`` from a
*transient* sync wrapper — only the async shim is cached. The shims lacked a
``close()``, and ``_close_cached_client`` reaches a client two ways:

* ``getattr(client, "close", None)`` — absent on the shims, and
* ``_force_close_async_httpx`` — looks for ``client._client`` (the httpx client
  inside an ``AsyncOpenAI``), which a shim does not have.

So both ``_evict_cached_clients`` (fired on every credential refresh) and
``shutdown_cached_clients`` — documented as closing "all cached clients (sync
and async)" — were no-ops for these types, leaking the underlying transport.
"""
import inspect

import pytest

import agent.auxiliary_client as ac


class _FakeAnthropic:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class _FakeOpenAI:
    api_key = "k"
    base_url = "https://example.invalid/v1"

    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _codex_pair():
    real = _FakeOpenAI()
    sync = ac.CodexAuxiliaryClient(real, "m")
    return real, ac.AsyncCodexAuxiliaryClient(sync)


def _anthropic_pair():
    real = _FakeAnthropic()
    sync = ac.AnthropicAuxiliaryClient(real, "m", "k", "https://example.invalid/v1")
    return real, ac.AsyncAnthropicAuxiliaryClient(sync)


@pytest.mark.parametrize("factory", [_codex_pair, _anthropic_pair])
def test_closing_cached_async_wrapper_releases_underlying_client(factory):
    real, async_wrapper = factory()

    ac._close_cached_client(async_wrapper)

    assert real.closed == 1, (
        "closing the cached async wrapper left the underlying client open; "
        "eviction and shutdown leak the transport"
    )


@pytest.mark.parametrize("factory", [_codex_pair, _anthropic_pair])
def test_async_wrapper_close_is_not_a_coroutine(factory):
    """Cache eviction skips coroutine close(), so this must stay synchronous."""
    _real, async_wrapper = factory()

    assert not inspect.iscoroutinefunction(async_wrapper.close)


def test_bedrock_async_wrapper_answers_close_protocol():
    """Bedrock builds per-call, so close() is a no-op — but it must exist."""
    sync = ac.BedrockAuxiliaryClient("us-east-1", "m")
    async_wrapper = ac.AsyncBedrockAuxiliaryClient(sync)

    assert callable(async_wrapper.close)
    ac._close_cached_client(async_wrapper)  # must not raise


def test_async_wrappers_mirror_the_sync_close_surface():
    """Every sync wrapper's close() must have an async counterpart."""
    pairs = [
        (ac.CodexAuxiliaryClient, ac.AsyncCodexAuxiliaryClient),
        (ac.AnthropicAuxiliaryClient, ac.AsyncAnthropicAuxiliaryClient),
        (ac.BedrockAuxiliaryClient, ac.AsyncBedrockAuxiliaryClient),
    ]
    for sync_cls, async_cls in pairs:
        assert hasattr(sync_cls, "close")
        assert hasattr(async_cls, "close"), f"{async_cls.__name__} lost close()"
