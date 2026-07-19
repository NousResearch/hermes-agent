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
    from agent.gemini_native_adapter import (
        AsyncGeminiNativeClient,
        GeminiNativeClient,
    )

    pairs = [
        (ac.CodexAuxiliaryClient, ac.AsyncCodexAuxiliaryClient),
        (ac.AnthropicAuxiliaryClient, ac.AsyncAnthropicAuxiliaryClient),
        (ac.BedrockAuxiliaryClient, ac.AsyncBedrockAuxiliaryClient),
        (GeminiNativeClient, AsyncGeminiNativeClient),
    ]
    for sync_cls, async_cls in pairs:
        assert hasattr(sync_cls, "close")
        assert hasattr(async_cls, "close"), f"{async_cls.__name__} lost close()"


# -- native Gemini: close() is a coroutine, so it needs a sync cache route ----


def _cached_gemini_pair():
    """A native-Gemini async wrapper parked in the client cache."""
    from agent.gemini_native_adapter import (
        AsyncGeminiNativeClient,
        GeminiNativeClient,
    )

    sync = GeminiNativeClient(api_key="unit-test-key")
    wrapper = AsyncGeminiNativeClient(sync)
    with ac._client_cache_lock:
        ac._client_cache.clear()
        ac._client_cache[("gemini", "native")] = (wrapper, "gemini-2.0", None)
    return sync, wrapper


def test_native_gemini_cached_wrapper_closes_on_shutdown():
    """``shutdown_cached_clients`` must release the native-Gemini transport.

    ``AsyncGeminiNativeClient.close()`` is a coroutine, so the canonical
    closer's ``iscoroutinefunction`` guard skips it, and the shim carries no
    ``_client`` for ``_force_close_async_httpx`` to find. Without a synchronous
    route the transport survives shutdown — the same leak this PR fixes for the
    Codex/Anthropic/Bedrock shims, reached by a different path.
    """
    sync, _ = _cached_gemini_pair()
    try:
        ac.shutdown_cached_clients()
    finally:
        with ac._client_cache_lock:
            ac._client_cache.clear()

    assert sync.is_closed, "native-Gemini transport leaked past cache shutdown"


def test_native_gemini_cached_wrapper_closes_on_credential_eviction():
    """``_evict_cached_clients`` fires on every credential refresh.

    Each refresh must hand back the transport it replaces, otherwise a
    long-running gateway accumulates one leaked native-Gemini connection per
    rotation (#10200).
    """
    sync, _ = _cached_gemini_pair()
    try:
        ac._evict_cached_clients("gemini")
    finally:
        with ac._client_cache_lock:
            ac._client_cache.clear()

    assert sync.is_closed, "native-Gemini transport leaked past credential eviction"


def test_native_gemini_async_close_still_awaits():
    """The synchronous cache route must not disturb the public async close()."""
    import asyncio

    from agent.gemini_native_adapter import (
        AsyncGeminiNativeClient,
        GeminiNativeClient,
    )

    sync = GeminiNativeClient(api_key="unit-test-key")
    asyncio.run(AsyncGeminiNativeClient(sync).close())

    assert sync.is_closed


def test_every_cached_async_wrapper_has_a_synchronous_close_route():
    """The closer runs from sync callers, so an async-only close() is unreachable.

    Generalises the Gemini case: any wrapper that can be cached must either
    expose a synchronous ``close()`` or mirror a leaf whose ``close()`` is
    synchronous, or ``_close_cached_client`` silently does nothing to it.
    """
    from agent.gemini_native_adapter import AsyncGeminiNativeClient

    wrappers = [
        ac.AsyncCodexAuxiliaryClient,
        ac.AsyncAnthropicAuxiliaryClient,
        ac.AsyncBedrockAuxiliaryClient,
        AsyncGeminiNativeClient,
    ]
    for cls in wrappers:
        close_fn = getattr(cls, "close", None)
        assert close_fn is not None, f"{cls.__name__} has no close()"
        if not inspect.iscoroutinefunction(close_fn):
            continue
        # Coroutine close() → the sync closer skips it, so the wrapper must
        # mirror its leaf as _real_client for the fallback to reach.
        assert "_real_client" in inspect.getsource(cls), (
            f"{cls.__name__}.close() is async and the class does not mirror a "
            "_real_client leaf — _close_cached_client cannot release it"
        )
