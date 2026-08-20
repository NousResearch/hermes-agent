"""``_retry_same_provider_async`` must forward ``main_runtime`` like its sync twin.

The auth-recovery retry rebuilds the provider client after refreshing
credentials. ``_retry_same_provider_sync`` passes ``main_runtime`` to
``_get_cached_client``; the async twin did not, even though ``main_runtime`` is
in scope at both async call sites (the very next statement there uses it for
``_recoverable_pool_provider``).

Without it the rebuilt client resolves against an EMPTY runtime
(``_normalize_main_runtime(None) -> {}``), dropping the main agent's
provider/model/base_url/api_key. ``main_runtime`` is also part of the client
cache key, so the retry cannot reuse the correct cached client either — the
recovery attempt targets the wrong endpoint and fails, defeating the refresh
that just succeeded.
"""
import asyncio
import inspect

import pytest

import agent.auxiliary_client as ac

_MAIN_RUNTIME = {
    "provider": "openai-compatible",
    "model": "main-model",
    "base_url": "https://main.endpoint/v1",
    "api_key": "sk-MAIN",
    "api_mode": "chat_completions",
}

_COMMON = dict(
    task="compression",
    resolved_provider="auto",
    resolved_model="m",
    resolved_base_url=None,
    resolved_api_key=None,
    resolved_api_mode=None,
    final_model="m",
    messages=[{"role": "user", "content": "x"}],
    temperature=None,
    max_tokens=None,
    tools=None,
    effective_timeout=30.0,
    effective_extra_body={},
    reasoning_config=None,
)


class _Resp:
    pass


def _install_stubs(monkeypatch, seen, *, is_async):
    class _Client:
        base_url = "https://main.endpoint/v1"

        class chat:
            class completions:
                if is_async:
                    @staticmethod
                    async def create(**kwargs):
                        return _Resp()
                else:
                    @staticmethod
                    def create(**kwargs):
                        return _Resp()

    def _fake_get_cached_client(provider, model=None, **kwargs):
        seen["main_runtime"] = kwargs.get("main_runtime")
        return _Client(), model

    monkeypatch.setattr(ac, "_get_cached_client", _fake_get_cached_client)
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task: resp)
    monkeypatch.setattr(ac, "_build_call_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(ac, "_is_anthropic_compat_endpoint", lambda *a, **k: False)


def test_async_retry_accepts_main_runtime():
    """The async twin must expose the same knob as the sync one."""
    assert "main_runtime" in inspect.signature(ac._retry_same_provider_async).parameters


def test_sync_retry_forwards_main_runtime(monkeypatch):
    seen = {}
    _install_stubs(monkeypatch, seen, is_async=False)

    ac._retry_same_provider_sync(main_runtime=_MAIN_RUNTIME, **_COMMON)

    assert seen["main_runtime"] == _MAIN_RUNTIME


def test_async_retry_forwards_main_runtime(monkeypatch):
    """Regression: the async retry dropped it and rebuilt against an empty runtime."""
    seen = {}
    _install_stubs(monkeypatch, seen, is_async=True)

    asyncio.run(ac._retry_same_provider_async(main_runtime=_MAIN_RUNTIME, **_COMMON))

    assert seen["main_runtime"] == _MAIN_RUNTIME, (
        "async retry rebuilt the client without the main runtime; the recovery "
        "call targets the wrong endpoint"
    )


def test_dropping_main_runtime_changes_client_resolution():
    """Why it matters: the omitted value drives both construction and the cache key."""
    assert ac._normalize_main_runtime(_MAIN_RUNTIME) != ac._normalize_main_runtime(None)
    assert ac._normalize_main_runtime(None) == {}

    key_with = ac._client_cache_key(
        "auto", async_mode=True, base_url=None, api_key=None, api_mode=None,
        main_runtime=_MAIN_RUNTIME, is_vision=False, task="compression", model=None,
    )
    key_without = ac._client_cache_key(
        "auto", async_mode=True, base_url=None, api_key=None, api_mode=None,
        main_runtime=None, is_vision=False, task="compression", model=None,
    )
    assert key_with != key_without
