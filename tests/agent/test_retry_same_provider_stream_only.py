"""stream_only_base_urls must survive same-provider retry paths.

Regression: the initial auxiliary attempt routes through _create_with_progress
with force_stream computed from auxiliary.stream_only_base_urls, but the
same-provider recovery retry rebuilt the request WITHOUT that wrapper — so a
stream-only endpoint (gateway behind a short proxy read timeout) silently
fell back to non-streaming on every retry, reintroducing the exact hangs the
setting imposes streaming to prevent.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_client as ac

STREAM_URL = "https://inference-api.nousresearch.com/v1"
PLAIN_URL = "https://opencode.ai/zen/v1"


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


class _Completions:
    def __init__(self, sink):
        self._sink = sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        return _response("ok")


class _AsyncCompletions:
    def __init__(self, sink):
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.append(kwargs)
        return _response("ok")


def _fake_client(sink, base_url=STREAM_URL):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions(sink)),
        base_url=base_url,
    )


def _fake_async_client(sink, base_url=STREAM_URL):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_AsyncCompletions(sink)),
        base_url=base_url,
    )


def _sync_kwargs(**overrides):
    kwargs = dict(
        task="compression",
        resolved_provider="custom",
        resolved_model="test-model",
        resolved_base_url=STREAM_URL,
        resolved_api_key="k",
        resolved_api_mode="chat_completions",
        main_runtime=None,
        final_model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=None,
        max_tokens=None,
        tools=None,
        effective_timeout=30.0,
        effective_extra_body={},
        reasoning_config=None,
        extra_headers=None,
    )
    kwargs.update(overrides)
    return kwargs


def _stream_only_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"auxiliary": {"stream_only_base_urls": ["inference-api.nousresearch.com"]}},
    )


def test_retry_same_provider_sync_streams_for_stream_only_base_url(monkeypatch):
    """A stream-only base_url forces stream=True on the same-provider sync retry."""
    captured = []
    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **k: (_fake_client(captured), "test-model"),
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)
    _stream_only_config(monkeypatch)

    ac._retry_same_provider_sync(**_sync_kwargs())

    assert captured, "retry never reached the client"
    assert captured[0].get("stream") is True, (
        "stream-only base_url downgraded to non-streaming on the retry path"
    )


def test_retry_same_provider_sync_nonstream_for_unlisted_base_url(monkeypatch):
    """Base URLs not in stream_only_base_urls keep the plain call."""
    captured = []
    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **k: (_fake_client(captured, PLAIN_URL), "test-model"),
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)
    _stream_only_config(monkeypatch)

    ac._retry_same_provider_sync(**_sync_kwargs(resolved_base_url=PLAIN_URL))

    assert captured and captured[0].get("stream") is not True


@pytest.mark.asyncio
async def test_retry_same_provider_async_streams_for_stream_only_base_url(monkeypatch):
    """Async twin: a stream-only base_url forces the streamed create on retry."""
    captured = []
    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **k: (_fake_async_client(captured), "test-model"),
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)
    _stream_only_config(monkeypatch)

    await ac._retry_same_provider_async(**_sync_kwargs())

    assert captured, "retry never reached the client"
    assert captured[0].get("stream") is True, (
        "stream-only base_url downgraded to non-streaming on the async retry path"
    )


@pytest.mark.asyncio
async def test_retry_same_provider_async_nonstream_for_unlisted_base_url(monkeypatch):
    """Async twin: unlisted URLs keep the plain non-streaming call."""
    captured = []
    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **k: (_fake_async_client(captured, PLAIN_URL), "test-model"),
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)
    _stream_only_config(monkeypatch)

    await ac._retry_same_provider_async(**_sync_kwargs(resolved_base_url=PLAIN_URL))

    assert captured and captured[0].get("stream") is not True


@pytest.mark.asyncio
async def test_retry_same_provider_async_skips_stream_for_native_clients(monkeypatch):
    """Native async adapters (Codex/Anthropic/Bedrock) never get OpenAI-style
    stream kwargs pushed at them, even on a stream-only base_url — mirrors the
    initial async path's exclusion tuple."""
    captured = []
    sync_wrapper = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions([])),
        api_key="k",
        base_url=STREAM_URL,
    )
    native = ac.AsyncCodexAuxiliaryClient(sync_wrapper)
    # Real native instance (isinstance exclusion applies); swap the wire
    # adapter for a recording async fake (setattr to keep type-checkers out).
    monkeypatch.setattr(
        native, "chat", SimpleNamespace(completions=_AsyncCompletions(captured))
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (native, "test-model"))
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)
    _stream_only_config(monkeypatch)

    with patch.object(ac, "_acreate_with_stream", new=AsyncMock()) as mock_stream:
        await ac._retry_same_provider_async(**_sync_kwargs())

    assert captured and captured[0].get("stream") is not True
    mock_stream.assert_not_called()
