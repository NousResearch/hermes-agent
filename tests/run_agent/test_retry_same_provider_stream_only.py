"""stream_only_base_urls must survive same-provider retry paths.

Regression: the initial auxiliary attempt routes through _create_with_progress
with force_stream computed from auxiliary.stream_only_base_urls, but the
same-provider recovery retry rebuilt the request WITHOUT that wrapper — so a
stream-only endpoint (gateway behind a short proxy read timeout) silently
fell back to non-streaming on every retry, reintroducing the exact hangs the
setting exists to prevent.
"""
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as ac


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


def _fake_client(sink, base_url="https://inference-api.nousresearch.com/v1"):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions(sink)),
        base_url=base_url,
    )


def test_retry_same_provider_sync_streams_for_stream_only_base_url(monkeypatch):
    """A stream-only base_url forces stream=True on the same-provider sync retry."""
    captured = []
    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **k: (_fake_client(captured), "test-model"),
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"auxiliary": {"stream_only_base_urls": ["inference-api.nousresearch.com"]}},
    )

    ac._retry_same_provider_sync(
        task="compression",
        resolved_provider="custom",
        resolved_model="test-model",
        resolved_base_url="https://inference-api.nousresearch.com/v1",
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

    assert captured, "retry never reached the client"
    assert captured[0].get("stream") is True, (
        "stream-only base_url downgraded to non-streaming on the retry path"
    )


def test_retry_same_provider_sync_nonstream_for_unlisted_base_url(monkeypatch):
    """Base URLs not in stream_only_base_urls keep the plain call."""
    captured = []
    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **k: (_fake_client(captured, "https://opencode.ai/zen/v1"), "test-model"),
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)

    ac._retry_same_provider_sync(
        task="compression",
        resolved_provider="custom",
        resolved_model="test-model",
        resolved_base_url="https://opencode.ai/zen/v1",
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

    assert captured and captured[0].get("stream") is not True
