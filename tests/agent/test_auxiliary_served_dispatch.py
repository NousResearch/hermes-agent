"""The completion seams name the dispatch that answered an auxiliary call, and nothing stale.

``observe_served_dispatch`` is what lets a multi-chunk compression follow the lane that rescued a
chunk: these checks drive the REAL seams (sync, async, Relay, stream) with local fake clients.
Synthetic placeholder credentials only, no network.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.auxiliary_relay_metadata import _SERVED_DISPATCH, _note_served_dispatch, observe_served_dispatch

HOST = "http://127.0.0.1:4/v1"
REQUEST = {"model": "lane/model", "messages": [{"role": "user", "content": "synthetic"}], "timeout": 45.0}


def _ok():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "synthetic answer"
    return response


def _client(api_key="synthetic-placeholder-b", base_url=HOST):
    client = MagicMock()
    client.base_url, client.api_key = base_url, api_key
    return client


LANE = {"provider": "custom", "model": "lane/model", "base_url": HOST, "api_key": "synthetic-placeholder-b",
        "api_mode": "chat_completions", "timeout": 45.0}


def test_the_sync_seam_records_the_client_that_answered():
    client = _client()
    with observe_served_dispatch() as observed:
        ac._relay_sync_completion(client, dict(REQUEST), provider="fallback_chain[1](custom)",
                                  api_mode="chat_completions", create=lambda request: _ok())
    assert observed == LANE


def test_the_async_seam_records_the_client_that_answered():
    async def create(request):
        return _ok()

    async def run():
        with observe_served_dispatch() as observed:
            await ac._relay_async_completion(_client(), dict(REQUEST), provider="custom",
                                             api_mode="chat_completions", create=create)
        return observed

    assert asyncio.run(run()) == LANE


def test_a_failed_dispatch_leaves_no_earlier_dispatch_behind():
    def refuse(request):
        raise RuntimeError("synthetic 402")

    with observe_served_dispatch() as observed:
        ac._relay_sync_completion(_client(), dict(REQUEST), provider="custom", create=lambda request: _ok())
        assert observed
        with pytest.raises(RuntimeError):
            ac._relay_sync_completion(_client("synthetic-placeholder-c"), dict(REQUEST), provider="custom",
                                      create=refuse)
        assert observed == {}, "the dispatch that failed must not leave the earlier one adoptable"


def test_a_streamed_answer_clears_the_observation():
    client = _client()
    client.chat.completions.create = MagicMock(return_value=iter(()))
    with observe_served_dispatch() as observed:
        ac._relay_sync_completion(_client(), dict(REQUEST), provider="custom", create=lambda request: _ok())
        ac._relay_sync_stream(client, dict(REQUEST), provider="custom")
        assert observed == {}


def test_relay_records_the_request_it_actually_sent():
    context = {"request_id": "synthetic-request", "task": "compression", "provider": "custom", "model": "lane/model"}

    def execute_current(kwargs, send, **_kw):
        return send(dict(kwargs, model="rewritten/model", timeout=90.0))

    token = ac._RELAY_AUX_CALL_CONTEXT.set(context)
    try:
        with patch("agent.relay_llm.execute_current", side_effect=execute_current), \
             patch("agent.auxiliary_hooks.run_with_aux_hooks", side_effect=lambda fn, **_kw: fn()), \
             observe_served_dispatch() as observed:
            ac._relay_sync_completion(_client(), dict(REQUEST), provider="custom", create=lambda request: _ok())
    finally:
        ac._RELAY_AUX_CALL_CONTEXT.reset(token)
    assert (observed["model"], observed["timeout"]) == ("rewritten/model", 90.0)


def test_served_dispatch_names_the_concrete_backend():
    tagged = _client("synthetic-placeholder-d", "https://api.anthropic.com")
    tagged._hermes_aux_effective_provider = "anthropic"
    untagged = _client(None)
    untagged._hermes_aux_effective_provider = ""
    cases = [((tagged, "fallback_chain[0](anthropic)"), "anthropic"), ((tagged, "auto"), "anthropic"),
             ((untagged, "auto"), None), ((untagged, None), None), ((tagged, "openrouter"), "openrouter")]
    for (client, provider), expected in cases:
        with observe_served_dispatch() as observed:
            sentinel = object()
            assert _note_served_dispatch(sentinel, client, {"model": "m"}, provider, None) is sentinel
        assert observed["provider"] == expected, (provider, observed)
    with observe_served_dispatch() as observed:
        _note_served_dispatch(None, MagicMock(base_url=HOST), {"model": "m"}, "custom", None)
    assert observed["api_key"] is None, "a non-string credential attribute is never recorded"


def test_a_dispatch_outside_an_observer_records_nothing():
    sentinel = object()
    assert ac._relay_sync_completion(_client(), dict(REQUEST), provider="custom", create=lambda request: sentinel) \
        is sentinel
    assert _SERVED_DISPATCH.get() is None
