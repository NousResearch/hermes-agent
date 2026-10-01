"""Regression #130132: real Relay intercepts auxiliary clients' chat-shaped surface.

The fixture owns an isolated managed Relay turn and releases it after each test.
Native mode selects the client, not Relay's codec; sync, async and streaming calls
must preserve answers and close inner awaitable streams without provider replay.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import auxiliary_client, relay_runtime


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    """A real Relay conversation turn with managed execution retained for its duration."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    consumer = "test.moa-reference-relay-boundary"
    lease.host.retain_managed_execution(consumer)
    try:
        yield lease.host.relay, turn
    finally:
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _completed(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _shim_client(content):
    """The public surface every auxiliary Responses/Codex client exposes; its wire is native."""
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: _completed(content))
        )
    )


def test_native_reference_boundary_is_chat_shaped_through_real_relay(relay_turn):
    """Sync, async, and streaming reference calls all surface to Relay as the chat shape it
    codecs.  Labelled ``codex_responses``, the Responses codec would fail to decode the
    ``messages`` body instead of returning the advisor's answer."""
    _relay, _turn = relay_turn

    @auxiliary_client._relay_auxiliary_call
    def run_sync(task):
        auxiliary_client._set_relay_auxiliary_route(
            "xai-oauth", "grok-4.7", "codex_responses"
        )
        return auxiliary_client._relay_sync_completion(
            _shim_client("sync-advisor"),
            {"model": "grok-4.7", "messages": [{"role": "user", "content": "hi"}]},
            provider="xai-oauth",
            api_mode="codex_responses",
        )

    @auxiliary_client._relay_auxiliary_call
    def run_stream(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openai-codex", "gpt-6.1-sol", "codex_responses"
        )
        return auxiliary_client._relay_sync_stream(
            _shim_client("stream-advisor"),
            {
                "model": "gpt-6.1-sol",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            provider="openai-codex",
            api_mode="codex_responses",
        )

    async def run_async_advisor():
        async def create(**_kwargs):
            return _completed("async-advisor")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        @auxiliary_client._relay_auxiliary_call_async
        async def _run(task):
            auxiliary_client._set_relay_auxiliary_route(
                "openai-codex", "gpt-6.1-sol", "codex_responses"
            )
            return await auxiliary_client._relay_async_completion(
                client,
                {
                    "model": "gpt-6.1-sol",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                provider="openai-codex",
                api_mode="codex_responses",
            )

        return (await _run("moa_reference")).choices[0].message.content

    observed = {
        "sync": run_sync("moa_reference").choices[0].message.content,
        "stream": run_stream("moa_aggregator").choices[0].message.content,
        "async": asyncio.run(run_async_advisor()),
    }

    assert observed == {
        "sync": "sync-advisor",
        "stream": "stream-advisor",
        "async": "async-advisor",
    }


class _DeltaStream:
    """An async token stream of chat deltas that records close."""

    def __init__(self, contents):
        self._contents = list(contents)
        self._index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._contents):
            raise StopAsyncIteration
        content = self._contents[self._index]
        self._index += 1
        return SimpleNamespace(
            model="native-actor",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        )

    async def aclose(self):
        self.closed = True


def test_inner_native_awaitable_stream_adapts_inside_real_relay(relay_turn):
    """The inner native client answers a stream request with an awaitable; Relay's provider
    callback iterates synchronously, so the adapter must resolve it to a token stream on one
    owning loop and release the source on close."""
    _relay, _turn = relay_turn

    completed_dispatches = []

    async def create_completed(**_kwargs):
        completed_dispatches.append(1)
        return _completed("native-completed")

    completed_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_completed))
    )

    @auxiliary_client._relay_auxiliary_call
    def run_completed(task):
        auxiliary_client._set_relay_auxiliary_route(
            "native", "native-completed", "chat_completions"
        )
        return auxiliary_client._relay_sync_stream(
            completed_client,
            {
                "model": "native-completed",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            provider="native",
            api_mode="chat_completions",
        )

    result = run_completed("moa_aggregator")
    assert result.choices[0].message.content == "native-completed"
    assert completed_dispatches == [1]

    source = _DeltaStream(["native-", "actor"])
    stream_dispatches = []

    async def create_stream(**_kwargs):
        stream_dispatches.append(1)
        return source

    stream_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_stream))
    )

    @auxiliary_client._relay_auxiliary_call
    def run_stream(task):
        auxiliary_client._set_relay_auxiliary_route(
            "native", "native-actor", "chat_completions"
        )
        return auxiliary_client._relay_sync_stream(
            stream_client,
            {
                "model": "native-actor",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            provider="native",
            api_mode="chat_completions",
        )

    managed = run_stream("moa_aggregator")
    first = next(managed)
    assert first.choices[0].delta.content == "native-"
    managed.close()
    assert stream_dispatches == [1]
    assert source.closed is True
