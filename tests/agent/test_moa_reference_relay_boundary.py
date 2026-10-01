"""Regression: MoA reference advisors must reach Relay with a chat-shaped boundary.

A reference slot whose provider speaks the native Responses wire (e.g.
``xai-oauth:grok-4.7`` or ``openai-codex:gpt-6.1-sol``) resolves
``api_mode="codex_responses"``, so ``moa_loop._slot_runtime`` forwards it to
``call_llm``. That api_mode must route *client selection* (the Codex/Responses
side is adapted into a ``chat.completions`` shim), but the request/response body
Relay actually intercepts is always chat-shaped — the auxiliary client only
exposes ``.chat.completions.create()``.

Labelling the Relay boundary with the provider's native ``codex_responses`` made
``relay_llm`` pick ``OpenAIResponsesCodec`` and decode a ``messages`` body,
raising ``RuntimeError: invalid argument: OpenAI Responses request is missing
input`` and collapsing the advisor into a ``[failed: …]`` note.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import auxiliary_client, relay_llm, relay_runtime


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
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
    try:
        yield lease.host.relay, turn
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _codex_shim_client(content: str = "ok"):
    """The public surface every auxiliary Responses/Codex client exposes."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )


def test_codex_reference_relay_reports_chat_boundary(monkeypatch):
    """Relay metadata must describe the chat shape it intercepts, not the wire mode."""
    captured = {}

    def execute_current(request, callback, **kwargs):
        captured.update(kwargs)
        return callback(request)

    monkeypatch.setattr(relay_llm, "execute_current", execute_current)

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openai-codex", "gpt-6.1-sol", "codex_responses"
        )
        return auxiliary_client._relay_sync_completion(
            _codex_shim_client(),
            {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "hi"}]},
            provider="openai-codex",
            api_mode="codex_responses",
        )

    run("moa_reference")

    assert captured["metadata"]["api_mode"] == "chat_completions"


@pytest.mark.asyncio
async def test_codex_reference_relay_reports_chat_boundary_async(monkeypatch):
    """The async auxiliary twin reports the same chat boundary to Relay."""
    captured = {}

    async def execute_current_async(request, callback, **kwargs):
        captured.update(kwargs)
        return await callback(request)

    monkeypatch.setattr(relay_llm, "execute_current_async", execute_current_async)

    async def create(**_kwargs):
        return SimpleNamespace(choices=[])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    @auxiliary_client._relay_auxiliary_call_async
    async def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openai-codex", "gpt-6.1-sol", "codex_responses"
        )
        return await auxiliary_client._relay_async_completion(
            client,
            {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "hi"}]},
            provider="openai-codex",
            api_mode="codex_responses",
        )

    await run("moa_reference")

    assert captured["metadata"]["api_mode"] == "chat_completions"


def test_codex_reference_relay_call_completes(relay_turn):
    """The managed call must not die in the Responses codec on a messages body."""
    relay, turn = relay_turn
    consumer = "test.moa-reference-relay-boundary"
    turn.lease.host.retain_managed_execution(consumer)
    try:

        @auxiliary_client._relay_auxiliary_call
        def run(task):
            auxiliary_client._set_relay_auxiliary_route(
                "xai-oauth", "grok-4.7", "codex_responses"
            )
            return auxiliary_client._relay_sync_completion(
                _codex_shim_client("advisor"),
                {"model": "grok-4.7", "messages": [{"role": "user", "content": "hi"}]},
                provider="xai-oauth",
                api_mode="codex_responses",
            )

        result = run("moa_reference")
    finally:
        turn.lease.host.release_managed_execution(consumer)

    assert result.choices[0].message.content == "advisor"


def test_codex_stream_relay_reports_chat_boundary(monkeypatch):
    """Streaming siblings intercept the same chat surface as non-streaming calls."""
    captured = {}

    def stream_current(request, callback, **kwargs):
        captured.update(kwargs)
        return callback(request)

    monkeypatch.setattr(relay_llm, "stream_current", stream_current)

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openai-codex", "gpt-6.1-sol", "codex_responses"
        )
        return auxiliary_client._relay_sync_stream(
            _codex_shim_client(),
            {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "hi"}]},
            provider="openai-codex",
            api_mode="codex_responses",
        )

    run("moa_aggregator")
    assert captured["metadata"]["api_mode"] == "chat_completions"


def test_async_native_stream_completes_through_real_relay(relay_turn):
    """The inner auxiliary Relay may iterate before the outer MoA facade returns."""
    _relay, turn = relay_turn
    consumer = "test.async-native-stream"
    turn.lease.host.retain_managed_execution(consumer)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _codex_shim_client("native-actor").chat.completions.create()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    try:

        @auxiliary_client._relay_auxiliary_call
        def run(task):
            auxiliary_client._set_relay_auxiliary_route(
                "native", "native-actor", "chat_completions"
            )
            return auxiliary_client._relay_sync_stream(
                client,
                {
                    "model": "native-actor",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                provider="native",
                api_mode="chat_completions",
            )

        result = run("moa_aggregator")
        assert result.choices[0].message.content == "native-actor"
        assert len(calls) == 1
    finally:
        turn.lease.host.release_managed_execution(consumer)
