"""Regression #130132: native auxiliary clients expose chat-shaped requests to Relay."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import auxiliary_client, relay_runtime


@pytest.fixture
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease, turn_id="turn", task_id="task"
    )
    consumer = "test.moa-reference-relay-boundary"
    lease.host.retain_managed_execution(consumer)
    try:
        yield
    finally:
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.mark.parametrize("path", ["sync", "async", "stream"])
def test_native_reference_boundary_is_chat_shaped_through_real_relay(relay_turn, path):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="advisor answer"))]
    )
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return response

    async def acreate(**kwargs):
        return create(**kwargs)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=acreate if path == "async" else create)
        )
    )
    kwargs = {
        "model": "native-advisor",
        "messages": [{"role": "user", "content": "hi"}],
    }

    @auxiliary_client._relay_auxiliary_call
    def run_sync(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openai-codex", kwargs["model"], "codex_responses"
        )
        if path == "stream":
            return auxiliary_client._relay_sync_stream(
                client,
                {**kwargs, "stream": True},
                provider="openai-codex",
                api_mode="codex_responses",
            )
        return auxiliary_client._relay_sync_completion(
            client, kwargs, provider="openai-codex", api_mode="codex_responses"
        )

    @auxiliary_client._relay_auxiliary_call_async
    async def run_async(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openai-codex", kwargs["model"], "codex_responses"
        )
        return await auxiliary_client._relay_async_completion(
            client, kwargs, provider="openai-codex", api_mode="codex_responses"
        )

    result = (
        asyncio.run(run_async("moa_reference"))
        if path == "async"
        else run_sync("moa_reference")
    )
    assert result.choices[0].message.content == "advisor answer"
    assert len(requests) == 1
    assert requests[0]["messages"] == kwargs["messages"]
