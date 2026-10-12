"""Auxiliary calls on native-wire adapters pass through Relay.

Relay intercepts the adapter's ``chat.completions.create`` call, so every auxiliary entry point
must hand it a chat-completions request whatever wire the adapter speaks to its provider.
Regression for MoA references on ``openai-codex`` failing with
``OpenAI Responses request is missing input``.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import auxiliary_client, relay_runtime

MESSAGES = [{"role": "system", "content": "advise"}, {"role": "user", "content": "hi"}]


@pytest.fixture()
def managed_relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="session-1", platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    consumer = "test.auxiliary-native-api-mode"
    lease.host.retain_managed_execution(consumer)
    try:
        yield
    finally:
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _response():
    message = SimpleNamespace(role="assistant", content="advice")
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")], model="gpt-6-astra",
    )


def _sync_client(sent):
    def create(**kwargs):
        sent.append(kwargs)
        return _response()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _async_client(sent):
    async def create(**kwargs):
        sent.append(kwargs)
        return _response()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _sync_completion(sent):
    return auxiliary_client._relay_sync_completion(
        _sync_client(sent), {"model": "gpt-6-astra", "messages": MESSAGES},
        provider="openai-codex", api_mode="codex_responses",
    )


def _async_completion(sent):
    @auxiliary_client._relay_auxiliary_call_async
    async def run(task):
        return await auxiliary_client._relay_async_completion(
            _async_client(sent), {"model": "gpt-6-astra", "messages": MESSAGES},
            provider="openai-codex", api_mode="codex_responses",
        )

    return asyncio.run(run("moa_reference"))


def _sync_stream(sent):
    return auxiliary_client._relay_sync_stream(
        _sync_client(sent), {"model": "gpt-6-astra", "messages": MESSAGES, "stream": True},
        provider="openai-codex", api_mode="codex_responses",
    )


@pytest.mark.parametrize("call", [_sync_completion, _async_completion, _sync_stream])
def test_native_adapter_request_reaches_provider(managed_relay_turn, call):
    sent = []

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        return call(sent)

    result = run("moa_reference")

    assert result.choices[0].message.content == "advice"
    assert len(sent) == 1
    assert [m["content"] for m in sent[0]["messages"]] == ["advise", "hi"]
