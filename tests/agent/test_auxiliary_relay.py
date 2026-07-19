from types import SimpleNamespace

import pytest

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


def test_auxiliary_retries_share_logical_relay_identity(monkeypatch):
    attempts = []
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: {"request": kwargs},
            )
        )
    )

    def execute_current(request, callback, **kwargs):
        attempts.append(kwargs)
        return callback(request)

    monkeypatch.setattr(relay_llm, "execute_current", execute_current)

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openrouter",
            "test-model",
            "chat_completions",
        )
        first = auxiliary_client._relay_sync_completion(
            client,
            {"model": "test-model", "messages": []},
        )
        second = auxiliary_client._relay_sync_completion(
            client,
            {"model": "test-model", "messages": []},
        )
        return first, second

    first, second = run("compression")

    assert first["request"]["model"] == "test-model"
    assert second["request"]["model"] == "test-model"
    assert attempts[0]["metadata"]["api_request_id"] == (
        attempts[1]["metadata"]["api_request_id"]
    )
    assert [attempt["metadata"]["retry_count"] for attempt in attempts] == [0, 1]
    assert attempts[0]["metadata"]["call_role"] == "auxiliary:compression"


@pytest.mark.asyncio
async def test_async_auxiliary_attempt_uses_inherited_relay_adapter(monkeypatch):
    captured = {}

    async def create(**kwargs):
        return {"request": kwargs}

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    async def execute_current_async(request, callback, **kwargs):
        captured.update(kwargs)
        return await callback(request)

    monkeypatch.setattr(
        relay_llm,
        "execute_current_async",
        execute_current_async,
    )

    @auxiliary_client._relay_auxiliary_call_async
    async def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "anthropic",
            "claude-test",
            "chat_completions",
        )
        return await auxiliary_client._relay_async_completion(
            client,
            {"model": "claude-test", "messages": []},
        )

    result = await run("title_generation")

    assert result["request"]["model"] == "claude-test"
    assert captured["name"] == "anthropic"
    assert captured["metadata"]["call_role"] == "auxiliary:title_generation"


def test_auxiliary_stream_uses_streaming_relay_primitive(monkeypatch):
    captured = {}
    raw_stream = iter([{"delta": "one"}, {"delta": "two"}])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: raw_stream)
        )
    )

    def stream_current(request, stream_factory, **kwargs):
        captured.update(kwargs)
        return stream_factory(request)

    monkeypatch.setattr(relay_llm, "stream_current", stream_current)

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openrouter",
            "moa-model",
            "chat_completions",
        )
        return auxiliary_client._relay_sync_stream(
            client,
            {"model": "moa-model", "messages": [], "stream": True},
        )

    assert list(run("moa")) == [{"delta": "one"}, {"delta": "two"}]
    assert captured["metadata"]["call_role"] == "auxiliary:moa"


def test_auxiliary_attempt_uses_real_relay_request_intercepts(relay_turn):
    relay, turn = relay_turn
    captured_requests = []
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: captured_requests.append(kwargs)
                or {"content": "ok"},
            )
        )
    )

    def rewrite_request(_name, request, annotated):
        annotated.params = {**(annotated.params or {}), "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request(
        "hermes-auxiliary-request",
        1,
        False,
        rewrite_request,
    )
    try:
        @auxiliary_client._relay_auxiliary_call
        def run(task):
            auxiliary_client._set_relay_auxiliary_route(
                "openrouter",
                "test-model",
                "chat_completions",
            )
            return auxiliary_client._relay_sync_completion(
                client,
                {"model": "test-model", "messages": []},
            )

        result = run("compression")
    finally:
        relay.intercepts.deregister_llm_request("hermes-auxiliary-request")

    assert result == {"content": "ok"}
    assert captured_requests[0]["temperature"] == 0.25
    assert turn.logical_llm_calls == {}
