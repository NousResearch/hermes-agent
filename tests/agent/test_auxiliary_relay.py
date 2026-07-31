from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import auxiliary_client, relay_llm, relay_runtime
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_contract import MODEL_ROUTE_METRIC
from hermes_cli.observability.shared_metrics_subscriber import SharedMetricsSubscriber


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
    logical_completions = []
    responses = iter([
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        ),
    ])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: next(responses),
            )
        )
    )

    def execute_current(request, callback, **kwargs):
        attempts.append(kwargs)
        return callback(request)

    monkeypatch.setattr(relay_llm, "execute_current", execute_current)
    monkeypatch.setattr(
        relay_llm,
        "complete_logical_call",
        lambda request_id, *, outcome, model_name, provider_name, response_model_name: logical_completions.append(
            (request_id, outcome, model_name, provider_name, response_model_name)
        ),
    )

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openrouter",
            "test-model",
            "chat_completions",
        )
        with pytest.raises(RuntimeError, match="invalid response"):
            auxiliary_client._validate_llm_response(
                auxiliary_client._relay_sync_completion(
                    client,
                    {"model": "test-model", "messages": []},
                ),
                task,
            )
        return auxiliary_client._validate_llm_response(
            auxiliary_client._relay_sync_completion(
                client,
                {"model": "test-model", "messages": []},
            ),
            task,
        )

    result = run("compression")

    assert result.choices[0].message.content == "ok"
    assert attempts[0]["metadata"]["api_request_id"] == (
        attempts[1]["metadata"]["api_request_id"]
    )
    assert [attempt["metadata"]["retry_count"] for attempt in attempts] == [0, 1]
    assert attempts[0]["metadata"]["call_role"] == "auxiliary:compression"
    assert all(attempt["defer_logical_completion"] is True for attempt in attempts)
    assert logical_completions == [
        (
            attempts[0]["metadata"]["api_request_id"],
            "success",
            "test-model",
            "openrouter",
            None,
        )
    ]


def test_auxiliary_provider_fallback_closes_one_real_logical_call(
    relay_turn,
    monkeypatch,
):
    relay, turn = relay_turn
    consumer = "test.auxiliary-provider-fallback"
    turn.lease.host.retain_managed_execution(consumer)
    logical_outputs = []
    original_pop = relay.scope.pop

    def record_pop(*args, **kwargs):
        logical_outputs.append(kwargs.get("output") or {})
        return original_pop(*args, **kwargs)

    monkeypatch.setattr(relay.scope, "pop", record_pop)
    responses = iter([
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))]
        ),
    ])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: next(responses),
            )
        )
    )

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "nvidia",
            "nvidia/test-model",
            "chat_completions",
        )
        with pytest.raises(RuntimeError, match="invalid response"):
            auxiliary_client._validate_llm_response(
                auxiliary_client._relay_sync_completion(
                    client,
                    {"model": "nvidia/test-model", "messages": []},
                ),
                task,
            )
        assert len(turn.logical_llm_calls) == 1

        auxiliary_client._set_relay_auxiliary_route(
            "openrouter",
            "openrouter/test-model",
            "chat_completions",
        )
        return auxiliary_client._validate_llm_response(
            auxiliary_client._relay_sync_completion(
                client,
                {"model": "openrouter/test-model", "messages": []},
            ),
            task,
        )

    try:
        result = run("compression")
    finally:
        turn.lease.host.release_managed_execution(consumer)

    assert result.choices[0].message.content == "recovered"
    assert turn.logical_llm_calls == {}
    assert logical_outputs == [
        {
            "model": "openrouter/test-model",
            "outcome": "success",
            "provider": "openrouter",
        }
    ]


def test_auxiliary_provider_fallback_records_one_terminal_model_route(
    relay_turn,
    tmp_path,
):
    relay, turn = relay_turn
    store = SharedMetricsStore(
        tmp_path / "metrics.sqlite3",
        tmp_path / "outbox",
    )
    subscriber = SharedMetricsSubscriber(
        store,
        "test-version",
        runtime_id=turn.lease.host.runtime_id,
    )
    subscriber_name = "test.auxiliary-model-route"
    relay.subscribers.register(subscriber_name, subscriber)
    turn.lease.host.retain_managed_execution(subscriber_name)
    responses = iter([
        SimpleNamespace(model="failed/model", choices=[]),
        SimpleNamespace(
            model="Accepted/Model",
            choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))],
        ),
    ])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: next(responses),
            )
        )
    )

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "nvidia",
            "failed/configured-model",
            "chat_completions",
        )
        with pytest.raises(RuntimeError, match="invalid response"):
            auxiliary_client._validate_llm_response(
                auxiliary_client._relay_sync_completion(
                    client,
                    {"model": "failed/configured-model", "messages": []},
                ),
                task,
            )
        auxiliary_client._set_relay_auxiliary_route(
            "OpenRouter",
            "fallback/configured-model",
            "chat_completions",
        )
        return auxiliary_client._validate_llm_response(
            auxiliary_client._relay_sync_completion(
                client,
                {"model": "fallback/configured-model", "messages": []},
            ),
            task,
        )

    try:
        result = run("compression")
        relay.subscribers.flush()
    finally:
        turn.lease.host.release_managed_execution(subscriber_name)
        relay.subscribers.deregister(subscriber_name)

    assert result.choices[0].message.content == "recovered"
    snapshot = store.counter_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["metric_name"] == MODEL_ROUTE_METRIC
    assert snapshot[0]["hermes_version"] == "test-version"
    assert snapshot[0]["dimensions"] == {
        "model": "accepted/model",
        "provider": "openrouter",
    }
    assert snapshot[0]["value"] == 1
    assert snapshot[0]["packaged_value"] == 0


@pytest.mark.asyncio
async def test_async_auxiliary_attempt_uses_inherited_relay_adapter(monkeypatch):
    captured = {}
    logical_completions = []

    async def create(**kwargs):
        return SimpleNamespace(
            request=kwargs,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

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
    monkeypatch.setattr(
        relay_llm,
        "complete_logical_call",
        lambda request_id, *, outcome, model_name, provider_name, response_model_name: logical_completions.append(
            (request_id, outcome, model_name, provider_name, response_model_name)
        ),
    )

    @auxiliary_client._relay_auxiliary_call_async
    async def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "anthropic",
            "claude-test",
            "chat_completions",
        )
        return auxiliary_client._validate_llm_response(
            await auxiliary_client._relay_async_completion(
                client,
                {"model": "claude-test", "messages": []},
            ),
            task,
        )

    result = await run("title_generation")

    assert result.request["model"] == "claude-test"
    assert captured["name"] == "anthropic"
    assert captured["metadata"]["call_role"] == "auxiliary:title_generation"
    assert captured["defer_logical_completion"] is True
    assert logical_completions == [
        (
            captured["metadata"]["api_request_id"],
            "success",
            "claude-test",
            "anthropic",
            None,
        )
    ]








def test_partial_auxiliary_stream_failure_closes_before_recovery(
    relay_turn, monkeypatch
):
    _relay, turn = relay_turn
    consumer = "test.partial-auxiliary-stream-failure"
    turn.lease.host.retain_managed_execution(consumer)
    logical_outputs = []
    original_pop = turn.lease.host.relay.scope.pop

    def record_pop(*args, **kwargs):
        logical_outputs.append(kwargs.get("output") or {})
        return original_pop(*args, **kwargs)

    monkeypatch.setattr(turn.lease.host.relay.scope, "pop", record_pop)

    class ProviderError(Exception):
        pass

    provider_error = ProviderError("stream failed")
    partial_chunk = SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="partial", tool_calls=None),
                finish_reason=None,
            )
        ],
        usage=None,
    )

    def partial_stream():
        yield partial_chunk
        raise provider_error

    stream_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: partial_stream(),
            )
        )
    )
    recovery_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="recovered"))
                    ]
                ),
            )
        )
    )

    @auxiliary_client._relay_auxiliary_call
    def start_stream(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openrouter",
            "test-model",
            "chat_completions",
        )
        return auxiliary_client._relay_sync_stream(
            stream_client,
            {"model": "test-model", "messages": [], "stream": True},
        )

    @auxiliary_client._relay_auxiliary_call
    def recover(task):
        auxiliary_client._set_relay_auxiliary_route(
            "openrouter",
            "test-model",
            "chat_completions",
        )
        return auxiliary_client._validate_llm_response(
            auxiliary_client._relay_sync_completion(
                recovery_client,
                {"model": "test-model", "messages": []},
            ),
            task,
        )

    try:
        stream = start_stream("moa")
        assert next(stream) is partial_chunk

        with pytest.raises(ProviderError) as caught:
            next(stream)

        assert caught.value is provider_error
        assert logical_outputs == [
            {
                "model": "test-model",
                "outcome": "failed",
                "provider": "openrouter",
            }
        ]
        assert turn.logical_llm_calls == {}

        result = recover("moa")

        assert result.choices[0].message.content == "recovered"
        assert logical_outputs == [
            {
                "model": "test-model",
                "outcome": "failed",
                "provider": "openrouter",
            },
            {
                "model": "test-model",
                "outcome": "success",
                "provider": "openrouter",
            },
        ]
        assert turn.logical_llm_calls == {}
    finally:
        turn.lease.host.release_managed_execution(consumer)
