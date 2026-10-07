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
        lambda request_id, *, outcome, model_name, provider_name, response_model_name, error_class: (
            logical_completions.append(
                (request_id, outcome, model_name, provider_name, response_model_name, error_class)
            )
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
            "unknown",  # recovered from the unclassifiable invalid first response
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
            "error_class": "unknown",
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
    assert snapshot[0]["resource"]["hermes_version"] == "test-version"
    assert snapshot[0]["dimensions"] == {
        "call_role": "auxiliary",
        # The success keeps the error it recovered from; an invalid shape classifies as unknown.
        "error_class": "unknown",
        "model": "accepted/model",
        "outcome": "success",
        "provider": "openrouter",
        "ttft_bucket": "unknown",
    }
    assert snapshot[0]["value"] == 1
    assert snapshot[0]["packaged_value"] == 0


class _ProviderStatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def _aux_route_rows(relay_turn, tmp_path, run):
    """Run ``run`` inside the turn with a real shared-metrics subscriber; the stored route rows."""
    relay, turn = relay_turn
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    subscriber = SharedMetricsSubscriber(store, "test-version", runtime_id=turn.lease.host.runtime_id)
    relay.subscribers.register("test.auxiliary-route-rows", subscriber)
    turn.lease.host.retain_managed_execution("test.auxiliary-route-rows")
    try:
        run()
        relay.subscribers.flush()
    finally:
        turn.lease.host.release_managed_execution("test.auxiliary-route-rows")
        relay.subscribers.deregister("test.auxiliary-route-rows")
    return sorted(
        (row["dimensions"]["outcome"], row["dimensions"]["error_class"], row["value"])
        for row in store.counter_snapshot() if row["metric_name"] == MODEL_ROUTE_METRIC
    )


def _aux_attempts(*outcomes):
    """A client whose successive attempts raise or answer, in order."""
    pending = iter(outcomes)

    def create(**_kwargs):
        outcome = next(pending)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(model="m", choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@auxiliary_client._relay_auxiliary_call
def _aux_call(task, client, attempts):
    auxiliary_client._set_relay_auxiliary_route("openrouter", "m", "chat_completions")
    for _ in range(attempts - 1):
        try:
            return auxiliary_client._validate_llm_response(
                auxiliary_client._relay_sync_completion(client, {"model": "m", "messages": []}), task)
        except Exception:
            continue
    return auxiliary_client._validate_llm_response(
        auxiliary_client._relay_sync_completion(client, {"model": "m", "messages": []}), task)


def test_auxiliary_model_routes_carry_classified_error_class(relay_turn, tmp_path):
    """A failed aux call reports the classifier's reason for the error that ended it, and a
    recovered one the error it recovered from, never a blanket ``unknown``/``none``."""
    def run():
        _aux_call("title_generation", _aux_attempts(_ProviderStatusError(429, "Rate limit exceeded"), "ok"), 2)
        with pytest.raises(_ProviderStatusError):
            _aux_call("title_generation", _aux_attempts(
                _ProviderStatusError(404, "The model `m` does not exist")), 1)

    assert _aux_route_rows(relay_turn, tmp_path, run) == [
        ("failed", "model_not_found", 1),
        ("success", "rate_limit", 1),
    ]


@pytest.mark.parametrize("abort", [
    auxiliary_client.AuxiliaryExplicitCancellation(), InterruptedError("aux stream interrupted"), KeyboardInterrupt(),
])
def test_aborted_auxiliary_call_is_cancelled_not_failed(relay_turn, tmp_path, abort):
    """A Hermes abort (/stop, interrupt, Ctrl+C) mid aux call is not a provider failure."""
    def run():
        with pytest.raises(type(abort)):
            _aux_call("compression", _aux_attempts(abort), 1)

    assert _aux_route_rows(relay_turn, tmp_path, run) == [("cancelled", "none", 1)]


def test_auxiliary_call_finishing_under_the_turns_live_scope_is_counted(relay_turn, tmp_path):
    """Title generation runs beside the turn, so the turn's own work is usually on top of its
    logical scope when it finishes. Its result must still become the route row."""
    relay, turn = relay_turn
    host, session = turn.lease.host, turn.lease.session
    turn_work = []

    @auxiliary_client._relay_auxiliary_call
    def title(task):
        auxiliary_client._set_relay_auxiliary_route("openrouter", "m", "chat_completions")
        response = auxiliary_client._relay_sync_completion(_aux_attempts("A title"), {"model": "m", "messages": []})
        turn_work.append(host.run_in_session(
            session, relay.scope.push, "test.turn_work", relay.ScopeType.Function, handle=turn.handle, input={}))
        return auxiliary_client._validate_llm_response(response, task)

    def run():
        title("title_generation")
        host.run_in_session(session, relay_runtime.pop_relay_scope, relay, turn_work[0], output={})
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")

    assert _aux_route_rows(relay_turn, tmp_path, run) == [("success", "none", 1)]


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
        lambda request_id, *, outcome, model_name, provider_name, response_model_name, error_class: (
            logical_completions.append(
                (request_id, outcome, model_name, provider_name, response_model_name, error_class)
            )
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
            "none",
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
                "error_class": "unknown",
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
                "error_class": "unknown",
                "model": "test-model",
                "outcome": "failed",
                "provider": "openrouter",
            },
            {
                "error_class": "none",
                "model": "test-model",
                "outcome": "success",
                "provider": "openrouter",
            },
        ]
        assert turn.logical_llm_calls == {}
    finally:
        turn.lease.host.release_managed_execution(consumer)
def test_auxiliary_stream_unwraps_completed_response(relay_turn):
    """MoA aggregator on an Anthropic-protocol provider: the client returns a
    completed response for ``stream=True`` (the adapter ignores the flag), so
    ``_relay_sync_stream`` must surface it raw for the consumer's
    ``hasattr(stream, "choices")`` handling — regression of #11732/#55933 via
    the Relay integration (SimpleNamespace is not iterable)."""
    _relay, _turn = relay_turn
    completed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="aggregated"),
                finish_reason="stop",
            )
        ],
        model="kimi-k3",
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completed)
        )
    )

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "kimi-coding",
            "kimi-k3",
            "chat_completions",
        )
        return auxiliary_client._relay_sync_stream(
            client,
            {"model": "kimi-k3", "messages": [], "stream": True},
        )

    assert run("moa_aggregator") is completed



def test_call_llm_stream_unwraps_completed_response(relay_turn, monkeypatch):
    """Outermost seam: ``call_llm(stream=True)`` — decorated with
    ``@_relay_auxiliary_call`` in production, so the Relay context is always
    set — with an Anthropic-shaped client that ignores ``stream=True`` and
    returns a completed response (the MoA aggregator on kimi-coding /
    MiniMax / ZAI / any /anthropic gateway). Must return the raw response for
    the consumer's ``hasattr(stream, "choices")`` handling, not crash with
    ``TypeError: 'types.SimpleNamespace' object is not iterable``."""
    _relay, _turn = relay_turn
    captured = {}
    completed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="aggregated"),
                finish_reason="stop",
            )
        ],
        model="kimi-k3",
    )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return completed

    client = SimpleNamespace(
        base_url="https://api.kimi.com/coding/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *args, **kwargs: (client, "kimi-k3"),
    )

    result = auxiliary_client.call_llm(
        "moa_aggregator",
        provider="kimi-coding",
        model="kimi-k3",
        api_key="sk-test",
        messages=[{"role": "user", "content": "q"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    assert result is completed
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
