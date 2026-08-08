"""Deterministic vertical-slice tests for zero-chunk stall recovery."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.provider_health_probe import ProbeOutcome
from agent.provider_stall import ProviderStalledError, format_provider_stall_status
from hermes_cli.timeouts import ProviderStallRecoveryConfig


def _chunk(content: str, *, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        model="test-model",
        usage=None,
    )


class _BlockingStream:
    def __init__(self, aborted: threading.Event, *, late_chunk: str | None = None):
        self.aborted = aborted
        self.late_chunk = late_chunk
        self._late_sent = False

    def __iter__(self):
        return self

    def __next__(self):
        assert self.aborted.wait(5), "watchdog did not abort the stalled stream"
        if self.late_chunk is not None and not self._late_sent:
            self._late_sent = True
            return _chunk(self.late_chunk, finish_reason="stop")
        raise ConnectionError("request-local transport aborted")

    def close(self):
        return None


class _GateStream:
    def __init__(self, release_chunk: threading.Event, *, chunk=None, yielded=None):
        self.release_chunk = release_chunk
        self.chunk = chunk or _chunk("original", finish_reason="stop")
        self.yielded = yielded
        self._sent = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._sent:
            raise StopIteration
        assert self.release_chunk.wait(5), "test did not release provider chunk"
        self._sent = True
        if self.yielded is not None:
            self.yielded.set()
        return self.chunk

    def close(self):
        return None


def _client(stream):
    client = MagicMock()
    client.chat.completions.create.return_value = stream
    return client


@pytest.fixture
def agent():
    from run_agent import AIAgent

    value = AIAgent(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    value.api_mode = "chat_completions"
    value._interrupt_requested = False
    return value


def _install_policy(monkeypatch, *, enabled=True, probe=True, retries=1):
    config = ProviderStallRecoveryConfig(
        enabled=enabled,
        health_probe_enabled=probe,
        health_probe_timeout_seconds=1.0,
        same_provider_retries=retries,
    )
    monkeypatch.setattr(
        "agent.chat_completion_helpers.get_provider_stall_recovery_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "agent.chat_completion_helpers.get_provider_stale_timeout",
        lambda provider, model: 0.001,
    )


@pytest.mark.parametrize(
    ("probe", "diagnosis"),
    [
        (
            ProbeOutcome("reachable", 404, "endpoint returned HTTP 404"),
            "endpoint reachable but request wedged",
        ),
        (
            ProbeOutcome("unreachable", None, "ConnectError"),
            "provider endpoint unreachable",
        ),
        (
            ProbeOutcome("unavailable", None, "unsupported URL"),
            "provider health probe unavailable",
        ),
        (
            ProbeOutcome("disabled", None, "health probe disabled"),
            "provider health probe disabled",
        ),
    ],
)
@pytest.mark.parametrize(
    ("action", "action_text"),
    [
        ("reconnecting", "Reconnecting once with a fresh connection."),
        ("falling_back", "Switching to configured fallback."),
        (
            "failed",
            "No configured fallback is available. Configure fallback_providers "
            "to continue on another provider.",
        ),
    ],
)
def test_provider_stall_status_is_concise_and_structured(
    probe, diagnosis, action, action_text
):
    error = ProviderStalledError(
        provider="test-provider",
        model="test/model",
        silent_seconds=360,
        attempt=2,
        probe=probe,
    )

    status = format_provider_stall_status(error, action)

    assert status == (
        f"⚠️ No response chunks from test-provider/test/model for 360s; "
        f"{diagnosis}. {action_text}"
    )
    if probe.detail != diagnosis.removeprefix("provider "):
        assert probe.detail not in status


def test_provider_stall_lifecycle_hook_includes_structured_probe_context(
    agent, monkeypatch
):
    events = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.has_hook",
        lambda name: name == "api_request_error",
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **payload: events.append((name, payload)),
    )

    agent._invoke_api_request_error_hook(
        task_id="task-1",
        turn_id="turn-1",
        api_request_id="request-1",
        api_call_count=2,
        api_start_time=0.0,
        api_kwargs={"messages": [{"role": "user", "content": "continue"}]},
        error_type="ProviderStalledError",
        error_message="provider stalled",
        retryable=False,
        reason="provider_stalled",
        error_context={
            "probe_status": "reachable",
            "probe_http_status": 404,
            "silent_seconds": 360.0,
            "attempt": 2,
        },
    )

    event_name, payload = events[-1]
    assert event_name == "api_request_error"
    assert payload["reason"] == "provider_stalled"
    assert payload["retryable"] is False
    assert payload["error_context"] == {
        "probe_status": "reachable",
        "probe_http_status": 404,
        "silent_seconds": pytest.approx(360, abs=1),
        "attempt": 2,
    }


def test_first_zero_chunk_stall_probes_cancels_and_retries_with_fresh_client(
    agent, monkeypatch
):
    _install_policy(monkeypatch)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    first_aborted = threading.Event()
    first_client = _client(_BlockingStream(first_aborted))
    second_client = _client(iter([_chunk("recovered", finish_reason="stop")]))
    agent._create_request_openai_client = MagicMock(
        side_effect=[first_client, second_client]
    )
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda client, reason: first_aborted.set()
    )
    statuses: list[str] = []
    agent._buffer_status = statuses.append
    probe = MagicMock(
        return_value=ProbeOutcome(
            status="reachable", http_status=401, detail="endpoint returned HTTP 401"
        )
    )
    monkeypatch.setattr("agent.chat_completion_helpers.probe_provider_endpoint", probe)

    response = agent._interruptible_streaming_api_call({"model": "test/model"})

    assert response.choices[0].message.content == "recovered"
    assert probe.call_count == 1
    assert agent._create_request_openai_client.call_count == 2
    assert first_client is not second_client
    assert first_aborted.is_set()
    assert statuses == [
        "⚠️ No response chunks from unknown/test/model for 0s; "
        "endpoint reachable but request wedged. Reconnecting once with a fresh connection."
    ]


def test_explicit_zero_stream_retries_prevents_same_provider_stall_retry(
    agent, monkeypatch
):
    _install_policy(monkeypatch, retries=1)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    aborted = threading.Event()
    client = _client(_BlockingStream(aborted))
    agent._create_request_openai_client = MagicMock(return_value=client)
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda request_client, reason: aborted.set()
    )
    statuses: list[str] = []
    agent._buffer_status = statuses.append
    probe = MagicMock(
        return_value=ProbeOutcome(
            status="reachable", http_status=401, detail="endpoint returned HTTP 401"
        )
    )
    monkeypatch.setattr("agent.chat_completion_helpers.probe_provider_endpoint", probe)

    with pytest.raises(ProviderStalledError) as caught:
        agent._interruptible_streaming_api_call({"model": "test/model"})

    assert caught.value.attempt == 1
    assert probe.call_count == 1
    assert agent._create_request_openai_client.call_count == 1
    assert aborted.is_set()
    assert statuses == []


def test_chunk_arriving_during_probe_prevents_cancellation(agent, monkeypatch):
    _install_policy(monkeypatch)
    probe_started = threading.Event()
    release_chunk = threading.Event()
    chunk_received = threading.Event()
    release_probe = threading.Event()
    client = _client(_GateStream(release_chunk))
    agent._create_request_openai_client = MagicMock(return_value=client)
    agent._abort_request_openai_client = MagicMock()
    original_touch = agent._touch_activity

    def touch(detail):
        if detail == "receiving stream response":
            chunk_received.set()
        return original_touch(detail)

    agent._touch_activity = touch

    def probe(**kwargs):
        probe_started.set()
        assert release_probe.wait(5), "test did not release probe"
        return ProbeOutcome(status="reachable", http_status=200, detail="HTTP 200")

    monkeypatch.setattr("agent.chat_completion_helpers.probe_provider_endpoint", probe)

    def coordinate():
        assert probe_started.wait(5), "watchdog did not start probe"
        release_chunk.set()
        assert chunk_received.wait(5), "provider chunk was not accepted during probe"
        release_probe.set()

    coordinator = threading.Thread(target=coordinate)
    coordinator.start()
    response = agent._interruptible_streaming_api_call({"model": "test/model"})
    coordinator.join(5)

    assert not coordinator.is_alive()
    assert response.choices[0].message.content == "original"
    agent._abort_request_openai_client.assert_not_called()
    assert agent._create_request_openai_client.call_count == 1


def test_chunk_released_after_probe_recheck_cannot_reach_any_callback(
    agent, monkeypatch
):
    _install_policy(monkeypatch)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    release_chunk = threading.Event()
    chunk_yielded = threading.Event()
    first_aborted = threading.Event()
    tool_delta = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments='{"path":"x"}'),
        extra_content=None,
    )
    raced_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content="must-not-be-visible",
                    tool_calls=[tool_delta],
                    reasoning_content="private-reasoning",
                    reasoning=None,
                ),
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=None,
    )
    clients = [
        _client(_GateStream(release_chunk, chunk=raced_chunk, yielded=chunk_yielded)),
        _client(iter([_chunk("winner", finish_reason="stop")])),
    ]
    agent._create_request_openai_client = MagicMock(side_effect=clients)
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda client, reason: first_aborted.set()
    )
    content_deltas: list[str] = []
    reasoning_deltas: list[str] = []
    tool_starts: list[str] = []
    agent._fire_stream_delta = content_deltas.append
    agent._fire_reasoning_delta = reasoning_deltas.append
    agent._fire_tool_gen_started = tool_starts.append

    def release_exactly_after_recheck(_status):
        release_chunk.set()
        assert chunk_yielded.wait(5), "racing chunk was not released"

    agent._buffer_status = release_exactly_after_recheck
    monkeypatch.setattr(
        "agent.chat_completion_helpers.probe_provider_endpoint",
        lambda **kwargs: ProbeOutcome("reachable", 200, "HTTP 200"),
    )

    response = agent._interruptible_streaming_api_call({"model": "test/model"})

    assert response.choices[0].message.content == "winner"
    assert content_deltas == ["winner"]
    assert reasoning_deltas == []
    assert tool_starts == []


def test_interrupt_during_blocked_probe_aborts_promptly_and_never_retries(
    agent, monkeypatch
):
    _install_policy(monkeypatch)
    probe_started = threading.Event()
    release_probe = threading.Event()
    stream_aborted = threading.Event()
    agent._create_request_openai_client = MagicMock(
        return_value=_client(_BlockingStream(stream_aborted))
    )
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda client, reason: stream_aborted.set()
    )

    def blocked_probe(**kwargs):
        probe_started.set()
        assert release_probe.wait(5), "test did not release blocked probe"
        return ProbeOutcome("reachable", 200, "HTTP 200")

    monkeypatch.setattr(
        "agent.chat_completion_helpers.probe_provider_endpoint", blocked_probe
    )

    def interrupt_probe():
        assert probe_started.wait(5), "watchdog did not start probe"
        agent._interrupt_requested = True

    interrupter = threading.Thread(target=interrupt_probe)
    interrupter.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            agent._interruptible_streaming_api_call({"model": "test/model"})
    finally:
        release_probe.set()
    elapsed = time.monotonic() - started
    interrupter.join(1)

    assert elapsed < 0.5
    assert stream_aborted.is_set()
    assert agent._create_request_openai_client.call_count == 1


def test_late_chunks_from_cancelled_stall_attempt_are_discarded(agent, monkeypatch):
    _install_policy(monkeypatch, probe=False)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    first_aborted = threading.Event()
    clients = [
        _client(_BlockingStream(first_aborted, late_chunk="late")),
        _client(iter([_chunk("winner", finish_reason="stop")])),
    ]
    agent._create_request_openai_client = MagicMock(side_effect=clients)
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda client, reason: first_aborted.set()
    )
    deltas: list[str] = []
    agent._fire_stream_delta = deltas.append

    response = agent._interruptible_streaming_api_call({"model": "test/model"})

    assert response.choices[0].message.content == "winner"
    assert "late" not in "".join(deltas)
    assert "winner" in "".join(deltas)
    assert agent._create_request_openai_client.call_count == 2


def test_zero_chunk_stall_with_retries_disabled_is_immediately_typed(
    agent, monkeypatch
):
    _install_policy(monkeypatch, retries=0)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "5")
    aborted = threading.Event()
    agent._create_request_openai_client = MagicMock(
        return_value=_client(_BlockingStream(aborted))
    )
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda client, reason: aborted.set()
    )
    monkeypatch.setattr(
        "agent.chat_completion_helpers.probe_provider_endpoint",
        lambda **kwargs: ProbeOutcome(status="unreachable", detail="ConnectError"),
    )

    with pytest.raises(ProviderStalledError) as caught:
        agent._interruptible_streaming_api_call({"model": "test/model"})

    assert caught.value.attempt == 1
    assert caught.value.probe.status == "unreachable"
    assert agent._create_request_openai_client.call_count == 1


def test_second_zero_chunk_stall_is_typed_without_generic_retry(agent, monkeypatch):
    _install_policy(monkeypatch, retries=1)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "5")
    first_aborted = threading.Event()
    second_aborted = threading.Event()
    clients = [
        _client(_BlockingStream(first_aborted)),
        _client(_BlockingStream(second_aborted)),
    ]
    agent._create_request_openai_client = MagicMock(side_effect=clients)

    def abort(client, reason):
        (first_aborted if client is clients[0] else second_aborted).set()

    agent._abort_request_openai_client = MagicMock(side_effect=abort)
    monkeypatch.setattr(
        "agent.chat_completion_helpers.probe_provider_endpoint",
        lambda **kwargs: ProbeOutcome(status="reachable", http_status=200),
    )

    with pytest.raises(ProviderStalledError) as caught:
        agent._interruptible_streaming_api_call({"model": "test/model"})

    assert caught.value.attempt == 2
    assert agent._create_request_openai_client.call_count == 2


def test_second_zero_chunk_stall_switches_to_configured_fallback(agent, monkeypatch):
    _install_policy(monkeypatch, retries=1)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "5")
    first_aborted = threading.Event()
    second_aborted = threading.Event()
    primary_clients = [
        _client(_BlockingStream(first_aborted)),
        _client(_BlockingStream(second_aborted)),
    ]
    fallback_client = _client(
        iter([_chunk("completed by fallback", finish_reason="stop")])
    )
    clients = [*primary_clients, fallback_client]
    agent._create_request_openai_client = MagicMock(side_effect=clients)

    def abort(client, reason):
        primary_clients[
            clients.index(client)
        ].chat.completions.create.return_value.aborted.set()

    agent._abort_request_openai_client = MagicMock(side_effect=abort)
    monkeypatch.setattr(
        "agent.chat_completion_helpers.probe_provider_endpoint",
        lambda **kwargs: ProbeOutcome(status="reachable", http_status=200),
    )
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    ]
    agent._fallback_index = 0
    agent._consecutive_stale_streams = 0
    logical_history_before_fallback: list[dict] = []
    fallback_history: list[dict] = []

    def activate(reason=None):
        assert reason is not None and reason.value == "provider_stalled"
        logical_history_before_fallback.extend(
            deepcopy(
                primary_clients[-1].chat.completions.create.call_args.kwargs["messages"]
            )
        )
        assert agent._consecutive_stale_streams >= 2
        agent._fallback_index = 1
        agent._fallback_activated = True
        agent.provider = "openrouter"
        agent.model = "anthropic/claude-sonnet-4"
        agent._cached_system_prompt = "fallback system prompt"
        agent._consecutive_stale_streams = 0
        return True

    def capture_fallback(**kwargs):
        fallback_history.extend(deepcopy(kwargs["messages"]))
        return iter([_chunk("completed by fallback", finish_reason="stop")])

    fallback_client.chat.completions.create.side_effect = capture_fallback

    with (
        patch.object(agent, "_try_activate_fallback", side_effect=activate) as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop.jittered_backoff") as generic_backoff,
    ):
        result = agent.run_conversation("continue the task")

    assert result["completed"] is True
    assert result["final_response"] == "completed by fallback"
    assert (
        sum(client.chat.completions.create.call_count for client in primary_clients)
        == 2
    )
    assert fallback_client.chat.completions.create.call_count == 1
    assert agent._create_request_openai_client.call_count == 3
    fallback.assert_called_once()
    generic_backoff.assert_not_called()
    assert logical_history_before_fallback[1:] == fallback_history[1:]
    assert (
        logical_history_before_fallback[0]["content"] != fallback_history[0]["content"]
    )
    assert agent._consecutive_stale_streams == 0


def test_second_stall_without_fallback_reports_probe_diagnosis(agent, monkeypatch):
    _install_policy(monkeypatch, retries=1)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "5")
    agent.provider = "primary-provider"
    agent.model = "primary/model"
    aborted = [threading.Event(), threading.Event()]
    clients = [
        _client(_BlockingStream(aborted[0])),
        _client(_BlockingStream(aborted[1])),
    ]
    agent._create_request_openai_client = MagicMock(side_effect=clients)

    def abort(client, reason):
        aborted[clients.index(client)].set()

    agent._abort_request_openai_client = MagicMock(side_effect=abort)
    monkeypatch.setattr(
        "agent.chat_completion_helpers.probe_provider_endpoint",
        lambda **kwargs: ProbeOutcome(
            status="reachable",
            http_status=200,
            detail="https://user:secret@provider.invalid/private?token=secret",
        ),
    )
    agent._fallback_chain = []
    agent._fallback_index = 0

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop.jittered_backoff") as generic_backoff,
    ):
        result = agent.run_conversation("continue the task")

    assert result["completed"] is False
    assert sum(client.chat.completions.create.call_count for client in clients) == 2
    assert "primary-provider" in result["error"]
    assert "primary/model" in result["error"]
    assert "No configured fallback is available" in result["error"]
    assert "for 0s" in result["error"]
    assert "endpoint reachable but request wedged" in result["error"]
    assert "fallback_providers" in result["error"]
    assert "HTTP None" not in result["error"]
    assert "provider.invalid" not in result["error"]
    assert "secret" not in result["error"]
    assert "Traceback" not in result["error"]
    generic_backoff.assert_not_called()


def test_recovery_disabled_preserves_generic_transient_retry_behavior(
    agent, monkeypatch
):
    _install_policy(monkeypatch, enabled=False)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    aborted = threading.Event()
    agent._create_request_openai_client = MagicMock(
        return_value=_client(_BlockingStream(aborted))
    )
    agent._abort_request_openai_client = MagicMock(
        side_effect=lambda client, reason: aborted.set()
    )
    probe = MagicMock()
    monkeypatch.setattr("agent.chat_completion_helpers.probe_provider_endpoint", probe)

    with pytest.raises(ConnectionError):
        agent._interruptible_streaming_api_call({"model": "test/model"})

    probe.assert_not_called()
    assert agent._create_request_openai_client.call_count == 1
