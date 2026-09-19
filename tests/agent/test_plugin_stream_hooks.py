import queue
import threading
import time

from types import SimpleNamespace
from unittest.mock import patch


def _agent():
    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def _make_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, reasoning_content=None, reasoning=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model")


def _callbacks(callbacks_by_hook):
    return lambda name: tuple(callbacks_by_hook.get(name, ()))


def test_stream_observer_hooks_are_valid_plugin_hooks():
    from hermes_cli.plugins import VALID_HOOKS

    assert {
        "on_stream_start",
        "on_stream_delta",
        "on_stream_end",
        "on_interim_message",
    }.issubset(VALID_HOOKS)


def test_last_stream_observer_disposal_stops_its_worker(monkeypatch):
    from agent import plugin_stream_hooks as hooks
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    context = plugins.PluginContext(plugins.PluginManifest(name="observer", key="observer"), manager)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    delivered = queue.Queue()

    def observe(**payload):
        delivered.put((payload["delta"], threading.current_thread()))

    hooks.shutdown_plugin_stream_hook_dispatcher()
    registration = context.register_hook("on_stream_delta", observe)
    try:
        assert hooks.enqueue_plugin_stream_hook("on_stream_delta", delta="before")
        delta, worker = delivered.get(timeout=2)
        assert delta == "before"
        registration.dispose()
        assert not hooks.enqueue_plugin_stream_hook("on_stream_delta", delta="after")
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert delivered.empty()
    finally:
        registration.dispose()
        hooks.shutdown_plugin_stream_hook_dispatcher(timeout=2)


def test_stream_observers_keep_profile_context_and_independent_lifetimes(monkeypatch, tmp_path):
    from pathlib import Path

    from agent import plugin_stream_hooks as hooks, secret_scope
    from gateway.run import _profile_runtime_scope
    from hermes_cli import plugins
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    delivered = queue.Queue()
    registrations = []

    def observe(**payload):
        delivered.put((payload["delta"], get_hermes_home(),
                       secret_scope.get_secret("OBSERVER_TEST_KEY"), threading.current_thread()))

    hooks.shutdown_plugin_stream_hook_dispatcher()
    try:
        for home in (home_a, home_b):
            with _profile_runtime_scope(home, {"OBSERVER_TEST_KEY": home.name}):
                context = plugins.PluginContext(
                    plugins.PluginManifest(name="observer", key="observer"), plugins.get_plugin_manager()
                )
                # A shared callable must still have a separate worker in each profile.
                registrations.append(context.register_hook("on_stream_delta", observe))

        workers = []
        for home, delta, expected_secret in (
            (home_a, "a1", "a"), (home_b, "b1", "b"), (home_a, "a2", "a-rotated")
        ):
            with _profile_runtime_scope(home, {"OBSERVER_TEST_KEY": expected_secret}):
                assert hooks.enqueue_plugin_stream_hook("on_stream_delta", delta=delta)
            actual_delta, actual_home, secret, worker = delivered.get(timeout=2)
            assert (actual_delta, actual_home, secret) == (delta, home, expected_secret)
            workers.append(worker)

        assert workers[0] is workers[2]
        assert workers[0] is not workers[1]
        assert workers[0].is_alive()
        with _profile_runtime_scope(home_b, {"OBSERVER_TEST_KEY": "b"}):
            registrations[1].dispose()
            assert not hooks.enqueue_plugin_stream_hook("on_stream_delta", delta="removed")
        workers[1].join(timeout=2)
        assert not workers[1].is_alive()
        assert workers[0].is_alive()
        with _profile_runtime_scope(home_a, {}):
            assert hooks.enqueue_plugin_stream_hook("on_stream_delta", delta="a3")
        assert delivered.get(timeout=2) == ("a3", home_a, None, workers[0])
    finally:
        for registration in registrations:
            registration.dispose()
        hooks.shutdown_plugin_stream_hook_dispatcher(timeout=2)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_stream_delta_plugin_hook_is_queued_off_token_path(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_stream_delta(**kwargs):
        time.sleep(0.2)
        calls.append(("on_stream_delta", kwargs))

    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_stream_delta": [on_stream_delta]}))

    agent = _agent()

    started = time.monotonic()
    agent._fire_stream_delta("hello")
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    _wait_for(lambda: calls)
    shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_stream_delta"
    assert calls[0][1]["delta"] == "hello"
    assert calls[0][1]["kind"] == "text"
    assert calls[0][1]["model"] == "test/model"
    assert calls[0][1]["provider"] == "openrouter"


def test_stream_delta_plugin_hook_error_does_not_break_streaming(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    ui_deltas = []

    def on_stream_delta(**_kwargs):
        raise RuntimeError("plugin failed")

    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_stream_delta": [on_stream_delta]}))

    agent = _agent()
    agent.stream_delta_callback = ui_deltas.append

    agent._fire_stream_delta("still visible")
    shutdown_plugin_stream_hook_dispatcher()

    assert ui_deltas == ["still visible"]


def test_stream_hook_queue_drops_oldest_pending_event_when_full(monkeypatch):
    from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    monkeypatch.setattr("agent.plugin_stream_hooks._QUEUE_SIZE", 1)
    delivered = []
    first_delivered = threading.Event()
    release_worker = threading.Event()

    def on_stream_delta(**kwargs):
        delivered.append(kwargs["delta"])
        first_delivered.set()
        release_worker.wait(timeout=1.0)

    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_stream_delta": [on_stream_delta]}))

    assert enqueue_plugin_stream_hook("on_stream_delta", delta="first") is True
    assert first_delivered.wait(timeout=1.0)
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="second") is True
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="third") is True

    release_worker.set()
    _wait_for(lambda: "third" in delivered)
    shutdown_plugin_stream_hook_dispatcher()

    assert delivered == ["first", "third"]


def test_stream_hook_queue_isolated_per_consumer(monkeypatch):
    from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    monkeypatch.setattr("agent.plugin_stream_hooks._QUEUE_SIZE", 1)
    slow_delivered = []
    fast_delivered = []
    slow_started = threading.Event()
    release_slow = threading.Event()

    def slow_consumer(**kwargs):
        slow_delivered.append(kwargs["delta"])
        slow_started.set()
        release_slow.wait(timeout=1.0)

    def fast_consumer(**kwargs):
        fast_delivered.append(kwargs["delta"])

    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        _callbacks({"on_stream_delta": [slow_consumer, fast_consumer]}),
    )

    assert enqueue_plugin_stream_hook("on_stream_delta", delta="first") is True
    assert slow_started.wait(timeout=1.0)
    _wait_for(lambda: fast_delivered == ["first"])
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="second") is True
    _wait_for(lambda: fast_delivered == ["first", "second"])
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="third") is True

    _wait_for(lambda: fast_delivered == ["first", "second", "third"])
    release_slow.set()
    _wait_for(lambda: "third" in slow_delivered)
    shutdown_plugin_stream_hook_dispatcher()

    assert slow_delivered == ["first", "third"]


def test_reasoning_stream_delta_plugin_hook_is_opt_in(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_stream_delta(**kwargs):
        calls.append(("on_stream_delta", kwargs))

    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_stream_delta": [on_stream_delta]}))

    agent = _agent()
    agent._fire_reasoning_delta("private chain")
    shutdown_plugin_stream_hook_dispatcher()

    assert calls == []

    with patch("hermes_cli.config.cfg_get", return_value=True):
        agent._fire_reasoning_delta("visible reasoning")
        _wait_for(lambda: calls)
        shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_stream_delta"
    assert calls[0][1]["kind"] == "reasoning"
    assert calls[0][1]["delta"] == "visible reasoning"


def test_interim_message_plugin_hook_is_queued(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_interim_message(**kwargs):
        calls.append(("on_interim_message", kwargs))

    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_interim_message": [on_interim_message]}))

    agent = _agent()
    agent._emit_interim_assistant_message({"content": "I will inspect the files first."})
    _wait_for(lambda: calls)
    shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_interim_message"
    assert calls[0][1]["text"] == "I will inspect the files first."
    assert calls[0][1]["already_streamed"] is False


def test_stream_plugin_hook_counts_as_stream_consumer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_stream_delta": [lambda **_kwargs: None]}))

    agent = _agent()

    assert agent._has_stream_consumers() is True


def test_interim_message_plugin_hook_does_not_count_as_stream_consumer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_interim_message": [lambda **_kwargs: None]}))

    agent = _agent()

    assert agent._has_stream_consumers() is False


def test_stream_lifecycle_plugin_hooks_are_queued(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_stream_start(**kwargs):
        calls.append(("on_stream_start", kwargs))

    def on_stream_end(**kwargs):
        calls.append(("on_stream_end", kwargs))

    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        _callbacks({"on_stream_start": [on_stream_start], "on_stream_end": [on_stream_end]}),
    )

    agent = _agent()
    agent._emit_stream_start()
    agent._emit_stream_end(final_text="done", finished=True, error=None)
    _wait_for(lambda: len(calls) == 2)
    shutdown_plugin_stream_hook_dispatcher()

    # start/end are delivered by separate per-callback workers; cross-hook
    # arrival order is not guaranteed. Assert content, not interleaving.
    assert sorted(call[0] for call in calls) == ["on_stream_end", "on_stream_start"]
    start_call = next(call for call in calls if call[0] == "on_stream_start")
    end_call = next(call for call in calls if call[0] == "on_stream_end")
    assert start_call[1]["model"] == "test/model"
    assert end_call[1]["final_text"] == "done"
    assert end_call[1]["finished"] is True
    assert end_call[1]["error"] is None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_chat_completion_stream_emits_lifecycle_hooks(_mock_close, mock_create, monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        _callbacks(
            {
                "on_stream_start": [lambda **kwargs: calls.append(("on_stream_start", kwargs))],
                "on_stream_delta": [lambda **kwargs: calls.append(("on_stream_delta", kwargs))],
                "on_stream_end": [lambda **kwargs: calls.append(("on_stream_end", kwargs))],
            }
        ),
    )

    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: iter([
                    _make_stream_chunk(content="hello "),
                    _make_stream_chunk(content="world"),
                    _make_stream_chunk(finish_reason="stop"),
                ])
            )
        )
    )
    mock_create.return_value = mock_client

    agent = _agent()
    agent.api_mode = "chat_completions"
    response = agent._interruptible_streaming_api_call({})

    _wait_for(lambda: len(calls) == 4)
    shutdown_plugin_stream_hook_dispatcher()

    assert response.choices[0].message.content == "hello world"
    # The dispatcher runs ONE worker per callback, so ordering is guaranteed
    # only per hook, not across hooks: the three callbacks here append from
    # three concurrent worker threads. Assert the per-hook contract instead
    # of a strict global interleaving (which is racy by design).
    names = [call[0] for call in calls]
    assert sorted(names) == [
        "on_stream_delta",
        "on_stream_delta",
        "on_stream_end",
        "on_stream_start",
    ]
    delta_texts = [call[1]["delta"] for call in calls if call[0] == "on_stream_delta"]
    assert delta_texts == ["hello ", "world"]  # in-order within the hook
    end_call = next(call for call in calls if call[0] == "on_stream_end")
    assert end_call[1]["final_text"] == "hello world"
    assert end_call[1]["finished"] is True


def test_bedrock_reasoning_delta_reaches_plugin_only_observer(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_stream_delta(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", _callbacks({"on_stream_delta": [on_stream_delta]}))
    monkeypatch.setattr("hermes_cli.config.cfg_get", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "agent.bedrock_adapter._get_bedrock_runtime_client",
        lambda _region: SimpleNamespace(converse_stream=lambda **_kwargs: {"stream": []}),
    )
    monkeypatch.setattr("agent.bedrock_adapter.is_stale_connection_error", lambda _exc: False)
    monkeypatch.setattr("agent.bedrock_adapter.is_streaming_access_denied_error", lambda _exc: False)
    monkeypatch.setattr("agent.bedrock_adapter.invalidate_runtime_client", lambda *_args, **_kwargs: None)

    def stream_converse_with_callbacks(
        _raw_response,
        *,
        on_text_delta=None,
        on_tool_start=None,
        on_reasoning_delta=None,
        on_interrupt_check=None,
        on_event=None,
        **_kwargs,
    ):
        # Main's Bedrock path also invokes this as a Relay finalizer with the
        # intercepted-event replay; only the live pass wires callbacks.
        if on_reasoning_delta is not None:
            assert on_tool_start is not None
            assert on_interrupt_check() is False
            on_reasoning_delta("bedrock reasoning")
        return SimpleNamespace(choices=[], usage=None, stop_reason="end_turn")

    monkeypatch.setattr("agent.bedrock_adapter.stream_converse_with_callbacks", stream_converse_with_callbacks)

    agent = _agent()
    agent.api_mode = "bedrock_converse"
    agent.reasoning_callback = None
    agent.stream_delta_callback = None

    agent._interruptible_streaming_api_call({"__bedrock_region__": "us-east-1", "__bedrock_converse__": True})
    _wait_for(lambda: calls)
    shutdown_plugin_stream_hook_dispatcher()

    assert calls[0]["kind"] == "reasoning"
    assert calls[0]["delta"] == "bedrock reasoning"
