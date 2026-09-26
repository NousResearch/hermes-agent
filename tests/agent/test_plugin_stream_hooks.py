import threading
import time
from concurrent.futures import ThreadPoolExecutor

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


def _wait_for(predicate, timeout=5.0):
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


def _patch_test_callback_manager(monkeypatch, callbacks_by_hook):
    """Use the explicit lock-free manager double for tests that inject callbacks directly."""
    from hermes_cli import plugins

    hooks = {name: list(callbacks) for name, callbacks in callbacks_by_hook.items()}
    manager = SimpleNamespace(
        _hooks=hooks,
        _observer_dispatcher_scope=object(),
        _report_hook_failure=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "iter_hook_callbacks", lambda name: tuple(hooks.get(name, ())))
    return manager


def test_stream_delta_plugin_hook_is_queued_off_token_path(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    callback_started = threading.Event()
    release_callback = threading.Event()
    dispatch_returned = threading.Event()

    def on_stream_delta(**kwargs):
        callback_started.set()
        release_callback.wait(timeout=10.0)
        calls.append(("on_stream_delta", kwargs))

    _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [on_stream_delta]})
    agent = _agent()
    producer = threading.Thread(
        target=lambda: (agent._fire_stream_delta("hello"), dispatch_returned.set())
    )
    try:
        producer.start()
        assert callback_started.wait(timeout=5.0)
        assert dispatch_returned.wait(timeout=5.0)
    finally:
        release_callback.set()
        producer.join(timeout=5.0)
        shutdown_plugin_stream_hook_dispatcher(timeout=5.0)

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

    _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [on_stream_delta]})

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
        release_worker.wait(timeout=5.0)

    _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [on_stream_delta]})

    assert enqueue_plugin_stream_hook("on_stream_delta", delta="first") is True
    assert first_delivered.wait(timeout=5.0)
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
        release_slow.wait(timeout=5.0)

    def fast_consumer(**kwargs):
        fast_delivered.append(kwargs["delta"])

    _patch_test_callback_manager(
        monkeypatch, {"on_stream_delta": [slow_consumer, fast_consumer]}
    )

    assert enqueue_plugin_stream_hook("on_stream_delta", delta="first") is True
    assert slow_started.wait(timeout=5.0)
    _wait_for(lambda: fast_delivered == ["first"])
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="second") is True
    _wait_for(lambda: fast_delivered == ["first", "second"])
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="third") is True

    _wait_for(lambda: fast_delivered == ["first", "second", "third"])
    release_slow.set()
    _wait_for(lambda: "third" in slow_delivered)
    shutdown_plugin_stream_hook_dispatcher()

    assert slow_delivered == ["first", "third"]


def test_stream_observers_keep_context_and_dispatcher_scope_per_profile(
    monkeypatch, tmp_path
):
    """Concurrent profile events use each profile's home and callback."""
    from hermes_cli import plugins
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from agent.plugin_stream_hooks import (
        enqueue_plugin_observer_hook,
        shutdown_plugin_observer_dispatcher,
    )

    shutdown_plugin_observer_dispatcher()
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    manager_a = plugins.PluginManager(scope_key=str(home_a))
    manager_b = plugins.PluginManager(scope_key=str(home_b))
    manager_a._discovered = True
    manager_b._discovered = True
    delivered = []
    delivered_lock = threading.Lock()
    delivered_event = threading.Event()

    def callback_a(**kwargs):
        with delivered_lock:
            delivered.append(("a", str(get_hermes_home()), kwargs["event_id"]))
            if len(delivered) == 2:
                delivered_event.set()

    def callback_b(**kwargs):
        with delivered_lock:
            delivered.append(("b", str(get_hermes_home()), kwargs["event_id"]))
            if len(delivered) == 2:
                delivered_event.set()

    plugins.PluginContext(
        plugins.PluginManifest(name="profile-a-plugin"), manager_a
    ).register_hook("memory_prefetch", callback_a)
    plugins.PluginContext(
        plugins.PluginManifest(name="profile-b-plugin"), manager_b
    ).register_hook("memory_prefetch", callback_b)

    def manager_for_active_home():
        return manager_a if get_hermes_home() == home_a else manager_b

    monkeypatch.setattr(plugins, "get_plugin_manager", manager_for_active_home)

    def emit(home, event_id):
        token = set_hermes_home_override(home)
        try:
            return enqueue_plugin_observer_hook("memory_prefetch", event_id=event_id)
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(emit, (home_a, home_b), ("event-a", "event-b"))) == {True}

    assert delivered_event.wait(timeout=5.0)
    shutdown_plugin_observer_dispatcher()
    assert sorted(delivered) == sorted(
        [
            ("a", str(home_a), "event-a"),
            ("b", str(home_b), "event-b"),
        ]
    )


def test_observer_failures_report_in_originating_profile(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    from hermes_cli import plugins
    from agent.plugin_stream_hooks import enqueue_plugin_observer_hook, shutdown_plugin_observer_dispatcher

    shutdown_plugin_observer_dispatcher()
    homes = [tmp_path / "a", tmp_path / "b"]
    managers = {home: plugins.PluginManager(scope_key=str(home)) for home in homes}
    reports = []

    def failing(**kwargs):
        raise ValueError(kwargs["event_id"])

    for home, manager in managers.items():
        manager._discovered = True
        plugins.PluginContext(plugins.PluginManifest(name="failure-plugin"), manager).register_hook(
            "memory_prefetch", failing
        )
        def report(hook, callback, payload, exc, owner=home):
            reports.append((owner, get_hermes_home(), payload["event_id"], str(exc)))
        monkeypatch.setattr(manager, "_report_hook_failure", report)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: managers[get_hermes_home()])
    try:
        for index, home in enumerate([homes[0], homes[1], homes[0]]):
            token = set_hermes_home_override(home)
            try:
                assert enqueue_plugin_observer_hook("memory_prefetch", event_id=str(index))
            finally:
                reset_hermes_home_override(token)
        shutdown_plugin_observer_dispatcher(timeout=5.0)
        assert sorted(reports) == sorted(
            (home, home, str(index), str(index))
            for index, home in enumerate([homes[0], homes[1], homes[0]])
        )
    finally:
        shutdown_plugin_observer_dispatcher()


def test_observer_dispatcher_shutdown_has_bounded_join(monkeypatch):
    from agent.plugin_stream_hooks import (
        enqueue_plugin_observer_hook,
        shutdown_plugin_observer_dispatcher,
    )

    shutdown_plugin_observer_dispatcher()
    started = threading.Event()
    release = threading.Event()

    def blocked_consumer(**_kwargs):
        started.set()
        release.wait(timeout=10.0)

    _patch_test_callback_manager(monkeypatch, {"memory_prefetch": [blocked_consumer]})
    assert enqueue_plugin_observer_hook("memory_prefetch", turn_id="turn") is True
    assert started.wait(timeout=5.0)

    shutdown_returned = threading.Event()

    def shutdown():
        shutdown_plugin_observer_dispatcher(timeout=2.0)
        shutdown_returned.set()

    shutdown_thread = threading.Thread(target=shutdown)
    try:
        shutdown_thread.start()
        assert shutdown_returned.wait(timeout=5.0)
        assert not release.is_set()
    finally:
        release.set()
        shutdown_thread.join(timeout=5.0)


def test_observer_dispatcher_shutdown_drains_pending_events(monkeypatch):
    from agent.plugin_stream_hooks import (
        enqueue_plugin_observer_hook,
        shutdown_plugin_observer_dispatcher,
    )

    shutdown_plugin_observer_dispatcher()
    delivered = []
    first_started = threading.Event()
    release_first = threading.Event()

    def consumer(**kwargs):
        delivered.append(kwargs["event_id"])
        if kwargs["event_id"] == "first":
            first_started.set()
            release_first.wait(timeout=5.0)

    _patch_test_callback_manager(monkeypatch, {"memory_prefetch": [consumer]})

    assert enqueue_plugin_observer_hook("memory_prefetch", event_id="first") is True
    assert first_started.wait(timeout=5.0)
    assert enqueue_plugin_observer_hook("memory_prefetch", event_id="second") is True
    release_first.set()
    shutdown_plugin_observer_dispatcher(timeout=5.0)

    assert delivered == ["first", "second"]


def test_reasoning_stream_delta_plugin_hook_is_opt_in(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def on_stream_delta(**kwargs):
        calls.append(("on_stream_delta", kwargs))

    _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [on_stream_delta]})

    agent = _agent()
    agent._fire_reasoning_delta("private chain")
    shutdown_plugin_stream_hook_dispatcher()

    assert calls == []

    # The opt-in is resolved once per stream; a new request picks up the flipped flag.
    agent._reset_stream_delivery_tracking()
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

    _patch_test_callback_manager(monkeypatch, {"on_interim_message": [on_interim_message]})

    agent = _agent()
    agent._emit_interim_assistant_message({"content": "I will inspect the files first."})
    _wait_for(lambda: calls)
    shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_interim_message"
    assert calls[0][1]["text"] == "I will inspect the files first."
    assert calls[0][1]["already_streamed"] is False


def test_stream_plugin_hook_counts_as_stream_consumer(monkeypatch):
    _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [lambda **_kwargs: None]})

    agent = _agent()

    assert agent._has_stream_consumers() is True


def test_interim_message_plugin_hook_does_not_count_as_stream_consumer(monkeypatch):
    _patch_test_callback_manager(monkeypatch, {"on_interim_message": [lambda **_kwargs: None]})

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

    _patch_test_callback_manager(
        monkeypatch,
        {"on_stream_start": [on_stream_start], "on_stream_end": [on_stream_end]},
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
    _patch_test_callback_manager(
        monkeypatch,
        {
            "on_stream_start": [lambda **kwargs: calls.append(("on_stream_start", kwargs))],
            "on_stream_delta": [lambda **kwargs: calls.append(("on_stream_delta", kwargs))],
            "on_stream_end": [lambda **kwargs: calls.append(("on_stream_end", kwargs))],
        },
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

    _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [on_stream_delta]})
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


def test_inline_think_reaches_reasoning_pane_unless_native_reasoning_streamed():
    """#89647: inline <think> text stripped from content feeds reasoning_callback (the live pane), but not
    once the provider streamed native reasoning for this response (no double reasoning)."""
    agent = _agent()
    seen = []
    agent.reasoning_callback = seen.append
    agent._reset_stream_delivery_tracking()
    for delta in ["<think>", "Let me", " check config", "</think>", "The answer is 42."]:
        agent._fire_stream_delta(delta)
    assert "".join(seen) == "Let me check config"

    seen.clear()
    agent._reset_stream_delivery_tracking()
    agent._fire_reasoning_delta("native")
    agent._fire_stream_delta("<think>dup</think>ok")
    assert seen == ["native"]


def test_finish_chat_stream_recovers_inline_reasoning_content():
    """#89647: with no reasoning delta, reasoning_content comes from the <think> blocks in raw content."""
    from agent import chat_completion_helpers as cch

    call = cch._StreamingCall.__new__(cch._StreamingCall)
    call.agent = _agent()
    deltas = ["<think>", "Let me", " check config", "</think>", "The answer is 42."]
    resp = call._finish_chat_stream(None, "assistant", deltas, [], {}, "stop", "MiniMax-M3", None,
                                    flush_pending=lambda: None)
    assert resp.choices[0].message.reasoning_content == "Let me check config"


def test_manager_unload_retires_only_its_observers_across_reload(monkeypatch, tmp_path):
    """Unloading one cached profile manager discards its pending old-generation events.

    The same callback object is deliberately re-registered after unload: manager identity and
    callback identity alone must not make the old dispatcher/queue current again. A second
    profile's live dispatcher must survive the A -> B -> A transition.
    """
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    psh.shutdown_plugin_observer_dispatcher()
    enqueue_plugin_observer_hook = psh.enqueue_plugin_observer_hook
    shutdown_plugin_observer_dispatcher = psh.shutdown_plugin_observer_dispatcher
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))

    def manager_at(home):
        token = set_hermes_home_override(home)
        try:
            return plugins.get_plugin_manager()
        finally:
            reset_hermes_home_override(token)

    manager_a, manager_b = manager_at(home_a), manager_at(home_b)
    delivered = []
    delivered_lock = threading.Lock()
    a_started = threading.Event()
    release_a = threading.Event()
    active_generation_delivered = threading.Event()

    def record(profile, event_id):
        with delivered_lock:
            delivered.append((profile, str(get_hermes_home()), event_id))
            ids = {entry[2] for entry in delivered}
            if {"b-still-active", "a-new-generation"}.issubset(ids):
                active_generation_delivered.set()

    def on_a(**kwargs):
        event_id = kwargs["event_id"]
        if event_id == "a-running":
            a_started.set()
            release_a.wait(timeout=10.0)
        record("a", event_id)

    def on_b(**kwargs):
        record("b", kwargs["event_id"])

    manager_a._discovered = manager_b._discovered = True
    plugins.PluginContext(plugins.PluginManifest(name="observer-a"), manager_a).register_hook(
        "memory_prefetch", on_a
    )
    plugins.PluginContext(plugins.PluginManifest(name="observer-b"), manager_b).register_hook(
        "memory_prefetch", on_b
    )

    def emit(home, event_id):
        token = set_hermes_home_override(home)
        try:
            return enqueue_plugin_observer_hook("memory_prefetch", event_id=event_id)
        finally:
            reset_hermes_home_override(token)

    stop_entered = threading.Event()
    allow_stop = threading.Event()
    unload_returned = threading.Event()
    queue_drained = threading.Event()
    unload_errors = []
    unload_thread = None
    drainer = None
    original_stop = psh._stop_dispatcher

    def pause_stop(dispatcher, timeout=5.0, *, discard_pending=False):
        if dispatcher is a_dispatcher:
            stop_entered.set()
            assert allow_stop.wait(timeout=5.0)
        return original_stop(dispatcher, timeout, discard_pending=discard_pending)

    def unload_a():
        try:
            manager_a.unload()
        except BaseException as exc:
            unload_errors.append(exc)
        finally:
            unload_returned.set()

    try:
        assert emit(home_a, "a-running")
        assert a_started.wait(timeout=5.0)
        token = set_hermes_home_override(home_a)
        try:
            a_dispatcher = psh._dispatchers_for("memory_prefetch")[0]
            a_worker = a_dispatcher.thread
        finally:
            reset_hermes_home_override(token)
        assert a_worker is not None
        assert emit(home_a, "a-pending-old-generation")

        # Pause after the manager lock has been released but before queue discard/stop signalling.
        # The running callback may finish; the queued callback must be rejected at its retirement gate.
        monkeypatch.setattr(psh, "_stop_dispatcher", pause_stop)
        unload_thread = threading.Thread(target=unload_a)
        unload_thread.start()
        assert stop_entered.wait(timeout=5.0)
        assert a_dispatcher.retired
        release_a.set()

        def wait_for_queue_drain():
            a_dispatcher.events.join()
            queue_drained.set()

        drainer = threading.Thread(target=wait_for_queue_drain)
        drainer.start()
        assert queue_drained.wait(timeout=5.0)
        with delivered_lock:
            assert ("a", str(home_a), "a-running") in delivered
            assert ("a", str(home_a), "a-pending-old-generation") not in delivered

        allow_stop.set()
        assert unload_returned.wait(timeout=5.0)
        unload_thread.join(timeout=5.0)
        assert not unload_errors
        plugins.PluginContext(plugins.PluginManifest(name="observer-a"), manager_a).register_hook(
            "memory_prefetch", on_a
        )

        # B is an active cached profile, not dead state to reap just because A is now active.
        assert emit(home_b, "b-still-active")
        token = set_hermes_home_override(home_b)
        try:
            b_worker = psh._dispatchers_for("memory_prefetch")[0].thread
        finally:
            reset_hermes_home_override(token)
        assert b_worker is not None
        assert emit(home_a, "a-new-generation")
        assert active_generation_delivered.wait(timeout=5.0)

        with delivered_lock:
            assert ("a", str(home_a), "a-new-generation") in delivered
            assert ("b", str(home_b), "b-still-active") in delivered
            assert ("a", str(home_a), "a-pending-old-generation") not in delivered
        a_worker.join(timeout=5.0)
        assert not a_worker.is_alive()
        assert b_worker.is_alive()
    finally:
        release_a.set()
        allow_stop.set()
        if unload_thread is not None:
            unload_thread.join(timeout=5.0)
        if drainer is not None:
            drainer.join(timeout=5.0)
        shutdown_plugin_observer_dispatcher(timeout=5.0)


def test_stale_enqueue_racing_manager_unload_cannot_enter_new_generation(monkeypatch, tmp_path):
    """A callback snapshot taken before unload is rejected after the manager token rotates."""
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    psh.shutdown_plugin_observer_dispatcher()
    home = tmp_path / "race-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)

    delivered = []
    delivered_lock = threading.Lock()
    callback_started = threading.Event()
    release_callback = threading.Event()
    new_generation_delivered = threading.Event()

    def observer(**kwargs):
        event_id = kwargs["event_id"]
        if event_id == "running":
            callback_started.set()
            release_callback.wait(timeout=10.0)
        with delivered_lock:
            delivered.append(event_id)
            if event_id == "new-generation-event":
                new_generation_delivered.set()

    manager._discovered = True
    plugins.PluginContext(plugins.PluginManifest(name="race-observer"), manager).register_hook(
        "memory_prefetch", observer
    )

    def emit(event_id):
        scoped_token = set_hermes_home_override(home)
        try:
            return psh.enqueue_plugin_observer_hook("memory_prefetch", event_id=event_id)
        finally:
            reset_hermes_home_override(scoped_token)

    original_lookup = psh._registered_callbacks
    lookup_paused = threading.Event()
    resume_lookup = threading.Event()

    def delayed_lookup(hook_name):
        callbacks = original_lookup(hook_name)
        lookup_paused.set()
        assert resume_lookup.wait(timeout=5.0)
        return callbacks

    try:
        assert emit("running")
        assert callback_started.wait(timeout=5.0)
        monkeypatch.setattr(psh, "_registered_callbacks", delayed_lookup)
        queued = {}

        def enqueue_stale_snapshot():
            queued["result"] = emit("stale-race-event")

        emitter = threading.Thread(target=enqueue_stale_snapshot)
        emitter.start()
        assert lookup_paused.wait(timeout=5.0)

        # The enqueue captured the old token, but has not acquired the manager lock or queued yet.
        token = set_hermes_home_override(home)
        try:
            manager.unload()
            plugins.PluginContext(plugins.PluginManifest(name="race-observer"), manager).register_hook(
                "memory_prefetch", observer
            )
        finally:
            reset_hermes_home_override(token)

        resume_lookup.set()
        emitter.join(timeout=5.0)
        assert not emitter.is_alive()
        assert queued["result"] is False
        assert emit("new-generation-event")
        release_callback.set()
        assert new_generation_delivered.wait(timeout=5.0)
        with delivered_lock:
            assert "new-generation-event" in delivered
            assert "stale-race-event" not in delivered
    finally:
        resume_lookup.set()
        release_callback.set()
        psh.shutdown_plugin_observer_dispatcher(timeout=5.0)


def test_lazy_discovery_callback_unloaded_before_scope_validation_is_not_enqueued(
    monkeypatch, tmp_path
):
    """A callback found by lazy discovery is revalidated after a targeted unload."""
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    psh.shutdown_plugin_observer_dispatcher()
    home = tmp_path / "lazy-discovery-race"
    manager = plugins.PluginManager(scope_key=str(home))
    manager._discovered = False
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manifest = plugins.PluginManifest(name="lazy-discovery-observer")
    delivered = threading.Event()

    def observer(**_kwargs):
        delivered.set()

    def discover_on_hook_gate():
        with manager._discovery_lock:
            plugins.PluginContext(manifest, manager).register_hook("memory_prefetch", observer)
            manager._discovered = True

    # Exercise _registered_callbacks' real empty-snapshot -> has_hook lazy-discovery retry.
    monkeypatch.setattr(manager, "discover_and_load", discover_on_hook_gate)
    original_lookup = psh._registered_callbacks
    lookup_complete = threading.Event()
    resume_lookup = threading.Event()
    looked_up = {}

    def pause_after_lookup(hook_name):
        callbacks = original_lookup(hook_name)
        looked_up["callbacks"] = callbacks
        lookup_complete.set()
        assert resume_lookup.wait(timeout=5.0)
        return callbacks

    monkeypatch.setattr(psh, "_registered_callbacks", pause_after_lookup)
    result = {}
    errors = []

    def enqueue_after_lookup():
        token = set_hermes_home_override(home)
        try:
            result["queued"] = psh.enqueue_plugin_observer_hook(
                "memory_prefetch", event_id="removed-after-discovery"
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    emitter = threading.Thread(target=enqueue_after_lookup)
    try:
        emitter.start()
        assert lookup_complete.wait(timeout=5.0)
        assert looked_up["callbacks"] == (observer,)
        old_scope = manager._observer_dispatcher_scope

        # Targeted unload removes this callback without rotating the manager-wide token.
        assert manager.unload(manifest) is True
        assert manager._observer_dispatcher_scope is old_scope
        assert manager.iter_hook_callbacks("memory_prefetch") == ()

        resume_lookup.set()
        emitter.join(timeout=5.0)
        assert not emitter.is_alive()
        assert not errors
        assert result["queued"] is False
        assert not delivered.is_set()
        with psh._dispatcher_lock:
            assert not any(key[0] is old_scope for key in psh._dispatchers)
    finally:
        resume_lookup.set()
        emitter.join(timeout=5.0)
        psh.shutdown_plugin_observer_dispatcher(timeout=5.0)


def test_manager_unload_from_its_observer_worker_does_not_self_join(monkeypatch, tmp_path):
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    psh.shutdown_plugin_observer_dispatcher()
    home = tmp_path / "self-unload-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)

    unloaded = threading.Event()

    def unload_from_worker(**_kwargs):
        manager.unload()
        unloaded.set()

    manager._discovered = True
    plugins.PluginContext(plugins.PluginManifest(name="self-unload-observer"), manager).register_hook(
        "memory_prefetch", unload_from_worker
    )
    token = set_hermes_home_override(home)
    try:
        assert psh.enqueue_plugin_observer_hook("memory_prefetch", event_id="unload")
    finally:
        reset_hermes_home_override(token)

    assert unloaded.wait(timeout=5.0)
    psh.shutdown_plugin_observer_dispatcher(timeout=5.0)


def test_observer_enqueue_does_not_wait_for_manager_lifecycle_lock(monkeypatch, tmp_path):
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    psh.shutdown_plugin_observer_dispatcher()
    home = tmp_path / "busy-manager-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)

    manager._discovered = True
    plugins.PluginContext(plugins.PluginManifest(name="busy-manager-observer"), manager).register_hook(
        "memory_prefetch", lambda **_kwargs: None
    )
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_manager_lock():
        with manager._discovery_lock:
            lock_held.set()
            release_lock.wait(timeout=10.0)

    holder = threading.Thread(target=hold_manager_lock)
    emitter = None
    enqueue_completed = threading.Event()
    enqueue_result = {}
    enqueue_errors = []

    def enqueue_during_lifecycle_lock():
        token = set_hermes_home_override(home)
        try:
            enqueue_result["queued"] = psh.enqueue_plugin_observer_hook(
                "memory_prefetch", event_id="during-unload"
            )
        except BaseException as exc:
            enqueue_errors.append(exc)
        finally:
            reset_hermes_home_override(token)
            enqueue_completed.set()

    holder.start()
    try:
        assert lock_held.wait(timeout=5.0)
        emitter = threading.Thread(target=enqueue_during_lifecycle_lock)
        emitter.start()
        # Completion must be signalled while the lifecycle lock is still held.
        assert enqueue_completed.wait(timeout=5.0)
        assert not enqueue_errors
        assert enqueue_result["queued"] is False
    finally:
        release_lock.set()
        holder.join(timeout=5.0)
        if emitter is not None:
            emitter.join(timeout=5.0)
        psh.shutdown_plugin_observer_dispatcher(timeout=5.0)
    assert not holder.is_alive()
    assert emitter is not None and not emitter.is_alive()
