"""Gateway streaming observer hooks (on_stream_delta / on_stream_segment /
on_stream_end).

These let a plugin observe the live token stream a turn produces on a chat
platform — e.g. to mirror tokens to its own side-channel/SSE — without
patching core streaming. They are fired from the stream consumer's producer
API and are gated on ``has_hook`` so the per-token path is free when no
plugin is listening.
"""
from types import SimpleNamespace

import pytest

import hermes_cli.plugins as plugins
from gateway.stream_consumer import GatewayStreamConsumer


@pytest.fixture
def recorded_stream_events():
    """Register recording callbacks for the three stream hooks on the global
    plugin manager and remove exactly them afterwards (test isolation)."""
    mgr = plugins.get_plugin_manager()
    events = []
    added = {}
    for hook in ("on_stream_delta", "on_stream_segment", "on_stream_end"):
        def _cb(_hook=hook, **kw):
            events.append((_hook, kw))
        mgr._hooks.setdefault(hook, []).append(_cb)
        added[hook] = _cb
    try:
        yield events
    finally:
        for hook, cb in added.items():
            try:
                mgr._hooks.get(hook, []).remove(cb)
            except ValueError:
                pass


def _consumer(metadata=None, message_id="m-1"):
    sc = GatewayStreamConsumer(SimpleNamespace(), "chat-9", metadata=metadata)
    sc._message_id = message_id
    return sc


def test_producer_api_fires_stream_hooks_in_order(recorded_stream_events):
    sc = _consumer(metadata={"platform": "matrix", "session": "s1"})

    sc.on_delta("hello")
    sc.on_segment_break()
    sc.finish()

    names = [name for name, _ in recorded_stream_events]
    assert names == ["on_stream_delta", "on_stream_segment", "on_stream_end"]

    delta_kw = recorded_stream_events[0][1]
    assert delta_kw["delta"] == "hello"
    assert delta_kw["chat_id"] == "chat-9"
    assert delta_kw["metadata"] == {"platform": "matrix", "session": "s1"}
    assert delta_kw["message_id"] == "m-1"

    assert recorded_stream_events[2][1]["reason"] == "done"


def test_delta_hook_not_fired_for_empty_or_boundary_delta(recorded_stream_events):
    sc = _consumer()

    sc.on_delta("")        # empty: nothing queued, no delta event
    sc.on_delta(None)      # tool boundary: routes to segment break, not delta

    names = [name for name, _ in recorded_stream_events]
    assert "on_stream_delta" not in names
    assert names == ["on_stream_segment"]


def test_streaming_still_queues_when_no_hook_registered(monkeypatch):
    """The gate must short-circuit before invoke_hook when nothing listens, and
    normal streaming (queueing the delta) must be unaffected."""
    called = []
    monkeypatch.setattr(plugins, "has_hook", lambda name: False)
    monkeypatch.setattr(
        plugins, "invoke_hook", lambda *a, **k: called.append((a, k))
    )

    sc = _consumer()
    sc.on_delta("hello")

    assert called == []                     # gated out — invoke_hook never ran
    assert sc._queue.get_nowait() == "hello"  # delta still queued for delivery


def test_hook_callback_exception_does_not_break_streaming(monkeypatch):
    """A misbehaving observer must not break token delivery."""
    mgr = plugins.get_plugin_manager()

    def _boom(**kw):
        raise RuntimeError("bad plugin")

    mgr._hooks.setdefault("on_stream_delta", []).append(_boom)
    try:
        sc = _consumer()
        sc.on_delta("hello")  # must not raise
        assert sc._queue.get_nowait() == "hello"
    finally:
        mgr._hooks.get("on_stream_delta", []).remove(_boom)


def test_stream_hooks_are_registered_valid_hooks():
    for hook in ("on_stream_delta", "on_stream_segment", "on_stream_end"):
        assert hook in plugins.VALID_HOOKS
