"""Reasoning deltas reach adapters only when the operator opts in.

Invariant contract (``plugins.stream_reasoning_deltas``):

1. Gate OFF (the default) — the consumer's ``on_reasoning`` no-ops, so no
   adapter ever sees chain-of-thought.  This is the prompt-caching / privacy
   invariant: nothing changes for non-opted installs.

2. Gate ON — a reasoning delta fed to the consumer surfaces on the adapter's
   ``render_reasoning_event`` hook (the first-class sink).

3. The structured-event dispatcher routes a ``Reasoning`` event onto the same
   adapter hook.

4. The turn wiring attaches a reasoning callback and unmutes the consumer
   only when ``stream_reasoning_deltas_enabled()`` is True.
"""

import asyncio
import queue

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _RecordingAdapter:
    """Minimal delivery adapter that records reasoning rendered onto it."""

    def __init__(self):
        self.reasoning_calls = []
        self.deltas = []

    def render_reasoning_event(self, event, sink):
        self.reasoning_calls.append(event.text)

    def on_delta(self, text):
        self.deltas.append(text)

    # Class-level attrs the consumer reads off the class (MagicMock-safe).
    draft_stream_is_message = False
    stream_is_message_for_chat = None

    MAX_MESSAGE_LENGTH = 4096

    def message_len_fn_for_chat(self, chat_id):
        return len


def _consumer(adapter, *, reasoning_enabled=False):
    cfg = StreamConsumerConfig(
        transport="edit", chat_type="dm",
        edit_interval=0.01, buffer_threshold=1, cursor="",
    )
    sc = GatewayStreamConsumer(adapter, "C1", cfg)
    sc.stream_reasoning_enabled = reasoning_enabled
    return sc


def test_reasoning_gate_off_noops_on_consumer():
    """Default consumer: an on_reasoning delta must never reach the adapter."""
    adapter = _RecordingAdapter()
    sc = _consumer(adapter, reasoning_enabled=False)
    sc.on_reasoning("private chain of thought")
    assert adapter.reasoning_calls == []
    # And nothing was queued for delivery.
    with pytest.raises(queue.Empty):
        sc._queue.get_nowait()


def test_reasoning_gate_off_is_default():
    """stream_reasoning_enabled must default False so non-opted installs are silent."""
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(transport="edit", chat_type="dm")
    sc = GatewayStreamConsumer(adapter, "C1", cfg)
    assert sc.stream_reasoning_enabled is False


def test_reasoning_reaches_adapter_when_opted_in():
    """Gate ON: an on_reasoning delta is delivered to the adapter's render hook."""
    adapter = _RecordingAdapter()
    sc = _consumer(adapter, reasoning_enabled=True)

    async def _run():
        task = asyncio.create_task(sc.run())
        # The text path needs real deltas for the edit transport; reasoning alone
        # rides the same tick and must still reach the adapter render.
        sc.on_delta("answer")  # noqa: F841 (keeps the run() edit loop honest)
        sc.on_reasoning("visible chain")
        await asyncio.sleep(0.06)
        sc.finish()
        await task

    asyncio.run(_run())
    assert adapter.reasoning_calls == ["visible chain"]


def test_dispatcher_routes_reasoning_event_to_adapter_hook():
    from gateway.stream_dispatch import GatewayEventDispatcher
    from gateway.stream_events import Reasoning

    adapter = _RecordingAdapter()
    d = GatewayEventDispatcher(adapter, _consumer(adapter))
    d.dispatch(Reasoning("dispatched chain"))
    assert adapter.reasoning_calls == ["dispatched chain"]


def test_turn_wiring_gates_consumer_and_callback(monkeypatch):
    """_setup_stream_consumer unmutes + returns a callback only when opted in."""
    from types import SimpleNamespace

    from gateway.config import StreamingConfig

    from gateway.run_turn_runner import TurnRunner

    class _Adapter(_RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.send_typing_calls = []

    adapter = _Adapter()
    ctx = SimpleNamespace(
        mute_notification_reply=False,
        scheduled_heartbeat=False,
        streaming_tts_consumer_holder=[None],
        user_config={},
        resolve_display_setting=lambda *args: True,
        interim_assistant_messages_enabled=True,
        source=SimpleNamespace(platform=SimpleNamespace(value="realtime"), chat_id="voice"),
        _run_still_current=lambda: True,
        progress_queue=None,
        event_message_id=None,
        _status_thread_metadata=None,
        session_key=None,
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=StreamingConfig()),
        _delivery_adapter_for=lambda source: adapter,
        _build_stream_consumer_config=lambda *args, **kwargs: (StreamingConfig(), None),
    )

    # Gate OFF: no reasoning callback, consumer stays muted.
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.stream_reasoning_deltas_enabled", lambda: False,
    )
    _sc, _delta, _interim, reasoning_cb, _want = TurnRunner(runner, ctx)._setup_stream_consumer("realtime")
    assert reasoning_cb is None
    assert _sc.stream_reasoning_enabled is False

    # Gate ON: callback attached, consumer unmuted, and the callback delivers.
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.stream_reasoning_deltas_enabled", lambda: True,
    )
    _sc2, _delta2, _interim2, reasoning_cb2, _want2 = TurnRunner(runner, ctx)._setup_stream_consumer("realtime")
    assert reasoning_cb2 is not None
    assert _sc2.stream_reasoning_enabled is True
    # The callback is the thread-safe enqueue side (like on_delta); the adapter
    # render happens when the consumer's run loop drains the queue. Assert the
    # reasoning delta was queued for delivery, not a synchronous render.
    reasoning_cb2("wired chain")
    _kind, _text = _sc2._queue.get_nowait()
    assert _text == "wired chain"
