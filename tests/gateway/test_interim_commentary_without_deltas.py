"""Regression coverage: interim commentary must survive a consumer that never
took any deltas.

With ``streaming.enabled: false`` and ``display.interim_assistant_messages:
true`` the gateway still builds a ``GatewayStreamConsumer`` so that commentary
reaches the platform, but ``stream_deltas_enabled`` is false — this consumer
never receives the reply's deltas. The agent can nonetheless report
``already_streamed`` when its own streaming is feeding an execution observer
rather than this adapter.

``on_segment_break()`` only queues a segment marker and carries no text, so
routing those commentary messages there dropped them: the platform showed
nothing but the final reply. Commentary must go to ``on_commentary()`` unless
this consumer really did accept the deltas.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway import stream_consumer as stream_consumer_mod
from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner


class _FakeConsumer:
    """Records what the interim callback hands the stream consumer."""

    instances: list["_FakeConsumer"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.commentary: list[str] = []
        self.segment_breaks = 0
        self.stream_deltas_enabled = True
        _FakeConsumer.instances.append(self)

    def on_commentary(self, text: str) -> None:
        self.commentary.append(text)

    def on_segment_break(self) -> None:
        self.segment_breaks += 1


def _turn_runner(*, streaming_enabled: bool, supports_editing: bool) -> TurnRunner:
    scfg = SimpleNamespace(enabled_for=lambda plat: streaming_enabled)
    adapter = SimpleNamespace(
        SUPPORTS_MESSAGE_EDITING=supports_editing,
        SUPPORTS_NATIVE_STREAMING=False,
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=scfg),
        _delivery_adapter_for=lambda _source: adapter,
        _build_stream_consumer_config=lambda *_a, **_kw: (object(), None),
    )
    ctx = SimpleNamespace(
        mute_notification_reply=False,
        streaming_tts_consumer_holder=[None],
        stream_consumer_holder=[None],
        user_config={},
        interim_assistant_messages_enabled=True,
        scheduled_heartbeat=False,
        resolve_display_setting=lambda *_a, **_kw: None,
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="42"),
        _status_thread_metadata=None,
        progress_queue=None,
        event_message_id=None,
        _run_still_current=lambda: True,
    )
    turn = object.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = ctx
    turn._approval_owner = None
    return turn


@pytest.fixture(autouse=True)
def _fake_consumer(monkeypatch):
    _FakeConsumer.instances = []
    monkeypatch.setattr(stream_consumer_mod, "GatewayStreamConsumer", _FakeConsumer)
    monkeypatch.setattr("gateway.run_turn_runner.stream_held", lambda: False)
    monkeypatch.setattr(
        "gateway.run_turn_runner._finite_viewer_turn", lambda: False
    )
    return _FakeConsumer


def _interim_cb(*, streaming_enabled: bool, supports_editing: bool):
    turn = _turn_runner(
        streaming_enabled=streaming_enabled, supports_editing=supports_editing
    )
    consumer, _delta_cb, interim_cb, want_interim = turn._setup_stream_consumer("telegram")
    assert want_interim is True
    assert consumer is not None
    return consumer, interim_cb


def test_commentary_reaches_the_consumer_when_deltas_are_off():
    """streaming.enabled: false — the consumer only relays commentary, so a
    message the agent marks ``already_streamed`` (its stream fed an execution
    observer) must still be sent, not traded for a bare segment break."""
    consumer, interim_cb = _interim_cb(streaming_enabled=False, supports_editing=True)
    assert consumer.stream_deltas_enabled is False

    interim_cb("working on it", already_streamed=True)

    assert consumer.commentary == ["working on it"]
    assert consumer.segment_breaks == 0


def test_commentary_is_deduped_when_this_consumer_accepted_the_deltas():
    """Control: with streaming on and an editable adapter the consumer did see
    the deltas, so the commentary is only a segment boundary."""
    consumer, interim_cb = _interim_cb(streaming_enabled=True, supports_editing=True)
    assert consumer.stream_deltas_enabled is True

    interim_cb("working on it", already_streamed=True)

    assert consumer.commentary == []
    assert consumer.segment_breaks == 1


def test_unstreamed_commentary_still_reaches_the_consumer():
    """Unchanged path: nothing streamed, so the text is posted."""
    consumer, interim_cb = _interim_cb(streaming_enabled=True, supports_editing=True)

    interim_cb("working on it")

    assert consumer.commentary == ["working on it"]
    assert consumer.segment_breaks == 0
