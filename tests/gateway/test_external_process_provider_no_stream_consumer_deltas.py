"""An external-process provider (ACP /
CLI-over-stdio, e.g. claude-subscription-directsdk-experimental) never streams
deltas (``agent/turn_api_call.py::_should_stream`` forces ``stream=False`` for
it), so a ``GatewayStreamConsumer`` built for it must come up with
``stream_deltas_enabled=False`` from the start — not just get marked that way
post-hoc for the #105341 interim-only case. Otherwise
``_run_agent_mark_streamed_delivery`` treats the never-fed consumer as one
that COULD have raced the final send and logs a guaranteed-false-positive
"possible duplicate send" warning on every turn of that provider.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import StreamConsumerConfig


def _make_ctx(*, interim_assistant_messages_enabled: bool = False):
    return SimpleNamespace(
        mute_notification_reply=False,
        streaming_tts_consumer_holder=[None],
        scheduled_heartbeat=False,
        user_config={},
        resolve_display_setting=lambda *a, **k: None,
        interim_assistant_messages_enabled=interim_assistant_messages_enabled,
        progress_queue=queue.Queue(),
        stream_consumer_holder=[None],
        source=SimpleNamespace(chat_id="chat-1"),
        event_message_id=None,
        _status_thread_metadata=None,
        _run_still_current=lambda: True,
    )


def _make_runner(adapter):
    class _StreamingConfig:
        def enabled_for(self, _plat_streaming):
            return True

    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=_StreamingConfig()),
        _delivery_adapter_for=lambda source: adapter,
        _build_stream_consumer_config=lambda *a, **k: (StreamConsumerConfig(), None),
    )
    return runner


def _make_editable_adapter():
    from gateway.platforms.base import BasePlatformAdapter

    EditableAdapter = type(
        "EditableAdapter", (BasePlatformAdapter,),
        {"SUPPORTS_MESSAGE_EDITING": True, "SUPPORTS_NATIVE_STREAMING": False},
    )
    EditableAdapter.__abstractmethods__ = frozenset()
    return EditableAdapter.__new__(EditableAdapter)


def _turn_runner_with_ctx(ctx, adapter):
    runner = _make_runner(adapter)
    tr = TurnRunner.__new__(TurnRunner)
    tr._runner = runner
    tr._ctx = ctx
    return tr


def _setup_external(ctx):
    tr = _turn_runner_with_ctx(ctx, _make_editable_adapter())
    with patch(
        "hermes_cli.runtime_provider_backends._is_external_process_provider",
        return_value=True,
    ):
        return tr._setup_stream_consumer(
            "telegram", provider="claude-subscription-directsdk-experimental",
        )


def test_external_process_provider_builds_no_delta_consumer():
    """Streaming on, interim commentary off: an external-process provider gets no
    stream consumer at all — there is nothing it could ever be fed, so the
    duplicate-risk diagnostic has nothing to fire on."""
    consumer, delta_cb, _interim_cb, _want_interim = _setup_external(_make_ctx())
    assert consumer is None
    assert delta_cb is None


def test_external_process_provider_interim_consumer_is_not_a_delta_consumer():
    """Interim commentary on: the consumer still exists (it relays commentary) but
    must come up with stream_deltas_enabled=False and no delta callback."""
    consumer, delta_cb, _interim_cb, _want_interim = _setup_external(
        _make_ctx(interim_assistant_messages_enabled=True))
    assert consumer is not None
    assert consumer.stream_deltas_enabled is False
    assert delta_cb is None


def test_ordinary_provider_keeps_stream_deltas_enabled():
    """Control: a normal streaming-capable provider is unaffected."""
    ctx = _make_ctx()
    tr = _turn_runner_with_ctx(ctx, _make_editable_adapter())
    with patch(
        "hermes_cli.runtime_provider_backends._is_external_process_provider",
        return_value=False,
    ):
        consumer, delta_cb, interim_cb, want_interim = tr._setup_stream_consumer(
            "telegram", provider="anthropic",
        )
    assert consumer is not None
    assert consumer.stream_deltas_enabled is True


def test_no_provider_argument_defaults_to_streaming_capable():
    """Backward compat: callers that don't pass provider keep the old behavior."""
    ctx = _make_ctx()
    tr = _turn_runner_with_ctx(ctx, _make_editable_adapter())
    consumer, delta_cb, interim_cb, want_interim = tr._setup_stream_consumer("telegram")
    assert consumer is not None
    assert consumer.stream_deltas_enabled is True
