"""Regression coverage for served-profile streaming configuration (#136972)."""

from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform, StreamingConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource


class _Consumer:
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.stream_deltas_enabled = False


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm")


def _adapter():
    return SimpleNamespace(SUPPORTS_MESSAGE_EDITING=True)


def _consumer_config(_source, config, _adapter, **_kwargs):
    return SimpleNamespace(transport=config.transport), None


@pytest.mark.parametrize(
    ("host_enabled", "served_streaming", "expect_consumer"),
    [
        (False, {"enabled": True, "transport": "edit"}, True),
        (True, {"enabled": False, "transport": "edit"}, False),
        (True, None, True),
    ],
)
def test_stream_consumer_uses_served_profile_streaming_config(
    monkeypatch, host_enabled, served_streaming, expect_consumer,
):
    """The normal turn path must not borrow the multiplex host's master switch."""
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _Consumer)
    source = _source()
    ctx = SimpleNamespace(
        mute_notification_reply=False,
        streaming_tts_consumer_holder=[None],
        user_config={} if served_streaming is None else {"streaming": served_streaming},
        resolve_display_setting=lambda *_args: None,
        scheduled_heartbeat=False,
        interim_assistant_messages_enabled=False,
        source=source,
        _status_thread_metadata=None,
        progress_queue=None,
        event_message_id=None,
        _run_still_current=lambda: True,
        stream_consumer_holder=[None],
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=StreamingConfig(enabled=host_enabled)),
        _delivery_adapter_for=lambda _source: _adapter(),
        _build_stream_consumer_config=_consumer_config,
    )

    consumer, callback, _, _ = TurnRunner(runner, ctx)._setup_stream_consumer("telegram")

    assert (consumer is not None) is expect_consumer
    assert (callback is not None) is expect_consumer


@pytest.mark.parametrize(
    ("host_enabled", "served_streaming", "expect_consumer"),
    [
        (False, {"enabled": True, "transport": "edit"}, True),
        (True, {"enabled": False, "transport": "edit"}, False),
        (True, None, True),
    ],
)
def test_proxy_stream_consumer_uses_served_profile_streaming_config(
    monkeypatch, host_enabled, served_streaming, expect_consumer,
):
    """The proxy path follows the same served-profile master switch."""
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _Consumer)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {} if served_streaming is None else {"streaming": served_streaming},
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=StreamingConfig(enabled=host_enabled)),
        _delivery_adapter_for=lambda _source: _adapter(),
        _build_stream_consumer_config=_consumer_config,
    )

    consumer = GatewayTurnMixin._proxy_stream_consumer(
        runner, _source(), None, None, lambda: True,
    )

    assert (consumer is not None) is expect_consumer
