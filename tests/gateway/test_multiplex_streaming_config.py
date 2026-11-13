"""Served-profile streaming configuration under multiplex (#136972).

A multiplexed gateway's ``runner.config`` is the launch profile's alone; a turn run by a served
profile must take ``streaming`` from that profile's own GatewayConfig, on both stream-consumer
gates (the normal turn path and the proxy path).
"""

from types import SimpleNamespace

import pytest

import hermes_agent.gateway.run as gateway_run
from hermes_agent.gateway.config import GatewayConfig, Platform, StreamingConfig
from hermes_agent.gateway.run_turn import GatewayTurnMixin
from hermes_agent.gateway.run_turn_runner import TurnRunner
from hermes_agent.gateway.session import SessionSource


class _Consumer:
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.stream_deltas_enabled = False


def _source(profile):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", profile=profile)


def _runner(host_enabled, served_enabled, *, multiplex=True):
    host = GatewayConfig(streaming=StreamingConfig(enabled=host_enabled))
    host.multiplex_profiles = multiplex
    return SimpleNamespace(
        config=host, _primary_profile_name="default",
        _profile_configs={"chief": GatewayConfig(streaming=StreamingConfig(enabled=served_enabled))},
        _delivery_adapter_for=lambda _source: SimpleNamespace(SUPPORTS_MESSAGE_EDITING=True),
        _build_stream_consumer_config=lambda _s, config, _a, **_k: (SimpleNamespace(transport=config.transport), None),
    )


# (host streaming, served streaming, profile running the turn, multiplex) -> stream consumer built
CASES = [
    (False, True, "chief", True, True),     # the report: served profile on, host off
    (True, False, "chief", True, False),    # served profile off wins over the host's on
    (True, False, "default", True, True),   # the launch profile keeps its own switch
    (False, True, "ghost", True, False),    # an uncached profile falls back to the launch config
    (False, True, "chief", False, False),   # single-profile gateway: runner.config is the profile's
]


@pytest.mark.parametrize(("host", "served", "profile", "multiplex", "expect"), CASES)
def test_turn_stream_consumer_follows_the_profile_running_the_turn(monkeypatch, host, served, profile,
                                                                    multiplex, expect):
    monkeypatch.setattr("hermes_agent.gateway.stream_consumer.GatewayStreamConsumer", _Consumer)
    ctx = SimpleNamespace(
        mute_notification_reply=False, streaming_tts_consumer_holder=[None], user_config={},
        resolve_display_setting=lambda *_args: None, scheduled_heartbeat=False,
        interim_assistant_messages_enabled=False, source=_source(profile), _status_thread_metadata=None,
        progress_queue=None, event_message_id=None, _run_still_current=lambda: True,
        stream_consumer_holder=[None],
    )
    consumer, callback, _, _ = TurnRunner(_runner(host, served, multiplex=multiplex), ctx)._setup_stream_consumer(
        "telegram")
    assert (consumer is not None) is expect
    assert (callback is not None) is expect


@pytest.mark.parametrize(("host", "served", "profile", "multiplex", "expect"), CASES)
def test_proxy_stream_consumer_follows_the_profile_running_the_turn(monkeypatch, host, served, profile,
                                                                     multiplex, expect):
    monkeypatch.setattr("hermes_agent.gateway.stream_consumer.GatewayStreamConsumer", _Consumer)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    consumer = GatewayTurnMixin._proxy_stream_consumer(
        _runner(host, served, multiplex=multiplex), _source(profile), None, None, lambda: True)
    assert (consumer is not None) is expect
