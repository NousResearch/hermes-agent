"""Replies the relay connector carries are labelled with the platform the conversation lives on."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_cli.observability import shared_metrics_gateway as smg
from tests.gateway.relay.test_relay_adapter import _CaptureTransport, make_desc
from tests.gateway.test_platform_shared_metrics import rows


def _relay(platform="telegram"):
    return RelayAdapter(PlatformConfig(), make_desc(platform=platform), transport=_CaptureTransport())


def _inbound(adapter, platform, chat_id):
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, message_id=f"in-{chat_id}",
                         source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm", user_id="u1"))
    adapter._capture_scope(event)
    return event


def test_multiplatform_relay_replies_are_labelled_by_the_inbound_platform(rows):
    relay = _relay()
    discord = _inbound(relay, Platform.DISCORD, "c1")
    slack = _inbound(relay, Platform.SLACK, "c2")
    smg.start_reply_clock(discord.source)
    assert asyncio.run(GatewayStreamConsumer(relay, "c1", StreamConsumerConfig(cursor=""))._send_or_edit("chunk"))
    smg.start_reply_clock(slack.source)
    asyncio.run(relay.send_final_ledgered(slack, "k", "final answer", {}, reply_to=None))

    assert rows("hermes.gateway.reply_latency") == [
        {"first_response_bucket": "lt_2s", "platform": "discord"},
        {"first_response_bucket": "lt_2s", "platform": "slack"},
    ]
    # Streamed chunks send directly; the final reply is the ledgered, counted delivery.
    assert rows("hermes.platform.delivery") == [{"failure_class": "none", "outcome": "sent", "platform": "slack"}]


def test_single_platform_relay_inbound_is_labelled_by_the_connector_platform(rows):
    """An inbound that only says 'relay' resolves to the platform the connector fronts."""
    relay = _relay("discord")
    event = _inbound(relay, Platform.RELAY, "c9")
    smg.start_reply_clock(event.source)
    asyncio.run(relay.send_final_ledgered(event, "k", "final answer", {}, reply_to=None))
    assert rows("hermes.gateway.reply_latency") == [{"first_response_bucket": "lt_2s", "platform": "discord"}]
    assert rows("hermes.platform.delivery") == [{"failure_class": "none", "outcome": "sent", "platform": "discord"}]


@pytest.mark.parametrize("chat_platform", [None, RuntimeError("descriptor gone")])
def test_an_unresolvable_relay_chat_keeps_the_relay_label(rows, chat_platform):
    class _Front:
        platform = Platform.RELAY

        def _metrics_platform(self, chat_id):
            if isinstance(chat_platform, BaseException):
                raise chat_platform
            return chat_platform

    smg._record_delivery(_Front(), SimpleNamespace(success=True), chat_id="c1")
    assert rows("hermes.platform.delivery") == [{"failure_class": "none", "outcome": "sent", "platform": "relay"}]


def test_an_unstamped_chat_on_a_multiplatform_connector_is_not_labelled_as_its_primary(rows):
    relay = _relay("discord")
    relay._transport._identities = [("discord", "bot-a"), ("slack", "bot-b")]
    event = _inbound(relay, Platform.RELAY, "c7")
    smg.start_reply_clock(event.source)
    asyncio.run(relay.send_final_ledgered(event, "k", "final answer", {}, reply_to=None))
    assert rows("hermes.platform.delivery") == [{"failure_class": "none", "outcome": "sent", "platform": "relay"}]


def test_a_relay_stamped_turn_is_a_gateway_message():
    from hermes_cli.observability.shared_metrics_contract import task_start_fields

    assert task_start_fields({"platform": "relay"}) == {
        "entrypoint": "gateway_message", "execution_surface": "gateway", "platform": "relay"}
