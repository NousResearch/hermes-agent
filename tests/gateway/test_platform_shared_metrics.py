"""Gateway platform health / delivery / first-reply shared metrics through the real runner seams."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, Platform, PlatformConfig, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_cli.observability import relay_shared_metrics as rsm
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_gateway as smg


@pytest.fixture
def rows(monkeypatch):
    got = []
    monkeypatch.setattr(rsm, "enabled", lambda: True)
    monkeypatch.setattr(rsm, "record_process_mark", lambda mark, data: got.append((mark, dict(data))))
    monkeypatch.setattr("gateway.platforms.base.random.uniform", lambda *_: 0.0)
    smg._reply_clocks.clear()

    def read(metric):
        smg.drain()
        found = [data for mark, data in got if contract._DECISION_MARK_METRICS[mark] == metric]
        assert all(contract.counter_dimensions_are_valid(metric, data) for data in found)
        return found

    return read


class _Adapter(BasePlatformAdapter):
    def __init__(self, results=(), connect: object = True):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.results, self._connect = list(results), connect

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if isinstance(self._connect, BaseException):
            raise self._connect
        if not self._connect:
            self._set_fatal_error("telegram_missing_dependency", "python-telegram-bot missing", retryable=False)
        return self._connect

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return self.results.pop(0) if self.results else SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _Runner(GatewayAdapterLifecycleMixin):
    def _platform_connect_timeout_secs(self, platform=None, *, initial=False):
        return 5.0


def _connect(adapter, **kw):
    try:
        return asyncio.run(_Runner()._connect_adapter_with_timeout(adapter, Platform.TELEGRAM, **kw))
    except Exception as exc:
        return type(exc).__name__


def test_every_connect_attempt_records_one_health_row_with_a_closed_error_class(rows):
    assert _connect(_Adapter()) is True
    assert _connect(_Adapter(), is_reconnect=True) is True
    assert _connect(_Adapter(connect=False)) is False
    assert _connect(_Adapter(connect=ConnectionResetError("peer reset by 10.0.0.1"))) == "ConnectionResetError"
    assert rows("hermes.platform.health") == [
        {"error_class": "none", "event": "connect_ok", "platform": "telegram"},
        {"error_class": "none", "event": "reconnect", "platform": "telegram"},
        {"error_class": "config", "event": "connect_failed", "platform": "telegram"},
        {"error_class": "network", "event": "connect_failed", "platform": "telegram"},
    ]


def test_one_logical_delivery_is_one_row_whatever_the_retries(rows):
    flaky = _Adapter([SendResult(success=False, error="ConnectionError: reset", retryable=True)])
    assert asyncio.run(flaky._send_with_retry("42", "hi", base_delay=0)).success
    blocked = _Adapter([SendResult(success=False, error="Forbidden: bot was blocked", error_kind="forbidden")] * 3)
    assert not asyncio.run(blocked._send_with_retry("42", "hi", base_delay=0)).success
    assert rows("hermes.platform.delivery") == [
        {"failure_class": "none", "outcome": "sent", "platform": "telegram"},
        {"failure_class": "forbidden", "outcome": "failed", "platform": "telegram"},
    ]


def test_first_reply_latency_is_one_row_per_turn_stopped_only_by_reply_text(rows):
    adapter = _Adapter()
    smg.start_reply_clock(SimpleNamespace(platform=Platform.TELEGRAM, chat_id="42"))
    asyncio.run(adapter._send_with_retry("42", "⏳ still working on the previous message"))  # busy ack
    assert rows("hermes.gateway.reply_latency") == []
    consumer = GatewayStreamConsumer(adapter, "42", StreamConsumerConfig(cursor=""))
    assert asyncio.run(consumer._send_or_edit("first streamed chunk"))
    asyncio.run(consumer._send_or_edit("first streamed chunk, more"))  # later edits are not the first reply

    event = MessageEvent(text="hi", message_type=MessageType.TEXT, message_id="in-1",
                         source=adapter.build_source(chat_id="8", user_id="u1"))
    smg.start_reply_clock(event.source)
    asyncio.run(adapter.send_final_ledgered(event, "k", "final answer", {}, reply_to=None))
    smg.start_reply_clock(SimpleNamespace(platform=Platform.TELEGRAM, chat_id="7"), internal=True)
    asyncio.run(adapter.send_final_ledgered(event, "k", "background notice", {}, reply_to=None))
    assert rows("hermes.gateway.reply_latency") == [{"first_response_bucket": "lt_2s", "platform": "telegram"}] * 2
