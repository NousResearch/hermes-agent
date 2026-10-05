"""Gateway platform health / delivery / first-reply shared metrics through the real runner seams."""

import asyncio
import warnings
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, Platform, PlatformConfig, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_cli.observability import relay_shared_metrics as rsm
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_gateway as smg
from hermes_constants import get_hermes_home


@pytest.fixture
def rows(monkeypatch):
    got = []
    monkeypatch.setattr(rsm, "enabled", lambda: True)
    monkeypatch.setattr(rsm, "record_process_mark",
                        lambda mark, data: got.append((mark, dict(data), str(get_hermes_home()))))
    monkeypatch.setattr("gateway.platforms.base.random.uniform", lambda *_: 0.0)
    smg._reply_clocks.clear()
    smg._chat_homes.clear()
    smg._failing_connects.clear()


    def read(metric, *, with_home=False):
        smg.drain()
        found = [(home, data) for mark, data, home in got if contract._DECISION_MARK_METRICS[mark] == metric]
        assert all(contract.counter_dimensions_are_valid(metric, data) for _, data in found)
        return found if with_home else [data for _, data in found]

    return read


class _Adapter(BasePlatformAdapter):
    def __init__(self, results=(), connect: object = True, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
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
        return asyncio.run(_Runner()._connect_adapter_with_timeout(adapter, adapter.platform, **kw))
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


def test_a_reconnect_backoff_loop_is_one_failed_connect_until_it_recovers(rows):
    outage = ConnectionResetError("network down")
    assert _connect(_Adapter(connect=outage)) == "ConnectionResetError"  # cold start
    for _ in range(5):  # the watcher's backoff retries during the same outage
        assert _connect(_Adapter(connect=outage), is_reconnect=True) == "ConnectionResetError"
    assert _connect(_Adapter(platform=Platform.DISCORD, connect=outage)) == "ConnectionResetError"
    assert _connect(_Adapter(), is_reconnect=True) is True
    assert _connect(_Adapter(connect=outage), is_reconnect=True) == "ConnectionResetError"  # a new outage
    smg.drain()
    smg._failing_connects.update(dict.fromkeys(smg._failing_connects, "2000-01-01"))  # ...still down a day later
    assert _connect(_Adapter(connect=outage), is_reconnect=True) == "ConnectionResetError"
    assert [(r["event"], r["platform"]) for r in rows("hermes.platform.health")] == [
        ("connect_failed", "telegram"), ("connect_failed", "discord"),
        ("reconnect", "telegram"), ("connect_failed", "telegram"), ("connect_failed", "telegram"),
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


def test_a_relay_delivered_turn_stops_the_clock_its_inbound_started(rows):
    # The connector's inbound names the platform it came from; the reply leaves through the relay.
    relay = _Adapter(platform=Platform.RELAY)
    smg.start_reply_clock(SimpleNamespace(platform=Platform.DISCORD, chat_id="c1"))
    consumer = GatewayStreamConsumer(relay, "c1", StreamConsumerConfig(cursor=""))
    assert asyncio.run(consumer._send_or_edit("first streamed chunk"))
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, message_id="in-2",
                         source=SimpleNamespace(platform=Platform.SLACK, chat_id="C2", thread_id=None))
    smg.start_reply_clock(event.source)
    asyncio.run(relay.send_final_ledgered(event, "k", "final answer", {}, reply_to=None))
    assert rows("hermes.gateway.reply_latency") == [
        {"first_response_bucket": "lt_2s", "platform": "discord"}, {"first_response_bucket": "lt_2s", "platform": "slack"}]
    assert not smg._reply_clocks


def test_a_delivery_sent_after_the_routed_scope_resets_lands_in_the_owning_profile(rows, tmp_path):
    # Multiplexed gateway: the turn starts in profile B's scope, the final send runs after it is reset.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_b = tmp_path / "profiles" / "b"
    token = set_hermes_home_override(str(home_b))
    try:
        smg.start_reply_clock(SimpleNamespace(platform=Platform.SLACK, chat_id="C9"))
    finally:
        reset_hermes_home_override(token)
    asyncio.run(_Adapter(platform=Platform.SLACK)._send_with_retry("C9", "final answer", base_delay=0))
    asyncio.run(_Adapter()._send_with_retry("unseen", "hi", base_delay=0))  # no turn: the caller's scope
    assert rows("hermes.platform.delivery", with_home=True) == [
        (str(home_b), {"failure_class": "none", "outcome": "sent", "platform": "slack"}),
        (str(get_hermes_home()), {"failure_class": "none", "outcome": "sent", "platform": "telegram"}),
    ]


class _FatalRunner(GatewayAdapterLifecycleMixin):
    def __init__(self):
        self.adapters, self._failed_platforms, self._running = {}, {}, True
        self.delivery_router, self.config = SimpleNamespace(adapters={}), SimpleNamespace(platforms={})
        self._profile_adapters, self._profile_failed_platforms = {}, {}

    def _update_platform_runtime_status(self, *args, **kwargs):
        pass

    async def _safe_adapter_disconnect(self, *args, **kwargs):
        pass

    async def stop(self):
        pass


def _fatal(adapter, code, *, retryable=False):
    adapter._set_fatal_error(code, "boom", retryable=retryable)
    return adapter


def test_fatal_handlers_count_lost_connections_in_the_owning_profile_only(rows, tmp_path, monkeypatch):
    home_b = tmp_path / "profiles" / "b"
    monkeypatch.setattr(_FatalRunner, "_routed_profile_home", staticmethod(lambda name: home_b))
    runner = _FatalRunner()
    secondary = _fatal(_Adapter(), "telegram_network_error")
    runner._profile_adapters["b"] = {Platform.TELEGRAM: secondary}
    # The secondary's notification arrives on a task scoped to the launch profile, not to b.
    asyncio.run(runner._handle_profile_adapter_fatal_error("b", Platform.TELEGRAM, secondary))
    for code in ("relay_disabled", "telegram_auth_error"):  # the user's relay opt-out is not a disconnect
        primary = _fatal(_Adapter(platform=Platform.RELAY), code)
        runner.adapters[Platform.RELAY] = primary
        asyncio.run(runner._handle_adapter_fatal_error_impl(primary))
    launch_home = str(get_hermes_home())
    assert rows("hermes.platform.health", with_home=True) == [
        (str(home_b), {"error_class": "network", "event": "disconnect", "platform": "telegram"}),
        (launch_home, {"error_class": "auth", "event": "disconnect", "platform": "relay"}),
    ]


class _ClosedWithDeprecatedCode(ConnectionError):
    """websockets-style: the status lives behind a deprecated property."""

    @property
    def code(self):
        warnings.warn("ConnectionClosed.code is deprecated", DeprecationWarning, stacklevel=2)
        return 1011

    @property
    def status_code(self):
        raise RuntimeError("no response attached")


def test_classifying_a_connect_failure_never_replaces_the_adapters_exception(rows):
    raised = _ClosedWithDeprecatedCode()

    async def attempt():
        try:
            await _Runner()._connect_adapter_with_timeout(_Adapter(connect=raised), Platform.TELEGRAM)
        except BaseException as exc:
            return exc

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seen = asyncio.run(attempt())
    assert seen is raised and seen.__context__ is None
    assert rows("hermes.platform.health") == [
        {"error_class": "network", "event": "connect_failed", "platform": "telegram"}]
