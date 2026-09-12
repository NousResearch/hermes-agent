"""Real Bolt Socket Mode -> native Slack filters -> Router SQLite admission."""
import asyncio
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from slack_bolt.async_app import AsyncApp
from slack_bolt.authorization.authorize_result import AuthorizeResult
from slack_sdk.socket_mode.request import SocketModeRequest

from gateway.config import PlatformConfig
from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.service import WorkRouter
from plugins.platforms.slack.adapter import SlackAdapter, SlackAdmissionStatus


@pytest.fixture
def router(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True, "channel_allowlist": ["CEXEC"],
        "db_path": str(tmp_path / "router.db"),
        "bot_registry": {n: f"UBOT{i}" for i, n in enumerate(DEFAULT_PROFILES)},
    }).require_ready()
    router = WorkRouter(config)
    yield router
    router.store.close()


def make_adapter(router):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter.attach_work_router(router)
    adapter._bot_user_id = "UBOT0"
    adapter._resolve_user_is_bot = AsyncMock(return_value=False)
    adapter._resolve_user_name = AsyncMock(return_value="Sender")
    adapter._resolve_channel_name = AsyncMock(return_value="work")
    adapter._hydrate_thread_context = AsyncMock(return_value=(None, [], []))
    adapter.handle_message = AsyncMock()
    adapter._app_token = "xapp-test"
    adapter._proxy_url = None

    async def authorize(enterprise_id, team_id, user_id):
        return AuthorizeResult(enterprise_id=enterprise_id, team_id=team_id,
                               bot_token="xoxb-test", bot_user_id="UBOT0")

    adapter._app = AsyncApp(authorize=authorize)
    adapter._register_bolt_handlers()
    return adapter


async def deliver(adapter, *, event_id="EvTest", ts="1700000000.1", channel="CEXEC",
                  fail_ack=False):
    event = {"type": "message", "channel": channel, "channel_type": "channel",
             "user": "USENDER", "client_msg_id": event_id, "ts": ts,
             "text": "<@UBOT0> hello"}
    req = SocketModeRequest(type="events_api", envelope_id="env-" + event_id,
                           payload={"type": "event_callback", "event_id": event_id,
                                    "team_id": "TTEST", "event": event})
    observed = []
    original = adapter._apply_work_router_admission

    async def admission(*args):
        result = await original(*args)
        observed.append(result)
        return result

    adapter._apply_work_router_admission = admission
    wire = []

    async def send(response):
        if channel == "CEXEC":
            # Canonical receipt exists before the wire attempt, but is not runnable.
            row = adapter._work_router.store.ack_journal.get(event_id)
            assert row["wire_state"] == "uncertain"
            assert adapter._work_router.store.get_event(event_id) is None
        if fail_ack:
            raise OSError("wire failed")
        wire.append(response.envelope_id)

    client = SimpleNamespace(send_socket_mode_response=send, logger=logging.getLogger("test.socket"))
    handler = adapter._make_socket_mode_handler()
    try:
        if fail_ack:
            with pytest.raises(OSError, match="wire failed"):
                await handler.handle(client, req)
        else:
            await handler.handle(client, req)
        await adapter.drain_work_router_ingress()
    finally:
        adapter._apply_work_router_admission = original
        await handler.close_async()
    return wire, observed


@pytest.mark.asyncio
async def test_socket_ack_native_handler_sqlite_and_reconnect_dedupe(router):
    first = make_adapter(router)
    wire, outcomes = await deliver(first)
    assert wire == ["env-EvTest"]
    assert outcomes[0].status is SlackAdmissionStatus.CONSUMED
    assert router.store.get_event("EvTest").text == "<@UBOT0> hello"
    first.handle_message.assert_not_awaited()
    probe = AsyncMock(wraps=first._apply_work_router_admission)
    first._apply_work_router_admission = probe
    await first._handle_slack_message({
        "type": "message", "channel": "CEXEC", "channel_type": "channel",
        "user": "USENDER", "client_msg_id": "EvTest", "ts": "1700000000.1",
        "text": "<@UBOT0> hello",
    }, {"event_id": "EvTest", "team_id": "TTEST"})
    probe.assert_not_awaited()
    # Rebuilt adapter has no native dedup memory; SQLite remains the owner.
    rebuilt = make_adapter(router)
    wire, outcomes = await deliver(rebuilt)
    assert wire == []
    assert outcomes == []
    assert router.store.inbox_count() == 1
    assert router.store.ack_journal.get("EvTest")["handoff_state"] == "promoted"
    rebuilt.handle_message.assert_not_awaited()
    # Unowned channel preserves upstream Base entry.
    unrelated = make_adapter(router)
    _, outcomes = await deliver(unrelated, event_id="EvOther", channel="COTHER")
    assert outcomes[0].status is SlackAdmissionStatus.ALLOW
    unrelated.handle_message.assert_awaited_once()
    assert router.store.get_event("EvOther") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["ack", "store", "shutdown"])
async def test_failed_admission_is_not_a_durable_receipt_or_base_fallback(router, monkeypatch, failure):
    adapter = make_adapter(router)
    if failure == "store":
        def fail(*args, **kwargs):
            raise OSError("store failed")
        monkeypatch.setattr(router.store, "enqueue_after_ack", fail)
    if failure == "shutdown":
        adapter.detach_work_router(router)
        with pytest.raises(RuntimeError, match="receiver is fenced"):
            await deliver(adapter, event_id="EvFailed")
        assert router.store.ack_journal.get("EvFailed") is None
        assert router.store.get_event("EvFailed") is None
        cast(AsyncMock, adapter.handle_message).assert_not_awaited()
        return
    _, outcomes = await deliver(adapter, event_id="EvFailed", fail_ack=failure == "ack")
    assert router.store.get_event("EvFailed") is None
    adapter.handle_message.assert_not_awaited()
    if failure == "ack":
        assert outcomes[0].status is SlackAdmissionStatus.CONSUMED
        assert router.store.ack_journal.get("EvFailed")["wire_state"] == "uncertain"
        assert "1700000000.1" in adapter._processed_message_ts
        # Same receiver and timestamp, not only a clean reconnect, can retry.
        _, retry = await deliver(adapter, event_id="EvFailed")
        assert retry == []
        assert router.store.get_event("EvFailed") is not None
    else:
        assert outcomes[0].status is SlackAdmissionStatus.FAILED
        assert router.store.ack_journal.get("EvFailed")["wire_state"] == "confirmed"
        assert "1700000000.1" not in adapter._processed_message_ts


def test_registration_single_owner(router):
    adapter = make_adapter(router)
    adapter.attach_work_router(router)
    with pytest.raises(RuntimeError, match="already attached"):
        adapter.attach_work_router(object())
    with pytest.raises(RuntimeError, match="owner mismatch"):
        adapter.detach_work_router(object())
