"""TSK-62: canonical pre-wire receipt, not a post-ACK observation."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_bolt.async_app import AsyncApp
from slack_bolt.authorization.authorize_result import AuthorizeResult
from slack_sdk.socket_mode.request import SocketModeRequest

from gateway.config import PlatformConfig
from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.service import WorkRouter
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def router(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True, "channel_allowlist": ["CEXEC"],
        "db_path": str(tmp_path / "router.db"),
        "bot_registry": {n: f"UBOT{i}" for i, n in enumerate(DEFAULT_PROFILES)},
    }).require_ready()
    value = WorkRouter(config)
    yield value
    value.store.close()


def adapter_for(router, *, synchronous=False):
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

    adapter._app = AsyncApp(authorize=authorize, process_before_response=synchronous)
    adapter._register_bolt_handlers()
    return adapter


def request(*, text="<@UBOT0> hello", event_id="EvReceipt", envelope="env-receipt", **extra):
    event = {"type": "message", "channel": "CEXEC", "channel_type": "channel",
             "user": "USENDER", "client_msg_id": event_id, "ts": "1700000000.1",
             "text": text, **extra}
    return SocketModeRequest(type="events_api", envelope_id=envelope,
                             payload={"type": "event_callback", "event_id": event_id,
                                      "team_id": "TTEST", "event": event})


async def deliver(adapter, req, send):
    handler = adapter._make_socket_mode_handler()
    client = SimpleNamespace(send_socket_mode_response=send, logger=logging.getLogger("preack.test"))
    try:
        await asyncio.wait_for(handler.handle(client, req), 5)
        # Real Bolt starts its listener independently when process_before_response=False.
        await adapter.drain_work_router_ingress()
    finally:
        await handler.close_async()


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_real_bolt_committed_receipt_precedes_wire_and_listener_never_waits_ack(router, synchronous):
    adapter = adapter_for(router, synchronous=synchronous)
    wire = []

    async def send(response):
        row = router.store.ack_journal.get("EvReceipt")
        assert row["wire_state"] == "uncertain"
        assert row["payload_json"] and "hello" in row["payload_json"]
        assert router.store.get_event("EvReceipt") is None
        wire.append(response.envelope_id)

    await deliver(adapter, request(), send)
    assert wire == ["env-receipt"]
    assert router.store.get_event("EvReceipt").text == "<@UBOT0> hello"
    assert router.store.ack_journal.get("EvReceipt")["handoff_state"] == "promoted"
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["prepare", "begin_wire"])
async def test_journal_failure_never_sends_ack(router, monkeypatch, method):
    adapter = adapter_for(router, synchronous=True)
    send = AsyncMock()
    def fail(*args):
        raise OSError("disk unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(router.store.ack_journal, method, fail)
        with pytest.raises(OSError):
            await deliver(adapter, request(), send)
    send.assert_not_awaited()
    assert router.store.get_event("EvReceipt") is None
    await deliver(adapter, request(), send)
    send.assert_awaited_once()
    assert router.store.inbox_count() == 1


@pytest.mark.asyncio
async def test_wire_failure_same_receiver_retry_uses_receipt_not_poisoned_native_caches(router):
    adapter = adapter_for(router, synchronous=True)
    send = AsyncMock(side_effect=OSError("transport ambiguous"))
    with pytest.raises(OSError):
        await deliver(adapter, request(), send)
    row = router.store.ack_journal.get("EvReceipt")
    assert (row['wire_state'], row['native_state']) == ('uncertain', 'admitted')
    assert router.store.get_event("EvReceipt") is None
    await adapter.recover_work_router_ingress()
    assert router.store.get_event("EvReceipt") is None
    send.side_effect = None
    await deliver(adapter, request(), send)
    assert router.store.inbox_count() == 1
    assert send.await_count == 2
    # A confirmed envelope callback does not send again, nor rerun native work.
    await deliver(adapter, request(), send)
    assert send.await_count == 2
    assert router.store.inbox_count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("same_receiver", [False, True])
async def test_concurrent_receivers_share_receipt_and_wire_ownership(router, same_receiver):
    first = adapter_for(router, synchronous=True)
    second = first if same_receiver else adapter_for(router, synchronous=True)
    entered, release = asyncio.Event(), asyncio.Event()
    async def send(response):
        entered.set()
        await release.wait()
    wire = AsyncMock(side_effect=send)
    one = asyncio.create_task(deliver(first, request(), wire))
    await asyncio.wait_for(entered.wait(), 5)
    two = asyncio.create_task(deliver(second, request(), wire))
    release.set()
    await asyncio.gather(one, two)
    assert wire.await_count == 1
    assert router.store.inbox_count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["confirm_wire", "enqueue_after_ack", "promoted"])
async def test_restart_recovery_preserves_crash_window_without_fabricating_ack(router, monkeypatch, stage):
    adapter = adapter_for(router, synchronous=True)
    wire = AsyncMock()
    target = router if stage == 'enqueue_after_ack' else router.store.ack_journal
    def fail(*args, **kwargs):
        raise OSError("simulated crash boundary")
    with monkeypatch.context() as patch:
        patch.setattr(target, stage, fail)
        with pytest.raises(OSError):
            await deliver(adapter, request(), wire)
    assert wire.await_count == 1
    recovered = WorkRouter(router.config)
    try:
        fresh = adapter_for(recovered, synchronous=True)
        await fresh.recover_work_router_ingress()
        if stage == 'confirm_wire':
            assert recovered.store.get_event('EvReceipt') is None
            assert recovered.store.ack_journal.get('EvReceipt')['wire_state'] == 'uncertain'
            await deliver(fresh, request(), wire)  # real redelivery, fresh wire evidence
        assert recovered.store.inbox_count() == 1
        assert recovered.store.ack_journal.get('EvReceipt')['handoff_state'] == 'promoted'
        await fresh.recover_work_router_ingress()
        assert recovered.store.inbox_count() == 1
    finally:
        recovered.store.close()


@pytest.mark.asyncio
async def test_native_bot_rejection_is_not_admission_and_never_recovered(router):
    adapter = adapter_for(router, synchronous=True)
    await deliver(adapter, request(bot_id='BSELF', user='UBOT0', subtype='bot_message'), AsyncMock())
    row = router.store.ack_journal.get('EvReceipt')
    assert (row['native_state'], row['wire_state']) == ('rejected', 'confirmed')
    assert row['payload_json'] is None
    await adapter.recover_work_router_ingress()
    assert router.store.inbox_count() == 0
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_privacy_projection_redaction_and_nonexecution(router, caplog):
    adapter = adapter_for(router, synchronous=True)
    token = 'xoxb-private-123456'
    req = request(text='<@UBOT0> api_key=secret-value ' + token,
                  files=[{'url_private': 'private-file-url'}],
                  metadata={'secret': 'nested-private-metadata'})
    req.payload['response_url'] = 'https://hooks.slack.com/private-url'
    await deliver(adapter, req, AsyncMock())
    row = router.store.ack_journal.get('EvReceipt')
    assert row['native_state'] == 'rejected'
    assert row['reason'] == 'privacy_redacted'
    assert router.store.inbox_count() == 0
    raw = open(router.store.path, 'rb').read()
    for secret in (token, 'secret-value', 'private-file-url', 'nested-private-metadata', 'private-url'):
        assert secret.encode() not in raw
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_shutdown_fence_drains_wire_and_forbids_reattach(router):
    adapter = adapter_for(router, synchronous=True)
    entered, release = asyncio.Event(), asyncio.Event()
    async def wire(response):
        entered.set()
        await release.wait()
    task = asyncio.create_task(deliver(adapter, request(), wire))
    await asyncio.wait_for(entered.wait(), 5)
    adapter.detach_work_router(router)
    assert not await adapter.drain_work_router_ingress(timeout=0.01)
    with pytest.raises(RuntimeError, match='fenced'):
        adapter.attach_work_router(router)
    release.set()
    await task
    assert await adapter.drain_work_router_ingress()
    assert router.store.ack_journal.get('EvReceipt')['wire_state'] == 'uncertain'
    assert router.store.inbox_count() == 0
    with pytest.raises(RuntimeError, match='fenced'):
        await deliver(adapter, request(event_id='EvNew'), AsyncMock())
    assert router.store.ack_journal.get('EvNew') is None


@pytest.mark.asyncio
async def test_canonical_collision_and_payload_bound_fail_before_wire(router):
    adapter = adapter_for(router, synchronous=True)
    await deliver(adapter, request(), AsyncMock())
    send = AsyncMock()
    with pytest.raises(ValueError, match='collision'):
        await deliver(adapter, request(text='<@UBOT0> changed'), send)
    with pytest.raises(ValueError, match='retention bound'):
        await deliver(adapter, request(event_id='EvHuge', text='x' * 70000), send)
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_handoff_is_in_shutdown_drain(router, monkeypatch):
    req = request()
    event = router.canonicalize_slack_event(req.payload['event'], req.payload)
    journal = router.store.ack_journal
    journal.prepare(event, req.envelope_id)
    journal.begin_wire(event.event_id, req.envelope_id)
    journal.confirm_wire(event.event_id, req.envelope_id)
    journal.native(event.event_id, True)
    adapter = adapter_for(router)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_handoff(event):
        entered.set()
        await release.wait()
        # Real handoff may still access SQLite while unwinding after a fence.
        assert router.store.ack_journal.get(event.event_id) is not None
        return False

    monkeypatch.setattr(router, 'enqueue_after_ack', blocked_handoff)
    task = asyncio.create_task(adapter.recover_work_router_ingress())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        adapter.detach_work_router(router)
        assert not await adapter.drain_work_router_ingress(timeout=0.01)
    finally:
        release.set()
        await task
    assert await adapter.drain_work_router_ingress()
    assert journal.get(event.event_id)['handoff_state'] == 'pending'


@pytest.mark.asyncio
async def test_receipt_only_and_native_only_do_not_become_runnable_on_restart(router):
    req = request()
    event = router.canonicalize_slack_event(req.payload['event'], req.payload)
    router.store.ack_journal.prepare(event, req.envelope_id)
    recovered = WorkRouter(router.config)
    try:
        adapter = adapter_for(recovered)
        await adapter.recover_work_router_ingress()
        assert recovered.store.inbox_count() == 0
        row = recovered.store.ack_journal.get(event.event_id)
        assert (row['wire_state'], row['native_state']) == ('prepared', 'pending')
        assert row['payload_json']
        recovered.store.ack_journal.native(event.event_id, True)
        await adapter.recover_work_router_ingress()
        assert recovered.store.inbox_count() == 0
    finally:
        recovered.store.close()
