"""Feishu may redeliver the same ID after an inbound failure (issue #62860)."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageType
from gateway.session_authority import SessionAuthority
from gateway.session_ingress_media import _lease_scope
from plugins.platforms.feishu.adapter import FeishuAdapter


def incoming(message_id):
    message = SimpleNamespace(message_id=message_id, chat_id="oc_chat", chat_type="p2p")
    sender = SimpleNamespace(sender_type="user", sender_id=SimpleNamespace(open_id="ou_user"))
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


def adapter_for_test():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._admit = lambda _sender, _message: None
    adapter._extract_message_content = AsyncMock(return_value=("hello", MessageType.TEXT, [], [], [], []))
    adapter.get_chat_info = AsyncMock(return_value={"type": "dm"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "ou_user", "user_name": "User", "user_id_alt": None})
    adapter._text_batch_delay_seconds = adapter._media_batch_delay_seconds = 3600
    return adapter


async def close_adapter(adapter):
    await adapter._cancel_pending_tasks(adapter._pending_text_batch_tasks)
    await adapter._cancel_pending_tasks(adapter._pending_media_batch_tasks)
    adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("before dispatch"), asyncio.CancelledError()])
async def test_early_failure_allows_same_id_redelivery(failure):
    adapter = adapter_for_test()
    adapter._process_inbound_message = AsyncMock(side_effect=[failure, None])
    try:
        with pytest.raises(type(failure)):
            await adapter._handle_message_event_data(incoming("om_retry"))
        reloaded = FeishuAdapter(PlatformConfig())
        assert "om_retry" not in reloaded._seen_message_ids
        await adapter._handle_message_event_data(incoming("om_retry"))
        assert adapter._process_inbound_message.await_count == 2
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "media"])
async def test_failed_batch_releases_every_message_id(kind):
    adapter = adapter_for_test()
    if kind == "media":
        adapter._extract_message_content.return_value = ("", MessageType.PHOTO, ["/tmp/photo.png"], ["image/png"], [], [])
    adapter.handle_message = AsyncMock(side_effect=RuntimeError("before gateway dispatch"))
    try:
        await adapter._handle_message_event_data(incoming("om_first"))
        await adapter._handle_message_event_data(incoming("om_second"))
        assert await adapter._is_duplicate("om_unrelated") is False
        batches = adapter._pending_text_batches if kind == "text" else adapter._pending_media_batches
        flush = adapter._flush_text_batch_now if kind == "text" else adapter._flush_media_batch_now
        with pytest.raises(RuntimeError, match="before gateway dispatch"):
            await flush(next(iter(batches)))
        assert await adapter._is_duplicate("om_first") is False
        assert await adapter._is_duplicate("om_second") is False
        assert await adapter._is_duplicate("om_unrelated") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_background_admission_failure_releases_merged_batch():
    adapter = adapter_for_test()
    authority = authority_with(AsyncMock(side_effect=RuntimeError("capture failed")))

    async def handler(event):
        return await authority.admit_native(event)

    adapter.set_message_handler(handler)
    adapter._start_typing_refresh = lambda *_args, **_kwargs: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._notify_turn_error = AsyncMock()
    try:
        await adapter._handle_message_event_data(incoming("om_batch_a"))
        await adapter._handle_message_event_data(incoming("om_batch_b"))
        await adapter._flush_text_batch_now(next(iter(adapter._pending_text_batches)))
        tasks = list(adapter._session_tasks.values())
        assert tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        assert await adapter._is_duplicate("om_batch_a") is False
        assert await adapter._is_duplicate("om_batch_b") is False
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_busy_burst_releases_all_claims_after_preparation_failure(monkeypatch):
    from gateway.session_ingress_burst import _flush
    from tests.gateway.test_busy_burst_serial_reader import _busy_owner

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_user")
    adapter = adapter_for_test()
    source = adapter.build_source(chat_id="oc_chat", user_id="ou_user", chat_type="dm")
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode="queue")
    original = owner.authority._admit_native
    try:
        first = await capture_event(adapter, "om_busy_a")
        second = await capture_event(adapter, "om_busy_b")
        owner.authority._admit_native = AsyncMock(side_effect=RuntimeError("capture failed"))
        await adapter.handle_message(first)
        await adapter.handle_message(second)
        burst = adapter._shared_bursts[owner.key]
        await _flush(adapter, owner.key, burst)
        assert await adapter._is_duplicate("om_busy_a") is False
        assert await adapter._is_duplicate("om_busy_b") is False
    finally:
        owner.authority._admit_native = original
        await owner.settle(1)
        await close_adapter(adapter)


async def capture_event(adapter, message_id):
    captured = []

    async def capture(event):
        captured.append(event)

    adapter._dispatch_inbound_event = capture
    await adapter._handle_message_event_data(incoming(message_id))
    return captured[0]


def authority_with(operation):
    authority = object.__new__(SessionAuthority)

    @asynccontextmanager
    async def route(_event):
        yield None

    authority._route_turn = route
    authority._admit_native = operation
    return authority


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("capture failed"), asyncio.CancelledError()])
async def test_background_preparation_failure_releases_claim(failure):
    adapter = adapter_for_test()
    try:
        event = await capture_event(adapter, "om_prepare")
        authority = authority_with(AsyncMock(side_effect=failure))
        with pytest.raises(type(failure)):
            await authority.admit_native(event)
        assert await adapter._is_duplicate("om_prepare") is False
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_cancelled_inflight_write_keeps_claim():
    adapter = adapter_for_test()

    async def writing(_event, _fence):
        _lease_scope.get().write_started = True
        raise asyncio.CancelledError()

    try:
        event = await capture_event(adapter, "om_writing")
        with pytest.raises(asyncio.CancelledError):
            await authority_with(writing).admit_native(event)
        assert await adapter._is_duplicate("om_writing") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["admitted", "command"])
async def test_reply_failure_after_handling_keeps_claim(kind):
    adapter = adapter_for_test()

    async def admitted(_event, _fence):
        _lease_scope.get().write_started = True
        return object()

    authority = authority_with(admitted)

    async def handler(event):
        if kind == "admitted":
            await authority.admit_native(event)
        return "finished"

    try:
        event = await capture_event(adapter, "om_admitted")
        if kind == "command":
            event.text = "/stop"
        adapter.set_message_handler(handler)
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=False, error="reply failed"))
        adapter._start_typing_refresh = lambda *_args, **_kwargs: None
        adapter._stop_typing_refresh = AsyncMock()
        await adapter._process_message_background(event, adapter._event_session_key(event))
        assert adapter._send_with_retry.await_count == 1
        assert await adapter._is_duplicate("om_admitted") is True
        assert "om_admitted" in FeishuAdapter(PlatformConfig())._seen_message_ids
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_old_claim_cannot_remove_a_newer_claim():
    adapter = adapter_for_test()
    adapter._dedup_cache_size = 1
    try:
        old = await adapter._claim_message_id("om_same")
        await adapter._claim_message_id("om_other")  # evict the first claim
        new = await adapter._claim_message_id("om_same")
        assert old is not None and new is not None
        await old.release()
        assert await adapter._is_duplicate("om_same") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("old_closed", [False, True])
async def test_reconnected_adapter_sees_predecessor_release(old_closed):
    from gateway.platforms.helpers import carry_inbound_dedup

    old = adapter_for_test()
    new = None
    try:
        event = await capture_event(old, "om_reconnect")
        new = adapter_for_test()
        carry_inbound_dedup(old, new)
        assert await new._is_duplicate("om_reconnect") is True
        if old_closed:
            old._shutdown_sdk_executor()
        with pytest.raises(RuntimeError):
            await authority_with(AsyncMock(side_effect=RuntimeError("before admission"))).admit_native(event)
        assert await new._is_duplicate("om_reconnect") is False
    finally:
        await close_adapter(old)
        if new is not None:
            await close_adapter(new)


@pytest.mark.asyncio
async def test_reconnect_does_not_share_claims_across_homes(tmp_path, monkeypatch):
    from gateway.platforms.helpers import carry_inbound_dedup

    old = adapter_for_test()
    new = None
    try:
        event = await capture_event(old, "om_separate")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
        new = adapter_for_test()
        carry_inbound_dedup(old, new)
        assert await new._is_duplicate("om_separate") is False
        with pytest.raises(RuntimeError):
            await authority_with(AsyncMock(side_effect=RuntimeError("before admission"))).admit_native(event)
        assert await new._is_duplicate("om_separate") is True
    finally:
        await close_adapter(old)
        if new is not None:
            await close_adapter(new)


@pytest.mark.asyncio
async def test_concurrent_redelivery_waits_for_original_failure():
    adapter = adapter_for_test()
    started, finish = asyncio.Event(), asyncio.Event()

    async def fail_first(**_kwargs):
        started.set()
        await finish.wait()
        raise RuntimeError("pre-dispatch failure")

    adapter._process_inbound_message = AsyncMock(side_effect=fail_first)
    task = asyncio.create_task(adapter._handle_message_event_data(incoming("om_parallel")))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await adapter._handle_message_event_data(incoming("om_parallel"))
        assert adapter._process_inbound_message.await_count == 1
        finish.set()
        with pytest.raises(RuntimeError):
            await task
        adapter._process_inbound_message.side_effect = None
        await adapter._handle_message_event_data(incoming("om_parallel"))
        assert adapter._process_inbound_message.await_count == 2
    finally:
        finish.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_rewritten_event_releases_original_claim():
    adapter = adapter_for_test()
    try:
        event = await capture_event(adapter, "om_rewrite")
        rewritten = replace(event, text="rewritten")
        with pytest.raises(RuntimeError):
            await authority_with(AsyncMock(side_effect=RuntimeError("before write"))).admit_native(rewritten)
        assert await adapter._is_duplicate("om_rewrite") is False
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_disconnect_releases_buffered_batch():
    adapter = adapter_for_test()
    try:
        await adapter._handle_message_event_data(incoming("om_buffer_a"))
        await adapter._handle_message_event_data(incoming("om_buffer_b"))
        assert await adapter._is_duplicate("om_kept") is False
        await adapter.disconnect()
        reloaded = FeishuAdapter(PlatformConfig())
        assert "om_buffer_a" not in reloaded._seen_message_ids
        assert "om_buffer_b" not in reloaded._seen_message_ids
        assert "om_kept" in reloaded._seen_message_ids
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_cancel_during_claim_write_releases_after_write_finishes():
    adapter = adapter_for_test()
    adapter._process_inbound_message = AsyncMock()
    started, finish = threading.Event(), threading.Event()
    persist = adapter._persist_seen_message_ids
    first = True

    def blocked_first_write():
        nonlocal first
        if first:
            first = False
            started.set()
            assert finish.wait(5)
        persist()

    adapter._persist_seen_message_ids = blocked_first_write
    task = asyncio.create_task(adapter._handle_message_event_data(incoming("om_cancelled")))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await adapter._handle_message_event_data(incoming("om_cancelled"))
        assert adapter._process_inbound_message.await_count == 0
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        reloaded = FeishuAdapter(PlatformConfig())
        assert "om_cancelled" not in reloaded._seen_message_ids
        await adapter._handle_message_event_data(incoming("om_cancelled"))
        assert adapter._process_inbound_message.await_count == 1
    finally:
        finish.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_name", ["pre_gateway_dispatch", "post_gateway_admission"])
async def test_plugin_hook_side_effects_keep_claim_on_later_failure(hook_name, monkeypatch):
    from gateway import run_inbound_consumer
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin
    from hermes_cli import lifecycle

    adapter = adapter_for_test()
    routed = {"active": False}

    @asynccontextmanager
    async def routed_scope(_runner, _source):
        routed["active"] = True
        try:
            yield
        finally:
            routed["active"] = False

    monkeypatch.setattr(run_inbound_consumer, "_routed_scope", routed_scope)
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == hook_name and (
        hook_name != "post_gateway_admission" or routed["active"]))
    monkeypatch.setattr(lifecycle, "ainvoke_hook", AsyncMock(return_value=[]))
    try:
        event = await capture_event(adapter, "om_plugin")
        runner = SimpleNamespace()
        if hook_name == "pre_gateway_dispatch":
            await GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook(runner, event, event.source)
        else:
            await GatewayInboundHooksMixin._hm_post_admission_consume(runner, event, event.source, "session")
        with pytest.raises(RuntimeError):
            await authority_with(AsyncMock(side_effect=RuntimeError("later"))).admit_native(event)
        assert await adapter._is_duplicate("om_plugin") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_unavailable_hook_inspection_still_runs_hook_and_keeps_claim(monkeypatch):
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin
    from hermes_cli import lifecycle

    adapter = adapter_for_test()
    invoke = AsyncMock(return_value=[])

    def unavailable(_name):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(lifecycle, "has_hook", unavailable)
    monkeypatch.setattr(lifecycle, "ainvoke_hook", invoke)
    try:
        event = await capture_event(adapter, "om_hook_unavailable")
        await GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook(SimpleNamespace(), event, event.source)
        invoke.assert_awaited_once()
        with pytest.raises(RuntimeError):
            await authority_with(AsyncMock(side_effect=RuntimeError("later"))).admit_native(event)
        assert await adapter._is_duplicate("om_hook_unavailable") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_busy_interrupt_keeps_claim_if_later_admission_fails():
    from gateway.run_inbound import GatewayInboundMixin

    adapter = adapter_for_test()
    try:
        event = await capture_event(adapter, "om_interrupt")
        state = SimpleNamespace(turn=SimpleNamespace(agent=object()))
        outcome = SimpleNamespace(effective_mode="interrupt", steered=False, redirected=False)
        runner = SimpleNamespace(
            _clarify_takeover=lambda *_: False,
            _effective_busy_input_mode=lambda *_: "interrupt",
            _hm_text_only=lambda *_: True,
            _hm_in_telegram_grace=lambda *_: False,
            _peek_session_state=lambda *_: state,
            _delivery_adapter_for=lambda *_: adapter,
            _effective_busy_text_mode=lambda *_: "interrupt",
            _resolve_busy_steer_or_redirect=AsyncMock(return_value=outcome),
            _busy_interrupt_and_ack=AsyncMock(),
            _draining=False,
        )
        assert not await GatewayInboundMixin._hm_busy_preempt(runner, event, event.source, "session")
        runner._busy_interrupt_and_ack.assert_awaited_once()
        with pytest.raises(RuntimeError):
            await authority_with(AsyncMock(side_effect=RuntimeError("later"))).admit_native(event)
        assert await adapter._is_duplicate("om_interrupt") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["before_write", "queue_full"])
async def test_real_sqlite_admission_after_redelivery(tmp_path, monkeypatch, refusal):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from persistence.authority.admissions import GateRefusal
    from persistence.core.errors import AuthorityRefusal
    from persistence_assembly import close_stores_under

    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = adapter_for_test()
    adapter.gateway_runner, runner.adapters = runner, {Platform.FEISHU: adapter}
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda *_args, **_kwargs: True)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="dedup-test")
    authority._schedule = lambda _ref: None
    try:
        event = await capture_event(adapter, "om_sqlite")
        gate = authority.gate_refusal
        if refusal == "before_write":
            event.media_types = ["image/png"]  # invalid input is refused before the write starts
        else:
            authority.gate_refusal = lambda *_args, **_kwargs: GateRefusal("queue_full")
        with pytest.raises(AuthorityRefusal):
            await authority.admit_native(event)
        authority.gate_refusal = gate
        retry = await capture_event(adapter, "om_sqlite")
        receipt = await authority.admit_native(retry)
        row = authority.persistence.admissions.get(admission_id=receipt.admission_id)
        assert row.request_id == "om_sqlite"
        assert row.payload["text"] == "hello"
        assert await adapter._is_duplicate("om_sqlite") is True
    finally:
        close_stores_under(authority.persistence.scope.profile_dir)
        await close_adapter(adapter)
