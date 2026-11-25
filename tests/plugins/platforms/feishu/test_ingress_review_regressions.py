"""Admission and cancellation boundaries for Feishu inbound dedup claims."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageType
from tests.plugins.platforms.feishu.test_inbound_dedup_retry import (
    adapter_for_test, authority_with, capture_event, close_adapter, incoming,
)


@pytest.mark.asyncio
async def test_session_key_failure_releases_claim_before_dispatch():
    adapter = adapter_for_test()
    adapter._extract_message_content.return_value = ("/help", MessageType.TEXT, [], [], [], [])
    handler = AsyncMock()
    adapter.set_message_handler(handler)

    def fail_key(_event):
        raise RuntimeError("session key unavailable")

    adapter._event_session_key = fail_key
    try:
        with pytest.raises(RuntimeError, match="session key unavailable"):
            await adapter._handle_message_event_data(incoming("om_session_key"))
        assert handler.await_count == 0
        assert await adapter._is_duplicate("om_session_key") is False
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_task", "during_start", "after_handler"])
async def test_shutdown_releases_only_before_handler(phase):
    adapter = adapter_for_test()
    start_entered, allow_start, handler_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(_event):
        handler_entered.set()
        await asyncio.Event().wait()

    async def hook(name, *_args):
        if name == "on_processing_start":
            start_entered.set()
            await allow_start.wait()

    adapter.set_message_handler(handler)
    adapter._run_processing_hook = hook
    adapter._start_typing_refresh = lambda *_args, **_kwargs: None
    adapter._stop_typing_refresh = AsyncMock()
    try:
        await adapter._handle_message_event_data(incoming("om_shutdown"))
        await adapter._flush_text_batch_now(next(iter(adapter._pending_text_batches)))
        if phase != "before_task":
            await asyncio.wait_for(start_entered.wait(), 5)
        if phase == "after_handler":
            allow_start.set()
            await asyncio.wait_for(handler_entered.wait(), 5)
        await adapter.cancel_background_tasks()
        assert await adapter._is_duplicate("om_shutdown") is (phase == "after_handler")
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_user_stop_before_handler_keeps_claim():
    adapter = adapter_for_test()
    start_entered = asyncio.Event()

    async def hook(name, *_args):
        if name == "on_processing_start":
            start_entered.set()
            await asyncio.Event().wait()

    adapter.set_message_handler(AsyncMock())
    adapter._run_processing_hook = hook
    adapter._start_typing_refresh = lambda *_args, **_kwargs: None
    adapter._stop_typing_refresh = AsyncMock()
    try:
        await adapter._handle_message_event_data(incoming("om_stopped"))
        await adapter._flush_text_batch_now(next(iter(adapter._pending_text_batches)))
        await asyncio.wait_for(start_entered.wait(), 5)
        session_key = next(iter(adapter._session_tasks))
        await adapter.cancel_session_processing(session_key)
        assert await adapter._is_duplicate("om_stopped") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_executor_cancels_queued_claim_write_before_dispatch():
    from gateway.platforms.helpers import carry_inbound_dedup

    adapter = adapter_for_test()
    started, finish = threading.Barrier(11), threading.Event()

    def block_worker():
        started.wait(timeout=5)
        finish.wait(5)

    executor = adapter._get_sdk_executor()
    blockers = [executor.submit(block_worker) for _ in range(10)]
    entered = asyncio.Event()
    run_blocking = adapter._run_blocking

    async def signal_submit(func, *args):
        entered.set()
        return await run_blocking(func, *args)

    adapter._run_blocking = signal_submit
    replacement = None
    try:
        await asyncio.to_thread(started.wait, 5)
        task = asyncio.create_task(adapter._claim_message_id("om_cancelled_write"))
        await asyncio.wait_for(entered.wait(), 5)
        adapter._shutdown_sdk_executor()
        with pytest.raises(asyncio.CancelledError):
            await task
        replacement = adapter_for_test()
        carry_inbound_dedup(adapter, replacement)
        assert await replacement._is_duplicate("om_cancelled_write") is False
    finally:
        finish.set()
        await asyncio.gather(*(asyncio.wrap_future(f) for f in blockers), return_exceptions=True)
        await close_adapter(adapter)
        if replacement is not None:
            await close_adapter(replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["demoted_interrupt", "attempted_steer"])
async def test_busy_claim_follows_actual_control_operation(tmp_path, mode):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    agent = SimpleNamespace(_active_children=[object()] if mode == "demoted_interrupt" else [], steer=lambda *_: None)
    state = SimpleNamespace(turn=SimpleNamespace(agent=agent))
    attempted = []
    runner._clarify_takeover = lambda *_: False
    runner._effective_busy_input_mode = lambda *_: "interrupt" if mode == "demoted_interrupt" else "steer"
    runner._hm_in_telegram_grace = lambda *_: False
    runner._peek_session_state = lambda *_: state
    runner._delivery_adapter_for = lambda *_: adapter
    runner._effective_busy_text_mode = lambda *_: "interrupt"
    runner._busy_interrupt_and_ack = AsyncMock()
    runner._prepare_busy_steer_text = AsyncMock(return_value="hello")
    runner._try_agent_verb = lambda *_args, **_kwargs: attempted.append("called") or False
    runner._draining = False
    try:
        event = await capture_event(adapter, "om_busy_mode")
        assert not await runner._hm_busy_preempt(event, event.source, "session")
        outcome = runner._busy_interrupt_and_ack.await_args.args[-1]
        assert outcome.effective_mode == "queue"
        assert bool(attempted) is (mode == "attempted_steer")
        with pytest.raises(RuntimeError, match="before admission"):
            await authority_with(AsyncMock(side_effect=RuntimeError("before admission"))).admit_native(event)
        assert await adapter._is_duplicate("om_busy_mode") is (mode == "attempted_steer")
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["refused", "committed"])
async def test_cancelled_caller_waits_for_known_write_outcome(tmp_path, monkeypatch, outcome):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from persistence.authority.admissions import GateRefusal
    from persistence_assembly import close_stores_under

    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = adapter_for_test()
    adapter.gateway_runner, runner.adapters = runner, {Platform.FEISHU: adapter}
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda *_args, **_kwargs: True)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="review-late-write")
    authority._schedule = lambda _ref: None
    started, finish = threading.Event(), threading.Event()

    def gate():
        started.set()
        assert finish.wait(5)
        return GateRefusal("queue_full") if outcome == "refused" else None

    authority.gate_refusal = lambda *_args, **_kwargs: gate
    try:
        event = await capture_event(adapter, "om_late_write")
        caller = asyncio.create_task(authority.admit_native(event))
        assert await asyncio.to_thread(started.wait, 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert await adapter._is_duplicate("om_late_write") is True  # the write is still unresolved
        finish.set()
        await asyncio.gather(*list(authority._mutation_tasks), return_exceptions=True)
        rows = authority.persistence.admissions.match(request_id="om_late_write")
        assert bool(rows) is (outcome == "committed")
        assert await adapter._is_duplicate("om_late_write") is (outcome == "committed")
    finally:
        finish.set()
        close_stores_under(authority.persistence.scope.profile_dir)
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["runtime_draining", "stale_generation", "admission_conflict"])
async def test_definite_drain_refusal_releases_without_reopening_controlled_or_existing_input(tmp_path, monkeypatch, reason):
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
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="review-gate")
    authority._schedule = lambda _ref: None
    authority.gate_refusal = lambda *_args, **_kwargs: GateRefusal(
        reason, field="route" if reason == "stale_generation" else None)
    try:
        event = await capture_event(adapter, "om_gate")
        with pytest.raises(AuthorityRefusal) as caught:
            await authority.admit_native(event)
        assert caught.value.reason == reason
        assert await adapter._is_duplicate("om_gate") is (reason != "runtime_draining")
    finally:
        close_stores_under(authority.persistence.scope.profile_dir)
        await close_adapter(adapter)
