"""Admission and cancellation boundaries for Feishu inbound dedup claims."""

import asyncio
import threading
from pathlib import Path
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
@pytest.mark.parametrize("phase", ["before_replay", "during_replay", "previously_handled"])
async def test_reused_event_tracks_each_handler_attempt(tmp_path, phase):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter.gateway_runner, runner.adapters = runner, {Platform.FEISHU: adapter}
    adapter._start_typing_refresh = lambda *_args, **_kwargs: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    runner._startup_restore_in_progress = phase != "previously_handled"
    runner._startup_restore_queue = []
    runner._intake_adapter_for = lambda _source: adapter
    adapter.set_message_handler(runner._handle_message if runner._startup_restore_in_progress else AsyncMock())
    entered = asyncio.Event()

    async def second_handler(_event):
        entered.set()
        await asyncio.Event().wait()

    try:
        event = await capture_event(adapter, "om_replayed")
        key = adapter._event_session_key(event)
        await adapter.handle_message(event)
        await asyncio.wait_for(adapter._session_tasks[key], 5)
        if phase != "previously_handled":
            assert runner._startup_restore_queue == [event]
            runner._startup_restore_in_progress = False
            if phase == "during_replay":
                adapter.set_message_handler(second_handler)
            assert await runner._drain_startup_restore_queue() == 1
        else:
            adapter.set_message_handler(second_handler)
            await adapter.handle_message(event)
        if phase == "during_replay":
            await asyncio.wait_for(entered.wait(), 5)
        await adapter.cancel_background_tasks()
        assert await adapter._is_duplicate("om_replayed") is (phase != "before_replay")
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
async def test_late_failure_after_disconnect_allows_shared_replacement_retry():
    from gateway.config import PlatformConfig
    from gateway.platforms.helpers import carry_inbound_dedup
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = adapter_for_test()
    replacement = None
    started, finish = asyncio.Event(), asyncio.Event()

    async def fail_after_disconnect(_event, _fence):
        started.set()
        await finish.wait()
        raise RuntimeError("before admission")

    try:
        event = await capture_event(adapter, "om_late_disconnect")
        task = asyncio.create_task(authority_with(fail_after_disconnect).admit_native(event))
        await asyncio.wait_for(started.wait(), 5)
        await adapter.disconnect()
        finish.set()
        with pytest.raises(RuntimeError, match="before admission"):
            await task
        replacement = FeishuAdapter(PlatformConfig())
        carry_inbound_dedup(adapter, replacement)
        assert await replacement._is_duplicate("om_late_disconnect") is False
    finally:
        finish.set()
        await close_adapter(adapter)
        if replacement is not None:
            await close_adapter(replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["merged_text", "separate_text", "text_and_media", "held_burst"])
async def test_timed_out_disconnect_releases_detached_inputs(tmp_path, kind):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.helpers import carry_inbound_dedup
    from gateway.run import GatewayRunner

    old, new = adapter_for_test(), adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._adapter_disconnect_timeout_secs = lambda: 0.05
    deliveries = [incoming("om_detached_a"), incoming("om_detached_b")]
    lock, locked = None, False
    try:
        if kind == "held_burst":
            class Handler:
                session_authority = object()

                async def handle(self, _event):
                    raise AssertionError("held burst must not execute")

            old.set_message_handler(Handler().handle)
            old._busy_text_mode = "queue"
            old._busy_text_debounce_seconds = old._busy_text_hard_cap_seconds = 3600
            first = await capture_event(old, "om_detached_a")
            second = await capture_event(old, "om_detached_b")
            key = old._event_session_key(first)
            old._active_sessions[key] = asyncio.Event()
            old._session_tasks[key] = asyncio.current_task()
            await old.handle_message(first)
            await old.handle_message(second)
        else:
            await old._handle_message_event_data(deliveries[0])
            if kind == "separate_text":
                deliveries[1].event.message.chat_id = "oc_other"
            if kind == "text_and_media":
                old._extract_message_content.return_value = ("", MessageType.PHOTO, ["/tmp/photo.png"], ["image/png"], [], [])
            await old._handle_message_event_data(deliveries[1])
        carry_inbound_dedup(old, new)
        lock = old._dedup_persist_lock
        await lock.acquire()
        locked = True
        await runner._safe_adapter_disconnect(old, Platform.FEISHU)
        assert old.__dict__.get("_feishu_ingress_cleanups")
        assert not old._pending_text_batches and not old._pending_media_batches
        assert not old.__dict__.get("_shared_bursts")
        lock.release()
        locked = False
        await asyncio.wait_for(asyncio.gather(*list(old._feishu_ingress_cleanups)), 5)
        await asyncio.wait_for(asyncio.gather(*list(old._feishu_disconnect_finalizers)), 5)
        new._process_inbound_message = AsyncMock()
        for delivery in deliveries:
            await new._handle_message_event_data(delivery)
        assert new._process_inbound_message.await_count == 2
    finally:
        if locked:
            lock.release()
        await close_adapter(old)
        await close_adapter(new)


@pytest.mark.asyncio
async def test_timed_out_background_wait_does_not_cancel_ingress_cleanup(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.platforms.helpers import carry_inbound_dedup
    from gateway.run import GatewayRunner

    old, new = adapter_for_test(), adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    old.set_message_handler(AsyncMock())
    old._start_typing_refresh = lambda *_args, **_kwargs: None
    old._stop_typing_refresh = AsyncMock()
    entered = asyncio.Event()

    async def hook(name, *_args):
        if name == "on_processing_start":
            entered.set()
            await asyncio.Event().wait()

    old._run_processing_hook = hook
    lock, locked = None, False
    try:
        event = await capture_event(old, "om_background_timeout")
        carry_inbound_dedup(old, new)
        await old.handle_message(event)
        await asyncio.wait_for(entered.wait(), 5)
        lock = old._dedup_persist_lock
        await lock.acquire()
        locked = True
        assert not await runner._await_adapter_cleanup_with_timeout(old.cancel_background_tasks(), 0.05)
        assert old.__dict__.get("_feishu_ingress_cleanups")
        lock.release()
        locked = False
        await asyncio.wait_for(asyncio.gather(*list(old._feishu_ingress_cleanups)), 5)
        new._process_inbound_message = AsyncMock()
        await new._handle_message_event_data(incoming("om_background_timeout"))
        assert new._process_inbound_message.await_count == 1
    finally:
        if locked:
            lock.release()
        await close_adapter(old)
        await close_adapter(new)


@pytest.mark.asyncio
async def test_disconnect_begins_cleanup_before_waiting_for_timers(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.helpers import carry_inbound_dedup
    from gateway.run import GatewayRunner

    old, new = adapter_for_test(), adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._adapter_disconnect_timeout_secs = lambda: 0.05
    cancel_pending = old._cancel_pending_tasks

    async def stalled_timers(_tasks):
        await asyncio.Event().wait()

    try:
        await old._handle_message_event_data(incoming("om_timer_timeout"))
        carry_inbound_dedup(old, new)
        old._cancel_pending_tasks = stalled_timers
        await runner._safe_adapter_disconnect(old, Platform.FEISHU)
        pending = list(old.__dict__.get("_feishu_ingress_cleanups", ()))
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 5)
        new._process_inbound_message = AsyncMock()
        await new._handle_message_event_data(incoming("om_timer_timeout"))
        assert new._process_inbound_message.await_count == 1
    finally:
        old._cancel_pending_tasks = cancel_pending
        for finalizer in list(old.__dict__.get("_feishu_disconnect_finalizers", ())):
            finalizer.cancel()
        await asyncio.gather(*list(old.__dict__.get("_feishu_disconnect_finalizers", ())), return_exceptions=True)
        await close_adapter(old)
        await close_adapter(new)


@pytest.mark.asyncio
async def test_timed_out_cleanup_does_not_skip_transport_teardown(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._adapter_disconnect_timeout_secs = lambda: 0.05
    ws = adapter._ws_client = SimpleNamespace(_auto_reconnect=True)
    adapter._teardown_ws_thread = AsyncMock()
    adapter._stop_webhook_server = AsyncMock()
    adapter._release_app_lock = AsyncMock()
    lock, locked = None, False
    try:
        await adapter._handle_message_event_data(incoming("om_slow_cleanup"))
        lock = adapter._dedup_persist_lock
        await lock.acquire()
        locked = True
        await runner._safe_adapter_disconnect(adapter, Platform.FEISHU)
        assert ws._auto_reconnect is False and adapter._ws_client is None
        adapter._teardown_ws_thread.assert_awaited_once()
        adapter._release_app_lock.assert_not_awaited()
        lock.release()
        locked = False
        await asyncio.wait_for(asyncio.gather(*list(adapter._feishu_disconnect_finalizers)), 5)
        adapter._release_app_lock.assert_awaited_once()
        assert "om_slow_cleanup" not in adapter._seen_message_ids
    finally:
        if locked:
            lock.release()
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_replacement_waits_for_old_app_lock_finalizer(tmp_path, monkeypatch):
    import plugins.platforms.feishu.adapter as feishu
    from gateway import status
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.helpers import carry_inbound_dedup
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(feishu, "_load_lark_oapi", lambda: True)
    old, new = adapter_for_test(), adapter_for_test()
    for adapter in (old, new):
        adapter._app_id, adapter._app_secret = "test-shared-app", "fake"
        adapter._connection_mode, adapter._verification_token = "webhook", "fake"
        adapter._connect_with_retry = AsyncMock()
        adapter._wire_plugin_handlers = lambda *_args, **_kwargs: None
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._adapter_disconnect_timeout_secs = lambda: 0.05
    locked, connecting = False, None
    try:
        assert await old.connect()
        await old._handle_message_event_data(incoming("om_app_lock_transfer"))
        lock = old._dedup_persist_lock
        await lock.acquire()
        locked = True
        await runner._safe_adapter_disconnect(old, Platform.FEISHU)
        carry_inbound_dedup(old, new)
        connecting = asyncio.create_task(new.connect())
        await asyncio.sleep(0)
        assert not connecting.done()
        lock.release()
        locked = False
        assert await asyncio.wait_for(connecting, 5)
        lock_path = status._get_scope_lock_path(feishu._FEISHU_APP_LOCK_SCOPE, new._app_id)
        assert lock_path.exists()
        new._process_inbound_message = AsyncMock()
        await new._handle_message_event_data(incoming("om_app_lock_transfer"))
        assert new._process_inbound_message.await_count == 1
    finally:
        if locked:
            lock.release()
        if connecting and not connecting.done():
            connecting.cancel()
            await asyncio.gather(connecting, return_exceptions=True)
        await new.disconnect()
        await close_adapter(old)
        await close_adapter(new)


@pytest.mark.asyncio
async def test_app_lock_finalizer_fences_other_profile_with_same_app(tmp_path, monkeypatch):
    import plugins.platforms.feishu.adapter as feishu
    from gateway import status
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(feishu, "_load_lark_oapi", lambda: True)

    def create(home, app_id):
        token = set_hermes_home_override(str(home))
        try:
            adapter = adapter_for_test()
        finally:
            reset_hermes_home_override(token)
        adapter._app_id, adapter._app_secret = app_id, "fake"
        adapter._connection_mode, adapter._verification_token = "webhook", "fake"
        adapter._connect_with_retry = AsyncMock()
        adapter._wire_plugin_handlers = lambda *_args, **_kwargs: None
        return adapter

    async def in_home(home, operation):
        token = set_hermes_home_override(str(home))
        try:
            return await operation()
        finally:
            reset_hermes_home_override(token)

    alice, bob, carol = (tmp_path / name for name in ("alice", "bob", "carol"))
    old = create(alice, "test-cross-profile-app")
    new = create(bob, "test-cross-profile-app")
    other = create(carol, "test-different-app")
    assert old._dedup_state_path != new._dedup_state_path
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._adapter_disconnect_timeout_secs = lambda: 0.05
    lock, locked, connecting = None, False, None
    try:
        assert await in_home(alice, old.connect)
        await in_home(alice, lambda: old._handle_message_event_data(incoming("om_cross_profile")))
        lock = old._dedup_persist_lock
        await lock.acquire()
        locked = True
        await in_home(alice, lambda: runner._bounded_adapter_teardown(old, Platform.FEISHU, profile="alice"))
        connecting = asyncio.create_task(in_home(bob, new.connect))
        await asyncio.sleep(0)
        assert not connecting.done()
        assert await asyncio.wait_for(in_home(carol, other.connect), 5)
        lock.release()
        locked = False
        assert await asyncio.wait_for(connecting, 5)
        path = status._get_scope_lock_path(feishu._FEISHU_APP_LOCK_SCOPE, new._app_id)
        assert path.exists()
    finally:
        if locked:
            lock.release()
        if connecting and not connecting.done():
            connecting.cancel()
            await asyncio.gather(connecting, return_exceptions=True)
        await in_home(bob, new.disconnect)
        await in_home(carol, other.disconnect)
        for adapter in (old, new, other):
            await close_adapter(adapter)


@pytest.mark.asyncio
async def test_connect_rechecks_app_finalizers_after_sdk_load(tmp_path, monkeypatch):
    import plugins.platforms.feishu.adapter as feishu
    from gateway import status
    from gateway.platforms.helpers import carry_inbound_dedup

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(feishu, "_load_lark_oapi", lambda: True)
    old, new = adapter_for_test(), adapter_for_test()
    for adapter in (old, new):
        adapter._app_id, adapter._app_secret = "test-sdk-load-race", "fake"
        adapter._connection_mode, adapter._verification_token = "webhook", "fake"
        adapter._connect_with_retry = AsyncMock()
        adapter._wire_plugin_handlers = lambda *_args, **_kwargs: None
    started, resume, loaded = threading.Event(), threading.Event(), threading.Event()

    def blocked_load():
        started.set()
        assert resume.wait(5)
        loaded.set()
        return True

    locked, connecting, closing = False, None, None
    try:
        assert await old.connect()
        await old._handle_message_event_data(incoming("om_sdk_load_race"))
        carry_inbound_dedup(old, new)
        monkeypatch.setattr(feishu, "_load_lark_oapi", blocked_load)
        connecting = asyncio.create_task(new.connect())
        assert await asyncio.to_thread(started.wait, 5)
        lock = old._dedup_persist_lock
        await lock.acquire()
        locked = True
        closing = asyncio.create_task(old.disconnect())

        async def wait_for_finalizer():
            while not old.__dict__.get("_feishu_disconnect_finalizers"):
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_finalizer(), 5)
        resume.set()
        assert await asyncio.to_thread(loaded.wait, 5)
        await asyncio.sleep(0)
        assert not connecting.done()
        lock.release()
        locked = False
        await asyncio.wait_for(closing, 5)
        assert await asyncio.wait_for(connecting, 5)
        path = status._get_scope_lock_path(feishu._FEISHU_APP_LOCK_SCOPE, new._app_id)
        assert path.exists()
    finally:
        resume.set()
        if locked:
            lock.release()
        if closing and not closing.done():
            await asyncio.gather(closing, return_exceptions=True)
        if connecting and not connecting.done():
            connecting.cancel()
            await asyncio.gather(connecting, return_exceptions=True)
        await new.disconnect()
        await close_adapter(old)
        await close_adapter(new)


@pytest.mark.asyncio
async def test_retired_adapter_does_not_overwrite_new_owner_seen_ids():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    old = adapter_for_test()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def delayed_failure(_event, _fence):
        entered.set()
        await finish.wait()
        raise RuntimeError("before admission")

    new = None
    try:
        event = await capture_event(old, "om_old")
        task = asyncio.create_task(authority_with(delayed_failure).admit_native(event))
        await asyncio.wait_for(entered.wait(), 5)
        await old.disconnect()
        new = FeishuAdapter(PlatformConfig())  # a distinct owner has its own in-memory cache
        assert await new._is_duplicate("om_new_owner") is False
        finish.set()
        with pytest.raises(RuntimeError, match="before admission"):
            await task
        reloaded = FeishuAdapter(PlatformConfig())
        assert "om_new_owner" in reloaded._seen_message_ids
    finally:
        finish.set()
        await close_adapter(old)
        if new is not None:
            await close_adapter(new)


@pytest.mark.asyncio
async def test_failed_teardown_retires_old_writes_and_blocks_replacement(monkeypatch):
    import plugins.platforms.feishu.adapter as feishu
    from gateway.platforms.helpers import carry_inbound_dedup

    old, new = adapter_for_test(), adapter_for_test()
    old._stop_webhook_server = AsyncMock(side_effect=RuntimeError("webhook teardown failed"))
    old._teardown_ws_thread = AsyncMock()
    old._release_app_lock = AsyncMock()
    new._app_id, new._app_secret = "fake", "fake"
    new._connection_mode, new._verification_token = "webhook", "fake"
    new._connect_with_retry = AsyncMock()
    new._wire_plugin_handlers = lambda *_args, **_kwargs: None
    new._release_app_lock = AsyncMock()
    monkeypatch.setattr(feishu, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(feishu, "acquire_scoped_lock", lambda *_args, **_kwargs: (True, None))
    try:
        await old._handle_message_event_data(incoming("om_failed_teardown"))
        with pytest.raises(RuntimeError, match="webhook teardown failed"):
            await old.disconnect()
        old._teardown_ws_thread.assert_awaited_once()
        old._release_app_lock.assert_not_awaited()  # incomplete teardown keeps the old ownership fence
        assert old._dedup_persistence_retired is True
        carry_inbound_dedup(old, new)
        assert not await new.connect()
    finally:
        await close_adapter(old)
        await close_adapter(new)


@pytest.mark.asyncio
async def test_executor_shutdown_during_cleanup_releases_all_buffers():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = adapter_for_test()
    first, second = incoming("om_queued_cleanup_a"), incoming("om_queued_cleanup_b")
    second.event.message.chat_id = "oc_other"
    started, finish = threading.Barrier(11), threading.Event()
    submitted, cancelled_write = asyncio.Event(), asyncio.Event()

    def block_worker():
        started.wait(timeout=5)
        finish.wait(5)

    blockers = []
    try:
        assert await adapter._is_duplicate("om_accepted") is False
        await adapter._handle_message_event_data(first)
        await adapter._handle_message_event_data(second)
        executor = adapter._get_sdk_executor()
        blockers = [executor.submit(block_worker) for _ in range(10)]
        await asyncio.to_thread(started.wait, 5)
        run_blocking = adapter._run_blocking

        async def watched_run(func, *args):
            task = asyncio.create_task(run_blocking(func, *args))
            await asyncio.sleep(0)  # let the task enqueue its write before teardown continues
            submitted.set()
            try:
                return await task
            except asyncio.CancelledError:
                cancelled_write.set()
                raise

        async def wait_for_submit(*_args):
            await asyncio.wait_for(submitted.wait(), 5)

        adapter._run_blocking = watched_run
        adapter._teardown_ws_thread = wait_for_submit
        await asyncio.wait_for(adapter.disconnect(), 5)
        assert cancelled_write.is_set()
        reloaded = FeishuAdapter(PlatformConfig())
        assert "om_accepted" in reloaded._seen_message_ids
        assert "om_queued_cleanup_a" not in reloaded._seen_message_ids
        assert "om_queued_cleanup_b" not in reloaded._seen_message_ids
    finally:
        finish.set()
        if blockers:
            await asyncio.gather(*(asyncio.wrap_future(future) for future in blockers), return_exceptions=True)
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_disconnect_cleanup_survives_default_executor_shutdown():
    import concurrent.futures
    import json

    adapter = adapter_for_test()
    loop = asyncio.get_running_loop()
    original_executor = loop._default_executor
    try:
        await adapter._handle_message_event_data(incoming("om_default_executor_closed"))
        closed_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(closed_executor)
        closed_executor.shutdown(wait=True)
        await adapter.disconnect()
        payload = json.loads(adapter._dedup_state_path.read_text())
        assert "om_default_executor_closed" not in payload["message_ids"]
        assert "_feishu_cleanup_executor" not in adapter.__dict__
    finally:
        loop._default_executor = original_executor
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_late_inbound_during_disconnect_is_released_even_if_caller_cancelled():
    adapter = adapter_for_test()
    dispatch = adapter._dispatch_inbound_event
    locked = False
    try:
        event = await capture_event(adapter, "om_late_during_close")
        adapter._ingress_closing = True
        lock = adapter._dedup_persist_lock
        await lock.acquire()
        locked = True
        caller = asyncio.create_task(dispatch(event))
        while not adapter.__dict__.get("_feishu_ingress_cleanups"):
            await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not adapter._pending_text_batches
        lock.release()
        locked = False
        await asyncio.wait_for(asyncio.gather(*list(adapter._feishu_ingress_cleanups)), 5)
        assert await adapter._is_duplicate("om_late_during_close") is False
    finally:
        if locked:
            lock.release()
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_reusing_adapter_does_not_keep_finished_busy_bursts(monkeypatch):
    import plugins.platforms.feishu.adapter as feishu

    adapter = adapter_for_test()

    class Handler:
        session_authority = object()

        async def handle(self, _event):
            raise AssertionError("held burst must not execute")

    adapter.set_message_handler(Handler().handle)
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = adapter._busy_text_hard_cap_seconds = 3600
    adapter._app_id, adapter._app_secret = "fake", "fake"
    adapter._connection_mode, adapter._verification_token = "webhook", "fake"
    adapter._connect_with_retry = AsyncMock()
    adapter._wire_plugin_handlers = lambda *_args, **_kwargs: None
    adapter._release_app_lock = AsyncMock()
    monkeypatch.setattr(feishu, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(feishu, "acquire_scoped_lock", lambda *_args, **_kwargs: (True, None))
    try:
        for index in range(3):
            assert await adapter.connect()
            first = await capture_event(adapter, f"om_reuse_{index}_a")
            second = await capture_event(adapter, f"om_reuse_{index}_b")
            key = adapter._event_session_key(first)
            adapter._active_sessions[key] = asyncio.Event()
            adapter._session_tasks[key] = asyncio.current_task()
            await adapter.handle_message(first)
            await adapter.handle_message(second)
            await adapter.disconnect()
            assert not adapter.__dict__.get("_feishu_ingress_busy_bursts")
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["held", "commit_scheduled", "handler_entered"])
async def test_shutdown_busy_burst_releases_only_uncommitted_inputs(phase):
    from gateway.session_ingress_burst import _flush

    adapter = adapter_for_test()
    entered = asyncio.Event()

    class Handler:
        session_authority = object()

        async def handle(self, _event):
            entered.set()
            await asyncio.Event().wait()

    handler = Handler()
    adapter.set_message_handler(handler.handle)
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = adapter._busy_text_hard_cap_seconds = 3600
    try:
        first = await capture_event(adapter, "om_held_a")
        second = await capture_event(adapter, "om_held_b")
        key = adapter._event_session_key(first)
        adapter._active_sessions[key] = asyncio.Event()
        adapter._session_tasks[key] = asyncio.current_task()
        await adapter.handle_message(first)
        await adapter.handle_message(second)
        burst = adapter._shared_bursts[key]
        assert len(burst.head._ingress_claims) == 2 and burst.commit is None
        if phase != "held":
            _flush(adapter, key, burst)
        if phase == "handler_entered":
            await asyncio.wait_for(entered.wait(), 5)
        await adapter.cancel_background_tasks()
        assert await adapter._is_duplicate("om_held_a") is (phase == "handler_entered")
        assert await adapter._is_duplicate("om_held_b") is (phase == "handler_entered")
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["unstarted", "held", "commit_scheduled"])
async def test_stop_fence_keeps_pre_handler_claims(tmp_path, phase):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_ingress_burst import _flush

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    epoch = [0]
    runner.session_authority = SimpleNamespace(route_epoch=lambda _route: epoch[0])
    runner._session_key_for_source = lambda _source: "route"
    adapter.gateway_runner = runner

    class Handler:
        session_authority = runner.session_authority

        async def handle(self, _event):
            await asyncio.Event().wait()

    adapter.set_message_handler(Handler().handle)
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = adapter._busy_text_hard_cap_seconds = 3600
    try:
        event = await capture_event(adapter, "om_controlled")
        key = adapter._event_session_key(event)
        if phase != "unstarted":
            adapter._active_sessions[key] = asyncio.Event()
            adapter._session_tasks[key] = asyncio.current_task()
        await adapter.handle_message(event)
        if phase == "commit_scheduled":
            _flush(adapter, key, adapter._shared_bursts[key])
        epoch[0] += 1  # a later accepted /stop or /new fenced this route
        await adapter.cancel_background_tasks()
        assert await adapter._is_duplicate("om_controlled") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_local_stop_keeps_scheduled_busy_claim():
    from gateway.session_ingress_burst import _flush

    adapter = adapter_for_test()

    class Handler:
        session_authority = object()

        async def handle(self, _event):
            await asyncio.Event().wait()

    adapter.set_message_handler(Handler().handle)
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = adapter._busy_text_hard_cap_seconds = 3600
    try:
        event = await capture_event(adapter, "om_local_stop")
        key = adapter._event_session_key(event)
        adapter._active_sessions[key] = asyncio.Event()
        adapter._session_tasks[key] = asyncio.current_task()
        await adapter.handle_message(event)
        _flush(adapter, key, adapter._shared_bursts[key])
        adapter._session_tasks.pop(key)  # the accepted control has no live owner task to cancel
        await adapter.cancel_session_processing(key)
        await adapter.cancel_background_tasks()
        assert await adapter._is_duplicate("om_local_stop") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_stop_between_unstarted_task_and_cleanup_keeps_claim():
    adapter = adapter_for_test()
    adapter.set_message_handler(AsyncMock())
    try:
        event = await capture_event(adapter, "om_stop_cleanup")
        key = adapter._event_session_key(event)
        await adapter.handle_message(event)
        task = adapter._session_tasks[key]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The task callback has scheduled cleanup, but this stop can still precede that cleanup.
        assert adapter.__dict__.get("_feishu_ingress_cleanups")
        await adapter.cancel_session_processing(key)
        await adapter.cancel_background_tasks()
        assert await adapter._is_duplicate("om_stop_cleanup") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_stop_while_batch_release_waits_for_lock_keeps_every_claim():
    adapter = adapter_for_test()
    lock = adapter._dedup_persist_lock
    locked = False
    try:
        first = await capture_event(adapter, "om_locked_a")
        second = await capture_event(adapter, "om_locked_b")
        first.absorb_turn_input(second)
        key = adapter._event_session_key(first)
        adapter._capture_ingress_controls(first, key)
        await lock.acquire()
        locked = True
        cleanup = asyncio.create_task(first._release_unadmitted_claims())
        while first._ingress_claims:
            await asyncio.sleep(0)
        await adapter.cancel_session_processing(key)
        lock.release()
        locked = False
        await cleanup
        assert await adapter._is_duplicate("om_locked_a") is True
        assert await adapter._is_duplicate("om_locked_b") is True
    finally:
        if locked:
            lock.release()
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unrelated_profile", "owner_stopped", "owner_replaced", "unknown_owner"])
async def test_ingress_fence_uses_the_pinned_profile(tmp_path, change):
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_identity import RoutingIdentity

    adapter = adapter_for_test()
    launch, alice, bob = tmp_path / "launch", tmp_path / "alice", tmp_path / "bob"
    registry = SessionAuthorities(launch, multiplexed=True)
    epoch = [0]
    owner = SimpleNamespace(route_epoch=lambda _route: epoch[0])
    registry.add(alice, owner)
    adapter.gateway_runner = SimpleNamespace(
        session_authorities=registry, _session_key_for_source=lambda _source: "route")
    try:
        event = await capture_event(adapter, "om_profile_fence")
        if change != "unknown_owner":
            event.source._identity = RoutingIdentity("default", "alice", launch, alice, multiplexed=True)
        adapter._capture_ingress_controls(event, "route")
        if change in {"unrelated_profile", "unknown_owner"}:
            registry.add(bob, SimpleNamespace(route_epoch=lambda _route: 0))
        elif change == "owner_stopped":
            epoch[0] += 1
        else:
            registry.remove(alice)
            registry.add(alice, SimpleNamespace(route_epoch=lambda _route: 0))
        await event._release_unadmitted_claims()
        assert await adapter._is_duplicate("om_profile_fence") is (change != "unrelated_profile")
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
async def test_user_stop_during_prewrite_admission_keeps_claim():
    adapter = adapter_for_test()
    entered = asyncio.Event()

    async def prewrite(_event, _fence):
        entered.set()
        await asyncio.Event().wait()

    authority = authority_with(prewrite)
    adapter.set_message_handler(authority.admit_native)
    adapter._start_typing_refresh = lambda *_args, **_kwargs: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    try:
        event = await capture_event(adapter, "om_stop_prewrite")
        key = adapter._event_session_key(event)
        await adapter.handle_message(event)
        await asyncio.wait_for(entered.wait(), 5)
        await adapter.cancel_session_processing(key)
        assert await adapter._is_duplicate("om_stop_prewrite") is True
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,case", [
    ("default", "failure"), ("secondary", "failure"), ("standalone", "failure"),
    ("default", "cancelled"), ("default", "fenced"), ("default", "handler_failed"),
])
async def test_profile_setup_releases_only_before_unfenced_handler(tmp_path, monkeypatch, profile, case):
    from contextlib import asynccontextmanager, contextmanager
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    epoch = [0]
    runner.session_authority = SimpleNamespace(route_epoch=lambda _route: epoch[0])
    runner._session_key_for_source = lambda _source: "route"
    runner._admit_primary_source = lambda _source, default: default
    runner._canonicalize = lambda *_args, **_kwargs: None
    runner._routed_profile_home = lambda _name: tmp_path
    runner._handle_message = AsyncMock(side_effect=RuntimeError("handler failed"))
    adapter.gateway_runner = runner
    entered, proceed = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def scoped(_home):
        entered.set()
        await proceed.wait()
        if case != "handler_failed":
            raise RuntimeError("scope failed")
        yield

    @contextmanager
    def failed_standalone():
        raise RuntimeError("scope failed")
        yield

    monkeypatch.setattr(run, "_async_profile_runtime_scope", scoped)
    if profile == "default":
        handler = runner._make_default_profile_message_handler()
    elif profile == "secondary":
        handler = runner._make_profile_message_handler("secondary")
    else:
        monkeypatch.setattr(runner, "_standalone_launch_scope", failed_standalone)
        handler = runner._standalone_scoped(runner._handle_message, release_ingress_on_scope_failure=True)
    try:
        event = await capture_event(adapter, "om_scope")
        adapter._capture_ingress_controls(event)
        task = asyncio.create_task(handler(event))
        if profile != "standalone":
            await asyncio.wait_for(entered.wait(), 5)
            if case == "fenced":
                epoch[0] += 1
            if case == "cancelled":
                task.cancel()
            else:
                proceed.set()
        with pytest.raises(asyncio.CancelledError if case == "cancelled" else RuntimeError):
            await task
        assert await adapter._is_duplicate("om_scope") is (case in {"fenced", "handler_failed"})
    finally:
        proceed.set()
        await close_adapter(adapter)


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
async def test_busy_voice_retry_preserves_each_successful_echo(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_inbound_transcription import hand_prepared_voice_to_turn

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._should_echo_stt_transcripts = lambda: True
    runner._echo_stt_transcripts = AsyncMock(return_value=True)
    runner._transcribe_one_clip = AsyncMock(return_value=("one", "one"))
    paths = [str(tmp_path / f"clip-{i}.ogg") for i in range(3)]
    for index, path in enumerate(paths):
        Path(path).write_bytes(bytes([index]) * (index + 1))

    def attach_voice(event):
        event.text, event.message_type = "", MessageType.VOICE
        event.media_urls, event.media_types = paths, ["audio/ogg"] * len(paths)

    try:
        first = await capture_event(adapter, "om_voice_retry")
        attach_voice(first)
        # The second clip was shown, the third was not, and the first was not yet transcribed.
        first._gateway_pending_stt_clips = {paths[1]: ("two", "two"), paths[2]: ("three", "three")}
        first._gateway_pending_stt_shown = {0}
        await asyncio.to_thread(hand_prepared_voice_to_turn, first)
        await first._release_unadmitted_claims()

        retry = await capture_event(adapter, "om_voice_retry")
        attach_voice(retry)
        _, transcripts = await runner._transcribe_pending_audio_event_once(retry)
        await runner._echo_pending_stt_transcripts_once(retry, adapter, retry.source, transcripts)
        assert transcripts == ["one", "two", "three"]
        assert [call.args[2] for call in runner._echo_stt_transcripts.await_args_list] == [["one"], ["three"]]
        await asyncio.to_thread(hand_prepared_voice_to_turn, retry)
        await retry._release_unadmitted_claims()

        again = await capture_event(adapter, "om_voice_retry")
        attach_voice(again)
        _, transcripts = await runner._transcribe_pending_audio_event_once(again)
        await runner._echo_pending_stt_transcripts_once(again, adapter, again.source, transcripts)
        assert runner._transcribe_one_clip.await_count == 1
        assert runner._echo_stt_transcripts.await_count == 2
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["first_redelivery", "second_redelivery", "restored_merged_turn"])
async def test_merged_voice_retry_keeps_each_message_echo_identity(tmp_path, destination):
    from dataclasses import replace
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_inbound_transcription import hand_prepared_voice_to_turn

    adapter = adapter_for_test()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._should_echo_stt_transcripts = lambda: True
    runner._echo_stt_transcripts = AsyncMock(return_value=True)
    runner._transcribe_one_clip = AsyncMock(side_effect=[
        ("first", "first"), ("second", "second"), ("second", "second")])
    first_path, second_path = tmp_path / "a.ogg", tmp_path / "b.ogg"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    def attach(event, path):
        event.text, event.message_type = "", MessageType.VOICE
        event.media_urls, event.media_types = [str(path)], ["audio/ogg"]

    try:
        first = await capture_event(adapter, "om_voice_first")
        second = await capture_event(adapter, "om_voice_second")
        attach(first, first_path)
        attach(second, second_path)
        first.absorb_turn_input(second)
        first.media_urls.extend(second.media_urls)
        first.media_types.extend(second.media_types)
        first.message_id = first.source.message_id = second.message_id  # Feishu anchors a batch to its latest input
        _, transcripts = await runner._transcribe_pending_audio_event_once(first)
        await runner._echo_pending_stt_transcripts_once(first, adapter, first.source, transcripts)
        await asyncio.to_thread(hand_prepared_voice_to_turn, first)
        await first._release_unadmitted_claims()

        if destination != "restored_merged_turn":
            is_first = destination == "first_redelivery"
            retry = await capture_event(adapter, "om_voice_first" if is_first else "om_voice_second")
            attach(retry, first_path if is_first else second_path)
        else:
            retry = replace(first, _ingress_claims=[])
        _, transcripts = await runner._transcribe_pending_audio_event_once(retry)
        await runner._echo_pending_stt_transcripts_once(retry, adapter, retry.source, transcripts)
        assert runner._transcribe_one_clip.await_count == 2
        assert runner._echo_stt_transcripts.await_count == 2
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_sizes", [(18,), (8, 8, 8)])
async def test_merged_voice_aliases_do_not_reduce_handoff_capacity(tmp_path, monkeypatch, batch_sizes):
    from collections import OrderedDict
    from dataclasses import replace
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway import run_inbound_transcription as transcription

    monkeypatch.setattr(transcription, "_HANDED_OFF", OrderedDict())
    adapter = adapter_for_test()
    adapter._text_batch_max_messages, adapter._text_batch_max_chars = 32, 100000
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._should_echo_stt_transcripts = lambda: True
    runner._echo_stt_transcripts = AsyncMock(return_value=True)
    runner._transcribe_one_clip = AsyncMock(return_value=("transcript", "transcript"))
    batches = []
    try:
        for group, size in enumerate(batch_sizes):
            for index in range(size):
                path = tmp_path / f"clip-{group}-{index}.ogg"
                path.write_bytes(f"voice-{group}-{index}".encode())
                event = await capture_event(adapter, f"om_many_{group}_{index}")
                event.source.chat_id = f"oc_chat_{group}"
                event.text, event.message_type = "audio", MessageType.TEXT
                event.media_urls, event.media_types = [str(path)], ["audio/ogg"]
                await adapter._enqueue_text_event(event)
            batch = next(event for event in adapter._pending_text_batches.values()
                         if event.source.chat_id == f"oc_chat_{group}")
            batches.append(batch)
            _, transcripts = await runner._transcribe_pending_audio_event_once(batch)
            await runner._echo_pending_stt_transcripts_once(batch, adapter, batch.source, transcripts)
            await asyncio.to_thread(transcription.hand_prepared_voice_to_turn, batch)
        assert len(transcription._HANDED_OFF) == sum(batch_sizes)

        restored = replace(batches[0], _ingress_claims=[])
        _, transcripts = await runner._transcribe_pending_audio_event_once(restored)
        await runner._echo_pending_stt_transcripts_once(restored, adapter, restored.source, transcripts)
        assert runner._transcribe_one_clip.await_count == sum(batch_sizes)
        assert runner._echo_stt_transcripts.await_count == sum(batch_sizes)
    finally:
        await close_adapter(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["refused", "committed", "stopped_refused"])
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
        return GateRefusal("queue_full") if outcome != "committed" else None

    authority.gate_refusal = lambda *_args, **_kwargs: gate
    try:
        event = await capture_event(adapter, "om_late_write")
        key = adapter._event_session_key(event)
        adapter._capture_ingress_controls(event, key)
        caller = asyncio.create_task(authority.admit_native(event))
        assert await asyncio.to_thread(started.wait, 5)
        if outcome == "stopped_refused":
            await adapter.cancel_session_processing(key)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert await adapter._is_duplicate("om_late_write") is True  # the write is still unresolved
        finish.set()
        await asyncio.gather(*list(authority._mutation_tasks), return_exceptions=True)
        rows = authority.persistence.admissions.match(request_id="om_late_write")
        assert bool(rows) is (outcome == "committed")
        assert await adapter._is_duplicate("om_late_write") is (outcome != "refused")
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
