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
async def test_late_failure_after_disconnect_persists_release():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = adapter_for_test()
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
        reloaded = FeishuAdapter(PlatformConfig())
        assert "om_late_disconnect" not in reloaded._seen_message_ids
    finally:
        finish.set()
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
