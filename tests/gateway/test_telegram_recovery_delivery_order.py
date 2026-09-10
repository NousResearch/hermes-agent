"""Real final-send and SQLite recovery contracts; only external calls are replaced."""
import asyncio
import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# The gateway fixture installs optional-library stubs before collection.
if not isinstance(sys.modules.get("telegram"), ModuleType):
    for name in list(sys.modules):
        if name == "telegram" or name.startswith("telegram."):
            del sys.modules[name]
pytest.importorskip("telegram")

from gateway import delivery_ledger as ledger
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest_asyncio.fixture
async def delivery_system(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("gateway:\n  delivery_ledger: true\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setattr("tools.tirith_security._install_tirith", lambda **kwargs: (None, "offline test"))
    real_sleep = asyncio.sleep

    async def no_backoff(delay):
        await real_sleep(0)

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", no_backoff)
    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner._active_profile_name = lambda: "default"
    runner.session_store = None
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    runner._async_session_store = store
    slots = {}
    for index, profile in enumerate(("default", "secondary"), start=1):
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"test-{profile}", typing_indicator=False))
        adapter._owner_profile = profile
        monkeypatch.setattr(adapter, "_bot", SimpleNamespace(
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=999))
        ))
        adapter._running = True
        adapter._rich_send_disabled = True
        monkeypatch.setattr(adapter, "_write_runtime_status_safe", MagicMock())
        adapter.gateway_runner = runner
        generation, _ = adapter._begin_polling_generation()
        answer = f"Completed answer for {profile}"
        adapter._message_handler = AsyncMock(return_value=answer)
        monkeypatch.setattr(adapter, "on_processing_complete", AsyncMock())
        event = MessageEvent(
            text=f"Request for {profile}.", message_type=MessageType.TEXT, message_id=str(index),
            source=SessionSource(platform=Platform.TELEGRAM, profile=profile,
                                 chat_id=str(-10000 - index), chat_type="group", thread_id=str(500 + index)),
        )
        key = f"agent:{profile}:telegram:group:{event.source.chat_id}:thread:{event.source.thread_id}"
        obligation_id = ledger.compute_obligation_id(key, str(index), answer)
        slots[profile] = SimpleNamespace(adapter=adapter, event=event, key=key, answer=answer,
                                        generation=generation, obligation_id=obligation_id)
    runner.adapters = {Platform.TELEGRAM: slots["default"].adapter}
    runner._profile_adapters = {"secondary": {Platform.TELEGRAM: slots["secondary"].adapter}}
    tasks = []
    releases = []
    workers = []
    system = SimpleNamespace(runner=runner, slots=slots, home=home, tasks=tasks, releases=releases,
                             workers=workers, extra_adapters=[])
    try:
        yield system
    finally:
        for release in releases:
            release.set()
        pending = set(tasks)
        for slot in slots.values():
            pending.update(slot.adapter._background_tasks)
        for adapter in system.extra_adapters:
            pending.update(adapter._background_tasks)
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=10)
        for started, completed in workers:
            if started.is_set():
                assert await asyncio.to_thread(completed.wait, 10), "ledger worker outlived its test"


def _row(system, slot):
    with sqlite3.connect(system.home / "state.db") as connection:
        connection.row_factory = sqlite3.Row
        return dict(connection.execute(
            "SELECT * FROM delivery_obligations WHERE obligation_id = ?", (slot.obligation_id,)
        ).fetchone())


def _produce(system, slot):
    slot.adapter._active_sessions[slot.key] = asyncio.Event()
    task = asyncio.create_task(slot.adapter._process_message_background(slot.event, slot.key))
    slot.adapter._session_tasks[slot.key] = task
    system.tasks.append(task)
    return task


async def _drain_background(slot):
    while slot.adapter._background_tasks:
        await asyncio.wait_for(asyncio.gather(*tuple(slot.adapter._background_tasks)), timeout=10)


def _pause_ledger_write(system, slot, monkeypatch, *, method="mark_failed", after_write=False):
    entered = asyncio.Event()
    released = threading.Event()
    system.releases.append(released)
    loop = asyncio.get_running_loop()
    real_write = getattr(ledger, method)
    started, completed = threading.Event(), threading.Event()
    system.workers.append((started, completed))

    def delayed_write(obligation_id, error=""):
        if obligation_id != slot.obligation_id:
            return real_write(obligation_id, error)
        started.set()
        try:
            result = None
            if after_write:
                result = real_write(obligation_id, error)
            loop.call_soon_threadsafe(entered.set)
            if not released.wait(15):
                raise TimeoutError("test did not release failure persistence")
            if not after_write:
                result = real_write(obligation_id, error)
            return result
        finally:
            completed.set()

    monkeypatch.setattr(ledger, method, delayed_write)
    return entered, released


def _assert_failed(system, slot):
    row = _row(system, slot)
    assert (row["state"], row["attempts"], row["last_error"]) == ("failed", 0, "send_path_degraded")
    assert row["content"] == slot.answer
    assert row["adapter_profile"] == slot.event.source.profile
    slot.adapter._bot.send_message.assert_not_awaited()


def _assert_delivered(system, slot, *, attempts=1):
    row = _row(system, slot)
    assert (row["state"], row["attempts"]) == ("delivered", attempts)
    assert row["content"] == slot.answer
    slot.adapter._message_handler.assert_awaited_once_with(slot.event)
    slot.adapter._bot.send_message.assert_awaited_once()
    sent = slot.adapter._bot.send_message.await_args.kwargs
    assert sent["chat_id"] == int(slot.event.source.chat_id)
    assert sent["message_thread_id"] == int(slot.event.source.thread_id)
    assert sent["text"] == ledger.RECONNECTED_MARKER + slot.answer
    assert slot.key not in slot.adapter._active_sessions


async def _claim_interrupted_startup_reply(system, slot):
    # Persist an interrupted send with a real, subsequently dead process owner.
    payload = {
        "obligation_id": slot.obligation_id, "session_key": slot.key,
        "platform": Platform.TELEGRAM.value, "chat_id": slot.event.source.chat_id,
        "thread_id": slot.event.source.thread_id, "content": slot.answer,
        "adapter_profile": slot.event.source.profile,
    }
    script = (
        "import json, sys; "
        "from gateway.delivery_ledger import record_obligation, mark_attempting; "
        "payload = json.loads(sys.argv[1]); "
        "record_obligation(**payload); mark_attempting(payload['obligation_id'])"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, json.dumps(payload),
        cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode()
    row = _row(system, slot)
    assert row["state"] == "attempting"
    assert not ledger._owner_alive(row["owner_pid"], row["owner_started_at"])
    claimed = await system.runner._claim_pending_obligations()
    assert [row["obligation_id"] for row in claimed] == [slot.obligation_id]
    assert not claimed[0].get("runtime_recovery")
    return claimed


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("order", [
    "failure_before_progress", "progress_before_failure", "overlapping_sweeps",
    "replay_failure_before_progress", "progress_before_replay_failure", "progress_before_claim_release",
])
async def test_completed_reply_recovers_once_for_its_transport_owner(
    delivery_system, monkeypatch, profile, order
):
    system = delivery_system
    slot = system.slots[profile]
    sibling = system.slots["secondary" if profile == "default" else "default"]
    await asyncio.wait_for(_produce(system, sibling), timeout=10)
    _assert_failed(system, sibling)
    adapter = slot.adapter
    sweep = AsyncMock(wraps=system.runner._redeliver_failed_obligations_for_platform)
    system.runner._redeliver_failed_obligations_for_platform = sweep
    attempts = 1
    if order == "progress_before_claim_release":
        await asyncio.wait_for(_produce(system, slot), timeout=10)
        _assert_failed(system, slot)
        claim_entered, claim_release = asyncio.Event(), asyncio.Event()
        system.releases.append(claim_release)

        async def pause_before_dispatch(session_key):
            assert session_key == slot.key
            claim_entered.set()
            await claim_release.wait()

        system.runner._async_session_store.clear_resume_pending.side_effect = pause_before_dispatch
        release_entered, release_write = _pause_ledger_write(
            system, slot, monkeypatch, method="release_runtime_claim"
        )
        adapter._record_polling_progress(slot.generation)
        await asyncio.wait_for(claim_entered.wait(), timeout=10)
        slot.generation, _ = adapter._begin_polling_generation()
        claim_release.set()
        await asyncio.wait_for(release_entered.wait(), timeout=10)
        row = _row(system, slot)
        assert (row["state"], row["attempts"]) == ("attempting", 1)
        previous_tasks = set(adapter._background_tasks)
        adapter._record_polling_progress(slot.generation)
        new_tasks = set(adapter._background_tasks) - previous_tasks
        if new_tasks:
            await asyncio.wait_for(asyncio.gather(*new_tasks), timeout=10)
        assert _row(system, slot)["state"] == "attempting"
        release_write.set()
        await _drain_background(slot)
    elif "replay" in order:
        attempts = 2
        await asyncio.wait_for(_produce(system, slot), timeout=10)
        _assert_failed(system, slot)
        claimed = await _claim_interrupted_startup_reply(system, slot)
        assert adapter.send_path_degraded
        if order == "progress_before_replay_failure":
            entered, released = _pause_ledger_write(system, slot, monkeypatch)
            boot_send = asyncio.create_task(system.runner._redeliver_claimed_obligations(claimed))
            system.tasks.append(boot_send)
            await asyncio.wait_for(entered.wait(), timeout=10)
            assert _row(system, slot)["state"] == "attempting"
            adapter._record_polling_progress(slot.generation)
            await _drain_background(slot)
            assert _row(system, slot)["state"] == "attempting"
            released.set()
            await asyncio.wait_for(boot_send, timeout=10)
            await _drain_background(slot)
        else:
            assert await system.runner._redeliver_claimed_obligations(claimed) == 0
            row = _row(system, slot)
            assert (row["state"], row["attempts"], row["last_error"]) == ("failed", 1, "send_path_degraded")
            adapter._bot.send_message.assert_not_awaited()
            adapter._record_polling_progress(slot.generation)
            await _drain_background(slot)
    elif order == "failure_before_progress":
        await asyncio.wait_for(_produce(system, slot), timeout=10)
        _assert_failed(system, slot)
        adapter._record_polling_progress(slot.generation)
        await _drain_background(slot)
    else:
        entered, released = _pause_ledger_write(
            system, slot, monkeypatch, after_write=order == "overlapping_sweeps"
        )
        task = _produce(system, slot)
        await asyncio.wait_for(entered.wait(), timeout=10)
        adapter._bot.send_message.assert_not_awaited()
        if order == "progress_before_failure":
            assert _row(system, slot)["state"] == "attempting"
            adapter._record_polling_progress(slot.generation)
            await _drain_background(slot)
            assert _row(system, slot)["state"] == "attempting"
            released.set()
            await asyncio.wait_for(task, timeout=10)
            await _drain_background(slot)
        else:
            send_entered, send_release = asyncio.Event(), asyncio.Event()
            system.releases.append(send_release)

            async def blocked_send(**kwargs):
                send_entered.set()
                await send_release.wait()
                return SimpleNamespace(message_id=999)

            adapter._bot.send_message.side_effect = blocked_send
            adapter._record_polling_progress(slot.generation)
            await asyncio.wait_for(send_entered.wait(), timeout=10)
            assert _row(system, slot)["state"] == "attempting"
            released.set()
            # The finalizer's second sweep finishes while recovery still awaits the ACK.
            await asyncio.wait_for(task, timeout=10)
            assert sweep.await_count == 2
            send_release.set()
            await _drain_background(slot)
    _assert_delivered(system, slot, attempts=attempts)
    assert adapter.on_processing_complete.await_args.args[1] == ProcessingOutcome.FAILURE
    _assert_failed(system, sibling)
    sweep_count = sweep.await_count
    adapter._record_polling_progress(slot.generation - 1)
    adapter._record_polling_progress(slot.generation)
    adapter._record_polling_progress(slot.generation)
    await _drain_background(slot)
    assert sweep.await_count == sweep_count
    assert await system.runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM, profile=profile) == 0
    _assert_delivered(system, slot, attempts=attempts)
    _assert_failed(system, sibling)
    system.runner._async_session_store.clear_resume_pending.side_effect = None
    sibling.adapter._record_polling_progress(sibling.generation)
    await _drain_background(sibling)
    _assert_delivered(system, sibling)


def _make_unavailable(slot, state):
    def begin_teardown():
        # disconnect() marks disconnected and fences polling before its first await.
        slot.adapter._mark_disconnected()
        slot.adapter._polling_teardown_started = True
        slot.adapter._fence_polling()

    actions = {
        "degraded": slot.adapter._begin_polling_generation,
        "stopped": lambda: setattr(slot.adapter, "_running", False),
        "fatal": lambda: slot.adapter._set_fatal_error(
            "telegram_auth_error", "synthetic rejection", retryable=False
        ),
        "teardown": begin_teardown,
        "runner_shutdown": lambda: setattr(slot.adapter.gateway_runner, "_draining", True),
    }
    actions[state]()


def _install_replacement(system, slot, monkeypatch):
    profile = slot.event.source.profile
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"replacement-{profile}", typing_indicator=False))
    adapter._owner_profile = profile
    adapter.gateway_runner = system.runner
    adapter._running = True
    adapter._rich_send_disabled = True
    monkeypatch.setattr(adapter, "_bot", SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=1000))
    ))
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", MagicMock())
    if profile == "default":
        system.runner.adapters[Platform.TELEGRAM] = adapter
    else:
        system.runner._profile_adapters[profile][Platform.TELEGRAM] = adapter
    system.extra_adapters.append(adapter)
    generation, _ = adapter._begin_polling_generation()
    return adapter, generation


async def _failed_batch(system, profile):
    slot = system.slots[profile]
    sibling = system.slots["secondary" if profile == "default" else "default"]
    await asyncio.wait_for(_produce(system, slot), 10)
    await asyncio.wait_for(_produce(system, sibling), 10)
    extra = SimpleNamespace(obligation_id=slot.obligation_id + "-extra", answer=slot.answer + " again")
    ledger.record_obligation(
        obligation_id=extra.obligation_id, session_key=slot.key, platform=Platform.TELEGRAM.value,
        chat_id=slot.event.source.chat_id, thread_id=slot.event.source.thread_id,
        content=extra.answer, adapter_profile=profile,
    )
    ledger.mark_failed(extra.obligation_id, "send_path_degraded")
    return slot, sibling, extra


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("phase", ["after_claim", "before_dispatch"])
@pytest.mark.parametrize("cancel_count", [1, 2])
@pytest.mark.parametrize("lifecycle", ["replacement", "unfenced_replacement", "shutdown"])
async def test_cancelled_adapter_refunds_the_whole_unsent_claim(
    delivery_system, monkeypatch, profile, phase, cancel_count, lifecycle
):
    system = delivery_system
    slot, sibling, extra = await _failed_batch(system, profile)
    entered = asyncio.Event()
    thread_release, async_release = threading.Event(), asyncio.Event()
    system.releases.extend((thread_release, async_release))
    claim_completed = threading.Event()
    if phase == "after_claim":
        loop = asyncio.get_running_loop()
        real_claim = ledger.sweep_failed_for_runtime
        started = threading.Event()
        system.workers.append((started, claim_completed))

        def hold_claim(*args, **kwargs):
            started.set()
            try:
                claimed = real_claim(*args, **kwargs)
                loop.call_soon_threadsafe(entered.set)
                assert thread_release.wait(10), "claim result barrier expired"
                return claimed
            finally:
                claim_completed.set()

        monkeypatch.setattr(ledger, "sweep_failed_for_runtime", hold_claim)
    else:
        async def hold_resume_clear(session_key):
            entered.set()
            await async_release.wait()

        system.runner._async_session_store.clear_resume_pending.side_effect = hold_resume_clear

    slot.adapter._record_polling_progress(slot.generation)
    await asyncio.wait_for(entered.wait(), 10)
    for item in (slot, extra):
        row = _row(system, item)
        assert (row["state"], row["attempts"]) == ("attempting", 1)
    recovery, = slot.adapter._background_tasks
    if lifecycle != "unfenced_replacement":
        _make_unavailable(slot, "runner_shutdown" if lifecycle == "shutdown" else "teardown")
    teardown = asyncio.create_task(slot.adapter.cancel_background_tasks())
    system.tasks.append(teardown)

    async def observe_cancel():
        while not recovery.cancelling():
            await asyncio.sleep(0)

    await asyncio.wait_for(observe_cancel(), 10)
    for _ in range(cancel_count - 1):
        recovery.cancel()
        await asyncio.sleep(0)
    thread_release.set()
    async_release.set()
    await asyncio.wait_for(teardown, 10)
    if phase == "after_claim":
        assert await asyncio.to_thread(claim_completed.wait, 10)
    assert recovery.cancelled()
    for item in (slot, extra):
        row = _row(system, item)
        assert (row["state"], row["attempts"], row["last_error"]) == ("failed", 0, "send_path_degraded")
    slot.adapter._bot.send_message.assert_not_awaited()
    _assert_failed(system, sibling)

    system.runner._draining = False
    system.runner._async_session_store.clear_resume_pending.side_effect = None
    replacement, generation = _install_replacement(system, slot, monkeypatch)
    replacement._record_polling_progress(generation)
    await _drain_background(SimpleNamespace(adapter=replacement))
    for item in (slot, extra):
        row = _row(system, item)
        assert (row["state"], row["attempts"], row["content"]) == ("delivered", 1, item.answer)
    assert replacement._bot.send_message.await_count == 2
    assert {call.kwargs["text"] for call in replacement._bot.send_message.await_args_list} == {
        ledger.RECONNECTED_MARKER + item.answer for item in (slot, extra)
    }
    for call in replacement._bot.send_message.await_args_list:
        assert call.kwargs["chat_id"] == int(slot.event.source.chat_id)
        assert call.kwargs["message_thread_id"] == int(slot.event.source.thread_id)
    slot.adapter._message_handler.assert_awaited_once_with(slot.event)
    _assert_failed(system, sibling)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("phase,recovery_owner", [
    (phase, "replacement") for phase in ("sending", "delivered_write", "refund_before_write", "refund_after_write")
] + [(phase, "same") for phase in ("refund_before_write", "refund_after_write")])
async def test_cancellation_settles_an_inflight_send_or_refund_once(
    delivery_system, monkeypatch, profile, phase, recovery_owner
):
    system = delivery_system
    slot, sibling, extra = await _failed_batch(system, profile)
    is_refund = phase.startswith("refund_")
    entered = asyncio.Event()
    thread_release, async_release = threading.Event(), asyncio.Event()
    system.releases.extend((thread_release, async_release))
    writes = []
    completed = threading.Event()
    if phase == "sending":
        async def hold_send(**kwargs):
            entered.set()
            await async_release.wait()
            return SimpleNamespace(message_id=1001)

        slot.adapter._bot.send_message.side_effect = hold_send
    else:
        name = "release_runtime_claim" if is_refund else "mark_delivered"
        real_write = getattr(ledger, name)
        loop = asyncio.get_running_loop()
        started = threading.Event()
        system.workers.append((started, completed))

        def hold_write(*args, **kwargs):
            writes.append(args[0])
            if len(writes) != 1:
                return real_write(*args, **kwargs)
            started.set()
            try:
                result = real_write(*args, **kwargs) if phase == "refund_after_write" else None
                loop.call_soon_threadsafe(entered.set)
                assert thread_release.wait(10), "settlement write barrier expired"
                return result if phase == "refund_after_write" else real_write(*args, **kwargs)
            finally:
                completed.set()

        monkeypatch.setattr(ledger, name, hold_write)

    resume_entered, resume_release = asyncio.Event(), asyncio.Event()
    system.releases.append(resume_release)
    if is_refund:
        async def hold_resume(session_key):
            resume_entered.set()
            await resume_release.wait()

        system.runner._async_session_store.clear_resume_pending.side_effect = hold_resume
    slot.adapter._record_polling_progress(slot.generation)
    if is_refund:
        await asyncio.wait_for(resume_entered.wait(), 10)
        _make_unavailable(slot, "degraded" if recovery_owner == "same" else "teardown")
        resume_release.set()
    await asyncio.wait_for(entered.wait(), 10)
    recovery, = slot.adapter._background_tasks
    if recovery_owner != "same":
        _make_unavailable(slot, "teardown")
    teardown = asyncio.create_task(slot.adapter.cancel_background_tasks())
    system.tasks.append(teardown)

    async def observe_cancel():
        while not recovery.cancelling():
            await asyncio.sleep(0)

    await asyncio.wait_for(observe_cancel(), 10)
    recovery.cancel()
    await asyncio.sleep(0)

    replacement = None
    successor_release = asyncio.Event()
    system.releases.append(successor_release)
    if is_refund and recovery_owner == "replacement":
        successor_entered = asyncio.Event()

        async def hold_successor(session_key):
            successor_entered.set()
            await successor_release.wait()

        system.runner._async_session_store.clear_resume_pending.side_effect = hold_successor
        replacement, generation = _install_replacement(system, slot, monkeypatch)
        replacement._record_polling_progress(generation)
        if phase == "refund_after_write":
            await asyncio.wait_for(successor_entered.wait(), 10)
            row = _row(system, SimpleNamespace(obligation_id=writes[0]))
            assert (row["state"], row["attempts"]) == ("attempting", 1)
        else:
            # The first replacement sweep completes before the old refund is claimable.
            await _drain_background(SimpleNamespace(adapter=replacement))
    elif recovery_owner == "same":
        # Isolate the late-write wakeup: health is restored without issuing a fresh signal yet.
        slot.adapter._send_path_degraded = False
    thread_release.set()
    async_release.set()
    await asyncio.wait_for(teardown, 10)
    assert recovery.cancelled()
    if phase != "sending":
        assert await asyncio.to_thread(completed.wait, 10)
    if is_refund:
        assert sorted(writes) == sorted(item.obligation_id for item in (slot, extra))
        if recovery_owner == "same":
            for item in (slot, extra):
                row = _row(system, item)
                assert (row["state"], row["attempts"]) == ("failed", 0)
            slot.adapter._bot.send_message.assert_not_awaited()
            replacement = slot.adapter
            generation, _ = replacement._begin_polling_generation()
            replacement._record_polling_progress(generation)
        elif phase == "refund_after_write":
            row = _row(system, SimpleNamespace(obligation_id=writes[0]))
            assert (row["state"], row["attempts"]) == ("attempting", 1)
        successor_release.set()
    else:
        states = sorted((_row(system, item)["state"], _row(system, item)["attempts"]) for item in (slot, extra))
        assert states == [("delivered", 1), ("failed", 0)]
        assert slot.adapter._bot.send_message.await_count == 1
        replacement, generation = _install_replacement(system, slot, monkeypatch)
        replacement._record_polling_progress(generation)
    await _drain_background(SimpleNamespace(adapter=replacement))
    for item in (slot, extra):
        row = _row(system, item)
        assert (row["state"], row["attempts"], row["content"]) == ("delivered", 1, item.answer)
    assert replacement is not None
    calls = replacement._bot.send_message.await_args_list
    if replacement is not slot.adapter:
        calls = slot.adapter._bot.send_message.await_args_list + calls
    assert len(calls) == 2
    assert {call.kwargs["text"] for call in calls} == {
        ledger.RECONNECTED_MARKER + item.answer for item in (slot, extra)
    }
    for call in calls:
        assert call.kwargs["chat_id"] == int(slot.event.source.chat_id)
        assert call.kwargs["message_thread_id"] == int(slot.event.source.thread_id)
    slot.adapter._message_handler.assert_awaited_once_with(slot.event)
    _assert_failed(system, sibling)

@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("state", ["degraded", "stopped", "fatal", "teardown", "runner_shutdown"])
@pytest.mark.parametrize("phase", ["before_progress", "after_schedule", "before_failure_write", "before_dispatch"])
async def test_unavailable_transport_keeps_reply_and_retry_budget(
    delivery_system, monkeypatch, profile, state, phase
):
    system = delivery_system
    slot = system.slots[profile]
    adapter = slot.adapter
    if phase == "before_failure_write":
        entered, released = _pause_ledger_write(system, slot, monkeypatch)
        task = _produce(system, slot)
        await asyncio.wait_for(entered.wait(), timeout=10)
        adapter._record_polling_progress(slot.generation)
        await _drain_background(slot)
        _make_unavailable(slot, state)
        released.set()
        await asyncio.wait_for(task, timeout=10)
    else:
        await asyncio.wait_for(_produce(system, slot), timeout=10)
        _assert_failed(system, slot)
        generation = slot.generation
        if phase == "before_progress":
            _make_unavailable(slot, state)
            adapter._record_polling_progress(generation)
        elif phase == "after_schedule":
            adapter._record_polling_progress(generation)
            _make_unavailable(slot, state)
        else:
            claim_entered, claim_release = asyncio.Event(), asyncio.Event()
            system.releases.append(claim_release)

            async def pause_before_dispatch(session_key):
                assert session_key == slot.key
                claim_entered.set()
                await claim_release.wait()

            system.runner._async_session_store.clear_resume_pending.side_effect = pause_before_dispatch
            adapter._record_polling_progress(generation)
            await asyncio.wait_for(claim_entered.wait(), timeout=10)
            assert _row(system, slot)["attempts"] == 1
            _make_unavailable(slot, state)
            claim_release.set()
    await _drain_background(slot)
    _assert_failed(system, slot)
    slot.adapter._message_handler.assert_awaited_once_with(slot.event)
    assert slot.key not in slot.adapter._active_sessions
