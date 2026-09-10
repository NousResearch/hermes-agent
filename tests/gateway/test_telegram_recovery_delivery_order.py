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
    system = SimpleNamespace(runner=runner, slots=slots, home=home, tasks=tasks, releases=releases, workers=workers)
    try:
        yield system
    finally:
        for release in releases:
            release.set()
        pending = set(tasks)
        for slot in slots.values():
            pending.update(slot.adapter._background_tasks)
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
    }
    actions[state]()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("state", ["degraded", "stopped", "fatal", "teardown"])
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
