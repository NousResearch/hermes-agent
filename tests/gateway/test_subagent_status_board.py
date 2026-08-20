import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.subagent_status import (
    SubagentStatusBoard,
    SubagentStatusEvent,
    SubagentStatusOwner,
    SubagentStatusPhase,
    SubagentStatusRegistry,
    SubagentStatusRow,
    SubagentStatusSnapshot,
    TelegramSubagentStatusPublisher,
    create_telegram_subagent_status,
    render_subagent_status,
)
from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _ManualCadence:
    def __init__(self):
        self.ticks = asyncio.Queue()
        self.waiting = asyncio.Event()

    async def wait(self, wake, delay):
        self.waiting.set()
        wake_task = asyncio.create_task(wake.wait())
        tick_task = asyncio.create_task(self.ticks.get())
        done, pending = await asyncio.wait(
            {wake_task, tick_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        return wake_task in done

    async def tick(self):
        self.waiting.clear()
        self.ticks.put_nowait(None)
        await asyncio.wait_for(self.waiting.wait(), timeout=1)


def test_render_subagent_status_counts_all_terminal_rows_and_freezes_elapsed():
    snapshot = SubagentStatusSnapshot(
        rows=(
            SubagentStatusRow(
                ordinal=1,
                phase=SubagentStatusPhase.WAITING,
                started_at=219.0,
            ),
            SubagentStatusRow(
                ordinal=2,
                phase=SubagentStatusPhase.FAILED,
                started_at=104.0,
                finished_at=223.0,
            ),
        )
    )

    assert render_subagent_status(snapshot, now=224.0) == (
        "🔀 Subagents · 1/2 done\n"
        "├ #1 🚀 waiting · 5s\n"
        "└ #2 ❌ failed · 1m59s"
    )


def test_board_stays_open_between_sequential_batches_until_turn_seals():
    board = SubagentStatusBoard()

    first = board.admit_batch(task_count=2, at=100.0)
    board.finalize_batch(
        first,
        outcomes=(SubagentStatusPhase.DONE, SubagentStatusPhase.FAILED),
        at=104.0,
    )

    assert board.is_terminal is False

    second = board.admit_batch(task_count=1, at=105.0)
    assert tuple(row.ordinal for row in board.snapshot().rows) == (1, 2, 3)

    board.finalize_batch(
        second,
        outcomes=(SubagentStatusPhase.DONE,),
        at=109.0,
    )
    assert board.is_terminal is False

    board.seal()
    assert board.is_terminal is True


def test_board_accepts_only_direct_child_events_and_terminal_rows_are_immutable():
    board = SubagentStatusBoard()
    batch = board.admit_batch(task_count=1, at=100.0)

    board.observe(
        batch,
        SubagentStatusEvent(
            task_index=0,
            depth=2,
            phase=SubagentStatusPhase.THINKING,
            at=101.0,
        ),
    )
    assert board.snapshot().rows[0].phase is SubagentStatusPhase.WAITING

    board.observe(
        batch,
        SubagentStatusEvent(
            task_index=0,
            depth=1,
            phase=SubagentStatusPhase.THINKING,
            at=102.0,
        ),
    )
    board.observe(
        batch,
        SubagentStatusEvent(
            task_index=0,
            depth=1,
            phase=SubagentStatusPhase.DONE,
            at=103.0,
        ),
    )
    board.observe(
        batch,
        SubagentStatusEvent(
            task_index=0,
            depth=1,
            phase=SubagentStatusPhase.THINKING,
            at=104.0,
        ),
    )

    assert board.snapshot().rows[0] == SubagentStatusRow(
        ordinal=1,
        phase=SubagentStatusPhase.DONE,
        started_at=100.0,
        finished_at=103.0,
    )


def test_batch_reconciliation_closes_missing_events_without_rewriting_terminal_rows():
    board = SubagentStatusBoard()
    batch = board.admit_batch(task_count=3, at=100.0)
    board.observe(
        batch,
        SubagentStatusEvent(
            task_index=0,
            depth=1,
            phase=SubagentStatusPhase.DONE,
            at=102.0,
        ),
    )
    board.observe(
        batch,
        SubagentStatusEvent(
            task_index=1,
            depth=1,
            phase=SubagentStatusPhase.THINKING,
            at=103.0,
        ),
    )

    board.finalize_batch(
        batch,
        outcomes=(
            SubagentStatusPhase.DONE,
            SubagentStatusPhase.DONE,
            SubagentStatusPhase.FAILED,
        ),
        at=110.0,
    )

    assert board.snapshot().rows == (
        SubagentStatusRow(1, SubagentStatusPhase.DONE, 100.0, 102.0),
        SubagentStatusRow(2, SubagentStatusPhase.DONE, 100.0, 110.0),
        SubagentStatusRow(3, SubagentStatusPhase.FAILED, 100.0, 110.0),
    )


@pytest.mark.asyncio
async def test_owner_applies_worker_updates_only_on_gateway_loop():
    loop_thread = threading.get_ident()
    callback_threads = []
    terminal_seen = asyncio.Event()

    def on_change(snapshot, is_terminal):
        callback_threads.append(threading.get_ident())
        if is_terminal:
            terminal_seen.set()

    owner = SubagentStatusOwner(
        loop=asyncio.get_running_loop(), on_change=on_change, clock=lambda: 20.0
    )
    sink = await asyncio.to_thread(owner.admit_batch, 1)

    await asyncio.to_thread(
        sink.observe,
        SubagentStatusEvent(0, 1, SubagentStatusPhase.THINKING, 21.0),
    )
    await asyncio.to_thread(sink.finalize, (SubagentStatusPhase.DONE,))
    owner.seal()

    await asyncio.wait_for(terminal_seen.wait(), timeout=1)
    assert callback_threads
    assert set(callback_threads) == {loop_thread}
    assert owner.snapshot().rows == (
        SubagentStatusRow(1, SubagentStatusPhase.DONE, 20.0, 20.0),
    )


@pytest.mark.asyncio
async def test_telegram_publisher_sends_once_then_coalesces_edits_and_flushes_terminal():
    calls = []
    now = [0.0]
    cadence = _ManualCadence()

    async def send(text):
        calls.append(("send", None, text))
        return SendResult(success=True, message_id="message-1")

    async def edit(message_id, text):
        calls.append(("edit", message_id, text))
        return SendResult(success=True, message_id=message_id)

    publisher = TelegramSubagentStatusPublisher(
        send=send,
        edit=edit,
        clock=lambda: now[0],
        wait=cadence.wait,
    )
    waiting = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.WAITING, 0.0),)
    )
    publisher.notify(waiting, is_terminal=False)
    await asyncio.wait_for(cadence.waiting.wait(), timeout=1)

    thinking = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.THINKING, 0.0),)
    )
    publisher.notify(thinking, is_terminal=False)
    now[0] = 5.0
    await cadence.tick()

    done = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.DONE, 0.0, 6.0),)
    )
    publisher.notify(done, is_terminal=True)
    await asyncio.wait_for(publisher.wait_closed(), timeout=1)

    assert calls == [
        ("send", None, "🔀 Subagents · 0/1 done\n└ #1 🚀 waiting · 0s"),
        ("edit", "message-1", "🔀 Subagents · 0/1 done\n└ #1 💭 thinking · 5s"),
        ("edit", "message-1", "🔀 Subagents · 1/1 done\n└ #1 ✅ done · 6s"),
    ]


@pytest.mark.asyncio
async def test_terminal_retryable_edit_retries_same_message_at_next_cadence():
    cadence = _ManualCadence()
    edit_attempts = 0

    async def send(text):
        return SendResult(success=True, message_id="message-1")

    async def edit(message_id, text):
        nonlocal edit_attempts
        edit_attempts += 1
        if edit_attempts == 1:
            return SendResult(success=False, retryable=True)
        return SendResult(success=True, message_id=message_id)

    publisher = TelegramSubagentStatusPublisher(
        send=send,
        edit=edit,
        clock=lambda: 2.0,
        wait=cadence.wait,
    )
    waiting = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.WAITING, 0.0),)
    )
    publisher.notify(waiting, is_terminal=False)
    await asyncio.wait_for(cadence.waiting.wait(), timeout=1)
    cadence.waiting.clear()

    done = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.DONE, 0.0, 2.0),)
    )
    publisher.notify(done, is_terminal=True)
    await asyncio.wait_for(cadence.waiting.wait(), timeout=1)

    cadence.ticks.put_nowait(None)
    await asyncio.wait_for(publisher.wait_closed(), timeout=1)
    assert edit_attempts == 2


@pytest.mark.asyncio
async def test_terminal_retryable_edit_stops_after_two_attempts():
    cadence = _ManualCadence()
    edit_attempts = 0

    async def send(text):
        return SendResult(success=True, message_id="message-1")

    async def edit(message_id, text):
        nonlocal edit_attempts
        edit_attempts += 1
        return SendResult(success=False, retryable=True, error="temporary")

    publisher = TelegramSubagentStatusPublisher(
        send=send,
        edit=edit,
        clock=lambda: 5.0,
        interval=5.0,
        wait=cadence.wait,
    )
    waiting = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.WAITING, 0.0),)
    )
    terminal = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.DONE, 0.0, 5.0),)
    )

    publisher.notify(waiting, is_terminal=False)
    await asyncio.wait_for(cadence.waiting.wait(), timeout=1)
    cadence.waiting.clear()
    publisher.notify(terminal, is_terminal=True)
    await asyncio.wait_for(cadence.waiting.wait(), timeout=1)
    cadence.waiting.clear()
    cadence.ticks.put_nowait(None)

    await asyncio.wait_for(publisher.wait_closed(), timeout=1)
    assert edit_attempts == 2


@pytest.mark.asyncio
async def test_retryable_edit_with_server_delay_is_not_retried():
    cadence = _ManualCadence()
    edit_attempts = 0

    async def send(text):
        return SendResult(success=True, message_id="message-1")

    async def edit(message_id, text):
        nonlocal edit_attempts
        edit_attempts += 1
        return SendResult(
            success=False,
            retryable=True,
            retry_after=3.0,
            error="flood wait",
        )

    publisher = TelegramSubagentStatusPublisher(
        send=send,
        edit=edit,
        clock=lambda: 5.0,
        interval=5.0,
        wait=cadence.wait,
    )
    waiting = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.WAITING, 0.0),)
    )
    terminal = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.DONE, 0.0, 5.0),)
    )

    publisher.notify(waiting, is_terminal=False)
    await asyncio.wait_for(cadence.waiting.wait(), timeout=1)
    publisher.notify(terminal, is_terminal=True)

    await asyncio.wait_for(publisher.wait_closed(), timeout=1)
    assert edit_attempts == 1


@pytest.mark.asyncio
async def test_publisher_shutdown_waits_for_entered_send_and_prohibits_future_calls():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def send(text):
        calls.append("send")
        entered.set()
        await release.wait()
        return SendResult(success=True, message_id="message-1")

    async def edit(message_id, text):
        calls.append("edit")
        return SendResult(success=True, message_id=message_id)

    publisher = TelegramSubagentStatusPublisher(send=send, edit=edit)
    snapshot = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.WAITING, 0.0),)
    )
    publisher.notify(snapshot, is_terminal=False)
    await asyncio.wait_for(entered.wait(), timeout=1)

    shutdown_task = asyncio.create_task(publisher.shutdown())
    release.set()
    await asyncio.wait_for(shutdown_task, timeout=1)

    publisher.notify(snapshot, is_terminal=True)
    assert calls == ["send"]


@pytest.mark.asyncio
async def test_factory_pins_route_metadata_for_initial_send():
    calls = []

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            calls.append((chat_id, content, metadata))
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            raise AssertionError("terminal first publication must not edit")

    route_metadata = {"message_thread_id": 17}
    pair = create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=route_metadata,
        loop=asyncio.get_running_loop(),
        clock=lambda: 4.0,
    )
    assert pair is not None
    _owner, publisher = pair
    route_metadata["message_thread_id"] = 99

    snapshot = SubagentStatusSnapshot(
        (SubagentStatusRow(1, SubagentStatusPhase.DONE, 0.0, 4.0),)
    )
    publisher.notify(snapshot, is_terminal=True)
    await asyncio.wait_for(publisher.wait_closed(), timeout=1)

    assert calls == [
        (
            "chat-1",
            "🔀 Subagents · 1/1 done\n└ #1 ✅ done · 4s",
            {"message_thread_id": 17, "_interim_send": True},
        )
    ]


@pytest.mark.asyncio
async def test_factory_uses_relay_per_chat_edit_capability():
    class Adapter:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def _descriptor_for_chat(self, chat_id):
            return self.descriptor

        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            return SendResult(success=True, message_id=message_id)

    def descriptor(platform, supports_edit, supports_op):
        return SimpleNamespace(
            platform=platform,
            supports_edit=supports_edit,
            supports_op=lambda op: supports_op,
        )

    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1")
    loop = asyncio.get_running_loop()

    for rejected in (
        descriptor("discord", True, True),
        descriptor("telegram", False, True),
        descriptor("telegram", True, False),
    ):
        assert create_telegram_subagent_status(
            source=source,
            adapter=Adapter(rejected),
            route_metadata=None,
            loop=loop,
        ) is None

    assert create_telegram_subagent_status(
        source=source,
        adapter=Adapter(descriptor("telegram", True, True)),
        route_metadata=None,
        loop=loop,
    ) is not None


@pytest.mark.asyncio
async def test_factory_rejects_native_adapter_without_edit_override():
    class Adapter:
        edit_message = BasePlatformAdapter.edit_message

        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="message-1")

    assert create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=None,
        loop=asyncio.get_running_loop(),
    ) is None


@pytest.mark.asyncio
async def test_registry_tracks_only_accepted_owner_once():
    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            return SendResult(success=True, message_id=message_id)

    registry = SubagentStatusRegistry()
    pair = create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=None,
        loop=asyncio.get_running_loop(),
        registry=registry,
    )
    assert pair is not None
    owner, publisher = pair
    assert registry.active_count == 0

    await asyncio.to_thread(owner.admit_batch, 1)
    await asyncio.to_thread(owner.admit_batch, 1)
    assert registry.active_count == 1

    await publisher.shutdown()
    await asyncio.wait_for(registry.wait_empty(), timeout=1)
    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_registry_retains_cleanup_waiter_until_publisher_closes():
    async def send(text):
        return SendResult(success=True, message_id="message-1")

    async def edit(message_id, text):
        return SendResult(success=True, message_id=message_id)

    loop = asyncio.get_running_loop()
    registry = SubagentStatusRegistry()
    publisher = TelegramSubagentStatusPublisher(send=send, edit=edit)
    owner = SubagentStatusOwner(
        loop=loop,
        on_change=publisher.notify,
    )

    assert registry.register(owner, publisher) is True
    assert registry.cleanup_task_count == 1

    await publisher.shutdown()
    await registry.wait_empty()
    await asyncio.sleep(0)
    assert registry.cleanup_task_count == 0


@pytest.mark.asyncio
async def test_closed_registry_refuses_first_admission_without_retention():
    calls = []

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            calls.append("send")
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            return SendResult(success=True, message_id=message_id)

    registry = SubagentStatusRegistry()
    pair = create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=None,
        loop=asyncio.get_running_loop(),
        registry=registry,
    )
    assert pair is not None
    owner, _publisher = pair

    registry.close()
    with pytest.raises(RuntimeError, match="registry is closed"):
        await asyncio.to_thread(owner.admit_batch, 1)

    assert registry.active_count == 0
    assert calls == []


@pytest.mark.asyncio
async def test_closed_registry_refuses_later_batch_on_active_owner():
    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            return SendResult(success=True, message_id=message_id)

    registry = SubagentStatusRegistry()
    pair = create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=None,
        loop=asyncio.get_running_loop(),
        registry=registry,
    )
    assert pair is not None
    owner, publisher = pair

    await asyncio.to_thread(owner.admit_batch, 1)
    registry.close()
    with pytest.raises(RuntimeError, match="registry is closed"):
        await asyncio.to_thread(owner.admit_batch, 1)

    await publisher.shutdown()


@pytest.mark.asyncio
async def test_registry_shutdown_allows_completion_during_grace():
    calls = []
    sent = asyncio.Event()

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            calls.append(("send", content))
            sent.set()
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            calls.append(("edit", content))
            return SendResult(success=True, message_id=message_id)

    registry = SubagentStatusRegistry()
    pair = create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=None,
        loop=asyncio.get_running_loop(),
        clock=lambda: 9.0,
        registry=registry,
    )
    assert pair is not None
    owner, _publisher = pair

    sink = await asyncio.to_thread(owner.admit_batch, 1)
    await asyncio.wait_for(sent.wait(), timeout=1)
    shutdown = asyncio.create_task(registry.shutdown(timeout=1))
    await asyncio.sleep(0)
    await asyncio.to_thread(sink.finalize, (SubagentStatusPhase.DONE,))
    await shutdown

    assert registry.active_count == 0
    assert calls == [
        ("send", "🔀 Subagents · 0/1 done\n└ #1 🚀 waiting · 0s"),
        ("edit", "🔀 Subagents · 1/1 done\n└ #1 ✅ done · 0s"),
    ]


@pytest.mark.asyncio
async def test_registry_shutdown_cancels_publisher_after_bound():
    entered = asyncio.Event()
    never = asyncio.Event()

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            entered.set()
            await never.wait()
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            raise AssertionError("send never produced an editable message")

    registry = SubagentStatusRegistry()
    pair = create_telegram_subagent_status(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1"),
        adapter=Adapter(),
        route_metadata=None,
        loop=asyncio.get_running_loop(),
        registry=registry,
    )
    assert pair is not None
    owner, _publisher = pair

    await asyncio.to_thread(owner.admit_batch, 1)
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(registry.shutdown(timeout=0.01), timeout=1)

    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_sealing_unadmitted_owner_stays_inert():
    changes = []
    owner = SubagentStatusOwner(
        loop=asyncio.get_running_loop(),
        on_change=lambda snapshot, terminal: changes.append((snapshot, terminal)),
    )

    owner.request_seal()
    await asyncio.sleep(0)

    assert changes == []


def test_timed_out_admission_cannot_execute_later():
    loop = asyncio.new_event_loop()
    changes = []
    owner = SubagentStatusOwner(
        loop=loop,
        on_change=lambda snapshot, terminal: changes.append((snapshot, terminal)),
        admission_timeout=0.01,
    )
    try:
        with pytest.raises(TimeoutError):
            owner.admit_batch(1)
        loop.run_until_complete(asyncio.sleep(0))
        assert changes == []
    finally:
        loop.close()


def test_gateway_runner_prepares_inert_owner_only_when_surface_enabled():
    from gateway.run import GatewayRunner

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="message-1")

        async def edit_message(
            self, chat_id, message_id, content, *, finalize=False
        ):
            return SendResult(success=True, message_id=message_id)

    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-1")
    loop = asyncio.new_event_loop()
    try:
        assert runner._prepare_subagent_status_owner(
            source=source,
            adapter=Adapter(),
            route_metadata=None,
            surface_mode="off",
            loop=loop,
        ) is None
        assert not hasattr(runner, "_subagent_status_registry")

        owner = runner._prepare_subagent_status_owner(
            source=source,
            adapter=Adapter(),
            route_metadata=None,
            surface_mode="generic",
            loop=loop,
        )
        assert owner is not None
        assert runner._subagent_status_registry.active_count == 0
    finally:
        loop.close()
