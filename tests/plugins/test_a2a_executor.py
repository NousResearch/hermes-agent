from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from plugins.platforms.a2a import server, setup, task_store
from plugins.platforms.a2a.executor import HermesA2AExecutor


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    setup.ensure_a2a_platform_config(public_url="https://agent.example.test/a2a")
    return home


class FakeAdapter:
    def __init__(self):
        self.events = []
        self.interrupts = []
        self.block = None

    async def dispatch_request(self, event):
        self.events.append(event)
        if self.block is not None:
            await self.block.wait()
        return f"reply:{event.text}"

    async def request_session_interrupt(self, source, **kwargs):
        self.interrupts.append((source, kwargs))
        if self.block is not None:
            self.block.set()
        return True


class CaptureQueue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def _context(*, task_id="task", context_id="context", current_task=None, text="hello"):
    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.types.a2a_pb2 import Message, Part, SendMessageRequest

    request = SendMessageRequest(
        message=Message(
            message_id="message",
            role=__import__("a2a.types.a2a_pb2", fromlist=["ROLE_USER"]).ROLE_USER,
            task_id=task_id,
            context_id=context_id,
            parts=[Part(text=text)],
        )
    )
    return RequestContext(
        ServerCallContext(user=server.AuthenticatedA2AUser("alice")),
        request=request,
        task_id=task_id,
        context_id=context_id,
        task=current_task,
    )


def _rpc(method, params, request_id="req"):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def _message(text="hello", *, context_id="ctx", metadata=None, parts=None, return_immediately=False):
    return {
        "message": {
            "messageId": "message-1",
            "role": "ROLE_USER",
            "contextId": context_id,
            "parts": parts if parts is not None else [{"text": text}],
            "metadata": metadata or {},
        },
        "configuration": {"returnImmediately": return_immediately},
        "metadata": metadata or {},
    }


def _headers(token):
    return {"Authorization": f"Bearer {token}", "A2A-Version": "1.0"}


def _build_app(adapter):
    from a2a.server.request_handlers import DefaultRequestHandler

    store = task_store.create_task_store()
    card = server.build_agent_card("https://agent.example.test/a2a")
    executor = HermesA2AExecutor(adapter, active_profile="default")
    handler = DefaultRequestHandler(agent_executor=executor, task_store=store, agent_card=card)
    app = server.create_a2a_app(
        handler,
        task_store_instance=store,
        agent_card=card,
    )
    return app, executor


def test_official_jsonrpc_send_returns_task_with_text_only_artifact(hermes_home):
    token = setup.add_principal("alice", profile="default")
    adapter = FakeAdapter()
    app, _executor = _build_app(adapter)

    with TestClient(app) as client:
        response = client.post("/a2a", json=_rpc("SendMessage", _message()), headers=_headers(token))

    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"] == [{"text": "reply:hello"}]
    assert task["artifacts"][0].get("metadata") is None


def test_same_context_continues_and_cross_principal_cannot_collide(hermes_home):
    alice = setup.add_principal("alice", profile="default")
    bob = setup.add_principal("bob", profile="default")
    adapter = FakeAdapter()
    app, _executor = _build_app(adapter)

    with TestClient(app) as client:
        client.post("/a2a", json=_rpc("SendMessage", _message("one"), "1"), headers=_headers(alice))
        client.post("/a2a", json=_rpc("SendMessage", _message("two"), "2"), headers=_headers(alice))
        client.post("/a2a", json=_rpc("SendMessage", _message("three"), "3"), headers=_headers(bob))

    assert adapter.events[0].source.chat_id == adapter.events[1].source.chat_id
    assert adapter.events[0].source.chat_id != adapter.events[2].source.chat_id
    assert adapter.events[0].source.profile == "default"


def test_forged_metadata_is_powerless_and_nontext_or_slash_fails(hermes_home):
    token = setup.add_principal("alice", profile="default")
    adapter = FakeAdapter()
    app, _executor = _build_app(adapter)

    with TestClient(app) as client:
        forged = client.post(
            "/a2a",
            json=_rpc("SendMessage", _message("safe", metadata={"profile": "admin", "toolsets": ["web"]})),
            headers=_headers(token),
        ).json()["result"]["task"]
        slash = client.post(
            "/a2a", json=_rpc("SendMessage", _message("/restart"), "slash"), headers=_headers(token)
        ).json()["result"]["task"]
        nontext = client.post(
            "/a2a",
            json=_rpc("SendMessage", _message(parts=[{"url": "https://example.test/file"}]), "file"),
            headers=_headers(token),
        ).json()["result"]["task"]

    assert forged["status"]["state"] == "TASK_STATE_COMPLETED"
    assert adapter.events[0].metadata == {}
    assert adapter.events[0].source.profile == "default"
    assert slash["status"]["state"] == "TASK_STATE_FAILED"
    assert nontext["status"]["state"] == "TASK_STATE_FAILED"


def test_get_list_cancel_owner_isolation_and_cancellation_race(hermes_home):
    alice = setup.add_principal("alice", profile="default")
    bob = setup.add_principal("bob", profile="default")
    adapter = FakeAdapter()
    app, executor = _build_app(adapter)

    with TestClient(app) as client:
        completed = client.post(
            "/a2a",
            json=_rpc("SendMessage", _message("owned"), "send-owned"),
            headers=_headers(alice),
        ).json()["result"]["task"]
        task_id = completed["id"]
        denied_get = client.post(
            "/a2a", json=_rpc("GetTask", {"id": task_id}, "get-bob"), headers=_headers(bob)
        ).json()
        bob_list = client.post(
            "/a2a", json=_rpc("ListTasks", {}, "list-bob"), headers=_headers(bob)
        ).json()["result"]
        denied_cancel = client.post(
            "/a2a",
            json=_rpc("CancelTask", {"id": task_id}, "cancel-bob"),
            headers=_headers(bob),
        ).json()

        adapter.block = asyncio.Event()
        running = client.post(
            "/a2a",
            json=_rpc(
                "SendMessage",
                _message("wait", context_id="cancel-context", return_immediately=True),
                "send-running",
            ),
            headers=_headers(alice),
        ).json()["result"]["task"]
        canceled = client.post(
            "/a2a",
            json=_rpc("CancelTask", {"id": running["id"]}, "cancel-running"),
            headers=_headers(alice),
        ).json()["result"]

    assert denied_get["error"]
    assert denied_cancel["error"]
    assert bob_list["tasks"] == []
    assert canceled["status"]["state"] == "TASK_STATE_CANCELED"
    assert len(adapter.interrupts) == 1
    assert adapter.interrupts[0][0].chat_id.startswith("a2a_")
    assert executor._runs == {}


@pytest.mark.asyncio
async def test_context_lock_registry_refcounts_without_leak():
    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default")
    first = await executor._context_locks.acquire("same")
    waiter = asyncio.create_task(executor._context_locks.acquire("same"))
    await asyncio.sleep(0)
    assert executor._context_locks.size == 1
    await executor._context_locks.release("same", first)
    second = await waiter
    await executor._context_locks.release("same", second)
    assert executor._context_locks.size == 0

    held = await executor._context_locks.acquire("cancel-waiter")
    canceled_waiter = asyncio.create_task(executor._context_locks.acquire("cancel-waiter"))
    await asyncio.sleep(0)
    canceled_waiter.cancel()
    await asyncio.gather(canceled_waiter, return_exceptions=True)
    await executor._context_locks.release("cancel-waiter", held)
    assert executor._context_locks.size == 0


@pytest.mark.asyncio
async def test_missing_cancel_does_not_accumulate_state_or_poison_future_execution():
    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default", cancel_wait_seconds=0.02)
    context = _context(task_id="pre-cancel")

    for _ in range(100):
        await executor.cancel(context, CaptureQueue())
    await asyncio.wait_for(executor.execute(context, CaptureQueue()), timeout=0.2)

    assert len(adapter.events) == 1
    assert executor._runs == {}


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_context_lock_returns_without_leak():
    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default", cancel_wait_seconds=0.02)
    context = _context(task_id="waiting")
    key = executor._source("alice", "waiting", "context").chat_id
    held = await executor._context_locks.acquire(key)
    run = asyncio.create_task(executor.execute(context, CaptureQueue()))
    for _ in range(50):
        if "waiting" in executor._runs:
            break
        await asyncio.sleep(0)

    await asyncio.wait_for(executor.cancel(context, CaptureQueue()), timeout=0.2)
    await asyncio.wait_for(run, timeout=0.2)
    await executor._context_locks.release(key, held)

    assert adapter.events == []
    assert executor._runs == {}
    assert executor._context_locks.size == 0


@pytest.mark.asyncio
async def test_producer_cancel_while_acquiring_context_lock_owns_waiters_and_recovers():
    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default", cancel_wait_seconds=0.02)
    context = _context(task_id="producer-wait")
    key = executor._source("alice", "producer-wait", "context").chat_id
    held = await executor._context_locks.acquire(key)
    run = asyncio.create_task(executor.execute(context, CaptureQueue()))
    for _ in range(50):
        if "producer-wait" in executor._runs:
            await asyncio.sleep(0)
            break
        await asyncio.sleep(0)

    run.cancel()
    await asyncio.wait_for(asyncio.gather(run, return_exceptions=True), timeout=0.2)
    await executor._context_locks.release(key, held)

    assert executor._runs == {}
    assert executor._context_locks.size == 0
    await asyncio.wait_for(
        executor.execute(_context(task_id="producer-retry"), CaptureQueue()),
        timeout=0.2,
    )
    assert executor._context_locks.size == 0


@pytest.mark.asyncio
async def test_producer_cancellation_performs_its_own_bounded_cleanup():
    class WedgedInterruptAdapter(FakeAdapter):
        async def request_session_interrupt(self, source, **kwargs):
            await asyncio.sleep(60)

    adapter = WedgedInterruptAdapter()
    adapter.block = asyncio.Event()
    executor = HermesA2AExecutor(adapter, active_profile="default", cancel_wait_seconds=0.01)
    run = asyncio.create_task(executor.execute(_context(task_id="timeout"), CaptureQueue()))
    for _ in range(50):
        if adapter.events:
            break
        await asyncio.sleep(0)
    run.cancel()

    await asyncio.wait_for(asyncio.gather(run, return_exceptions=True), timeout=0.2)

    assert executor._runs == {}
    assert executor._context_locks.size == 0


@pytest.mark.asyncio
async def test_input_required_continuation_preserves_existing_task_and_no_bare_submit():
    from a2a.types.a2a_pb2 import (
        TASK_STATE_INPUT_REQUIRED,
        Artifact,
        Part,
        Task,
        TaskStatus,
    )

    existing = Task(
        id="continue",
        context_id="context",
        status=TaskStatus(state=TASK_STATE_INPUT_REQUIRED),
        artifacts=[Artifact(artifact_id="old", parts=[Part(text="preserve")])],
    )
    queue = CaptureQueue()
    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default")

    await executor.execute(
        _context(task_id="continue", current_task=existing, text="more"), queue
    )

    assert not any(isinstance(event, Task) for event in queue.events)
    assert existing.artifacts[0].parts[0].text == "preserve"


@pytest.mark.asyncio
async def test_completion_cancel_barrier_emits_exactly_one_terminal_event():
    from a2a.types.a2a_pb2 import (
        TASK_STATE_CANCELED,
        TASK_STATE_COMPLETED,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )

    artifact_started = asyncio.Event()
    release_artifact = asyncio.Event()

    class BarrierQueue(CaptureQueue):
        async def enqueue_event(self, event):
            if isinstance(event, TaskArtifactUpdateEvent):
                artifact_started.set()
                await release_artifact.wait()
            await super().enqueue_event(event)

    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default")
    context = _context(task_id="terminal-race")
    queue = BarrierQueue()
    run = asyncio.create_task(executor.execute(context, queue))
    await artifact_started.wait()
    cancel = asyncio.create_task(executor.cancel(context, CaptureQueue()))
    await asyncio.sleep(0)
    release_artifact.set()

    results = await asyncio.gather(run, cancel, return_exceptions=True)
    terminals = [
        event.status.state
        for event in queue.events
        if isinstance(event, TaskStatusUpdateEvent)
        and event.status.state in {TASK_STATE_COMPLETED, TASK_STATE_CANCELED}
    ]
    assert results == [None, None]
    assert terminals == [TASK_STATE_COMPLETED]


@pytest.mark.asyncio
async def test_resistant_dispatch_cancel_is_bounded_owned_and_context_serialized():
    from a2a.types.a2a_pb2 import TASK_STATE_CANCELED, TaskStatusUpdateEvent

    release = asyncio.Event()
    entered = asyncio.Event()

    class ResistantAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0

        async def dispatch_request(self, event):
            self.events.append(event)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            entered.set()
            try:
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
                return f"reply:{event.text}"
            finally:
                self.active -= 1

        async def request_session_interrupt(self, source, **kwargs):
            self.interrupts.append((source, kwargs))
            return True

    adapter = ResistantAdapter()
    executor = HermesA2AExecutor(
        adapter, active_profile="default", cancel_wait_seconds=0.01
    )
    first_queue = CaptureQueue()
    first = asyncio.create_task(
        executor.execute(
            _context(task_id="resistant-first", context_id="shared"), first_queue
        )
    )
    await entered.wait()

    cancel_context = _context(task_id="resistant-first", context_id="shared")
    await asyncio.wait_for(
        asyncio.gather(
            executor.cancel(cancel_context, CaptureQueue()),
            executor.cancel(cancel_context, CaptureQueue()),
        ),
        timeout=0.15,
    )
    terminals = [
        event.status.state
        for event in first_queue.events
        if isinstance(event, TaskStatusUpdateEvent)
        and event.status.state == TASK_STATE_CANCELED
    ]
    assert terminals == [TASK_STATE_CANCELED]

    # Canceling the producer must also return promptly, but it must not drop
    # ownership of the resistant dispatch or release its context lock.
    first.cancel()
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=0.15)
    second = asyncio.create_task(
        executor.execute(
            _context(task_id="resistant-second", context_id="shared"), CaptureQueue()
        )
    )
    await asyncio.sleep(0.03)
    assert len(adapter.events) == 1
    assert adapter.max_active == 1
    assert "resistant-first" in executor._runs

    shutdown = asyncio.create_task(executor.shutdown())
    await asyncio.sleep(0.03)
    assert not shutdown.done()
    shutdown.cancel()
    await asyncio.wait_for(
        asyncio.gather(shutdown, return_exceptions=True), timeout=0.15
    )
    assert adapter.active == 1
    assert "resistant-first" in executor._runs

    # The adapter may retain/retry shutdown after its own bounded deadline.
    shutdown = asyncio.create_task(executor.shutdown())

    release.set()
    await asyncio.wait_for(shutdown, timeout=0.2)
    await asyncio.wait_for(second, timeout=0.2)
    assert executor._runs == {}
    assert executor._context_locks.size == 0

    # Once the resistant dispatch has genuinely exited, the context is usable.
    await asyncio.wait_for(
        executor.execute(
            _context(task_id="resistant-third", context_id="shared"), CaptureQueue()
        ),
        timeout=0.2,
    )
    assert adapter.max_active == 1


@pytest.mark.asyncio
async def test_observer_cancellation_cannot_orphan_resistant_child(monkeypatch):
    from plugins.platforms.a2a import executor as executor_module

    adapter = FakeAdapter()
    executor = HermesA2AExecutor(adapter, active_profile="default")
    child_started = asyncio.Event()
    child_resisted = asyncio.Event()
    release_child = asyncio.Event()
    observer_yielded = asyncio.Event()
    release_observer = asyncio.Event()
    original_sleep = asyncio.sleep

    async def resistant_child():
        child_started.set()
        try:
            await release_child.wait()
        except asyncio.CancelledError:
            child_resisted.set()
            await release_child.wait()

    async def controlled_sleep(delay):
        if asyncio.current_task().get_name() == "cancel-observer" and delay == 0:
            observer_yielded.set()
            await release_observer.wait()
            return
        await original_sleep(delay)

    monkeypatch.setattr(executor_module.asyncio, "sleep", controlled_sleep)
    child = asyncio.create_task(resistant_child(), name="resistant-child")
    await child_started.wait()
    observer = asyncio.create_task(
        executor._observe_without_waiting(child, cancel_pending=True),
        name="cancel-observer",
    )
    await observer_yielded.wait()
    await child_resisted.wait()

    observer.cancel()
    await asyncio.wait_for(
        asyncio.gather(observer, return_exceptions=True), timeout=0.1
    )
    assert child in executor._owned_cleanup_tasks
    assert not child.done()

    shutdown = asyncio.create_task(executor.shutdown())
    await original_sleep(0)
    assert not shutdown.done()
    shutdown.cancel()
    await asyncio.wait_for(
        asyncio.gather(shutdown, return_exceptions=True), timeout=0.1
    )
    assert child in executor._owned_cleanup_tasks

    retry_shutdown = asyncio.create_task(executor.shutdown())
    release_child.set()
    await asyncio.wait_for(retry_shutdown, timeout=0.1)
    assert child.done()
    assert child not in executor._owned_cleanup_tasks


@pytest.mark.asyncio
async def test_contended_runs_guard_cannot_orphan_new_dispatch():
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    class GuardRaceAdapter(FakeAdapter):
        async def dispatch_request(self, event):
            self.events.append(event)
            dispatch_started.set()
            while not release_dispatch.is_set():
                try:
                    await release_dispatch.wait()
                except asyncio.CancelledError:
                    continue
            return f"reply:{event.text}"

        async def request_session_interrupt(self, source, **kwargs):
            self.interrupts.append((source, kwargs))
            return True

    adapter = GuardRaceAdapter()
    executor = HermesA2AExecutor(
        adapter, active_profile="default", cancel_wait_seconds=0.01
    )
    context_id = "guard-shared"
    context_key = executor._source("alice", "held", context_id).chat_id
    held_context = await executor._context_locks.acquire(context_key)
    first = asyncio.create_task(
        executor.execute(
            _context(task_id="guard-first", context_id=context_id), CaptureQueue()
        )
    )
    for _ in range(50):
        if "guard-first" in executor._runs:
            break
        await asyncio.sleep(0)
    assert "guard-first" in executor._runs

    await executor._runs_guard.acquire()
    try:
        await executor._context_locks.release(context_key, held_context)
        await dispatch_started.wait()
        first.cancel()
        await asyncio.wait_for(
            asyncio.gather(first, return_exceptions=True), timeout=0.15
        )

        record = executor._runs["guard-first"]
        assert record.task is not None
        assert not record.task.done()
        assert record.context_lock is not None

        second = asyncio.create_task(
            executor.execute(
                _context(task_id="guard-second", context_id=context_id),
                CaptureQueue(),
            )
        )
    finally:
        executor._runs_guard.release()

    await asyncio.sleep(0.03)
    assert len(adapter.events) == 1
    assert "guard-first" in executor._runs

    release_dispatch.set()
    await asyncio.wait_for(second, timeout=0.2)
    for _ in range(50):
        if not executor._runs:
            break
        await asyncio.sleep(0)
    assert executor._runs == {}
    assert executor._context_locks.size == 0
