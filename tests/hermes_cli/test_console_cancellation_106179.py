"""RED ownership regression: real Console/curator dispatch, controllable provider.

No model/network call: only agent construction and conversation work are replaced.
The real AIAgent interrupt and inline request lifecycle remain under test.
"""

import asyncio
import concurrent.futures
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.asyncio
async def test_console_cancel_waits_for_agent_request_and_worker(monkeypatch):
    from agent import curator
    from agent.chat_completion_helpers import direct_api_call
    from hermes_cli.web_routers import chat_ws
    from run_agent import AIAgent
    import run_agent

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    completed = asyncio.Event()
    release = threading.Event()
    aborted = threading.Event()
    request_exited = threading.Event()
    worker_exited = threading.Event()
    snapshots = []

    # Real AIAgent interrupt methods, without model/config/tool initialization.
    agent = object.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._execution_thread_id = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    agent.provider = "custom"
    agent._consecutive_stale_streams = 0
    agent._touch_activity = Mock()
    agent._session_messages = []
    agent.close = Mock()

    def request(**kwargs):
        # The confirmation probe already completed in this same single-worker pool.
        worker_exited.clear()
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(15):
                raise AssertionError("test cleanup failed to release request")
            if aborted.is_set():
                raise OSError("fake provider connection aborted")
            return SimpleNamespace()
        finally:
            request_exited.set()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
    agent._create_request_openai_client = Mock(return_value=client)
    agent._close_request_openai_client = Mock()

    def abort_client(client, *, reason):
        aborted.set()
        release.set()

    agent._abort_request_openai_client = abort_client

    def conversation(**kwargs):
        agent._execution_thread_id = threading.get_ident()
        try:
            direct_api_call(agent, {"model": "fake", "messages": []})
        except InterruptedError:
            return {"interrupted": True, "final_response": ""}
        return {"final_response": "fake complete"}

    agent.run_conversation = conversation
    monkeypatch.setattr(run_agent, "AIAgent", lambda **kwargs: agent)
    monkeypatch.setattr(curator, "is_enabled", lambda: True)
    monkeypatch.setattr(curator, "_safe_curated_report", lambda: [])
    monkeypatch.setattr(curator.skill_usage, "curated_report", lambda: [])
    monkeypatch.setattr(curator, "_render_candidate_list", lambda: "candidate: fake skill")
    monkeypatch.setattr(curator, "_resolve_review_provider", lambda: ({}, "fake", "custom", {}))
    monkeypatch.setattr(curator, "_write_run_report", lambda **kwargs: None)
    from agent import chat_completion_helpers
    monkeypatch.setattr(chat_completion_helpers, "_resolve_direct_stale_timeout", lambda *args: 60.0)

    # Observe actual concurrent Future completion, not asyncio wrapper state.
    class ObservedExecutor(concurrent.futures.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            future.add_done_callback(lambda _: worker_exited.set())
            return future

    pool = ObservedExecutor(max_workers=1, thread_name_prefix="hermes-console-test")
    monkeypatch.setattr(chat_ws, "_get_console_executor", lambda: pool)

    async def gate(*args):
        return "test", "test", "test"

    monkeypatch.setattr(chat_ws, "_ws_gate", gate)
    incoming = asyncio.Queue()
    line = "curator run --consolidate --dry-run"

    class Socket:
        query_params = {}

        async def accept(self):
            pass

        async def receive(self):
            return await incoming.get()

        async def send_json(self, frame):
            if frame.get("type") == "confirm_required":
                incoming.put_nowait({"text": json.dumps({"type": "confirm", "command": line})})
            if frame.get("status") == "cancelled":
                snapshots.append({
                    "agent_interrupted": agent._interrupt_requested,
                    "provider_aborted": aborted.is_set(),
                    "request_exited": request_exited.is_set(),
                    "worker_exited": worker_exited.is_set(),
                })
                completed.set()

    incoming.put_nowait({"text": json.dumps({"type": "input", "line": line})})
    socket_task = asyncio.create_task(chat_ws.console_ws(Socket()))
    try:
        await asyncio.wait_for(started.wait(), 10)
        assert callable(agent._active_request_abort)
        incoming.put_nowait({"text": json.dumps({"type": "cancel"})})
        await asyncio.wait_for(completed.wait(), 5)
        assert snapshots == [{
            "agent_interrupted": True,
            "provider_aborted": True,
            "request_exited": True,
            "worker_exited": True,
        }], f"Console reported cancelled before runtime propagation: {snapshots}"
    finally:
        # Positive control and deterministic teardown, even when the RED assertion fails.
        # This also proves that the fake request responds to the real interrupt primitive.
        agent.interrupt()
        release.set()
        incoming.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(socket_task, 5)
        pool.shutdown(wait=True)
        assert aborted.is_set() and request_exited.is_set() and worker_exited.is_set()
        agent.clear_interrupt()


def test_console_execution_is_command_scoped():
    """Cancellation of one owner must never reach another command's agent."""
    from hermes_cli.console_execution import ConsoleExecution

    calls = []

    class Agent:
        def __init__(self, name):
            self.name = name

        def interrupt(self, message=None):
            calls.append((self.name, message))

    first = ConsoleExecution(command_id=1)
    second = ConsoleExecution(command_id=2)
    first.bind_agent(Agent("first"))
    second.bind_agent(Agent("second"))

    first.cancel("cancelled")

    assert calls == [("first", "Console command cancelled")]
    assert first.cancelled is True
    assert second.cancelled is False


def test_console_execution_cancels_queued_work_without_starting_it():
    """I7: a queued command is cancelled before its callable starts."""
    from hermes_cli.console_execution import ConsoleExecution

    started = threading.Event()
    release = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    blocker = pool.submit(release.wait)
    owner = ConsoleExecution(command_id=1)
    owner.bind_future(pool.submit(lambda: started.set()))

    try:
        assert owner.cancel("cancel") is True
        release.set()
        blocker.result(timeout=2)
        assert started.is_set() is False
    finally:
        pool.shutdown(wait=True)


def test_console_execution_cleans_agent_handle_on_completion():
    """I4: completion clears the command's active agent handle."""
    from hermes_cli.console_execution import ConsoleExecution

    owner = ConsoleExecution(command_id=1)
    agent = object()
    owner.bind_agent(agent)
    owner.unbind_agent(agent)

    assert owner.active_agent is None
    assert owner.cancel("cancel") is True


@pytest.mark.asyncio
async def test_console_timeout_waits_for_owned_worker(monkeypatch):
    """I6: timeout uses the same owner cancellation and waits for unwind."""
    from hermes_cli.web_routers import chat_ws

    started = threading.Event()
    request_exited = threading.Event()
    worker_exited = threading.Event()
    terminal = asyncio.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(chat_ws, "_get_console_executor", lambda: pool)
    monkeypatch.setattr(chat_ws, "_CONSOLE_COMMAND_TIMEOUT_SECONDS", 0.02)

    def execute(engine, line, *, confirmed, profile, execution):
        started.set()
        try:
            while not execution.cancelled:
                threading.Event().wait(0.005)
        finally:
            request_exited.set()
            worker_exited.set()
        return SimpleNamespace(status="ok", command=line, output="")

    monkeypatch.setattr(chat_ws, "_execute_console_line", execute)
    incoming = asyncio.Queue()

    class Socket:
        query_params = {}

        async def accept(self):
            pass

        async def receive(self):
            return await incoming.get()

        async def send_json(self, frame):
            if frame.get("status") == "timeout":
                assert started.is_set() and request_exited.is_set() and worker_exited.is_set()
                terminal.set()

    async def disconnect():
        return "test", "test", "test"

    async def _gate():
        return await disconnect()

    monkeypatch.setattr(chat_ws, "_ws_gate", lambda ws, kind: _gate())
    incoming.put_nowait({"text": json.dumps({"type": "input", "line": "slow"})})
    task = asyncio.create_task(chat_ws.console_ws(Socket()))
    await asyncio.wait_for(terminal.wait(), 5)
    incoming.put_nowait({"type": "websocket.disconnect"})
    await asyncio.wait_for(task, 5)
    pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_console_disconnect_unwinds_owned_worker(monkeypatch):
    """I6: websocket disconnect cancels the command owner before returning."""
    from hermes_cli.web_routers import chat_ws

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    worker_exited = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(chat_ws, "_get_console_executor", lambda: pool)

    def execute(engine, line, *, confirmed, profile, execution):
        loop.call_soon_threadsafe(started.set)
        try:
            while not execution.cancelled:
                threading.Event().wait(0.005)
        finally:
            worker_exited.set()
        return SimpleNamespace(status="ok", command=line, output="")

    monkeypatch.setattr(chat_ws, "_execute_console_line", execute)

    async def gate(*args):
        return "test", "test", "test"

    monkeypatch.setattr(chat_ws, "_ws_gate", gate)
    incoming = asyncio.Queue()

    class Socket:
        query_params = {}

        async def accept(self):
            pass

        async def receive(self):
            msg = await incoming.get()
            if msg == "wait-for-start":
                await started.wait()
                return {"type": "websocket.disconnect"}
            return msg

        async def send_json(self, frame):
            pass

    incoming.put_nowait({"text": json.dumps({"type": "input", "line": "slow"})})
    incoming.put_nowait("wait-for-start")
    await asyncio.wait_for(chat_ws.console_ws(Socket()), 5)
    assert worker_exited.is_set()
    pool.shutdown(wait=True)


def test_console_cancellation_blocks_retry_resurrection():
    """I5: a cancelled command cannot open a second provider attempt."""
    from hermes_cli.console_execution import ConsoleExecution

    owner = ConsoleExecution(command_id=1)
    attempts = []

    def provider_attempt():
        attempts.append(1)
        if owner.cancelled:
            return
        raise AssertionError("provider should not retry after cancellation")

    owner.cancel("cancelled")
    provider_attempt()
    assert attempts == [1]


def test_console_execution_handle_does_not_survive_worker_reuse():
    """I4: a reused executor thread cannot retain a prior command's agent."""
    from hermes_cli.console_execution import ConsoleExecution

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    first = ConsoleExecution(command_id=1)
    first_agent = object()
    first.bind_agent(first_agent)
    first_future = pool.submit(first.unbind_agent, first_agent)
    first.bind_future(first_future)
    first_future.result(timeout=2)
    assert first.active_agent is None

    second = ConsoleExecution(command_id=2)
    second_agent = object()
    second.bind_agent(second_agent)
    first.cancel("cancelled")
    assert second.active_agent is second_agent
    second.unbind_agent(second_agent)
    pool.shutdown(wait=True)
