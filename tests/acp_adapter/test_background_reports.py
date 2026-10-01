"""Background processes and completion-queue events reach ACP clients."""

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from acp_adapter.background import NOTIFICATION_METHOD, PROCESS_METHOD, BackgroundNotifier, track_background_process


class _Client:
    """Records extension notifications on a real event loop thread."""

    def __init__(self):
        self.sent = []
        self.arrived = threading.Condition()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    async def ext_notification(self, method, params):
        with self.arrived:
            self.sent.append((method, params))
            self.arrived.notify_all()

    def wait_for(self, count):
        with self.arrived:
            assert self.arrived.wait_for(lambda: len(self.sent) >= count, timeout=5), self.sent
        return self.sent[:count]


@pytest.fixture
def client():
    client = _Client()
    yield client
    client.loop.call_soon_threadsafe(client.loop.stop)


@pytest.fixture
def registry(monkeypatch):
    import tools.process_registry as module

    registry = module.ProcessRegistry()
    registry._completions_restored = True  # no durable ledger replay in these tests
    monkeypatch.setattr(module, "process_registry", registry)
    return registry


class _Sessions:
    """The two SessionManager attributes the notifier reads."""

    def __init__(self, *session_ids):
        self._lock = threading.Lock()
        self._sessions = {
            sid: SimpleNamespace(session_id=sid, is_running=False, runtime_lock=threading.Lock())
            for sid in session_ids
        }


def _record_passes(notifier, monkeypatch):
    """Queue of each routing pass's ``(held, idle_owner)`` result, put once the pass is done."""
    passes = queue.Queue()
    deliver = notifier._deliver
    monkeypatch.setattr(notifier, "_deliver", lambda reg: (result := deliver(reg), passes.put(result))[0])
    return passes


def _completion(registry, session_key, *, proc_id="proc_aaaa1111", exit_code=1):
    registry.completion_queue.put({
        "type": "completion", "session_id": proc_id, "session_key": session_key,
        "command": "gh pr checks 94", "exit_code": exit_code, "output": "X  Test Server 1",
    })


@pytest.mark.platforms("posix")
def test_background_terminal_reports_running_then_exit(client, registry, tmp_path):
    proc = registry.spawn_local("exit 3", cwd=str(tmp_path), task_id="acp-1", session_key="acp-1")
    result = json.dumps({"output": "Background process started", "session_id": proc.id, "pid": proc.pid})

    track_background_process(client, "acp-1", client.loop, "tc-1", result)

    started, exited = client.wait_for(2)
    common = {"sessionId": "acp-1", "toolCallId": "tc-1", "processId": proc.id, "command": "exit 3"}
    assert started == (PROCESS_METHOD, {**common, "status": "running"})
    assert exited == (PROCESS_METHOD, {**common, "status": "exited", "exitCode": 3, "reason": "exited"})


def test_foreground_terminal_reports_nothing(client, registry):
    track_background_process(client, "acp-1", client.loop, "tc-1", json.dumps({"output": "hi", "exit_code": 0}))
    track_background_process(client, "acp-1", client.loop, "tc-2", json.dumps({"session_id": "proc_missing"}))

    assert client.sent == []


def test_idle_session_gets_the_cli_notification_text(client, registry):
    notifier = BackgroundNotifier(_Sessions("acp-1"))
    _completion(registry, "acp-1")

    notifier.turn_ended(client, client.loop)

    [(method, params)] = client.wait_for(1)
    assert method == NOTIFICATION_METHOD
    assert params["sessionId"] == "acp-1"
    assert params["kind"] == "completion"
    assert params["title"] == "Background Process Failed (exit 1): gh pr checks 94"
    assert params["text"].startswith("[IMPORTANT: Background process proc_aaaa1111 exited (exit code 1)")
    assert "X  Test Server 1" in params["text"]
    assert registry.completion_queue.empty()


def test_busy_session_is_notified_once_its_turn_ends(client, registry, monkeypatch):
    sessions = _Sessions("acp-1")
    sessions._sessions["acp-1"].is_running = True
    notifier = BackgroundNotifier(sessions)
    passes = _record_passes(notifier, monkeypatch)
    notifier.turn_ended(client, client.loop)
    _completion(registry, "acp-1")

    assert passes.get(timeout=5) == (True, False)  # held for the running turn
    assert client.sent == []

    sessions._sessions["acp-1"].is_running = False
    notifier.turn_ended(client, client.loop)

    [(_method, params)] = client.wait_for(1)
    assert params["sessionId"] == "acp-1"
    assert registry.completion_queue.empty()


def test_events_for_other_sessions_never_reach_this_one(client, registry, monkeypatch):
    notifier = BackgroundNotifier(_Sessions("acp-1", "acp-2"))
    passes = _record_passes(notifier, monkeypatch)
    _completion(registry, "gone-session", proc_id="proc_orphan01")
    _completion(registry, "acp-2", proc_id="proc_second02")

    notifier.turn_ended(client, client.loop)

    assert passes.get(timeout=5) == (False, False)  # the orphan is dropped, not requeued forever
    [(_method, params)] = client.sent
    assert params["sessionId"] == "acp-2"
    assert "proc_second02" in params["text"]


def test_completion_the_agent_already_read_is_not_resent(client, registry):
    notifier = BackgroundNotifier(_Sessions("acp-1"))
    registry._completion_consumed.add("proc_consumed1")
    _completion(registry, "acp-1", proc_id="proc_consumed1")
    _completion(registry, "acp-1", proc_id="proc_fresh0002")

    notifier.turn_ended(client, client.loop)

    [(_method, params)] = client.wait_for(1)
    assert "proc_fresh0002" in params["text"]
    assert "proc_consumed1" not in params["text"]
