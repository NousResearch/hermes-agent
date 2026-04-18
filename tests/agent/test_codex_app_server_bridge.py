import json
import queue
import time

import pytest

from agent.codex_app_server_bridge import (
    COMPLETED,
    DIFF_UPDATED,
    FAILED,
    RUNNING,
    START_THREAD_METHOD,
    START_TURN_METHOD,
    CodexAppServerTimeoutError,
    CodexAppServerBridge,
)
from hermes_cli import __version__ as HERMES_VERSION
from tools.process_registry import process_registry


class FakeStdout:
    def __init__(self):
        self._lines = queue.Queue()
        self.closed = False

    def push_json(self, message):
        self._lines.put(json.dumps(message) + "\n")

    def readline(self):
        while not self.closed:
            try:
                line = self._lines.get(timeout=0.01)
            except queue.Empty:
                continue
            if line is None:
                return ""
            return line
        return ""

    def close(self):
        self.closed = True
        self._lines.put(None)


class FakeStdin:
    def __init__(self, stdout, *, auto_respond=True):
        self.stdout = stdout
        self.auto_respond = auto_respond
        self.writes = []
        self.closed = False

    def write(self, line):
        self.writes.append(line)
        if not self.auto_respond:
            return
        message = json.loads(line)
        method = message.get("method")
        if method is None:
            return
        if method == "initialize":
            result = {"server": "fake-codex"}
        elif method == START_THREAD_METHOD:
            result = {"thread": {"id": "thread-123"}}
        elif method == START_TURN_METHOD:
            result = {"turn": {"id": "turn-456"}}
        else:
            result = {"ok": True}
        self.stdout.push_json({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, *, auto_respond=True):
        self.stdout = FakeStdout()
        self.stdin = FakeStdin(self.stdout, auto_respond=auto_respond)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.stdout.close()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.close()

    def wait(self, timeout=None):
        deadline = time.time() + (timeout or 0)
        while self.returncode is None and (timeout is None or time.time() < deadline):
            time.sleep(0.001)
        if self.returncode is None:
            raise TimeoutError()
        return self.returncode


def _drain_completion_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


@pytest.fixture(autouse=True)
def clean_completion_queue():
    _drain_completion_queue()
    yield
    _drain_completion_queue()


def test_response_correlation_works_for_matching_request_id():
    bridge = CodexAppServerBridge()
    request_id = bridge._next_request_id()
    pending = bridge._register_pending_request(request_id)

    result = bridge._handle_incoming_line(
        json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}})
    )

    assert result["matched"] is True
    assert result["id"] == request_id
    assert pending["event"].is_set()
    assert pending["result"] == {"ok": True}
    assert pending["error"] is None


def test_start_bootstraps_subprocess_and_waits_for_initialize():
    created = []

    def popen_factory(args, **kwargs):
        process = FakeProcess()
        created.append((args, kwargs, process))
        return process

    bridge = CodexAppServerBridge(popen_factory=popen_factory)

    status = bridge.start(timeout=1)

    assert status["bridge_status"] == "ready"
    assert status["initialize_result"] == {"server": "fake-codex"}
    assert created[0][0] == ["codex", "app-server", "--listen", "stdio://"]
    assert created[0][1]["text"] is True
    initialize_request = json.loads(created[0][2].stdin.writes[0])
    assert initialize_request["method"] == "initialize"
    assert initialize_request["params"] == {
        "clientInfo": {
            "name": "hermes-agent",
            "version": HERMES_VERSION,
        }
    }

    stopped = bridge.stop()
    assert stopped["bridge_status"] == "stopped"
    assert created[0][2].terminated is True


def test_start_marks_error_when_initialize_times_out():
    created = []

    def popen_factory(args, **kwargs):
        process = FakeProcess(auto_respond=False)
        created.append(process)
        return process

    bridge = CodexAppServerBridge(popen_factory=popen_factory)

    try:
        bridge.start(timeout=0.02)
    except CodexAppServerTimeoutError as exc:
        assert "initialize" in str(exc)
    else:
        raise AssertionError("expected initialize timeout")

    status = bridge.get_status()
    assert status["bridge_status"] == "error"
    assert status["normalized_status"] == FAILED
    assert "initialize failed" in status["last_error"]
    assert created[0].terminated is True


def test_start_turn_sends_schema_thread_and_turn_requests():
    created = []

    def popen_factory(args, **kwargs):
        process = FakeProcess()
        created.append(process)
        return process

    bridge = CodexAppServerBridge(popen_factory=popen_factory)
    bridge.start(timeout=1)

    result = bridge.start_turn(
        repo_path="/tmp/example-repo",
        prompt="make the change",
        timeout=1,
    )

    assert result["thread_id"] == "thread-123"
    assert result["turn_id"] == "turn-456"
    assert result["methods"] == {
        "start_thread": "thread/start",
        "start_turn": "turn/start",
    }

    writes = [json.loads(line) for line in created[0].stdin.writes]
    assert writes[1]["method"] == "thread/start"
    assert writes[1]["params"] == {"cwd": "/tmp/example-repo"}
    assert writes[2]["method"] == "turn/start"
    assert writes[2]["params"] == {
        "threadId": "thread-123",
        "input": [
            {
                "type": "text",
                "text": "make the change",
            }
        ],
    }
    assert bridge.get_status()["normalized_status"] == RUNNING
    assert bridge.get_status()["bridge_status"] == "ready"
    bridge.stop()


def test_server_request_fails_fast_and_replies_with_error():
    created = []

    def popen_factory(args, **kwargs):
        process = FakeProcess()
        created.append(process)
        return process

    bridge = CodexAppServerBridge(popen_factory=popen_factory)
    bridge.start(timeout=1)
    bridge.state.turn_id = "turn-req-1"
    pending = bridge._register_pending_request(1234)

    event = bridge._handle_incoming_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "approval/request",
                "params": {"reason": "need approval"},
            }
        )
    )

    assert event.normalized_status == FAILED
    assert bridge.get_status()["bridge_status"] == "error"
    assert "Unsupported Codex app-server request: approval/request" in bridge.get_status()["last_error"]
    assert pending["event"].is_set()
    assert pending["error"] == {
        "message": (
            "Unsupported Codex app-server request: approval/request. "
            "Hermes bridge does not handle approval/input requests yet."
        )
    }

    queued = process_registry.completion_queue.get_nowait()
    assert queued["session_id"] == "codex_turn_turn-req-1"
    assert queued["exit_code"] == 1

    writes = [json.loads(line) for line in created[0].stdin.writes]
    assert writes[-1] == {
        "jsonrpc": "2.0",
        "id": 99,
        "error": {
            "code": -32000,
            "message": (
                "Unsupported Codex app-server request: approval/request. "
                "Hermes bridge does not handle approval/input requests yet."
            ),
        },
    }

    bridge.stop()


def test_notification_normalization_updates_state_and_stores_recent_events():
    bridge = CodexAppServerBridge(clock=lambda: 123.0)

    event = bridge._handle_incoming_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "codex/turn/delta",
                "params": {"thread_id": "thread-1", "turn_id": "turn-1", "text": "hi"},
            }
        )
    )

    assert event.raw_method == "codex/turn/delta"
    assert event.normalized_status == RUNNING
    assert bridge.get_status()["normalized_status"] == RUNNING
    assert bridge.get_status()["thread_id"] == "thread-1"
    assert bridge.get_status()["turn_id"] == "turn-1"

    diff_event = bridge._handle_incoming_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "codex/diff_updated",
                "params": {"files": ["agent/example.py"]},
            }
        )
    )

    recent_events = bridge.get_recent_events()
    assert diff_event.normalized_status == DIFF_UPDATED
    assert bridge.get_status()["normalized_status"] == DIFF_UPDATED
    assert [event["raw_method"] for event in recent_events] == [
        "codex/turn/delta",
        "codex/diff_updated",
    ]
    assert recent_events[-1]["payload"] == {"files": ["agent/example.py"]}
    assert recent_events[-1]["received_at"] == 123.0


def test_completion_notification_yields_completed_status():
    bridge = CodexAppServerBridge()

    event = bridge._handle_notification(
        {
            "jsonrpc": "2.0",
            "method": "codex/turn/completed",
            "params": {"turn_id": "turn-2"},
        }
    )

    assert event.normalized_status == COMPLETED
    assert bridge.get_status()["normalized_status"] == COMPLETED
    assert bridge.get_last_completion_event()["exit_code"] == 0
    queued = process_registry.completion_queue.get_nowait()
    assert queued == {
        "type": "completion",
        "session_id": "codex_turn_turn-2",
        "command": "codex app-server turn",
        "exit_code": 0,
        "output": "Codex turn completed via app-server",
    }


def test_error_notification_yields_failed_status():
    bridge = CodexAppServerBridge()

    event = bridge._handle_notification(
        {
            "jsonrpc": "2.0",
            "method": "codex/error",
            "params": {"turn_id": "turn-3", "error": {"message": "boom"}},
        }
    )

    assert event.normalized_status == FAILED
    assert bridge.get_status()["normalized_status"] == FAILED
    assert bridge.get_status()["last_error"] == "boom"
    assert bridge.get_last_completion_event()["exit_code"] == 1
    queued = process_registry.completion_queue.get_nowait()
    assert queued["session_id"] == "codex_turn_turn-3"
    assert queued["command"] == "codex app-server turn"
    assert queued["exit_code"] == 1
    assert queued["output"] == "Codex turn failed via app-server: boom"


def test_repeated_terminal_notification_enqueues_once():
    bridge = CodexAppServerBridge()

    notification = {
        "jsonrpc": "2.0",
        "method": "codex/turn/completed",
        "params": {"turn_id": "turn-dupe"},
    }
    bridge._handle_notification(notification)
    bridge._handle_notification(notification)

    assert process_registry.completion_queue.qsize() == 1
    queued = process_registry.completion_queue.get_nowait()
    assert queued["session_id"] == "codex_turn_turn-dupe"


def test_unknown_notification_methods_are_stored_safely_without_failure():
    bridge = CodexAppServerBridge()

    event = bridge._handle_incoming_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "codex/unexpected_new_event",
                "params": {"shape": {"can": "drift"}},
            }
        )
    )

    assert event.raw_method == "codex/unexpected_new_event"
    assert event.normalized_status == "idle"
    assert bridge.get_status()["normalized_status"] == "idle"
    assert bridge.get_recent_events()[0]["payload"] == {"shape": {"can": "drift"}}


def test_synthetic_completion_event_payload_has_expected_minimal_keys():
    bridge = CodexAppServerBridge()
    bridge.set_route_metadata(platform="local", chat_id="cli", user_id="user-1")
    bridge._handle_notification(
        {
            "jsonrpc": "2.0",
            "method": "codex/turn/completed",
            "params": {"turn_id": "turn-4"},
        }
    )

    payload = bridge.build_completion_event()

    assert payload == {
        "type": "completion",
        "session_id": "codex_turn_turn-4",
        "command": "codex app-server turn",
        "exit_code": 0,
        "output": "Codex turn completed via app-server",
        "platform": "local",
        "chat_id": "cli",
        "user_id": "user-1",
    }
