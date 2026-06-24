import os
import queue
import stat
import time

import pytest


pytestmark = pytest.mark.skipif(
    not hasattr(__import__("socket"), "AF_UNIX"),
    reason="Unix-domain sockets are not available on this Python build",
)


def _cli_stub(*, running=True, busy_mode="queue", agent=None):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "sid"
    cli._agent_running = running
    cli.busy_input_mode = busy_mode
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    cli.agent = agent
    cli._app = None
    return HermesCLI, cli


def test_session_ipc_server_accepts_and_registers_live_session(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.session_ipc import (
        SessionIPCServer,
        list_live_sessions,
        send_message_to_session,
    )

    received = []

    server = SessionIPCServer(
        "20260622_153814_f1c0de",
        lambda payload: received.append(payload) or {
            "sent": True,
            "session": payload["session_id"],
            "message_id": "m1",
            "status": "queued",
        },
    )
    server.start()
    try:
        live = list_live_sessions()
        assert [entry["session_id"] for entry in live] == ["20260622_153814_f1c0de"]
        if os.name != "nt":
            socket_mode = stat.S_IMODE(os.stat(live[0]["socket_path"]).st_mode)
            registry_mode = stat.S_IMODE(os.stat(tmp_path / "runtime" / "session_ipc" / "live_sessions.json").st_mode)
            assert socket_mode == 0o600
            assert registry_mode == 0o600

        result = send_message_to_session(
            session="20260622_153814_f1c0de",
            message="Run the review",
        )

        assert result["sent"] is True
        assert result["status"] == "queued"
        assert received[0]["content"] == "Run the review"
        assert received[0]["source"] == "cli-send"
    finally:
        server.stop()

    assert list_live_sessions() == []


def test_session_ipc_current_targets_most_recent_session(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.session_ipc import SessionIPCServer, send_message_to_session

    seen = []
    older = SessionIPCServer("older", lambda payload: {"sent": True, "session": "older", "message_id": "old", "status": "queued"})
    newer = SessionIPCServer("newer", lambda payload: seen.append(payload) or {"sent": True, "session": "newer", "message_id": "new", "status": "queued"})
    older.start()
    try:
        time.sleep(0.05)
        newer.start()
        try:
            result = send_message_to_session(session=None, current=True, message="hello")
            assert result["sent"] is True
            assert result["session"] == "newer"
            assert seen[0]["content"] == "hello"
        finally:
            newer.stop()
    finally:
        older.stop()


def test_session_ipc_exact_match_wins_over_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.session_ipc import SessionIPCServer, resolve_live_session

    first = SessionIPCServer("abc", lambda payload: {"sent": True})
    second = SessionIPCServer("abcdef", lambda payload: {"sent": True})
    first.start()
    try:
        second.start()
        try:
            resolved = resolve_live_session("abc")
            assert resolved is not None
            assert resolved["session_id"] == "abc"
        finally:
            second.stop()
    finally:
        first.stop()


def test_session_ipc_ambiguous_prefix_reports_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.session_ipc import SessionIPCServer, send_message_to_session

    first = SessionIPCServer("abc123", lambda payload: {"sent": True})
    second = SessionIPCServer("abc456", lambda payload: {"sent": True})
    first.start()
    try:
        second.start()
        try:
            result = send_message_to_session(session="abc", message="hello")
            assert result["sent"] is False
            assert "Multiple running Hermes sessions match" in result["error"]
        finally:
            second.stop()
    finally:
        first.stop()


def test_cli_ipc_message_routes_by_busy_mode(monkeypatch):
    HermesCLI, cli = _cli_stub(running=True, busy_mode="queue")

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": "queued message", "mode": "auto"},
    )

    assert result["sent"] is True
    assert result["status"] == "queued"
    assert cli._pending_input.get_nowait() == "queued message"


def test_cli_ipc_auto_reject_mode_reports_busy():
    HermesCLI, cli = _cli_stub(running=True, busy_mode="reject")

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": "do not queue", "mode": "auto"},
    )

    assert result["sent"] is False
    assert result["status"] == "busy"
    assert cli._pending_input.empty()


def test_cli_ipc_explicit_reject_mode_reports_busy():
    HermesCLI, cli = _cli_stub(running=True, busy_mode="queue")

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": "do not queue", "mode": "reject"},
    )

    assert result["sent"] is False
    assert result["status"] == "busy"
    assert cli._pending_input.empty()


def test_cli_ipc_interrupt_mode_uses_interrupt_queue():
    HermesCLI, cli = _cli_stub(running=True, busy_mode="queue")

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": "interrupt this", "mode": "interrupt"},
    )

    assert result["sent"] is True
    assert result["status"] == "interrupt_queued"
    assert cli._interrupt_queue.get_nowait() == "interrupt this"
    assert cli._pending_input.empty()


def test_cli_ipc_steer_mode_uses_agent_steer():
    class Agent:
        def __init__(self):
            self.seen = []

        def steer(self, text):
            self.seen.append(text)
            return True

    agent = Agent()
    HermesCLI, cli = _cli_stub(running=True, busy_mode="queue", agent=agent)

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": "steer this", "mode": "steer"},
    )

    assert result["sent"] is True
    assert result["status"] == "steered"
    assert agent.seen == ["steer this"]
    assert cli._pending_input.empty()


def test_cli_ipc_idle_session_delivers_next_turn():
    HermesCLI, cli = _cli_stub(running=False, busy_mode="queue")

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": "next turn", "mode": "auto"},
    )

    assert result["sent"] is True
    assert result["status"] == "delivered"
    assert cli._pending_input.get_nowait() == "next turn"


def test_cli_ipc_preserves_message_whitespace():
    HermesCLI, cli = _cli_stub(running=False, busy_mode="queue")
    content = "\n\n  indented/trailing  \n"

    result = HermesCLI._handle_session_ipc_message(
        cli,
        {"content": content, "mode": "auto"},
    )

    assert result["sent"] is True
    assert cli._pending_input.get_nowait() == content
