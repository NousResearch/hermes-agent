import os

import pytest

from tools import approval
from tools.approval import reset_state


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in ("HERMES_REQUEST_ID", "HERMES_TASK_ID", "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(key, raising=False)
    reset_state()
    yield
    reset_state()


def test_submit_pending_logs_request(monkeypatch):
    seen = {}
    monkeypatch.setenv("HERMES_REQUEST_ID", "req_test123")
    monkeypatch.setattr(approval.hermes_log, "approval_requested", lambda **kwargs: seen.update(kwargs))

    approval.submit_pending(
        "session-1",
        {"command": "rm -rf /tmp/x", "description": "recursive delete", "task_id": "task-1"},
    )

    assert seen["request_id"] == "req_test123"
    assert seen["task_id"] == "task-1"
    assert seen["session_id"] == "session-1"


def test_check_dangerous_command_logs_resolution(monkeypatch):
    seen = {}
    monkeypatch.setenv("HERMES_REQUEST_ID", "req_test456")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_TASK_ID", "task-2")
    monkeypatch.setattr(approval.hermes_log, "approval_resolved", lambda **kwargs: seen.update(kwargs))

    monkeypatch.setattr(
        approval,
        "detect_dangerous_command",
        lambda command: (True, "recursive delete", "recursive delete"),
    )
    result = approval.check_dangerous_command(
        "rm -rf /tmp/x",
        "local",
        approval_callback=lambda command, description, allow_permanent=True: "session",
    )

    assert result["approved"] is True
    assert seen["request_id"] == "req_test456"
    assert seen["task_id"] == "task-2"
    assert seen["decision"] == "session"
