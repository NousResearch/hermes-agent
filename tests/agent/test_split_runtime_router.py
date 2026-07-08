import json
import threading

import model_tools
import agent.split_runtime_router as split_router
from agent.split_runtime_router import _NOT_ROUTED, route_tool_locally
from gateway.tool_channel_state import (
    clear_tool_channel_state,
    register_tool_notify,
    reset_current_split_runtime,
    resolve_tool_result,
    set_current_split_runtime,
    unregister_tool_notify,
)
from tools.approval import reset_current_session_key, set_current_session_key
from tools.registry import discover_builtin_tools


def setup_module():
    discover_builtin_tools()


def teardown_function():
    clear_tool_channel_state()


def _bind_split(session_key="run_local", *, timeout=1.0):
    approval_token = set_current_session_key(session_key)
    split_token = set_current_split_runtime({
        "enabled": True,
        "routed_toolsets": ["file"],
        "request_timeout_seconds": timeout,
    })
    return approval_token, split_token


def _reset(approval_token, split_token):
    reset_current_split_runtime(split_token)
    reset_current_session_key(approval_token)


def test_route_tool_locally_returns_sentinel_when_disabled():
    result = route_tool_locally(
        "read_file",
        {"path": "README.md"},
        "call_disabled",
        task_id="task",
        session_id="session",
    )
    assert result is _NOT_ROUTED


def test_route_tool_locally_returns_sentinel_for_non_routable_tool():
    approval_token, split_token = _bind_split()
    try:
        result = route_tool_locally("write_file", {"path": "x", "content": "y"}, "call_write")
    finally:
        _reset(approval_token, split_token)
    assert result is _NOT_ROUTED


def test_route_tool_locally_fails_closed_without_executor():
    approval_token, split_token = _bind_split()
    try:
        result = route_tool_locally("read_file", {"path": "README.md"}, "call_no_executor")
    finally:
        _reset(approval_token, split_token)
    payload = json.loads(result)
    assert payload["code"] == "split_runtime_no_executor"
    assert payload["tool_call_id"] == "call_no_executor"


def test_route_tool_locally_round_trips_to_attached_executor():
    session_key = "run_roundtrip"
    approval_token, split_token = _bind_split(session_key=session_key)
    saw_request = threading.Event()
    captured = {}

    def notify(request):
        captured.update(request)
        saw_request.set()

    assert register_tool_notify(session_key, notify, "client-a") is True

    def resolver():
        assert saw_request.wait(timeout=2.0)
        assert captured["tool_name"] == "read_file"
        assert captured["arguments"] == {"path": "README.md"}
        resolve_tool_result(session_key, captured["tool_call_id"], "LOCAL RESULT")

    thread = threading.Thread(target=resolver)
    thread.start()
    try:
        result = route_tool_locally(
            "read_file",
            {"path": "README.md"},
            "call_roundtrip",
            task_id="task-1",
            session_id="session-1",
            turn_id="turn-1",
            api_request_id="api-1",
        )
    finally:
        thread.join(timeout=2.0)
        unregister_tool_notify(session_key)
        _reset(approval_token, split_token)

    assert result == "LOCAL RESULT"
    assert captured["session_id"] == "session-1"
    assert captured["task_id"] == "task-1"
    assert captured["turn_id"] == "turn-1"
    assert captured["api_request_id"] == "api-1"


def test_handle_function_call_fails_closed_when_split_router_crashes(monkeypatch):
    approval_token, split_token = _bind_split()

    def boom(*args, **kwargs):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(split_router, "route_tool_locally", boom)
    try:
        result = model_tools.handle_function_call(
            "read_file",
            {"path": "README.md"},
            tool_call_id="call_router_crash",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
    finally:
        _reset(approval_token, split_token)

    payload = json.loads(result)
    assert payload["code"] == "split_runtime_router_error"
    assert payload["tool_call_id"] == "call_router_crash"
