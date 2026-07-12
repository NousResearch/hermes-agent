import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

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
    approval_token, split_token = _bind_split(timeout=0.1)
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
        resolve_tool_result(session_key, captured["request_id"], "LOCAL RESULT")

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


def test_real_agent_tool_executor_routes_read_file_through_local_channel():
    from run_agent import AIAgent

    session_key = "run-real-agent"
    approval_token, split_token = _bind_split(session_key=session_key, timeout=5.0)
    captured = {}
    saw_request = threading.Event()

    def notify(request):
        captured.update(request)
        saw_request.set()

    assert register_tool_notify(session_key, notify, "") is True

    def resolve_request():
        assert saw_request.wait(timeout=5.0)
        resolve_tool_result(session_key, captured["request_id"], "REAL AGENT LOCAL RESULT")

    resolver = threading.Thread(target=resolve_request)
    resolver_started = False
    try:
        with (
            patch("run_agent.OpenAI"),
            patch("run_agent.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(
                model="test-model",
                provider="openrouter",
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                enabled_toolsets=["file"],
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        resolver.start()
        resolver_started = True

        tool_call = SimpleNamespace(
            id="call_real_agent",
            function=SimpleNamespace(
                name="read_file",
                arguments=json.dumps({"path": "README.md"}),
            ),
        )
        assistant_message = SimpleNamespace(tool_calls=[tool_call])
        messages = []
        agent._execute_tool_calls(assistant_message, messages, "task-real-agent")
    finally:
        if resolver_started:
            resolver.join(timeout=5.0)
        unregister_tool_notify(session_key)
        _reset(approval_token, split_token)

    assert captured["tool_name"] == "read_file"
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["content"] == "REAL AGENT LOCAL RESULT"


def test_route_tool_locally_waits_for_executor_to_attach():
    session_key = "run_late_attach"
    approval_token, split_token = _bind_split(session_key=session_key, timeout=1.0)
    captured = {}

    def attach_and_resolve():
        def notify(request):
            captured.update(request)
            resolve_tool_result(session_key, request["request_id"], "LATE ATTACH RESULT")

        assert register_tool_notify(session_key, notify, "client-late") is True

    timer = threading.Timer(0.02, attach_and_resolve)
    timer.start()
    try:
        result = route_tool_locally(
            "read_file",
            {"path": "README.md"},
            "call_late_attach",
        )
    finally:
        timer.join(timeout=1.0)
        unregister_tool_notify(session_key)
        _reset(approval_token, split_token)

    assert result == "LATE ATTACH RESULT"
    assert captured["tool_call_id"] == "call_late_attach"
    assert captured["request_id"].startswith("toolreq_")


def test_route_tool_locally_times_out_when_executor_never_answers():
    session_key = "run_timeout"
    approval_token, split_token = _bind_split(session_key=session_key, timeout=0.1)
    assert register_tool_notify(session_key, lambda request: None, "client-timeout") is True
    try:
        result = route_tool_locally(
            "read_file",
            {"path": "README.md"},
            "call_timeout",
        )
    finally:
        unregister_tool_notify(session_key)
        _reset(approval_token, split_token)

    assert json.loads(result)["code"] == "split_runtime_tool_timeout"


def test_route_tool_locally_reports_executor_disconnect():
    session_key = "run_disconnect"
    approval_token, split_token = _bind_split(session_key=session_key, timeout=1.0)

    def disconnect(_request):
        unregister_tool_notify(session_key, client_token="client-disconnect")

    assert register_tool_notify(session_key, disconnect, "client-disconnect") is True
    try:
        result = route_tool_locally(
            "search_files",
            {"pattern": "*.py", "target": "files"},
            "call_disconnect",
        )
    finally:
        unregister_tool_notify(session_key)
        _reset(approval_token, split_token)

    assert json.loads(result)["code"] == "split_runtime_executor_disconnected"


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


def test_split_runtime_keeps_server_tool_results_inline_only():
    from agent.tool_executor import _tool_result_is_local, _tool_result_storage_target

    approval_token, split_token = _bind_split()
    server_env = object()
    try:
        with patch("agent.tool_executor.get_active_env", return_value=server_env):
            result_env, inline_only = _tool_result_storage_target("terminal", "")

        assert result_env is None
        assert inline_only is True
        assert _tool_result_is_local("terminal") is False
        assert _tool_result_is_local("read_file") is True
    finally:
        _reset(approval_token, split_token)
