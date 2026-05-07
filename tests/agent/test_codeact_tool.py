import json

from agent.codeact_tool import _run_code_handler, set_kernel_dispatcher
from model_tools import handle_function_call


def teardown_function(_function):
    set_kernel_dispatcher("session-a", None)
    set_kernel_dispatcher("session-b", None)


def test_run_code_handler_routes_by_session_id():
    set_kernel_dispatcher("session-a", lambda args: json.dumps({"session": "a", "args": args}))
    set_kernel_dispatcher("session-b", lambda args: json.dumps({"session": "b", "args": args}))

    result_a = json.loads(
        _run_code_handler({"code": "print('a')"}, session_id="session-a")
    )
    result_b = json.loads(
        _run_code_handler({"code": "print('b')"}, session_id="session-b")
    )

    assert result_a["session"] == "a"
    assert result_a["args"]["code"] == "print('a')"
    assert result_b["session"] == "b"
    assert result_b["args"]["code"] == "print('b')"


def test_run_code_handler_requires_session_id():
    result = json.loads(_run_code_handler({"code": "print('x')"}))
    assert "routing missing" in result["error"]


def test_handle_function_call_threads_session_id_to_run_code():
    set_kernel_dispatcher("session-a", lambda args: json.dumps({"session": "a", "args": args}))
    set_kernel_dispatcher("session-b", lambda args: json.dumps({"session": "b", "args": args}))

    result_a = json.loads(
        handle_function_call(
            "run_code",
            {"thoughts": "A", "code": "print('a')"},
            session_id="session-a",
        )
    )
    result_b = json.loads(
        handle_function_call(
            "run_code",
            {"thoughts": "B", "code": "print('b')"},
            session_id="session-b",
        )
    )

    assert result_a["session"] == "a"
    assert result_a["args"]["thoughts"] == "A"
    assert result_b["session"] == "b"
    assert result_b["args"]["thoughts"] == "B"
