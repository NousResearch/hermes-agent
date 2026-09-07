"""Middleware payloads evolve additively, like hook payloads: a ``tool_execution`` / ``tool_request``
middleware with an explicit parameter list (no ``**kwargs``) that was valid before the principal
fields were added must still run and receive the fields it declares, not raise ``TypeError`` and be
logged-and-skipped (fail-open for a policy middleware). ``**kwargs`` callbacks keep getting everything.
"""

import json
import logging
import types

from agent.inline_tool_executors import tool_hook_ids
from hermes_cli.middleware import apply_tool_request_middleware, run_tool_execution_middleware
from hermes_cli.plugins_dispatch import PluginDispatchMixin
from model_tools import handle_function_call

BLOCKED = json.dumps({"error": "blocked by policy middleware"})


def _narrow_execution_middleware(
    tool_name, args, next_call, original_args=None, task_id="", session_id="", tool_call_id="",
    turn_id="", api_request_id="", telemetry_schema_version=None, middleware_schema_version=None,
):
    """Declares exactly the pre-principal payload field set; blocks without calling next_call."""
    return BLOCKED


def _kwargs_execution_middleware(tool_name, args, next_call, **kwargs):
    return BLOCKED


def _install_manager(monkeypatch, kind, middleware):
    manager = PluginDispatchMixin()  # real dispatcher: request middleware goes through invoke_middleware
    manager._middleware = {kind: [middleware]}
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.plugins.has_middleware", lambda k: k == kind)
    return manager


def _run_handle_function_call(monkeypatch, middleware):
    _install_manager(monkeypatch, "tool_execution", middleware)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    dispatched = []
    monkeypatch.setattr(
        "model_tools.registry.dispatch",
        lambda tool_name, args, **kw: dispatched.append(tool_name) or json.dumps({"ok": True}),
    )
    result = json.loads(handle_function_call(
        "web_search", {"q": "x"}, task_id="t1", session_id="s1", tool_call_id="c1",
    ))
    return result, dispatched


def test_kwargs_execution_middleware_blocks(monkeypatch):
    result, dispatched = _run_handle_function_call(monkeypatch, _kwargs_execution_middleware)
    assert result == json.loads(BLOCKED)
    assert dispatched == []


def test_closed_signature_execution_middleware_still_blocks(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hermes_cli.middleware")
    result, dispatched = _run_handle_function_call(monkeypatch, _narrow_execution_middleware)
    assert result == json.loads(BLOCKED), (result, [r.getMessage() for r in caplog.records])
    assert dispatched == [], "closed-signature middleware was skipped; the tool executed (fail-open)"


def test_closed_signature_execution_middleware_gets_declared_fields_only(monkeypatch):
    seen = {}

    def _narrow(tool_name, args, next_call, task_id="", session_id=""):
        seen.update(tool_name=tool_name, args=args, task_id=task_id, session_id=session_id)
        return next_call(args)

    _install_manager(monkeypatch, "tool_execution", _narrow)
    agent = types.SimpleNamespace(session_id="s1", _current_turn_id="t1", _current_api_request_id="r1")
    result = run_tool_execution_middleware(
        "terminal", {"command": "ls"}, lambda final_args: "RAN", original_args={"command": "ls"},
        **tool_hook_ids(agent, "task", "call-1"),
    )
    assert result == "RAN"
    assert seen == {"tool_name": "terminal", "args": {"command": "ls"}, "task_id": "task", "session_id": "s1"}


def test_closed_signature_request_middleware_still_rewrites_args(monkeypatch):
    def _narrow_request(tool_name, args, original_args=None, task_id="", session_id="", tool_call_id="",
                        turn_id="", api_request_id="", telemetry_schema_version=None,
                        middleware_schema_version=None):
        return {"args": {**args, "redacted": True}, "source": "policy"}

    _install_manager(monkeypatch, "tool_request", _narrow_request)
    agent = types.SimpleNamespace(session_id="s1", _current_turn_id="t1", _current_api_request_id="r1")
    result = apply_tool_request_middleware(
        "read_file", {"path": "x"}, skip_relay=True, **tool_hook_ids(agent, "task", "call-1"),
    )
    assert result.changed is True
    assert result.payload == {"path": "x", "redacted": True}
    assert result.trace == [{"source": "policy"}]


def test_invoke_middleware_filters_by_signature():
    manager = PluginDispatchMixin()
    manager._middleware = {"tool_request": [lambda tool_name, args: {"args": {"n": tool_name}}]}
    assert manager.invoke_middleware("tool_request", tool_name="t", args={}, user_id="u") == [{"args": {"n": "t"}}]
