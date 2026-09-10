"""Desktop batches publish consent together, never shell effects together."""

import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent
from gateway.session_context import set_session_vars, clear_session_vars
from tools import approval
from tools.thread_context import propagate_context_to_thread


def _call(call_id, command):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name="terminal", arguments=json.dumps({"command": command})))


def _agent():
    from tools.terminal_tool import TERMINAL_SCHEMA
    from tools.file_tools import READ_FILE_SCHEMA
    with (
        patch("model_tools.get_tool_definitions", return_value=[
            {"type": "function", "function": schema}
            for schema in (TERMINAL_SCHEMA, READ_FILE_SCHEMA)
        ]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                       quiet_mode=True, skip_context_files=True, skip_memory=True, platform="gui")


@pytest.mark.parametrize("read_count", [0, 1, 2])
@pytest.mark.parametrize("threaded_middleware", [False, True])
def test_desktop_publishes_final_commands_before_wait_and_runs_in_order(tmp_path, monkeypatch, read_count, threaded_middleware):
    from tools.terminal_scope import terminal_scope
    from tools.terminal_tool_lifecycle import cleanup_vm
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")
    monkeypatch.setattr("tools.approval._tirith_scan", lambda command: {"action": "allow"})
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    nested = tmp_path / "nested"
    nested.mkdir()
    key = "desktop-terminal-batch"
    agent = _agent()
    published = queue.Queue()
    approval.register_gateway_notify(key, published.put)
    tokens = set_session_vars(source="gui", session_key=key, cwd=str(tmp_path), cron_session="")
    calls = [_call("first", "original-first"), _call("second", "original-second")]
    if read_count:
        source = tmp_path / "input.txt"
        source.write_text("input")
        calls[:0] = [SimpleNamespace(id=f"read-{i}", type="function", function=SimpleNamespace(
            name="read_file", arguments=json.dumps({"path": str(source)}))) for i in range(read_count)]
    commands = {"original-first": f"rm -rf absent-first; cd '{nested}'", "original-second": "rm -rf absent-second; pwd"}
    policy = []
    executed = []
    flushed = []
    messages = []
    errors = []
    replay_errors = []

    if threaded_middleware:
        def middleware(name, args, execute, **kwargs):
            if name != "terminal":
                return execute(args)
            results = []
            def dispatch():
                results.append(execute(args))
                try:
                    execute(args)
                except RuntimeError as exc:
                    replay_errors.append(str(exc))
            # Middleware may move its continuation onto a fresh thread.
            thread = threading.Thread(target=dispatch, daemon=True)
            thread.start()
            thread.join(20)
            return results[0]
        monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", middleware)

    def pre_hook(name, args, **kwargs):
        if name != "terminal":
            return None, args
        policy.append(args["command"])
        return None, {**args, "command": commands[args["command"]]}

    def started(call_id, name, args):
        if name != "terminal":
            return
        executed.append((call_id, args["command"], list(flushed)))

    agent.tool_start_callback = started
    agent._flush_messages_to_session_db = lambda rows, **kw: flushed.append([r["tool_call_id"] for r in rows]) or True

    def run():
        try:
            agent._execute_tool_calls(SimpleNamespace(tool_calls=calls), messages, key)
        except BaseException as exc:
            errors.append(exc)

    with terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)}), patch(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=pre_hook
    ):
        worker = threading.Thread(target=propagate_context_to_thread(run), daemon=True)
        worker.start()
        try:
            first = published.get(timeout=10)
            second = published.get(timeout=5)
            assert [first["command"], second["command"]] == list(commands.values())
            assert policy == list(commands)
            assert executed == []
            assert approval.ack_gateway_approval(key, second["request_id"])
            assert executed == []
            assert approval.resolve_gateway_approval(key, "once", request_id=second["request_id"]) == 1
            assert executed == []
            assert approval.resolve_gateway_approval(key, "once", request_id=first["request_id"]) == 1
            worker.join(timeout=15)
            assert not worker.is_alive()
            assert errors == []
            assert [item[:2] for item in executed] == [("first", commands["original-first"]), ("second", commands["original-second"])]
            assert len(replay_errors) == (2 if threaded_middleware else 0)
            assert executed[1][2][-1] == [c.id for c in calls[:-1]]
            assert "first" not in {call_id for flush in executed[0][2] for call_id in flush}
            assert [m["tool_call_id"] for m in messages] == [c.id for c in calls]
            assert str(nested) in json.loads(messages[-1]["content"])["output"]
            assert approval.list_gateway_approvals(key) == []
        finally:
            agent.interrupt("test cleanup")
            approval.unregister_gateway_notify(key)
            worker.join(timeout=5)
            cleanup_vm(key)
            clear_session_vars(tokens)


def test_cancelled_preparation_drains_requests_without_reusing_once(tmp_path, monkeypatch):
    from tools.terminal_scope import terminal_scope
    from tools.terminal_tool_lifecycle import cleanup_vm

    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")
    monkeypatch.setattr("tools.approval._tirith_scan", lambda command: {"action": "allow"})
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    key = "cancelled-terminal-batch"
    command = "rm -rf absent; printf executed > effect.txt"
    published = queue.Queue()
    release_notify = threading.Event()
    messages, errors = [], []
    agent = _agent()
    agent._flush_messages_to_session_db = lambda *a, **kw: True
    calls = [_call(call_id, command) for call_id in ("first", "second", "third")]
    notices = []

    def notify(data):
        notices.append(data)
        published.put(data)
        if len(notices) == 3:
            release_notify.wait(10)

    def run(target, batch_calls, rows):
        try:
            target._execute_tool_calls(SimpleNamespace(tool_calls=batch_calls), rows, key)
        except BaseException as exc:
            errors.append(exc)

    approval.register_gateway_notify(key, notify)
    tokens = set_session_vars(source="gui", session_key=key, cwd=str(tmp_path), cron_session="")
    with terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)}):
        worker = threading.Thread(target=propagate_context_to_thread(lambda: run(agent, calls, messages)), daemon=True)
        retry_worker = None
        retry_agent = None
        worker.start()
        try:
            requests = [published.get(timeout=10) for _ in calls]
            assert len({r["request_id"] for r in requests}) == len(calls)
            assert approval.resolve_gateway_approval(key, "once", request_id=requests[1]["request_id"]) == 1
            agent.interrupt("cancel while publishing")
            worker.join(5)
            assert not worker.is_alive()
            assert errors == []
            assert [row["tool_call_id"] for row in messages] == [c.id for c in calls]
            assert approval.list_gateway_approvals(key) == []
            assert not (tmp_path / "effect.txt").exists()
            assert approval.resolve_gateway_approval(key, "once", request_id=requests[0]["request_id"]) == 0
            release_notify.set()

            # Same command AND call id in a later run still needs fresh consent.
            retry_agent = _agent()
            retry_agent._flush_messages_to_session_db = lambda *a, **kw: True
            retry_rows = []
            retry_worker = threading.Thread(target=propagate_context_to_thread(
                lambda: run(retry_agent, [_call("second", command)], retry_rows)), daemon=True)
            retry_worker.start()
            request = published.get(timeout=10)
            assert request["request_id"] not in {r["request_id"] for r in requests}
            assert not (tmp_path / "effect.txt").exists()
            assert approval.resolve_gateway_approval(key, "deny", request_id=request["request_id"]) == 1
            retry_worker.join(5)
            assert not retry_worker.is_alive()
            assert errors == []
            assert json.loads(retry_rows[0]["content"])["status"] == "blocked"
            assert not (tmp_path / "effect.txt").exists()
            assert approval.list_gateway_approvals(key) == []
        finally:
            release_notify.set()
            agent.interrupt("test cleanup")
            if retry_agent is not None:
                retry_agent.interrupt("test cleanup")
            approval.unregister_gateway_notify(key)
            worker.join(5)
            if retry_worker is not None:
                retry_worker.join(5)
            cleanup_vm(key)
            clear_session_vars(tokens)
