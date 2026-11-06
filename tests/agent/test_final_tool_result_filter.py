"""A discovered final filter controls the bytes committed and displayed by both executors."""
import copy
import json
from threading import Event
from types import SimpleNamespace

import pytest

from hermes_cli import plugins
from tests.agent.test_tool_call_incremental_persistence import (
    _make_agent, _mock_tool_call, _disable_background_titles, _attach_real_session_db,
)

SECRET = "unchecked-private-output"


@pytest.fixture
def filters(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    directory = tmp_path / "plugins" / "final_filter"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: final_filter\n")
    (directory / "__init__.py").write_text('''
seen = []
mode = "replace"
def first(result, **kwargs):
    seen.append(result)
    if mode == "timeout":
        entered.set()
        release.wait(5)
        kwargs["args"]["nested"].append("late mutation")
    if mode == "error": raise RuntimeError("unchecked-private-output")
    if mode == "invalid": return {"unexpected": result}
    if mode == "observe": return None
    return "checked: " + kwargs["tool_name"]
def second(result, **kwargs):
    seen.append(result)
    return None
def register(ctx):
    ctx.register_hook("filter_tool_result", first)
    ctx.register_hook("filter_tool_result", second)
''')
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["final_filter"]}}))
    monkeypatch.setattr(plugins, "_plugin_manager", plugins.PluginManager())
    plugins.discover_plugins()
    callbacks = plugins._delivery_manager()._hooks["filter_tool_result"]
    return callbacks[0].__globals__


@pytest.fixture
def host(tmp_path):
    agent = _make_agent()
    db = _attach_real_session_db(agent, tmp_path / "state.db", "filtered-output")
    try:
        yield agent, db
    finally:
        db.close()


@pytest.mark.parametrize("executor", ["sequential", "concurrent", "inline"])
@pytest.mark.parametrize("mode", ["replace", "error", "invalid", "observe"])
def test_final_filter_controls_model_persistence_and_early_ui(
    executor, mode, filters, monkeypatch, host,
):
    filters["mode"] = mode
    agent, db = host
    persisted, completed = [], []
    flush = agent._flush_messages_to_session_db
    def record_flush(messages, **kwargs):
        persisted.append(copy.deepcopy(messages))
        return flush(messages, **kwargs)
    agent._flush_messages_to_session_db = record_flush
    agent.tool_progress_callback = lambda kind, *args, **kwargs: completed.append(kwargs.get("result")) if kind == "tool.completed" else None
    agent.tool_complete_callback = lambda _id, _name, _args, result: completed.append(result)
    monkeypatch.setattr(agent, "_should_emit_quiet_tool_messages", lambda: True)
    monkeypatch.setattr("agent.tool_executor._get_cute_tool_message_impl", lambda *args, **kw: completed.append(kw.get("result")) or "done")
    original = json.dumps({"output": SECRET})
    monkeypatch.setattr("model_tools.handle_function_call", lambda *args, **kwargs: original)
    monkeypatch.setattr(agent, "_invoke_tool", lambda *args, **kwargs: original)
    if executor == "inline":
        from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
        monkeypatch.setitem(INLINE_TOOL_EXECUTORS, "web_search", lambda *args: original)
        executor = "sequential"
    call = _mock_tool_call(name="web_search")
    messages = []
    getattr(agent, "_execute_tool_calls_" + executor)(SimpleNamespace(content="", tool_calls=[call]), messages, "task-1")
    delivered = json.dumps([messages, persisted, completed, db.get_messages_as_conversation("filtered-output")])
    assert messages and persisted and completed
    if mode == "observe":
        assert SECRET in delivered
    else:
        assert SECRET not in delivered
        assert ("withheld by output policy" if mode in {"error", "invalid"} else "checked: web_search") in delivered
    assert filters["seen"][0] == original
    if mode == "replace":
        assert filters["seen"][1] == "checked: web_search"
    # A completed filter fault does not poison the next call in this Agent.
    filters["mode"] = "replace"
    getattr(agent, "_execute_tool_calls_" + executor)(SimpleNamespace(content="", tool_calls=[_mock_tool_call(call_id="next")]), messages, "task-1")
    assert "checked: web_search" in json.dumps(messages[-1])


def test_timed_out_filter_cannot_mutate_adopted_data(filters, monkeypatch):
    filters.update(mode="timeout", entered=Event(), release=Event())
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.02)
    original = {"nested": []}
    try:
        result, withheld = plugins._delivery_manager().filter_tool_result(
            result=SECRET, args=original, tool_name="test", tool_call_id="timeout-1",
        )
        assert filters["entered"].is_set()
        assert withheld and result is None
    finally:
        filters["release"].set()
    from tests.waiting import wait_until
    assert wait_until(lambda: not plugins._delivery_manager()._hook_running_callbacks, timeout=3)
    assert original == {"nested": []}
    filters["mode"] = "replace"
    assert plugins._delivery_manager().filter_tool_result(
        result=SECRET, args=original, tool_name="test", tool_call_id="next-1",
    ) == (None, True)  # Native timeout suppression remains in force.


@pytest.mark.parametrize("outcome", ["timeout", "cancel"])
def test_terminal_error_results_use_the_same_final_filter(outcome, filters, host):
    from agent.tool_executor import (
        _ManagedToolResult, _ToolCallRef, _ToolTimeoutResult, _ToolCancelledResult,
        _publish_sequential_result,
    )
    from tools.tool_result_storage import BudgetConfig

    agent, db = host
    result_type = _ToolTimeoutResult if outcome == "timeout" else _ToolCancelledResult
    result = result_type(SECRET)
    messages = []
    ref = _ToolCallRef("web_search", {}, "task-1", "terminal-1", [])
    managed = _ManagedToolResult(result=result, args={}, middleware_trace=[], blocked=False, dispatched=True)
    assert _publish_sequential_result(agent, messages, ref, managed,
        tool_duration=0, index=1, budget=BudgetConfig(), transform_applied=False)
    assert SECRET not in json.dumps([messages, db.get_messages_as_conversation("filtered-output")])
    assert filters["seen"][0] == SECRET


def test_real_http_model_receives_only_filtered_result(filters, monkeypatch, host):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from openai import OpenAI

    requests = []
    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            message = ({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_http", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]}
                if len(requests) == 1 else {"role": "assistant", "content": "done"})
            payload = json.dumps({"id": "chat_fixture", "object": "chat.completion", "created": 0,
                "model": "test/model", "choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            content_type = "application/json"
            if body.get("stream"):
                if message.get("tool_calls"):
                    message["tool_calls"][0]["index"] = 0
                chunk = {"id": "chat_fixture", "object": "chat.completion.chunk", "created": 0,
                    "model": "test/model", "choices": [{"index": 0, "delta": message, "finish_reason": None}]}
                ending = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]}
                payload = ("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(ending) + "\n\ndata: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent, db = host
    agent.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    agent.model = "test/model"
    agent._client_kwargs = {"base_url": agent.base_url, "api_key": "fixture", "timeout": 5, "max_retries": 0}
    agent.client = OpenAI(**agent._client_kwargs)
    monkeypatch.setattr("model_tools.handle_function_call", lambda *args, **kwargs: SECRET)
    monkeypatch.setattr(agent, "_invoke_tool", lambda *args, **kwargs: SECRET)
    try:
        result = agent.run_conversation("inspect the tool")
        assert result["final_response"] == "done"
        assert len(requests) == 2
        assert SECRET not in json.dumps(requests)
        assert "checked: web_search" in json.dumps(requests[1])
        assert SECRET not in json.dumps(db.get_messages_as_conversation("filtered-output"))
    finally:
        agent.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
