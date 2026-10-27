"""A failed policy dispatch must not start the registered tool handler."""

import json
from types import SimpleNamespace

import pytest

import model_tools
from agent.agent_runtime_helpers import _pre_tool_block_message
from agent.tool_executor import _pre_tool_block
from hermes_cli import plugins
from tools.registry import registry


@pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
def test_dispatch_failure_blocks_handler_and_redacts_error(monkeypatch, tool_name):
    started = []
    observed = []

    def broken_dispatch(*args, **kwargs):
        raise RuntimeError("private dispatcher detail")

    def handler(args, **kwargs):
        started.append(args)
        return json.dumps({"output": "executed"})

    # Substitute only the target handler; use the real tool dispatch and registry.
    entry = registry.get_entry(tool_name)
    assert entry is not None
    monkeypatch.setattr(entry, "handler", handler)
    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", broken_dispatch)
    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", lambda **kw: observed.append(kw))

    result = model_tools.handle_function_call(
        tool_name, {"path": "unused.txt", "content": "unused"},
        skip_tool_request_middleware=True,
    )

    assert started == []
    assert "policy dispatch failed" in json.loads(result)["error"]
    assert "private dispatcher detail" not in result
    assert observed[-1]["status"] == "blocked"
    assert observed[-1]["error_type"] == "plugin_dispatch_error"
    assert "private dispatcher detail" not in observed[-1]["error_message"]

    agent = SimpleNamespace(session_id="policy-test")
    ref = SimpleNamespace(name=tool_name, args={"path": "unused.txt"}, task_id="policy-test", call_id="tc", trace=[])
    for block_message, args in (
        _pre_tool_block_message(agent, tool_name, ref.args, ref.task_id, ref.call_id, []),
        _pre_tool_block(agent, ref),
    ):
        assert "policy dispatch failed" in block_message
        assert "private dispatcher detail" not in block_message
        assert args == ref.args


def test_empty_policy_result_keeps_dispatch_working(monkeypatch):
    started = []

    def handler(args, **kwargs):
        started.append(args)
        return json.dumps({"output": "executed"})

    entry = registry.get_entry("read_file")
    assert entry is not None
    monkeypatch.setattr(entry, "handler", handler)
    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", lambda *a, **kw: (None, None))
    result = model_tools.handle_function_call(
        "read_file", {"path": "unused.txt"},
        skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
    )

    assert json.loads(result)["output"] == "executed"
    assert started == [{"path": "unused.txt"}]
    agent = SimpleNamespace(session_id="policy-test")
    ref = SimpleNamespace(name="read_file", args={"path": "unused.txt"}, task_id="policy-test", call_id="tc", trace=[])
    assert _pre_tool_block_message(agent, ref.name, ref.args, ref.task_id, ref.call_id, []) == (None, ref.args)
    assert _pre_tool_block(agent, ref) == (None, ref.args)
