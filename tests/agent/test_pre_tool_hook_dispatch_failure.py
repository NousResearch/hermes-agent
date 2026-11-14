"""A failure to dispatch ``pre_tool_call`` hooks blocks the tool call.

Regression for #141420. Individual hook callback failures are already handled
inside ``_dispatch_pre_tool_call_hooks``. These tests cover the layer above it:
when the dispatch itself fails (it raises, returns a malformed verdict, or the
plugin module cannot be imported), the three outer wrappers must block the call
instead of letting it run without the hooks' verdict.
"""

import json
import logging
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import _pre_tool_block_message, invoke_tool
from agent.tool_executor import _pre_tool_block, _ToolCallRef
from model_tools import _pre_dispatch_guards, handle_function_call
from tests.agent.test_tool_call_guardrail_runtime import _make_agent, _mock_tool_call

BLOCK = "Pre-tool hook dispatch failed; tool execution blocked"
SECRET = "private-detail-that-must-not-reach-the-model"
DISPATCH = "hermes_cli.plugins._dispatch_pre_tool_call_hooks"
FAILURES = ["raises", "malformed"]


@contextmanager
def _failing_dispatch(kind: str):
    """Make hook dispatch fail: by raising, or by returning a verdict that is not a pair."""
    if kind == "raises":
        with patch(DISPATCH, side_effect=RuntimeError(SECRET)) as mock:
            yield mock
    else:
        with patch(DISPATCH, return_value=None) as mock:
            yield mock


def _wrapper_verdicts():
    """Call each outer wrapper once; returns ``[(block_message, args), ...]``."""
    agent = SimpleNamespace(session_id="session-1")
    args = {"query": "q"}
    ref = _ToolCallRef("web_search", dict(args), "task-1", "call-1", [])
    ids = SimpleNamespace(hook_kwargs=dict)
    with patch("acp_adapter.edit_approval.maybe_require_edit_approval", return_value=None):
        model_args, blocked = _pre_dispatch_guards("web_search", dict(args), False, ids, [])
    return [
        _pre_tool_block(agent, ref),
        _pre_tool_block_message(agent, "web_search", dict(args), "task-1", "call-1", []),
        (None if blocked is None else blocked[2], model_args),
    ]


@pytest.mark.parametrize("kind", FAILURES)
def test_every_wrapper_blocks_when_dispatch_fails(kind):
    with _failing_dispatch(kind):
        verdicts = _wrapper_verdicts()

    assert verdicts == [(BLOCK, {"query": "q"})] * 3


def test_every_wrapper_blocks_when_plugin_module_cannot_be_imported(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", None)

    with caplog.at_level(logging.WARNING):
        verdicts = _wrapper_verdicts()

    assert verdicts == [(BLOCK, {"query": "q"})] * 3
    warnings = [
        record for record in caplog.records
        if record.levelno == logging.WARNING and "pre_tool_call hook dispatch failed" in record.getMessage()
    ]
    assert len(warnings) == 3
    assert all(record.exc_info for record in warnings)


@pytest.mark.parametrize(
    "verdict, expected",
    [
        ((None, None), (None, {"query": "q"})),
        (("denied by policy", None), ("denied by policy", {"query": "q"})),
        ((None, {"query": "rewritten"}), (None, {"query": "rewritten"})),
    ],
    ids=["allow", "block", "modify"],
)
def test_every_wrapper_keeps_ordinary_verdicts(verdict, expected):
    with patch(DISPATCH, return_value=verdict):
        verdicts = _wrapper_verdicts()

    assert verdicts == [expected] * 3


@pytest.mark.parametrize("kind", FAILURES)
def test_direct_model_dispatch_blocks_and_reports_once(kind):
    with (
        _failing_dispatch(kind) as dispatch_hooks,
        patch("model_tools.registry.dispatch", side_effect=AssertionError("must not dispatch")) as registry,
        patch("model_tools._emit_post_tool_call_hook") as post,
    ):
        raw = handle_function_call("read_file", {"path": "notes.txt"}, task_id="task-1")

    assert json.loads(raw) == {"error": BLOCK}
    assert SECRET not in raw
    registry.assert_not_called()
    dispatch_hooks.assert_called_once()
    post.assert_called_once()
    outcome = post.call_args.kwargs
    assert (outcome["status"], outcome["error_type"], outcome["error_message"]) == ("blocked", "plugin_block", BLOCK)


@pytest.mark.parametrize("kind", FAILURES)
def test_invoke_tool_blocks_before_any_executor(kind):
    agent = _make_agent("web_search")

    with (
        _failing_dispatch(kind),
        patch("agent.inline_tool_executors.resolve_invoke_tool_executor", side_effect=AssertionError("must not resolve")),
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as handler,
    ):
        raw = invoke_tool(agent, "web_search", {"query": "q"}, "task-1", "call-1")

    assert json.loads(raw) == {"error": BLOCK}
    assert SECRET not in raw
    handler.assert_not_called()


@pytest.mark.parametrize("kind", FAILURES)
def test_sequential_path_blocks_and_returns_one_tool_result(kind):
    agent = _make_agent("web_search")
    starts = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    call = _mock_tool_call("web_search", json.dumps({"query": "q"}), "c-failed")
    msg = SimpleNamespace(content="", tool_calls=[call])
    messages = []

    with (
        _failing_dispatch(kind),
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as handler,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    handler.assert_not_called()
    assert starts == []
    assert [m["tool_call_id"] for m in messages] == ["c-failed"]
    assert BLOCK in messages[0]["content"]
    assert SECRET not in messages[0]["content"]


def test_concurrent_path_blocks_only_the_call_whose_dispatch_failed():
    agent = _make_agent("web_search")
    failed_args = {"query": "dispatch-fails"}
    allowed_args = {"query": "allowed"}
    calls = [
        _mock_tool_call("web_search", json.dumps(failed_args), "c-failed"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allowed"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def dispatch_hooks(name, args, **kwargs):
        if args == failed_args:
            raise RuntimeError(SECRET)
        return None, None

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with (
        patch(DISPATCH, side_effect=dispatch_hooks),
        patch("model_tools.handle_function_call", side_effect=fake_handle),
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allowed")]
    assert [m["tool_call_id"] for m in messages] == ["c-failed", "c-allowed"]
    assert BLOCK in messages[0]["content"]
    assert SECRET not in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
