"""Regression tests for run_agent.py shard s5 mixin extraction (w1a).

Covers the two highest-agreement move clusters extracted verbatim into
plugins/agent/mixins/:
  * c16 -> ToolExecutionMixin  (tool-call execution dispatch)
  * c7  -> ReasoningEchoMixin  (reasoning_content echo-back policy)

Pure/lightweight methods only; the heavy forwarders are tested for wiring
(delegation to the agent.* target with identical arguments).  Bare-adapter
pattern (object.__new__(AIAgent) + stub config attrs) matches the existing
tests/run_agent suite.
"""

import types

import pytest

from agent import agent_runtime_helpers
from agent import chat_completion_helpers
from agent import tool_executor
from plugins.agent.mixins.tool_execution_mixin import ToolExecutionMixin
from plugins.agent.mixins.reasoning_echo_mixin import ReasoningEchoMixin
from run_agent import AIAgent

MOVED_METHODS = [
    # c7 / ReasoningEchoMixin
    "_needs_thinking_reasoning_pad",
    "_needs_kimi_tool_reasoning",
    "_needs_deepseek_tool_reasoning",
    "_needs_mimo_tool_reasoning",
    "_copy_reasoning_content_for_api",
    "_reapply_reasoning_echo_for_provider",
    # c16 / ToolExecutionMixin
    "_execute_tool_calls",
    "_dispatch_delegate_task",
    "_invoke_tool",
    "_wrap_verbose",
    "_execute_tool_calls_concurrent",
    "_execute_tool_calls_sequential",
    "_handle_max_iterations",
]


def _bare_agent(**attrs):
    agent = object.__new__(AIAgent)
    for k, v in attrs.items():
        setattr(agent, k, v)
    return agent


# ---------------------------------------------------------------------------
# Mixin wiring
# ---------------------------------------------------------------------------

def test_mixins_wired_into_aiagent():
    assert issubclass(AIAgent, ToolExecutionMixin)
    assert issubclass(AIAgent, ReasoningEchoMixin)
    for name in MOVED_METHODS:
        assert hasattr(AIAgent, name), name


# ---------------------------------------------------------------------------
# c7 / ReasoningEchoMixin — provider reasoning-content echo-back policy
# ---------------------------------------------------------------------------

def test_needs_deepseek_tool_reasoning_by_provider():
    agent = _bare_agent(provider="deepseek", model="deepseek-v4",
                        base_url="http://localhost:30000/v1")
    assert agent._needs_deepseek_tool_reasoning() is True


def test_needs_deepseek_tool_reasoning_false_for_other_provider():
    agent = _bare_agent(provider="openai", model="gpt-4o",
                        base_url="http://localhost:30000/v1")
    assert agent._needs_deepseek_tool_reasoning() is False


def test_needs_kimi_tool_reasoning_by_provider():
    agent = _bare_agent(provider="kimi-coding", model="kimi-k2",
                        base_url="http://localhost:30000/v1")
    assert agent._needs_kimi_tool_reasoning() is True


def test_needs_mimo_tool_reasoning_by_provider():
    agent = _bare_agent(provider="xiaomi", model="mimo-1",
                        base_url="http://localhost:30000/v1")
    assert agent._needs_mimo_tool_reasoning() is True


def test_needs_thinking_reasoning_pad_or_chain():
    agent = _bare_agent(provider="deepseek", model="deepseek-v4",
                        base_url="http://localhost:30000/v1")
    assert agent._needs_thinking_reasoning_pad() is True


def test_needs_thinking_reasoning_pad_false_and_cached():
    agent = _bare_agent(provider="openai", model="gpt-4o",
                        base_url="http://localhost:30000/v1")
    assert agent._needs_thinking_reasoning_pad() is False
    # Second call must hit the per-instance cache keyed by
    # (provider, model, base_url) — the family predicates must NOT re-run.
    agent._needs_deepseek_tool_reasoning = lambda: (_ for _ in ()).throw(
        AssertionError("cache miss: deepseek predicate re-ran"))
    agent._needs_kimi_tool_reasoning = lambda: (_ for _ in ()).throw(
        AssertionError("cache miss: kimi predicate re-ran"))
    agent._needs_mimo_tool_reasoning = lambda: (_ for _ in ()).throw(
        AssertionError("cache miss: mimo predicate re-ran"))
    assert agent._needs_thinking_reasoning_pad() is False


def test_reasoning_echo_forwarders_delegate_verbatim(monkeypatch):
    copy_calls = {}

    def fake_copy(agent, source_msg, api_msg):
        copy_calls["self"] = agent
        copy_calls["args"] = (source_msg, api_msg)
        return None

    monkeypatch.setattr(agent_runtime_helpers, "copy_reasoning_content_for_api", fake_copy)
    agent = _bare_agent()
    src, dst = {"role": "assistant"}, {"role": "assistant"}
    assert agent._copy_reasoning_content_for_api(src, dst) is None
    assert copy_calls["self"] is agent
    assert copy_calls["args"] == (src, dst)

    reapply_calls = {}

    def fake_reapply(agent, api_messages):
        reapply_calls["self"] = agent
        reapply_calls["args"] = api_messages
        return 3

    monkeypatch.setattr(agent_runtime_helpers, "reapply_reasoning_echo_for_provider", fake_reapply)
    msgs = [{"role": "user", "content": "x"}]
    assert agent._reapply_reasoning_echo_for_provider(msgs) == 3
    assert reapply_calls["self"] is agent
    assert reapply_calls["args"] is msgs


# ---------------------------------------------------------------------------
# c16 / ToolExecutionMixin — tool-call execution dispatch
# ---------------------------------------------------------------------------

def test_wrap_verbose_label_and_preserves_line_breaks():
    out = ToolExecutionMixin._wrap_verbose("OUT", "hello\nworld")
    assert out.startswith("     OUT")
    assert "\n     world" in out
    assert out.count("\n") == 1


def test_invoke_tool_forwarder_delegates_verbatim(monkeypatch):
    invoke_calls = {}

    def fake_invoke(self, function_name, function_args, effective_task_id,
                    tool_call_id, messages, pre_tool_block_checked,
                    skip_tool_request_middleware, tool_request_middleware_trace,
                    skip_tool_execution_middleware):
        invoke_calls["self"] = self
        invoke_calls["args"] = (function_name, function_args, effective_task_id)
        return "tool-result"

    monkeypatch.setattr(agent_runtime_helpers, "invoke_tool", fake_invoke)
    agent = _bare_agent()
    out = agent._invoke_tool("read_file", {"path": "/tmp/x"}, "task-1")
    assert out == "tool-result"
    assert invoke_calls["self"] is agent
    assert invoke_calls["args"] == ("read_file", {"path": "/tmp/x"}, "task-1")


def test_execute_tool_calls_concurrent_forwarder(monkeypatch):
    calls = {}

    def fake(self, assistant_message, messages, effective_task_id, api_call_count):
        calls["self"] = self
        calls["args"] = (assistant_message, messages, effective_task_id, api_call_count)
        return None

    monkeypatch.setattr(tool_executor, "execute_tool_calls_concurrent", fake)
    agent = _bare_agent()
    msg = types.SimpleNamespace(tool_calls=[])
    assert agent._execute_tool_calls_concurrent(msg, [], "task-1") is None
    assert calls["self"] is agent
    assert calls["args"] == (msg, [], "task-1", 0)


def test_execute_tool_calls_sequential_forwarder(monkeypatch):
    calls = {}

    def fake(self, assistant_message, messages, effective_task_id, api_call_count):
        calls["args"] = (assistant_message, messages, effective_task_id, api_call_count)
        return None

    monkeypatch.setattr(tool_executor, "execute_tool_calls_sequential", fake)
    agent = _bare_agent()
    msg = types.SimpleNamespace(tool_calls=[])
    assert agent._execute_tool_calls_sequential(msg, [], "task-1") is None
    assert calls["args"] == (msg, [], "task-1", 0)


def test_handle_max_iterations_forwarder(monkeypatch):
    calls = {}

    def fake(self, messages, api_call_count):
        calls["args"] = (messages, api_call_count)
        return "max-iterations-halt"

    monkeypatch.setattr(chat_completion_helpers, "handle_max_iterations", fake)
    agent = _bare_agent()
    msgs = [{"role": "assistant", "content": "x"}]
    assert agent._handle_max_iterations(msgs, 7) == "max-iterations-halt"
    assert calls["args"] == (msgs, 7)
