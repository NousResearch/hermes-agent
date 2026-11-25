"""A ``tool_execution`` middleware that runs a call somewhere else (another machine, a sandbox) must get Hermes's
own authorization for it first, through ``hermes_cli.middleware.authorize_tool_call``, and its result must get the
same terminal ``post_tool_call`` and ``transform_tool_result`` as a local result."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins
from hermes_cli.middleware import authorize_tool_call
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": n,
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _call(name, args, call_id=None):
    return SimpleNamespace(id=call_id or f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _agent(*tools):
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs(*tools)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, platform="cli")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


@pytest.fixture
def manager(monkeypatch):
    """An empty plugin manager the test fills with middleware and hooks (never the real plugins)."""
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager.posts = []
    manager._hooks.setdefault("post_tool_call", []).append(
        lambda tool_name="", result=None, status=None, **kw: manager.posts.append((tool_name, result, status)))
    return manager


def _run(agent, mode, *calls):
    messages = []
    msg = SimpleNamespace(content="", tool_calls=list(calls))
    getattr(agent, f"_execute_tool_calls_{mode}")(msg, messages, "task-1")
    return [m["content"] for m in messages if m.get("role") == "tool"]


def _remote(seen):
    """A middleware that authorizes the call through Hermes and runs it 'remotely' (never calls next_call)."""
    def middleware(tool_name="", args=None, authorization=None, **kw):
        seen.append(("authorization", authorization))
        decision = authorize_tool_call()
        seen.append(("decision", decision.allowed, decision.args))
        if not decision.allowed:
            return decision.result
        return f"REMOTE {tool_name} {json.dumps(decision.args, sort_keys=True)}"
    return middleware


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_a_blocked_call_never_runs_remotely_and_reports_once(manager, mode):
    seen = []
    manager._middleware["tool_execution"] = [_remote(seen)]
    agent = _agent("web_search")
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=("host policy: no", None)),
        patch("model_tools.handle_function_call", return_value="LOCAL") as local,
    ):
        results = _run(agent, mode, _call("web_search", {"query": "x"}))
    local.assert_not_called()
    assert ("authorization", "pending") in seen
    assert ("decision", False, {"query": "x"}) in seen
    assert "host policy: no" in results[0]
    assert [p[2] for p in manager.posts] == ["blocked"]


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_an_authorized_remote_result_gets_one_post_hook_and_one_transform(manager, mode):
    seen = []
    manager._middleware["tool_execution"] = [_remote(seen)]
    manager._hooks.setdefault("transform_tool_result", []).append(
        lambda tool_name="", result="", **kw: f"{result} [transformed]")
    agent = _agent("web_search")
    modify = (None, {"query": "modified"})
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=modify) as hooks,
        patch("model_tools.handle_function_call", return_value="LOCAL") as local,
    ):
        results = _run(agent, mode, _call("web_search", {"query": "x"}))
    local.assert_not_called()
    hooks.assert_called_once()
    assert len(results) == 1 and results[0].count('REMOTE web_search {"query": "modified"} [transformed]') == 1
    assert results[0].count("[transformed]") == 1     # once, inside the untrusted-content envelope web results get
    assert [(p[0], p[1]) for p in manager.posts] == [("web_search", 'REMOTE web_search {"query": "modified"}')]


def test_next_call_after_authorize_reuses_the_decision_and_refuses_changed_args(manager):
    calls = []

    def local_after_authorize(args=None, next_call=None, **kw):
        decision = authorize_tool_call()
        return next_call(decision.args)

    manager._middleware["tool_execution"] = [local_after_authorize]
    agent = _agent("web_search")
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)) as hooks,
        patch("model_tools.handle_function_call", side_effect=lambda *a, **k: calls.append(a) or "LOCAL"),
    ):
        results = _run(agent, "sequential", _call("web_search", {"query": "x"}))
    assert results == ["LOCAL"]
    assert len(calls) == 1
    hooks.assert_called_once()                      # authorized once, not again by the local dispatch

    def changes_args(args=None, next_call=None, **kw):
        authorize_tool_call()
        return next_call({"query": "something else"})

    manager._middleware["tool_execution"] = [changes_args]
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)), \
            patch("model_tools.handle_function_call", return_value="LOCAL") as local:
        results = _run(agent, "sequential", _call("web_search", {"query": "x"}))
    local.assert_not_called()
    assert "authorized" in results[0]


def test_handle_function_call_tells_middleware_authorization_is_done(manager):
    seen = []
    manager._middleware["tool_execution"] = [_remote(seen)]
    import model_tools
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)) as hooks:
        out = model_tools.handle_function_call("web_search", {"query": "x"}, task_id="t1", session_id="s1")
    assert ("authorization", "done") in seen
    assert ("decision", True, {"query": "x"}) in seen
    hooks.assert_called_once()                      # by handle_function_call itself, before the middleware
    assert out.startswith("REMOTE web_search")


def test_authorize_outside_a_tool_execution_middleware_is_an_error():
    with pytest.raises(RuntimeError, match="tool_execution middleware"):
        authorize_tool_call()


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_a_short_circuit_result_gets_transform_tool_result_like_a_local_one(manager, mode):
    """Before: in the agent loop a registry tool's transform ran inside the dispatch, so a middleware that returned
    without next_call skipped it. The executor now applies it, once."""
    manager._middleware["tool_execution"] = [lambda **kw: "CACHED"]
    manager._hooks.setdefault("transform_tool_result", []).append(lambda result="", **kw: f"{result} [t]")
    agent = _agent("lookup")
    with patch("model_tools.handle_function_call", return_value="LOCAL") as local:
        results = _run(agent, mode, _call("lookup", {"q": 1}))
    local.assert_not_called()
    assert results == ["CACHED [t]"]
    assert [p[1] for p in manager.posts] == ["CACHED"]
