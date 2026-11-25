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


# ---- gauntlet: the tool-body checks, refusals that stay refusals, no fall-back after authorizing ----

def test_a_remote_terminal_call_gets_hermess_command_gate_and_a_denial_never_runs(manager):
    """The terminal's dangerous-command gate lives in the tool body; authorize_tool_call runs it (Hermes's own code)."""
    seen = []
    manager._middleware["tool_execution"] = [_remote(seen)]
    agent = _agent("terminal")
    denied = {"approved": False, "message": "Command denied by the user", "description": "recursive delete"}
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)),
        patch("tools.terminal_tool._check_all_guards", return_value=denied) as gate,
        patch("model_tools.handle_function_call", return_value="LOCAL") as local,
    ):
        results = _run(agent, "sequential", _call("terminal", {"command": "rm -rf build"}))
    gate.assert_called_once()
    local.assert_not_called()
    assert not any(isinstance(s, str) and s.startswith("REMOTE") for s in results)
    assert "Command denied by the user" in results[0]
    assert [p[2] for p in manager.posts] == ["blocked"]


def test_an_approved_terminal_call_dispatched_locally_is_not_asked_twice(manager):
    def authorize_then_local(args=None, next_call=None, **kw):
        decision = authorize_tool_call()
        return next_call(decision.args)

    manager._middleware["tool_execution"] = [authorize_then_local]
    agent = _agent("terminal")
    from agent.tool_authorization import consume_authorized_terminal
    approved = {"approved": True, "user_approved": True, "description": "recursive delete"}
    handed = []

    def local_terminal(name, args, *a, **k):
        from tools import terminal_tool as tt
        config = tt._get_env_config()
        handed.append(consume_authorized_terminal(args["command"], config["env_type"],
                                                  tt._docker_has_host_access(config), k.get("tool_call_id") or ""))
        return "LOCAL"

    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)),
        patch("tools.terminal_tool._check_all_guards", return_value=approved) as gate,
        patch("model_tools.handle_function_call", side_effect=local_terminal),
    ):
        _run(agent, "sequential", _call("terminal", {"command": "rm -rf build"}, call_id="c-term"))
    gate.assert_called_once()                       # asked once, by authorize_tool_call
    assert handed == [approved]                     # the local terminal gets that answer instead of asking again


def test_acp_edit_approval_applies_to_a_remote_file_edit(manager):
    seen = []
    manager._middleware["tool_execution"] = [_remote(seen)]
    agent = _agent("write_file")
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)),
        patch("acp_adapter.edit_approval.maybe_require_edit_approval", return_value='{"error": "Edit approval denied"}'),
    ):
        results = _run(agent, "sequential", _call("write_file", {"path": "/x/a.txt", "content": "x"}))
    assert results == ['{"error": "Edit approval denied"}']
    assert [p[2] for p in manager.posts] == ["blocked"]


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_a_refusal_stays_a_refusal_when_the_middleware_raises_or_calls_next_call(manager, mode):
    def refused_then_raise(**kw):
        authorize_tool_call()
        raise RuntimeError("remote side broke")

    def refused_then_next_call(args=None, next_call=None, **kw):
        authorize_tool_call()
        return next_call(args)

    for middleware in (refused_then_raise, refused_then_next_call):
        manager.posts.clear()
        manager._middleware["tool_execution"] = [middleware]
        agent = _agent("web_search")
        with (
            patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=("host policy: no", None)),
            patch("model_tools.handle_function_call", return_value="LOCAL") as local,
        ):
            results = _run(agent, mode, _call("web_search", {"query": "x"}))
        local.assert_not_called()
        assert "host policy: no" in results[0]
        assert [p[2] for p in manager.posts] == ["blocked"], middleware.__name__


def test_a_call_whose_middleware_failed_after_authorizing_does_not_fall_back_to_running_here(manager):
    def allowed_then_raise(**kw):
        authorize_tool_call()
        raise RuntimeError("the remote machine went away")

    manager._middleware["tool_execution"] = [allowed_then_raise]
    agent = _agent("web_search")
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)),
        patch("model_tools.handle_function_call", return_value="LOCAL") as local,
    ):
        results = _run(agent, "sequential", _call("web_search", {"query": "x"}))
    local.assert_not_called()
    assert "was not run here" in results[0]


def test_a_repeat_with_other_arguments_is_refused_and_a_hook_cannot_reenter(manager):
    seen = []

    def two_asks(**kw):
        seen.append(authorize_tool_call({"query": "safe"}).allowed)
        seen.append(authorize_tool_call({"query": "DANGEROUS"}).allowed)
        return "REMOTE"

    def reentering_hook(*a, **k):
        try:
            authorize_tool_call()
        except RuntimeError as exc:
            seen.append(str(exc)[:40])
        return (None, None)

    manager._middleware["tool_execution"] = [two_asks]
    agent = _agent("web_search")
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=reentering_hook):
        _run(agent, "sequential", _call("web_search", {"query": "safe"}))
    assert seen[0].startswith("authorize_tool_call() cannot be called")    # the hook re-entry raised, no deadlock
    assert seen[1:] == [True, False]


def test_done_path_changed_arguments_are_refused_and_never_run_here(manager):
    def rewrites(args=None, next_call=None, **kw):
        return next_call({"query": "rewritten-unchecked"})

    def authorizes(args=None, **kw):
        decision = authorize_tool_call(args)
        if not decision.allowed:
            raise RuntimeError("cannot run it")      # the chain's fall-through must not run it either
        return "REMOTE"

    manager._middleware["tool_execution"] = [rewrites, authorizes]
    import model_tools
    ran = []
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)), \
            patch("tools.registry.registry.dispatch", side_effect=lambda *a, **k: ran.append(a) or "LOCAL"):
        out = model_tools.handle_function_call("web_search", {"query": "x"}, task_id="t1", session_id="s1")
    assert ran == []
    assert "not run" in out
