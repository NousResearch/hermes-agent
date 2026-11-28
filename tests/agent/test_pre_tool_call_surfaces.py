"""Every tool surface runs ``pre_tool_call`` before a tool and honors a fail-closed guard (#160512).

The report: a fail-closed guard plugin that could not start (its external binary was missing)
let ``terminal`` run. Each surface below drives a real ``AIAgent`` (scripted model client, real
tool executor, real local terminal) or a real ``handle_function_call`` caller, with the guard
loaded through real plugin discovery from a temp ``HERMES_HOME``:

1. TUI gateway / Desktop (``source=gui`` -> ``platform="desktop"``)
2. Gateway worker / canonical local session (``platform="local"``)
3. Cron (``platform="cron"``)
4. ACP adapter (``platform="acp"``)
5. API server (``platform="api_server"``)
6. Platform adapters (``platform="telegram"``)
7. CLI / direct ``AIAgent`` paths: sequential, concurrent, Desktop terminal batch, ``invoke_tool``
8. ``model_tools.handle_function_call`` callers: direct, the execute_code RPC dispatch

``loaded`` is a healthy guard that records each call and blocks ``terminal``; ``broken`` is the
reported failure (a ``fail_closed`` guard whose ``register()`` raises because its binary is missing).
"""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins as plugins_mod

pytestmark = pytest.mark.platforms("posix")

_GUARD_BODIES = {
    "loaded": (
        "import json, os\n"
        "def _pre(tool_name='', **kw):\n"
        "    with open(os.environ['GUARD_TRACE_160512'], 'a', encoding='utf-8') as f:\n"
        "        f.write(json.dumps(tool_name) + '\\n')\n"
        "    if tool_name in ('terminal', 'read_file'):\n"
        "        return {'action': 'block', 'message': 'BLOCKED-BY-GUARD'}\n"
        "def register(ctx):\n"
        "    ctx.register_hook('pre_tool_call', _pre)\n"
    ),
    "broken": (
        "import shutil\n"
        "def register(ctx):\n"
        "    if shutil.which('guard-binary-160512-not-installed') is None:\n"
        "        raise RuntimeError('guard-binary-160512-not-installed not found on PATH')\n"
    ),
}


@pytest.fixture(params=sorted(_GUARD_BODIES))
def guard(request, tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    pdir = home / "plugins" / "guard"
    pdir.mkdir(parents=True)
    (home / ".env").write_text("", encoding="utf-8")
    (pdir / "plugin.yaml").write_text(
        "name: guard\nversion: '0.0.1'\ndescription: g\nprovides_hooks: [pre_tool_call]\nfail_closed: true\n",
        encoding="utf-8")
    (pdir / "__init__.py").write_text(_GUARD_BODIES[request.param], encoding="utf-8")
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["guard"]}}), encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GUARD_TRACE_160512", str(trace))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "off")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    plugins_mod._reset_plugin_managers_for_tests()
    yield SimpleNamespace(mode=request.param, tmp=tmp_path, trace=trace)
    plugins_mod._reset_plugin_managers_for_tests()
    from tools.terminal_tool_lifecycle import cleanup_vm
    cleanup_vm("default")


def _hooked_tools(guard):
    return [json.loads(x) for x in guard.trace.read_text(encoding="utf-8").splitlines()] if guard.trace.exists() else []


def _assert_blocked(guard, text, tool="terminal"):
    if guard.mode == "loaded":
        assert tool in _hooked_tools(guard), "pre_tool_call never ran for the tool"
        assert "BLOCKED-BY-GUARD" in text
    else:
        assert "guard" in text and "not running" in text, text


def _touch(guard, name):
    marker = guard.tmp / name
    return marker, {"command": f"touch {shlex.quote(str(marker))}"}


def _call(call_id, name, args):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _response(calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="" if calls else "done", tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")], model="test/model", usage=None)


def _agent(platform):
    from run_agent import AIAgent
    from tools.file_tools import READ_FILE_SCHEMA
    from tools.terminal_tool import TERMINAL_SCHEMA

    with (
        patch("model_tools.get_tool_definitions", return_value=[
            {"type": "function", "function": schema} for schema in (TERMINAL_SCHEMA, READ_FILE_SCHEMA)]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=3,
                        platform=platform)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Use the available tools."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_turn(agent, calls):
    agent.client.chat.completions.create.side_effect = [_response(calls), _response()]
    result = agent.run_conversation("Run the tools.")
    return [m["content"] for m in result["messages"] if m["role"] == "tool"]


SURFACE_PLATFORMS = {
    "tui_gateway_desktop": "desktop", "gateway_worker": "local", "cron": "cron", "acp": "acp",
    "api_server": "api_server", "platform_adapter": "telegram", "cli": "cli",
}


@pytest.mark.parametrize("surface", sorted(SURFACE_PLATFORMS))
def test_surface_runs_pre_tool_call_before_terminal(guard, surface):
    marker, args = _touch(guard, f"{surface}-ran")
    results = _run_turn(_agent(SURFACE_PLATFORMS[surface]), [_call("c1", "terminal", args)])
    assert not marker.exists(), f"{surface}: terminal ran past the guard"
    assert len(results) == 1
    _assert_blocked(guard, results[0])


def test_concurrent_path_runs_pre_tool_call_for_each_call(guard):
    sources = [guard.tmp / f"secret-{i}.txt" for i in range(2)]
    for source in sources:
        source.write_text("SECRET-CONTENT", encoding="utf-8")
    calls = [_call(f"r{i}", "read_file", {"path": str(source)}) for i, source in enumerate(sources)]
    from agent import tool_executor

    with patch.object(tool_executor, "execute_tool_calls_concurrent",
                      wraps=tool_executor.execute_tool_calls_concurrent) as concurrent:
        results = _run_turn(_agent("cli"), calls)
    assert concurrent.called, "two read-only calls should take the concurrent path"
    assert len(results) == 2
    for text in results:
        assert "SECRET-CONTENT" not in text
        _assert_blocked(guard, text, tool="read_file")


def test_desktop_terminal_batch_runs_pre_tool_call_for_each_command(guard, monkeypatch):
    from agent import terminal_approval_batch as batch_mod
    from gateway.session_context import clear_session_vars
    from tools import approval
    from tui_gateway import server

    agent = _agent("desktop")
    key = "desktop-batch-160512"
    monkeypatch.setattr(server, "_sessions", {key: {
        "session_key": key, "source": "desktop", "agent": agent, "cwd": str(guard.tmp)}})
    started = []
    real_start = batch_mod._TerminalBatch.start
    monkeypatch.setattr(batch_mod._TerminalBatch, "start", lambda self: started.append(1) or real_start(self))
    approval.register_gateway_notify(key, lambda _event: None)
    tokens = server._set_session_context(key)
    try:
        (m1, a1), (m2, a2) = _touch(guard, "batch-1"), _touch(guard, "batch-2")
        results = _run_turn(agent, [_call("t1", "terminal", a1), _call("t2", "terminal", a2)])
    finally:
        clear_session_vars(tokens)
        approval.unregister_gateway_notify(key)
    assert started, "two Desktop terminal calls should take the terminal-batch path"
    assert not m1.exists() and not m2.exists(), "a Desktop batch command ran past the guard"
    assert len(results) == 2
    for text in results:
        _assert_blocked(guard, text)


def test_invoke_tool_runs_pre_tool_call(guard):
    marker, args = _touch(guard, "invoke-ran")
    agent = _agent("cli")
    result = agent._invoke_tool("terminal", args, "task", tool_call_id="c1")
    assert not marker.exists()
    _assert_blocked(guard, result)


def _direct(name, args):
    from model_tools import handle_function_call
    return handle_function_call(name, args, task_id="task", tool_call_id="c1")


def _execute_code_rpc(name, args):
    from tools.code_execution_rpc import _default_dispatch
    return _default_dispatch("task", session_id="s1")(name, args)


@pytest.mark.parametrize("caller", [_direct, _execute_code_rpc], ids=["direct", "execute_code_rpc"])
def test_handle_function_call_callers_run_pre_tool_call(guard, caller):
    marker, args = _touch(guard, "hfc-ran")
    result = caller("terminal", args)
    assert not marker.exists(), "handle_function_call ran terminal past the guard"
    _assert_blocked(guard, result)
