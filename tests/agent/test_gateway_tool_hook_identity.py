"""Regression for #118717: stable gateway identity survives real transcript rotation."""

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture
def capture_plugin(tmp_path, monkeypatch):
    from hermes_cli import plugins

    home = tmp_path / "home"
    plugin = home / "plugins" / "identity_capture"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [identity_capture]\nterminal:\n  backend: local\n"
    )
    (plugin / "plugin.yaml").write_text("name: identity_capture\nversion: '0.1.0'\n")
    (plugin / "__init__.py").write_text('''
import threading

events = []
lock = threading.Lock()

def record(phase, kwargs):
    with lock:
        events.append((phase, dict(kwargs)))

def pre(**kwargs):
    record("pre_tool_call", kwargs)
    mode = kwargs["args"].get("identity_test_mode")
    if mode == "block":
        return {"action": "block", "message": "fixture veto"}
    if mode == "raise":
        raise RuntimeError("fixture hook failure")
    if mode == "malformed":
        return {"action": "block", "message": []}

def post(**kwargs):
    record("post_tool_call", kwargs)
    if kwargs["args"].get("identity_test_mode") == "raise":
        raise RuntimeError("fixture post failure")

def register(ctx):
    ctx.register_hook("pre_tool_call", pre)
    ctx.register_hook("post_tool_call", post)
''')
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()
    assert len(manager._hooks["pre_tool_call"]) == 1
    return manager._hooks["pre_tool_call"][0].__globals__["events"]


def make_agent(db, key=None):
    return AIAgent(
        api_key="fixture-key", base_url="https://example.invalid/v1", model="fixture/model",
        quiet_mode=True, session_db=db, session_id="fixture-transcript",
        gateway_session_key=key, enabled_toolsets=["file", "todo"],
        skip_context_files=True, skip_memory=True,
    )


def execute(agent, path, calls):
    message = SimpleNamespace(content="", tool_calls=[
        SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
            name=name, arguments=json.dumps(args),
        )) for call_id, name, args in calls
    ])
    messages = []
    getattr(agent, f"_execute_tool_calls_{path}")(message, messages, "fixture-task")
    assert Counter(m["tool_call_id"] for m in messages) == Counter(c[0] for c in calls)
    return messages


def assert_hooks(events, calls, key, session_id):
    expected = Counter((phase, call[0]) for call in calls
                       for phase in ("pre_tool_call", "post_tool_call"))
    assert Counter((phase, kw["tool_call_id"]) for phase, kw in events) == expected
    assert all(kw["gateway_session_key"] == (key or "") for _, kw in events)
    assert all(kw["session_id"] == session_id for _, kw in events)
    assert all(kw["task_id"] == "fixture-task" for _, kw in events)


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_gateway_key_survives_session_rotation(path, tmp_path, capture_plugin):
    key = "agent:default:discord:dm:fixture"
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = make_agent(db, key)
    sample = tmp_path / "sample.txt"
    sample.write_text("fixture content")
    second_sample = tmp_path / "sample-2.txt"
    second_sample.write_text("fixture content")

    def calls(prefix):
        # Two registry tools exercise the actual parallel worker pool, plus an inline tool.
        return [(prefix + "-read-1", "read_file", {"path": str(sample)}),
                (prefix + "-read-2", "read_file", {"path": str(second_sample)}),
                (prefix + "-inline", "todo", {"action": "list"})]

    try:
        before = agent.session_id
        first = calls("before")
        results = execute(agent, path, first)
        assert all("fixture content" in m["content"] for m in results[:2])
        assert_hooks(capture_plugin, first, key, before)
        capture_plugin.clear()

        # Only the remote summary is stubbed: compression, child-session persistence,
        # lineage notification and adoption run through the real runtime and SessionDB.
        compressor = MagicMock()
        compressor.compress.return_value = [{"role": "user", "content": "[CONTEXT COMPACTION] fixture"}]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor
        agent._compression_feasibility_checked = True
        agent.compression_in_place = False
        agent._compress_context([{"role": "user", "content": "fixture history"}], "sys", approx_tokens=10000)
        after = agent.session_id
        assert after != before
        assert db.get_session(after)["parent_session_id"] == before
        assert agent._gateway_session_key == key
        second = calls("after")
        execute(agent, path, second)
        assert_hooks(capture_plugin, second, key, after)
        capture_plugin.clear()

        # A fresh non-gateway agent in the same plugin/process must never inherit the key.
        control = make_agent(None)
        try:
            ordinary = calls("control")
            execute(control, path, ordinary)
            assert_hooks(capture_plugin, ordinary, None, control.session_id)
        finally:
            control.close()
    finally:
        agent.close()
        db.close()


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
@pytest.mark.parametrize("mode", ["normal", "block", "raise", "malformed", "tool-error", "execution-error"])
def test_terminal_hooks_keep_gateway_key_once(path, mode, tmp_path, capture_plugin, monkeypatch):
    key = "agent:default:discord:dm:fixture"
    agent = make_agent(None, key)
    sample = tmp_path / "sample.txt"
    if mode != "tool-error":
        sample.write_text("fixture content")
    if mode == "execution-error":
        def fail_dispatch(*args, **kwargs):
            raise RuntimeError("fixture dispatch failure")
        monkeypatch.setattr("model_tools.registry.dispatch", fail_dispatch)
    calls = [("terminal", "read_file", {"path": str(sample), "identity_test_mode": mode})]
    try:
        results = execute(agent, path, calls)
        assert_hooks(capture_plugin, calls, key, agent.session_id)
        post = next(kw for phase, kw in capture_plugin if phase == "post_tool_call")
        if mode in {"block", "raise"}:
            # Current main fails closed when a policy callback raises.
            assert post["status"] == "blocked"
            expected = "fixture veto" if mode == "block" else "fixture hook failure"
            assert expected in results[0]["content"]
        elif mode in {"tool-error", "execution-error"}:
            assert post["status"] == "error"
        else:
            assert "fixture content" in results[0]["content"]
    finally:
        agent.close()


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_cancelled_tool_keeps_gateway_key_once(path, tmp_path, capture_plugin):
    key = "agent:default:discord:dm:fixture"
    agent = make_agent(None, key)
    calls = [("cancel-1", "read_file", {"path": str(tmp_path / "never-read")}),
             ("cancel-2", "todo", {"action": "list"})]
    try:
        agent.interrupt()
        execute(agent, path, calls)
        assert Counter((phase, kw["tool_call_id"]) for phase, kw in capture_plugin) == Counter(
            ("post_tool_call", call[0]) for call in calls
        )
        assert all(kw["gateway_session_key"] == key and kw["status"] == "cancelled"
                   and kw["session_id"] == agent.session_id for _, kw in capture_plugin)
    finally:
        agent.clear_interrupt()
        agent.close()
