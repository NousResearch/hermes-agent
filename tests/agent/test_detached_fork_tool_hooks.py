"""Tool hooks of a detached fork carry ``detached=True`` (#133603).

A detached fork (``build_cache_parity_fork``: background review, ``/btw``, plugin session forks)
shares its parent's ``session_id``, so a status/observer plugin could not tell the fork's tool calls
from the live turn's. Its ``pre_tool_call`` / ``post_tool_call`` / ``transform_tool_result`` hooks
now carry ``detached=True``; the hooks still fire and a plugin ``block`` still blocks. A live
agent's hooks carry no ``detached`` kwarg at all.

The flag is derived from the agent object (``_persist_disabled``), not from thread state; the
concurrent test still runs the batch on real pool workers (``propagate_context_to_thread``).
"""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent

PLUGIN = '''
import json
import threading

events = []
gate = {"barrier": None}


def _record(kind, kw):
    events.append({"kind": kind, "tool_call_id": kw.get("tool_call_id"), "session_id": kw.get("session_id"),
                   "status": kw.get("status"), "keys": frozenset(kw), "detached": kw.get("detached", "absent"),
                   "thread": threading.get_ident()})


def handle(args, **kw):
    events.append({"kind": "handler", "thread": threading.get_ident()})
    if gate["barrier"] is not None:
        gate["barrier"].wait()
    if args.get("fail"):
        raise RuntimeError("probe failure")
    return json.dumps({"ok": True})


def pre(**kw):
    _record("pre_tool_call", kw)
    if (kw.get("args") or {}).get("block"):
        return {"action": "block", "message": "blocked by fork_probe policy"}
    return None


def post(**kw):
    _record("post_tool_call", kw)


def transform(**kw):
    _record("transform_tool_result", kw)


def register(ctx):
    ctx.register_tool(name="fork_probe", toolset="fork_probe", handler=handle, schema={
        "name": "fork_probe", "description": "probe",
        "parameters": {"type": "object", "properties": {"block": {"type": "boolean"}, "fail": {"type": "boolean"},
                                                         "n": {"type": "integer"}}}})
    ctx.register_hook("pre_tool_call", pre)
    ctx.register_hook("post_tool_call", post)
    ctx.register_hook("transform_tool_result", transform)
'''
_TOOL_HOOKS = ("pre_tool_call", "post_tool_call", "transform_tool_result")


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """The fork_probe plugin loaded through real discovery; hooks run inline on the caller thread."""
    from hermes_cli import plugins
    from tools.registry import registry

    home = tmp_path / "home"
    plugin = home / "plugins" / "fork_probe"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # hook_callback_timeout 0: callbacks run on the dispatching thread, so a recorded thread id is
    # the tool worker's, not a hook-timeout worker's.
    (home / "config.yaml").write_text("plugins:\n  enabled: [fork_probe]\n  hook_callback_timeout: 0\n")
    (plugin / "plugin.yaml").write_text("name: fork_probe\nversion: '0.1.0'\n")
    (plugin / "__init__.py").write_text(PLUGIN)
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()
    module_globals = manager._hooks["pre_tool_call"][0].__globals__
    yield SimpleNamespace(events=module_globals["events"], gate=module_globals["gate"])
    registry.deregister("fork_probe")


@pytest.fixture
def agents(probe):
    """``(parent, fork)``: a live agent and its detached cache-parity fork."""
    from agent.background_review import build_cache_parity_fork, release_cache_parity_fork

    parent = AIAgent(api_key="k", base_url="https://example.invalid/v1", model="fixture/model", quiet_mode=True,
                     session_db=None, session_id="parent-session", enabled_toolsets=["fork_probe"],
                     skip_context_files=True, skip_memory=True)
    fork, _rt, _routed = build_cache_parity_fork(parent, {}, max_iterations=2)
    assert fork._persist_disabled is True and parent._persist_disabled is False
    try:
        yield parent, fork
    finally:
        release_cache_parity_fork(parent, fork, task="test")
        parent.close()


def _calls(*arg_sets):
    return [SimpleNamespace(id=f"call-{i}", type="function",
                            function=SimpleNamespace(name="fork_probe", arguments=json.dumps(args)))
            for i, args in enumerate(arg_sets)]


def _run(agent, path, *arg_sets):
    messages = []
    getattr(agent, f"_execute_tool_calls_{path}")(
        SimpleNamespace(content="", tool_calls=_calls(*arg_sets)), messages, "task")
    return [m for m in messages if m.get("role") == "tool"]


def _hook_events(probe):
    return [e for e in probe.events if e["kind"] in _TOOL_HOOKS]


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
@pytest.mark.parametrize("outcome,args", [("ok", {}), ("error", {"fail": True})])
def test_fork_tool_hooks_carry_detached(path, outcome, args, agents, probe):
    _parent, fork = agents
    _run(fork, path, args)
    events = _hook_events(probe)
    assert sorted(e["kind"] for e in events) == sorted(_TOOL_HOOKS)
    assert [e["detached"] for e in events] == [True] * 3
    # The fork shares its parent's session_id: that is why the flag is needed at all.
    assert {e["session_id"] for e in events} == {"parent-session"}
    assert [e["status"] for e in events if e["kind"] == "post_tool_call"] == [outcome]


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_plugin_block_still_blocks_a_fork_call(path, agents, probe):
    _parent, fork = agents
    results = _run(fork, path, {"block": True})
    assert "blocked by fork_probe policy" in results[0]["content"]
    assert not [e for e in probe.events if e["kind"] == "handler"]  # the tool never ran
    events = _hook_events(probe)
    assert [(e["kind"], e["detached"]) for e in events] == [("pre_tool_call", True), ("post_tool_call", True)]
    assert events[1]["status"] == "blocked"


def test_direct_invoke_tool_pre_hook_blocks_a_fork_call(agents, probe):
    """``invoke_tool`` without a prior policy pass builds the pre_tool_call ids itself."""
    _parent, fork = agents
    result = fork._invoke_tool("fork_probe", {"block": True}, "task", "call-direct")
    assert "blocked by fork_probe policy" in result
    assert [(e["kind"], e["detached"], e["status"]) for e in _hook_events(probe)] == [
        ("pre_tool_call", True, None), ("post_tool_call", True, "blocked")]


def test_cancelled_fork_call_reports_detached(agents, probe):
    _parent, fork = agents
    with (patch("model_tools.handle_function_call", side_effect=KeyboardInterrupt),
          patch("run_agent._set_interrupt"), patch("agent.interrupt_control._set_interrupt")):
        _run(fork, "concurrent", {}, {})
    post = [e for e in probe.events if e["kind"] == "post_tool_call"]
    assert [(e["status"], e["detached"]) for e in post] == [("cancelled", True)] * 2


def test_concurrent_fork_batch_flags_every_worker_event(agents, probe):
    """Three calls meet at a barrier, so they provably run at once on pool workers; every pre/post/transform
    event those workers emit carries ``detached=True``. The flag comes from the agent object, not from
    thread state, but this runs the real worker path (``propagate_context_to_thread``)."""
    _parent, fork = agents
    probe.gate["barrier"] = threading.Barrier(3, timeout=10)
    # Distinct args: three identical calls would trip the repeated-call guardrail note.
    results = _run(fork, "concurrent", {"n": 0}, {"n": 1}, {"n": 2})
    assert [json.loads(r["content"]) for r in results] == [{"ok": True}] * 3
    handler_threads = {e["thread"] for e in probe.events if e["kind"] == "handler"}
    assert len(handler_threads) == 3 and threading.get_ident() not in handler_threads
    events = _hook_events(probe)
    assert sorted((e["kind"], e["tool_call_id"]) for e in events) == sorted(
        (kind, f"call-{i}") for kind in _TOOL_HOOKS for i in range(3))
    assert all(e["detached"] is True for e in events)
    assert {e["thread"] for e in events} <= handler_threads


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_live_agent_hooks_carry_no_detached_kwarg(path, agents, probe):
    """A normal turn's tool hooks are unchanged: no ``detached`` key, and the fork's hooks differ from
    them by that one key only."""
    parent, fork = agents
    _run(parent, path, {}, {"block": True})
    live = _hook_events(probe)
    assert len(live) == 5 and all(e["detached"] == "absent" for e in live)
    probe.events.clear()
    _run(fork, path, {}, {"block": True})
    forked = _hook_events(probe)

    def shapes(events, drop=frozenset()):
        return sorted((e["kind"], e["tool_call_id"], tuple(sorted(e["keys"] - drop))) for e in events)

    assert shapes(forked, {"detached"}) == shapes(live)
    assert all("detached" in e["keys"] for e in forked)
