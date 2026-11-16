"""A plugin session fork holds only names that dispatch to the plugin's own registry handler.

Some names are served before the tool registry (``agent.inline_tool_executors.pre_registry_route``):
connector names, the session's memory-provider and context-engine tools, agent-level tools,
legacy aliases and the Tool Search bridge. A plugin that registers one of them still never runs
its handler there, so a fork may not hold it: the request is refused with the token unspent, and
a job built past that check is refused at dispatch, on the sequential and the concurrent path,
without reaching the connector, memory provider or context engine.
"""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import background_review, memory_provider, plugin_session_fork
from agent.inline_tool_executors import pre_registry_route, resolve_invoke_tool_executor
from agent.memory_manager import MemoryManager
from agent.side_question_lease import issue_side_question_token
from tests.agent import test_plugin_session_fork as _base
from tests.agent.test_plugin_session_fork import (
    PLUGIN, _Agent, _agent, _ctx, _install_plugin, _job_holding, _load, _settled, _tool,
)

forks = _base.forks  # the scripted fork builder, shared with the main fork tests

CONNECTOR, MEMORY, CONTEXT = "connectors__acme__lookup", "mem_probe_recall", "ctx_probe_expand"
ROUTED = (CONNECTOR, MEMORY, CONTEXT)

_PLUGIN = '''
import json
import sys

PROBE = sys.modules["_fork_routes_probe"]


def register(ctx):
    for name in ("cw_offer", "connectors__acme__lookup", "mem_probe_recall", "ctx_probe_expand"):
        def handle(args, _name=name, **kwargs):
            PROBE.seen.append(("plugin", _name))
            return json.dumps({"route": "plugin"})
        ctx.register_tool(name, "cw", {"name": name, "description": name, "parameters": {
            "type": "object", "properties": {}}}, handle)
'''


class _SpyMemory(MemoryManager):
    """A memory provider serving ``MEMORY``."""

    def __init__(self, seen):
        super().__init__()
        self._seen = seen

    def has_tool(self, tool_name):
        return tool_name == MEMORY

    def handle_tool_call(self, tool_name, args, **kwargs):
        self._seen.append(("memory", tool_name))
        return json.dumps({"route": "memory"})


@pytest.fixture
def probe(monkeypatch):
    """``wisdom`` (granted ``llm.session_fork``) registers ``cw_offer`` plus one name per pre-registry
    route; ``probe.seen`` records which handler each call reached."""
    from tools import connectors

    state = SimpleNamespace(seen=[])
    monkeypatch.setitem(sys.modules, "_fork_routes_probe", state)
    monkeypatch.setattr(connectors, "dispatch_connector_call",
                        lambda name, *a, **k: state.seen.append(("connector", name)) or json.dumps({"route": "connector"}))
    _install_plugin(PLUGIN, declares=True, body=_PLUGIN)
    state.manager = _load([PLUGIN], granted=[PLUGIN])
    return state


def _wire(agent, seen):
    """Give ``agent`` a live memory provider serving MEMORY and a context engine serving CONTEXT."""
    agent._memory_manager = _SpyMemory(seen)
    agent._context_engine_tool_names = {CONTEXT}
    agent.context_compressor = SimpleNamespace(
        handle_tool_call=lambda name, args, **kw: seen.append(("context", name)) or json.dumps({"route": "context"}))
    return agent


def _session(**overrides):
    return _agent(tools=[_tool("cw_offer"), *(_tool(name) for name in ROUTED)], **overrides)


# ── request time: refused before the lease ───────────────────────────────────────────────

@pytest.mark.parametrize("name", ROUTED)
def test_a_name_served_before_the_registry_is_refused_at_request_and_the_token_stays_usable(forks, probe, name):
    parent = _wire(_session(), probe.seen)
    token = issue_side_question_token(parent, [], "turn-1")
    assert name in _ctx(probe.manager)._own_tool_names()
    assert _ctx(probe.manager).fork_session(token, "q", tools=["cw_offer", name]) is None
    assert forks.built == [] and probe.seen == []
    assert _settled(_ctx(probe.manager).fork_session(token, "q", tools=["cw_offer"])).outcome == "ok"


def test_without_a_live_route_the_name_is_the_plugins_own(forks, probe):
    """Memory-provider and context-engine names are refused only while the session serves them."""
    parent = _session()
    token = issue_side_question_token(parent, [], "turn-1")
    assert _settled(_ctx(probe.manager).fork_session(token, "q", tools=["cw_offer", MEMORY, CONTEXT])).outcome == "ok"


# ── dispatch time: a prebuilt job, both tool paths ───────────────────────────────────────

def _dispatcher(seen):
    """A real session agent with a live memory provider and context engine."""
    from run_agent import AIAgent

    names = ("cw_offer", *ROUTED)
    with (
        patch("model_tools.get_tool_definitions", return_value=[_tool(n) for n in names]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._review_input_token_budget = 1000
    return _wire(agent, seen)


def _calls(path):
    return SimpleNamespace(content="", tool_calls=[
        SimpleNamespace(id=f"{path}-{i}", type="function", function=SimpleNamespace(name=name, arguments="{}"))
        for i, name in enumerate(("cw_offer", *ROUTED))])


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_a_prebuilt_job_cannot_reach_a_route_served_before_the_registry(forks, probe, monkeypatch, path):
    """The fork that dispatches serves MEMORY and CONTEXT from its own memory provider and context
    engine (the session stand-in serves neither). Its plugin's own tool runs; the three routed
    names are denied before any connector, memory or context handler, or the plugin's handler."""
    fork = _dispatcher(probe.seen)
    results: list = []

    def _turn(user_message, conversation_history=None):
        messages: list = []
        getattr(fork, f"_execute_tool_calls_{path}")(_calls(path), messages, "task-1")
        results.extend(m for m in messages if m.get("role") == "tool")
        return {"final_response": "done", "messages": []}

    monkeypatch.setattr(fork, "run_conversation", _turn)
    monkeypatch.setattr(background_review, "build_cache_parity_fork",
                        lambda parent, task_cfg, *, max_iterations, write_origin: (fork, {}, False))
    parent = _session()
    job = _job_holding(parent, ("cw_offer", *ROUTED), _ctx(probe.manager)._own_tool_names)
    plugin_session_fork._launch(job)
    assert _settled(job.handle).outcome == "ok"

    assert probe.seen == [("plugin", "cw_offer")]
    denied = {r["tool_call_id"]: str(r["content"]) for r in results}
    for i, name in enumerate(ROUTED, start=1):
        assert f"{name} was denied" in denied[f"{path}-{i}"], denied


@pytest.mark.parametrize(("path", "context_route"), [("sequential", "context"), ("concurrent", "plugin")])
def test_unfenced_the_same_calls_reach_the_route_the_predicate_names(probe, path, context_route):
    """Control for the test above: outside a fork the same dispatcher sends the connector name to the
    connector and the memory name to the memory provider on both paths, and the context-engine name
    to the context engine on the sequential path (the concurrent path hands it to the registry)."""
    fork = _dispatcher(probe.seen)
    getattr(fork, f"_execute_tool_calls_{path}")(_calls(path), [], "task-1")
    assert sorted(probe.seen) == sorted([("plugin", "cw_offer"), ("connector", CONNECTOR), ("memory", MEMORY),
                                         (context_route, CONTEXT)])


# ── the predicate is the dispatcher's own routing ────────────────────────────────────────

def test_the_route_predicate_names_where_the_dispatcher_sends_each_name(probe):
    """Each kind the predicate reports is where dispatch actually goes, and the allowlist drops it."""
    from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist
    from model_tools import handle_function_call

    agent = _wire(SimpleNamespace(_gateway_session_key=""), probe.seen)
    assert {name: pre_registry_route(name, agent) for name in (*ROUTED, "cw_offer")} == {
        CONNECTOR: "connector", MEMORY: "memory_provider", CONTEXT: "context_engine", "cw_offer": None}
    assert {pre_registry_route(n) for n in ("delegate_task", "cronjob", "tool_call")} == {"inline", "alias", "bridge"}

    # Unfenced, the connector name reaches the connector and the memory name the memory provider.
    handle_function_call(CONNECTOR, {})
    memory_executor = resolve_invoke_tool_executor(agent, MEMORY)
    memory_executor(agent, {}, SimpleNamespace())
    assert probe.seen == [("connector", CONNECTOR), ("memory", MEMORY)]

    allowed = plugin_session_fork.fork_tool_allowlist(_ctx(probe.manager)._own_tool_names(), agent)
    assert allowed - plugin_session_fork.FORK_SAFE_CORE_TOOLS == {"cw_offer"}
    probe.seen.clear()
    set_thread_tool_whitelist(set(allowed), "blocked {tool_name}")
    try:
        assert "blocked" in handle_function_call(CONNECTOR, {})
    finally:
        clear_thread_tool_whitelist()
    assert probe.seen == []


def test_a_live_turn_cancel_waits_for_a_failing_launch_to_settle():
    """A launch whose worker cannot start is settled once, by the launcher; a live-turn cancel that
    races it waits for that settlement and returns promptly after it."""
    import contextvars
    import weakref

    from agent.side_question_request import SideQuestionCapture, runtime_key
    from tests.waiting import wait_until

    parent = _Agent(session_id="sess", _current_turn_id="turn-1", _plugin_fork_jobs=set())
    capture = SideQuestionCapture(parent_ref=weakref.ref(parent), messages=[], system_prompt="sys",
                                  system_prompt_static=None, tools=[], session_id="sess", turn_id="turn-1",
                                  runtime=runtime_key(parent), context=contextvars.copy_context())
    run = background_review._BackgroundReviewRun()
    job = plugin_session_fork._ForkJob("probe", capture, "q", frozenset(), 1, None, {}, None, run)
    job.handle = plugin_session_fork.PluginForkHandle("probe", lambda: plugin_session_fork._cancel_job(job))
    entered, release, cancel_done = threading.Event(), threading.Event(), threading.Event()

    def _blocked_spawn(*args, **kwargs):
        entered.set()
        release.wait(5)
        raise RuntimeError("spawn failed")

    with patch.object(memory_provider, "spawn_context_thread", _blocked_spawn):
        launcher = threading.Thread(target=plugin_session_fork._launch, args=(job,))
        launcher.start()
        assert entered.wait(2)
        canceller = threading.Thread(target=lambda: (
            background_review.cancel_background_review_for_live_turn(parent), cancel_done.set()))
        canceller.start()
        wait_until(run.cancel_requested.is_set, timeout=5)  # the canceller fenced the job and awaits it
        waited = not cancel_done.is_set()
        release.set()
        launcher.join(2)
        canceller.join(2)
    assert waited and cancel_done.is_set()
    assert job.handle.outcome == "error:spawn" and run.request_done.is_set()
    assert job not in parent._plugin_fork_jobs
