"""Turns a plugin injects into a gateway session (``PluginContext.inject_message``).

A plugin's tool handler running inside its session fork sees ``turn_origin="plugin_fork:<id>"``,
the fork's context, a dispatch whitelist and auto-denied approvals. When it injects a message, the
turn the gateway starts for it runs from the gateway's own context, never the caller's: that turn's
handlers see none of the fork's state.

Nor does an injected turn supersede the session's running background review and plugin forks the
way a person's turn does: it lets them finish first, for a bounded wait an interrupt ends early.

Real plugin discovery, fork worker (``start_session_fork``), tool dispatch
(``model_tools.handle_function_call``), plugin injection API, GatewayRunner, DiscordAdapter and
SessionAuthority FIFO; the fork's model turn and the Discord transport are scripted.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from tests.waiting import async_wait_until

PLUGIN = "wisdom"
PROMPT = "Collective Wisdom: my skill may be worth sharing"

# Stands for everything else a caller's context carries (the fork copies the profile, approval and
# observability scopes of the turn its lease was issued in).
CALLER_SCOPED: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_caller_scoped", default=None)

_PLUGIN_BODY = '''
import json
import sys
import threading

from hermes_cli.plugins import get_pre_tool_call_directive
from tools.terminal_tool import _get_approval_callback


def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}


def register(ctx):
    probe = sys.modules["_inject_probe"]

    def offer(args, turn_origin=None, **kwargs):
        probe.calls.append({
            "turn_origin": turn_origin,
            "caller_scoped": probe.caller_scoped.get(),
            "approval": getattr(_get_approval_callback(), "__name__", None),
            "cw_other": get_pre_tool_call_directive("cw_other", {})[0],
        })
        if turn_origin is not None:  # inside the fork: tell the user through a live turn
            probe.injected.append(ctx.inject_message(probe.prompt, session_key=probe.session_key))
        return json.dumps({"ok": True})

    ctx.register_tool("cw_offer", "cw", _schema("cw_offer"), offer)
    ctx.register_tool("cw_other", "cw", _schema("cw_other"), lambda args, **kw: json.dumps({"ok": True}))
'''

LIVE_TURN_VIEW = {"turn_origin": None, "caller_scoped": None, "approval": None, "cw_other": None}


def _load_plugin():
    import hermes_cli.plugins as plugins_mod
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    plugin_dir = home / "plugins" / PLUGIN
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(
        {"name": PLUGIN, "capabilities": ["llm.session_fork"]}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN_BODY, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [PLUGIN], "entries": {PLUGIN: {
        "granted_capabilities": ["llm.session_fork"], "allow_gateway_injection": True}}}}), encoding="utf-8")
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins(force=True)
    manager = plugins_mod.get_plugin_manager()
    return plugins_mod.PluginContext(manager._plugins[PLUGIN].manifest, manager)


class _Parent(SimpleNamespace):
    """The live session agent the lease was issued for (held weakly by the lease)."""


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": name,
                                             "parameters": {"type": "object", "properties": {}}}}


class _ScriptedFork:
    """The fork agent: its one model step calls ``cw_offer`` through the real dispatcher."""

    _review_input_token_budget = 1000

    def interrupt(self, message=None):
        pass

    def run_conversation(self, user_message, conversation_history=None):
        from model_tools import handle_function_call

        handle_function_call("cw_offer", {}, task_id="fork", session_id="sess")
        return {"final_response": "Offered.", "messages": []}


@pytest.fixture
def probe(monkeypatch):
    from agent import background_review, side_question_lease

    monkeypatch.setattr(background_review, "build_cache_parity_fork",
                        lambda parent, task_cfg, *, max_iterations, write_origin: (_ScriptedFork(), {}, False))
    monkeypatch.setattr(background_review, "release_cache_parity_fork",
                        lambda parent, fork, *, task, session_id=None: {})
    state = SimpleNamespace(calls=[], injected=[], live=[], turns=[], prompt=PROMPT, session_key=None,
                            caller_scoped=CALLER_SCOPED, resident=None)
    monkeypatch.setitem(sys.modules, "_inject_probe", state)
    state.ctx = _load_plugin()
    yield state
    side_question_lease._leases.clear()


def _live_turn_dispatch(probe):
    """A live turn's tool round: the plugin's tool dispatched on a turn worker thread."""
    from model_tools import handle_function_call

    seen_before = len(probe.calls)
    handle_function_call("cw_offer", {}, task_id="live", session_id="sess")
    return probe.calls[seen_before:]


@contextlib.asynccontextmanager
async def _gateway_session(monkeypatch, probe):
    """A started gateway (real runner, Discord adapter, authority, published injector) whose Discord
    DM already holds a human turn and has gone idle. Its agent turn runs the live tool round for the
    plugin's message."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def turn(event, source, quick_key, run_generation):
        if probe.resident is not None:
            # The resident agent's run_conversation starts by superseding its background work.
            from agent.background_review import cancel_background_review_for_live_turn
            await runner._run_in_executor_with_context(cancel_background_review_for_live_turn, probe.resident)
        if event.text == PROMPT:
            probe.live.append(await runner._run_in_executor_with_context(_live_turn_dispatch, probe))
        probe.turns.append(event.text)
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(adapter, "send", send)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._install_plugin_message_injector()
    try:
        source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
        await adapter.handle_message(MessageEvent(text="A", source=source, message_id="A"))
        await async_wait_until(lambda: "A" in probe.turns, timeout=10, what="the human turn")
        probe.session_key = runner.session_store._generate_session_key(source)
        # Idle: the FIFO drain exits, so the next admission schedules a new one.
        live = authority.sessions[authority.logical_owner(
            runner.session_store.lookup_by_session_key(probe.session_key).session_id)]
        await async_wait_until(lambda: live.task is not None and live.task.done(), timeout=10,
                               what="the session's drain to go idle")
        probe.adapter, probe.source, probe.session = adapter, source, live
        yield runner
    finally:
        runner._clear_plugin_message_injector()
        runner._running = False


def _issue_lease_in_a_turn_scope():
    """The live turn that just ended issues the lease from its own context; the fork copies it."""
    from agent.side_question_lease import issue_side_question_token

    CALLER_SCOPED.set("turn scope")
    parent = _Parent(_persist_disabled=False, _cached_system_prompt="sys", _cached_system_prompt_static=None,
                     tools=[_tool("cw_offer"), _tool("cw_other")], session_id="sess", _current_turn_id="turn-1",
                     model="m", provider="openrouter", base_url="https://openrouter.ai/api/v1",
                     api_mode="chat_completions", reasoning_config=None)
    return parent, issue_side_question_token(parent, [], "turn-1")


@pytest.mark.asyncio
async def test_a_turn_injected_from_inside_a_fork_sees_no_fork_context(monkeypatch, probe):
    """The live report: the plugin's handler, inside its fork, injects; the injected turn's call to
    the same handler arrived with the fork's ``turn_origin`` and took the fork branch again."""
    async with _gateway_session(monkeypatch, probe):
        parent, token = contextvars.copy_context().run(_issue_lease_in_a_turn_scope)
        handle = probe.ctx.fork_session(token, "Worth sharing?", tools=["cw_offer"])
        assert handle is not None and await asyncio.to_thread(handle.wait, 10) and handle.outcome == "ok"
        await async_wait_until(lambda: probe.live, timeout=10, what="the injected turn's tool round")

    # Inside the fork: its origin, the copied turn scope, auto-denied approvals, cw_other blocked; it
    # injects once. The injected live turn sees none of it, so its handler takes the live branch.
    assert {"fork": probe.calls[0], "injected": probe.injected, "live_turn": probe.live} == {
        "fork": {"turn_origin": f"plugin_fork:{PLUGIN}", "caller_scoped": "turn scope",
                 "approval": "_bg_review_auto_deny", "cw_other": "block"},
        "injected": [True],
        "live_turn": [[LIVE_TURN_VIEW]],
    }
    assert probe.turns == ["A", PROMPT] and parent is not None


@pytest.mark.asyncio
async def test_a_turn_injected_from_the_gateway_loop_does_not_inherit_the_callers_context(monkeypatch, probe):
    """Called on the loop itself (an async hook, a scoped task), the scheduler still starts the turn
    from the gateway's context, not the calling task's."""
    from agent.plugin_session_fork import _TURN_ORIGIN

    async with _gateway_session(monkeypatch, probe):
        async def scoped_caller():
            CALLER_SCOPED.set("caller task scope")
            _TURN_ORIGIN.set(f"plugin_fork:{PLUGIN}")
            return probe.ctx.inject_message(PROMPT, session_key=probe.session_key)

        assert await asyncio.create_task(scoped_caller()) is True
        await async_wait_until(lambda: probe.live, timeout=10, what="the injected turn's tool round")

    assert probe.live == [[LIVE_TURN_VIEW]] and probe.injected == []


# --- Background work the session is running when a turn starts ---------------------------------
# A plugin-injected turn lets the session's running background review and plugin forks finish (up
# to a bounded wait an interrupt ends early) instead of superseding them; a person's turn still
# supersedes them at once. Covered on both paths a turn supersedes them: the authority's own cancel
# of a cold /refine review (``_execute_claimed``) and the resident agent's first step.

WAITS = "Turn waits for"


class _Interruptible:
    """A background model run: runs until released or interrupted."""

    _review_input_token_budget = 1000

    def __init__(self):
        self.started, self.stop = threading.Event(), threading.Event()
        self.interrupted = False

    def interrupt(self, message=None):
        self.interrupted = True
        self.stop.set()

    def run_conversation(self, user_message, conversation_history=None):  # as the fork agent
        self.started.set()
        self.stop.wait(30)
        return {"final_response": "Offered.", "messages": []}


def _start_refine_review(guard):
    """A cold /refine review on the session's guard, as ``gateway.session_refine`` starts one."""
    from agent.background_review import finish_background_review_run, prepare_background_review_run

    run, review, outcome = prepare_background_review_run(guard), _Interruptible(), []
    assert run is not None and run.begin_request(review)

    def worker():
        review.stop.wait(30)
        outcome.append("cancelled" if review.interrupted else "completed")
        finish_background_review_run(guard, run)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return SimpleNamespace(review=review, outcome=outcome, thread=thread)


async def _start_background_work(monkeypatch, probe):
    """The resident agent's running plugin fork and the session's running /refine review."""
    from agent import background_review

    fork = _Interruptible()
    monkeypatch.setattr(background_review, "build_cache_parity_fork",
                        lambda parent, task_cfg, *, max_iterations, write_origin: (fork, {}, False))
    parent, token = contextvars.copy_context().run(_issue_lease_in_a_turn_scope)
    parent._interrupt_requested = False
    handle = probe.ctx.fork_session(token, "Worth sharing?", tools=["cw_offer"])
    assert handle is not None
    await asyncio.to_thread(fork.started.wait, 10)
    probe.resident = parent
    return SimpleNamespace(parent=parent, fork=fork, handle=handle, refine=_start_refine_review(probe.session.review))


@pytest.mark.asyncio
async def test_a_plugin_injected_turn_lets_the_sessions_running_review_and_fork_finish(monkeypatch, probe, caplog):
    """The live report: the plugin's message superseded the user's own review (result=none)."""
    caplog.set_level(logging.INFO, logger="agent.background_review")
    async with _gateway_session(monkeypatch, probe):
        bg = await _start_background_work(monkeypatch, probe)
        assert probe.ctx.inject_message(PROMPT, session_key=probe.session_key) is True
        await async_wait_until(lambda: caplog.text.count(WAITS) == 1, timeout=10, what="the wait for the review")
        during_review = (bg.refine.review.interrupted, bg.fork.interrupted, PROMPT in probe.turns)
        bg.refine.review.stop.set()  # the review finishes on its own
        await async_wait_until(lambda: caplog.text.count(WAITS) == 2, timeout=10, what="the wait for the fork")
        during_fork = (bg.fork.interrupted, PROMPT in probe.turns)
        bg.fork.stop.set()  # so does the fork
        await async_wait_until(lambda: PROMPT in probe.turns, timeout=10, what="the injected turn")
        assert await asyncio.to_thread(bg.handle.wait, 10)
        bg.refine.thread.join(10)

    assert {"during_review": during_review, "during_fork": during_fork, "review": bg.refine.outcome,
            "fork": bg.handle.outcome} == {
        "during_review": (False, False, False), "during_fork": (False, False),
        "review": ["completed"], "fork": "ok"}


@pytest.mark.asyncio
async def test_a_persons_turn_still_supersedes_them_at_once(monkeypatch, probe, caplog):
    from gateway.platforms.event import MessageEvent

    caplog.set_level(logging.INFO, logger="agent.background_review")
    async with _gateway_session(monkeypatch, probe):
        bg = await _start_background_work(monkeypatch, probe)
        await probe.adapter.handle_message(MessageEvent(text="B", source=probe.source, message_id="B"))
        await async_wait_until(lambda: "B" in probe.turns, timeout=10, what="the person's turn")
        assert await asyncio.to_thread(bg.handle.wait, 10)
        bg.refine.thread.join(10)

    assert (bg.refine.outcome, bg.handle.outcome, bg.fork.interrupted, WAITS in caplog.text) == (
        ["cancelled"], "cancelled", True, False)


@pytest.mark.asyncio
async def test_an_interrupt_of_the_session_agent_ends_the_injected_turns_wait(monkeypatch, probe, caplog):
    """A person stopping or overtaking the turn interrupts the session agent: the injected turn stops
    waiting and supersedes the fork still running, well inside the 120 s bound."""
    caplog.set_level(logging.INFO, logger="agent.background_review")
    async with _gateway_session(monkeypatch, probe):
        bg = await _start_background_work(monkeypatch, probe)
        bg.refine.review.stop.set()
        bg.refine.thread.join(10)
        assert probe.ctx.inject_message(PROMPT, session_key=probe.session_key) is True
        await async_wait_until(lambda: WAITS in caplog.text, timeout=10, what="the wait for the fork")
        bg.parent._interrupt_requested = True
        await async_wait_until(lambda: PROMPT in probe.turns, timeout=10, what="the injected turn")
        assert await asyncio.to_thread(bg.handle.wait, 10)

    assert (bg.handle.outcome, bg.fork.interrupted) == ("cancelled", True)
