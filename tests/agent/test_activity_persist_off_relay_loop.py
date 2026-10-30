"""A managed worker's activity heartbeat must not reach the synchronous persistence RPC on
Relay's event loop (#139029).

Relay drives Hermes' provider and tool callbacks synchronously on its private ``asyncio.run``
loop, so ``_touch_activity`` inside them runs with a loop on the thread. In a managed worker
``_session_db`` is a ``RuntimeSessionStore`` whose ``WorkerRPC`` refuses on-loop calls
(``synchronous_rpc_on_event_loop``) and the refusal is sticky: every later write of the turn
failed. These drive the real Relay pipeline, the real store and the real guard; only the owner's
socket is replaced by an in-process reply.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

import run_agent
from agent import relay_llm, relay_runtime, relay_tools
from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC
from agent.session_activity import ActivityProvenance

SESSION = "session-1"


class _OwnerReplies:
    """Stands in for the owner's socket (``WorkerRPC._connection``); records what reached it."""

    def __init__(self):
        self.persisted = []
        self.on_loop = []

    def send(self, request_id, message):
        import json
        try:
            asyncio.get_running_loop()
            self.on_loop.append(True)
        except RuntimeError:
            self.on_loop.append(False)
        frame = json.loads(message)
        self.persisted.append(frame["params"]["operation"])
        self._reply = {"jsonrpc": "2.0", "id": request_id, "result": {"ok": True, "value": None}}

    def receive(self, timeout):
        return self._reply


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id=SESSION, platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    lease.host.retain_managed_execution("test.activity_off_loop")
    try:
        yield lease.host
    finally:
        lease.host.release_managed_execution("test.activity_off_loop")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.fixture()
def worker_agent(tmp_path, monkeypatch):
    owner = _OwnerReplies()
    monkeypatch.setattr(WorkerRPC, "_connection", lambda self: owner)
    store = RuntimeSessionStore(WorkerRPC(tmp_path / "home"), {"session_id": SESSION}, tmp_path / "outbox")
    agent = SimpleNamespace(
        session_id=SESSION, _session_db=store, _last_activity_ts=0.0, _last_activity_desc="",
        _last_activity_provenance=ActivityProvenance.UNKNOWN, _session_activity_last_persist_mono=0.0,
    )
    agent._touch_activity = run_agent.AIAgent._touch_activity.__get__(agent, SimpleNamespace)
    agent._persist_session_activity_if_due = (
        run_agent.AIAgent._persist_session_activity_if_due.__get__(agent, SimpleNamespace))
    agent._reset_activity_labels_after_turn = (
        run_agent.AIAgent._reset_activity_labels_after_turn.__get__(agent, SimpleNamespace))
    try:
        yield agent, store, owner
    finally:
        store._outbox_owner.close()


def _guard_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "agent.runtime_session_store" and "synchronous_rpc_on_event_loop" in r.getMessage()]


def _assert_persisted_off_loop(store, owner, caplog, *operations):
    assert _guard_warnings(caplog) == []
    assert store.failure is None
    assert owner.persisted == list(operations)
    assert owner.on_loop == [False] * len(operations)


def test_provider_callback_heartbeat_persists_off_relay_loop(relay_turn, worker_agent, caplog):
    """Reporter stack 1: relay_llm.run_callback -> interruptible_api_call -> _emit_wait_notice
    -> _touch_activity -> worker.persist, on the loop ``relay_llm.execute`` runs."""
    agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")

    def provider(request):
        assert asyncio.get_running_loop() is not None  # Relay's loop is live on this thread
        agent._touch_activity("waiting for provider response")
        return {"id": "r", "object": "chat.completion", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    relay_llm.execute(
        {"messages": [{"role": "user", "content": "hi"}], "model": "m"}, provider,
        name="prov", model_name="m", session_id=SESSION,
        metadata={"api_mode": "chat_completions", "api_request_id": "req-1"},
    )
    _assert_persisted_off_loop(store, owner, caplog, "session.activity")


def test_concurrent_tool_callback_heartbeat_persists_off_relay_loop(relay_turn, worker_agent, caplog):
    """Reporter stack 3: concurrent tool dispatch on a ThreadPoolExecutor thread ->
    _begin_tool_execution -> _touch_activity -> worker.persist, inside relay_tools.execute."""
    agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")

    def tool(args):
        assert asyncio.get_running_loop() is not None
        agent._touch_activity("executing tool: image_generate")
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(relay_tools.execute, "image_generate", {"prompt": "x"}, tool,
                    session_id=SESSION, tool_call_id="call-1").result(timeout=30)
    _assert_persisted_off_loop(store, owner, caplog, "session.activity")


def test_later_turn_writes_still_land_after_an_on_loop_heartbeat(relay_turn, worker_agent, caplog):
    """The sticky follow-on in the reporter's timeline: append_message / finalize failed with
    the same error after the first refusal. The heartbeat must not poison the store."""
    agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")

    def tool(args):
        agent._touch_activity("executing tool: terminal")
        return {"ok": True}

    relay_tools.execute("terminal", {"command": "true"}, tool, session_id=SESSION)
    store.clear_session_activity_labels(SESSION)  # an ordinary off-loop write after the tool
    _assert_persisted_off_loop(store, owner, caplog, "session.activity", "session.activity_clear")


def test_label_reset_on_a_running_loop_persists_off_loop(worker_agent, caplog):
    """Sibling writer in the same mixin: turn-teardown label clear reached on a loop.

    Defence in depth, not a reproduced caller: no reported stack shows this path on a loop, so
    this drives it from a synthetic ``asyncio.run``."""
    agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")

    async def teardown():
        agent._reset_activity_labels_after_turn()

    asyncio.run(teardown())
    _assert_persisted_off_loop(store, owner, caplog, "session.activity_clear")


def test_sync_aux_usage_inside_managed_tool_persists_off_relay_loop(relay_turn, worker_agent, caplog):
    """Sibling: a synchronous auxiliary call made by a tool (vision, web_extract, ...) records its
    usage at ``_validate_llm_response`` -> ``record_aux_usage`` while the tool callback still holds
    Relay's loop. The async aux path already hops with ``asyncio.to_thread``; the sync one did not."""
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import _validate_llm_response

    _agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")
    response = SimpleNamespace(
        model="m", usage={"prompt_tokens": 7, "completion_tokens": 3},
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    def tool(args):
        assert asyncio.get_running_loop() is not None
        _validate_llm_response(response, "vision", provider="prov")
        return {"ok": True}

    token = set_accounting_context(store, SESSION)
    try:
        relay_tools.execute("vision_analyze", {"image_url": "x"}, tool, session_id=SESSION)
    finally:
        reset_accounting_context(token)
    _assert_persisted_off_loop(store, owner, caplog, "usage.auxiliary")


def test_begin_tool_execution_inside_relay_tool_persists_off_relay_loop(relay_turn, worker_agent, caplog):
    """Reporter stack 3 through the real frame: tool_executor._begin_tool_execution on a
    ThreadPoolExecutor thread inside relay_tools.execute."""
    from agent import tool_executor

    agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")
    agent.quiet_mode, agent.tool_progress_callback, agent.tool_start_callback = True, None, None
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    ref = tool_executor._ToolCallRef(name="image_generate", args={"prompt": "x"}, task_id="task-1",
                                     call_id="call-1", trace=[])

    def tool(args):
        assert asyncio.get_running_loop() is not None
        tool_executor._begin_tool_execution(agent, ref, 1)
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(relay_tools.execute, "image_generate", {"prompt": "x"}, tool,
                    session_id=SESSION, tool_call_id="call-1").result(timeout=30)
    _assert_persisted_off_loop(store, owner, caplog, "session.activity")


def test_async_delegation_ledger_writes_inside_relay_tool_persist_off_relay_loop(
        relay_turn, worker_agent, caplog, monkeypatch):
    """Reviewer probe: ``_persist_dispatch`` -> ``WorkerDelegationLedger.dispatch`` (and the
    ``unscheduled`` mark) run inside the tool callback, on Relay's loop."""
    from tools import async_delegation
    from tools.async_delegation_worker import WorkerDelegationLedger

    _agent, store, owner = worker_agent
    caplog.set_level(logging.WARNING, logger="agent.runtime_session_store")
    ledger = WorkerDelegationLedger(store)
    monkeypatch.setattr(async_delegation, "worker_ledger", lambda: ledger)

    def tool(args):
        assert asyncio.get_running_loop() is not None
        async_delegation._persist_dispatch({"delegation_id": "d-1", "parent_session_id": SESSION,
                                            "goal": "g", "dispatched_at": 1.0})
        ledger.unscheduled("d-1")
        return {"ok": True}

    relay_tools.execute("cronjob", {"action": "run"}, tool, session_id=SESSION)
    _assert_persisted_off_loop(store, owner, caplog, "delegation.dispatch", "delegation.unscheduled")
