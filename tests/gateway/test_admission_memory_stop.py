"""A turn held for memory obeys a Stop through its session (Desktop, TUI, API).

Before the turn has an agent, the authority's Stop only latches ``pending_stops`` for the claim;
the memory wait must see it and settle the turn interrupted instead of holding its place in the
queue until memory frees or the wait expires. Runs the real FIFO drain and inbound handler.
"""

import asyncio

import pytest

from gateway import admission_memory as am
from hermes_platform.memory import MemoryReading
from tests.waiting import async_wait_until

MB = 1024 * 1024


@pytest.mark.asyncio
async def test_a_stop_ends_a_turn_held_for_memory(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, Submission
    from gateway.session_local import create_local_session
    from hermes_state import SessionDB
    from hermes_state_runtime import list_session_admissions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Far below every floor, and a wait long enough that only the Stop can end it in time.
    gate = am.MemoryGate(settings=am.MemoryGuardSettings(critical_mb=256, low_mb=512, max_wait_seconds=20),
                         reader=lambda: MemoryReading(10 * MB, 4096 * MB, "cgroup"))
    monkeypatch.setattr(am, "_gate", gate)
    monkeypatch.setattr(am, "_POLL_INTERVAL_S", 0.05)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    built = []
    monkeypatch.setattr(runner, "_handle_message_with_agent", lambda *a, **k: built.append(a))
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", db=db)
    viewer = Principal("test-owner", "default", frozenset(
        {"session:create", "session:submit", "session:read", "session:control"}), "viewer")
    try:
        ref = create_local_session(authority, viewer, dict(request_id="cli", source="cli", model="fixture",
                                                           toolsets=[]))
        await authority.submit(viewer, Submission("typed", ref, {"text": "hello"}, "queue"))
        await async_wait_until(lambda: gate._waiters, timeout=10, what="the turn waiting for memory")

        await authority.interrupt(viewer, ref, authority._handle(ref).execution_generation)
        loop = asyncio.get_running_loop()
        stopped_at = loop.time()
        await async_wait_until(lambda: not gate._waiters, timeout=10, what="the stopped turn leaving the queue")
        assert loop.time() - stopped_at < 5, "the stopped turn kept waiting for memory"
        await async_wait_until(lambda: not list_session_admissions(db, session_id=ref.session_id),
                               timeout=10, what="the stopped turn settling")
        rows = list_session_admissions(db, session_id=ref.session_id, pending_only=False)
    finally:
        db.close()
    assert built == [], "a stopped turn went on to build its agent"
    assert [(row["status"], row["outcome"]) for row in rows] == [("terminal", "interrupted")]


def test_a_gateway_restart_ends_a_chat_turn_held_for_memory(monkeypatch):
    """A held chat turn never started, so a restart/stop refuses it like any new turn instead of
    holding the drain open until memory frees."""
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin

    clock = [1000.0]
    gate = am.MemoryGate(settings=am.MemoryGuardSettings(critical_mb=256, low_mb=512, max_wait_seconds=600),
                         reader=lambda: MemoryReading(10 * MB, 4096 * MB, "cgroup"), clock=lambda: clock[0])
    monkeypatch.setattr(am, "_gate", gate)
    runner = SimpleNamespace(_draining=False, _restart_requested=True, sent=[])

    async def poll(_):
        clock[0] += 2
        runner._draining = True  # /restart arrives while the turn waits

    monkeypatch.setattr(am.asyncio, "sleep", poll)

    async def send(event, adapter, text):
        runner.sent.append(text)

    runner._delivery_adapter_for = lambda source: object()
    runner._send_busy_reply = send
    runner._is_session_run_current = lambda key, generation: True
    runner._status_action_gerund = lambda: GatewayRunner._status_action_gerund(runner)
    runner._hm_memory_wait_abandoned = lambda claim, kind: GatewayInboundHooksMixin._hm_memory_wait_abandoned(
        runner, claim, kind)
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="1")
    reply = asyncio.run(GatewayInboundHooksMixin._hm_memory_admission(
        runner, SimpleNamespace(source=source), source, "k", False, 1))
    assert not gate._waiters
    assert reply and "memory" not in reply, f"a restart left the held turn waiting: {reply!r}"
