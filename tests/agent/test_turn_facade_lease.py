"""Unit tests for agent.turn_facade_lease (admission + lease bracket)."""
import threading
from types import SimpleNamespace

from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    DurableTurnLease,
    admit_durable_turn_lease,
)


class _Db:
    def __init__(self, exists=True, acquired=True):
        self.exists = exists
        self.acquired = acquired
        self.events = []

    def get_session(self, session_id):
        return {"id": session_id} if self.exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        return self.acquired

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


def _agent(db, **overrides):
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        _persist_disabled=False,
        _interrupt_requested=False,
        _interrupt_message=None,
        _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0,
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _admit(agent, history=None):
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "cli"},
        conversation_history=history,
    )


def test_fresh_row_leases_with_seed_kept_and_persist_disabled_skips(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    seed = [{"role": "user", "content": "hi"}]
    agent = _agent(_Db(exists=False))
    admission = _admit(agent, seed)
    assert admission.lease is not None and admission.early_result is None
    assert admission.conversation_history is seed
    assert getattr(agent, "_session_db_created", False) is False
    admission.lease.release()

    db = _Db()
    admission = _admit(_agent(db, _persist_disabled=True), seed)
    assert admission.lease is None and db.events == []


def test_admission_sets_holder_attrs_and_release_clears_them(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    db = _Db()
    agent = _agent(db)
    admission = _admit(agent)
    lease = admission.lease
    assert isinstance(lease, DurableTurnLease)
    assert agent._session_db_created is True
    assert agent._active_session_turn_lease_holder == lease.holder
    assert agent._active_session_turn_lease_ttl_seconds == LEASE_TTL_SECONDS
    assert lease.holder.startswith("pid=") and ":platform=cli" in lease.holder
    assert lease.watchdog is None and lease.timer_handles == []
    assert lease.is_turn_active() is False

    lease.stop_refresher()
    lease.join_threads()
    lease.clear_interrupt()
    lease.release()
    assert db.events == [("acquire", "s1", lease.holder), ("release", "s1", lease.holder)]
    assert agent._active_session_turn_lease_holder is None
    assert agent._active_session_turn_lease_ttl_seconds is None


def test_timeout_and_interrupt_early_results():
    agent = _agent(_Db(acquired=False))
    admission = _admit(agent, [{"role": "user", "content": "x"}])
    assert admission.lease is None
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    # Stamped so UI descriptors show "session busy" instead of code="unknown".
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(agent)
    assert admission.early_result["interrupted"] is True
    assert admission.early_result["interrupt_message"] == "stop"


def test_interrupt_turn_only_while_active():
    agent = _agent(_Db())
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h")
    lease._interrupt_turn("lost")  # inactive: ignored
    assert calls == [] and lease.interrupt_message is None
    lease.turn_active = True
    lease._interrupt_turn("lost")
    assert calls == ["lost"] and lease.interrupt_message == "lost"
    lease.deactivate_after_liveness_abort()
    assert lease.stop.is_set() and lease.is_turn_active() is False


def test_start_clears_orphaned_codex_inflight_tool_count():
    """A turn aborted mid-tool never delivers item/completed, so the
    bridge's open-call count would otherwise stay > 0 and exempt the NEXT
    turn's watchdog forever (#127643)."""
    agent = _agent(_Db())
    agent._codex_inflight_tool_calls = 1  # orphaned by the previous turn
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h")
    try:
        lease.start()
        assert agent._codex_inflight_tool_calls == 0
        assert lease.is_turn_active() is True
    finally:
        lease.stop_refresher()
        lease.join_threads()
