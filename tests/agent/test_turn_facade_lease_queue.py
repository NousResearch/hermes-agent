"""Turn-lease starvation must queue an inbound delivery, never fail it (card t_d4f85953).

Reproduced in production: a turn that makes no model call and no tool completion still holds the
lease, and its transcript writes keep refreshing it. An inbound delivery (peer dm / bot-mode
delivery / peer run) that cannot take the lease inside its wait window is answered
``session_turn_lease_timeout`` and its message is dropped on the floor with no record anywhere.

The lease must not be stolen from a live holder, so the delivery is spooled into the existing
durable live-delivery store, pinned to the conversation instead of to a live-owner: the next turn
admitted on that conversation drains it as its first user message, exactly once.
"""

from __future__ import annotations

import json
import os
import threading

from agent.turn_facade_lease import admit_durable_turn_lease
from tools.bot_live_delivery import claim_lease_delivery, enqueue_lease_delivery


class _Db:
    def __init__(self, exists=True, acquired=True, holder="pid=1:turn=other:platform=api_server"):
        self.exists = exists
        self.acquired = acquired
        self.holder = holder
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

    def current_session_turn_lease_holder(self, session_id):
        return self.holder

    def session_turn_waiter_count(self, session_id):
        return 3

    def oldest_session_turn_waiter_age(self, session_id):
        return 41.5


def _agent(db, **overrides):
    agent = __import__("types").SimpleNamespace(
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


def _admit(agent, history=None, inbound_delivery=None):
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "api_server"},
        conversation_history=history,
        inbound_delivery=inbound_delivery,
    )


def _queue_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setattr("agent.turn_facade_lease._lease_queue_home", lambda: home)
    return home


def test_inbound_delivery_that_loses_the_wait_is_queued_not_failed(tmp_path, monkeypatch):
    home = _queue_home(tmp_path, monkeypatch)
    history = [{"role": "user", "content": "hi"}]
    admission = _admit(
        _agent(_Db(acquired=False)),
        history,
        inbound_delivery={
            "message": "please sync the fork",
            "author": "lane-beta",
            "delivery_id": "peer-42",
        },
    )

    assert admission.lease is None
    result = admission.early_result
    assert result is not None
    assert result["queued"] is True
    assert not result.get("failed")
    assert not result.get("error")
    assert result["delivery_id"] == "peer-42"
    assert result["holder"] == "pid=1:turn=other:platform=api_server"
    assert result["waiter_age_s"] == 41.5
    assert result["queue_depth"] == 3
    assert result["messages"] == history

    # Spooled durably, pinned to the conversation, and claimable by the next turn.
    record = claim_lease_delivery(home, conversation_id="s1")
    assert record is not None
    assert record["delivery_id"] == "peer-42"
    assert record["message"] == "please sync the fork"
    assert record["author"] == "lane-beta"
    assert record["status"] == "claimed"


def test_interrupted_inbound_delivery_is_not_queued(tmp_path, monkeypatch):
    """A delivery that arrives with an interrupt pending must keep the legacy shape."""
    _queue_home(tmp_path, monkeypatch)
    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(
        agent,
        [{"role": "user", "content": "x"}],
        inbound_delivery={"message": "hi", "author": "peer", "delivery_id": "peer-1"},
    )
    assert admission.early_result["interrupted"] is True
    assert "queued" not in admission.early_result


def test_human_turn_that_loses_the_wait_keeps_the_legacy_failure(tmp_path, monkeypatch):
    """No inbound delivery means no queue: the fail-closed contract is unchanged."""
    home = _queue_home(tmp_path, monkeypatch)
    admission = _admit(_agent(_Db(acquired=False)), [{"role": "user", "content": "x"}])
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    assert not (home / "runtime" / "bot_live_delivery").exists()


def test_queued_delivery_is_drained_once_into_the_next_admitted_turn(tmp_path, monkeypatch):
    home = _queue_home(tmp_path, monkeypatch)
    enqueue_lease_delivery(
        home,
        conversation_id="s1",
        message="queued hello",
        holder="pid=1:turn=other:platform=api_server",
        author="lane-beta",
        delivery_id="peer-42",
    )

    history = [{"role": "assistant", "content": "prior reply"}]
    agent = _agent(_Db())
    admission = _admit(agent, list(history))
    lease = admission.lease
    assert lease is not None
    try:
        drained = admission.conversation_history
        assert drained[-1] == {"role": "user", "content": "queued hello"}
        assert drained[:-1] == history
        assert admission.drained_delivery_ids == ["peer-42"]
        # Settled: the next turn cannot see it again.
        assert claim_lease_delivery(home, conversation_id="s1") is None
    finally:
        lease.stop_refresher()
        lease.join_threads()

    second = _admit(_agent(_Db()), [{"role": "assistant", "content": "prior reply"}])
    lease2 = second.lease
    try:
        assert second.conversation_history == [{"role": "assistant", "content": "prior reply"}]
        assert second.drained_delivery_ids == []
        assert "queued hello" not in json.dumps(second.conversation_history)
    finally:
        lease2.stop_refresher()
        lease2.join_threads()


def test_settled_delivery_is_not_reclaimable_and_reenqueue_is_idempotent(tmp_path, monkeypatch):
    """Exactly once: a settled record is never claimed again, and a retrying peer cannot
    resurrect it by re-sending the same delivery id."""
    home = _queue_home(tmp_path, monkeypatch)
    enqueue_lease_delivery(
        home, conversation_id="s1", message="dup body", holder="", delivery_id="peer-9"
    )
    agent = _agent(_Db())
    admission = _admit(agent, [{"role": "assistant", "content": "reply"}])
    lease = admission.lease
    try:
        assert admission.conversation_history[-1]["content"] == "dup body"
        assert admission.drained_delivery_ids == ["peer-9"]
    finally:
        lease.stop_refresher()
        lease.join_threads()

    # The record is terminal: nothing is left for a later turn to claim.
    assert claim_lease_delivery(home, conversation_id="s1") is None

    # A peer retrying the SAME delivery id must not refill the queue.
    enqueue_lease_delivery(
        home, conversation_id="s1", message="dup body", holder="", delivery_id="peer-9"
    )
    assert claim_lease_delivery(home, conversation_id="s1") is None

    second = _admit(_agent(_Db()), list(admission.conversation_history))
    lease2 = second.lease
    try:
        assert second.drained_delivery_ids == []
        assert [m["content"] for m in second.conversation_history].count("dup body") == 1
    finally:
        lease2.stop_refresher()
        lease2.join_threads()


def test_drain_preserves_strict_role_alternation(tmp_path, monkeypatch):
    home = _queue_home(tmp_path, monkeypatch)
    enqueue_lease_delivery(
        home, conversation_id="s1", message="queued second", holder="", delivery_id="peer-7"
    )
    agent = _agent(_Db())
    admission = _admit(agent, [{"role": "user", "content": "typed first"}])
    lease = admission.lease
    try:
        history = admission.conversation_history
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert "typed first" in history[0]["content"]
        assert "queued second" in history[0]["content"]
    finally:
        lease.stop_refresher()
        lease.join_threads()


def test_queue_home_follows_the_process_hermes_home(tmp_path, monkeypatch):
    """The spool must land in the store the live-delivery router uses for this profile."""
    from agent.turn_facade_lease import _lease_queue_dir, _lease_queue_home

    profile_home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    assert _lease_queue_home() == profile_home
    assert _lease_queue_dir(_lease_queue_home()) == str(
        profile_home / "runtime" / "bot_live_delivery"
    )


def test_real_store_reports_holder_and_queue_depth_on_the_receipt(tmp_path, monkeypatch):
    """The receipt's holder / queue fields must come off the real durable store, not a stub."""
    from hermes_state import SessionDB

    _queue_home(tmp_path, monkeypatch)
    # The real store honours the full wait budget (1800s) before it gives up; a stub returns at
    # once. Scale the budget down so the test exercises the real denial path without the wait.
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0.25)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = f"pid={os.getpid()}:turn=holder:platform=api_server"
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=60) is True
    waiter = f"pid={os.getpid()}:turn=waiter:platform=api_server"
    assert db.register_session_turn_waiter("s1", waiter) is True

    admission = _admit(
        _agent(db),
        [{"role": "user", "content": "hi"}],
        inbound_delivery={"message": "queued body", "author": "peer", "delivery_id": "real-1"},
    )
    result = admission.early_result
    assert result["queued"] is True
    assert result["holder"] == holder
    assert result["queue_depth"] == 1
    assert isinstance(result["waiter_age_s"], float)
    assert result["waiter_age_s"] >= 0.0

    # Released for real: the next admitted turn on this conversation drains it exactly once.
    db.release_session_turn_lease("s1", holder)
    db.clear_session_turn_waiter("s1", waiter)
    second = _admit(_agent(db), [{"role": "assistant", "content": "prior"}])
    lease = second.lease
    assert lease is not None
    try:
        assert second.drained_delivery_ids == ["real-1"]
        assert second.conversation_history[-1] == {"role": "user", "content": "queued body"}
    finally:
        lease.stop_refresher()
        lease.join_threads()
    db.close()


def test_drain_failure_does_not_break_the_turn(tmp_path, monkeypatch):
    """A corrupt spool must not fail the turn that was admitted normally."""
    home = _queue_home(tmp_path, monkeypatch)
    root = home / "runtime" / "bot_live_delivery"
    root.mkdir(parents=True, exist_ok=True)
    (root / "peer-bad.json").write_text("{not json", encoding="utf-8")

    agent = _agent(_Db())
    admission = _admit(agent, [{"role": "assistant", "content": "reply"}])
    lease = admission.lease
    try:
        assert admission.conversation_history == [{"role": "assistant", "content": "reply"}]
        assert admission.drained_delivery_ids == []
    finally:
        lease.stop_refresher()
        lease.join_threads()
