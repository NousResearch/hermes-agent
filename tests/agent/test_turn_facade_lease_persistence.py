"""The admitted turn lease runs on the agent's storage interface handle; a client store keeps its own calls."""
from types import SimpleNamespace

from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT
from hermes_constants import get_hermes_home
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationRequest, NewMessage
from persistence_assembly import open_persistence, profile_scope
from tests.agent.test_turn_facade_lease import _admit, _agent, _Db, _Unused
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_lease_carries_the_agent_handle_and_runs_on_it(monkeypatch):
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    handle = open_persistence(profile_scope())
    try:
        admission = _admit(_agent(_Unused(), persistence=handle))
        lease = admission.lease
        assert lease.persistence is handle
        committed = handle.coordination.turn_lease_expires_at(session_id="s1", holder=lease.holder)
        assert committed is not None and lease._authority_deadline == committed
        lease.release()
        assert handle.coordination.turn_lease_expires_at(session_id="s1", holder=lease.holder) is None
        assert (get_hermes_home() / "state.db").exists()
    finally:
        handle.close()


def test_agent_without_a_handle_leases_without_one(monkeypatch):
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    admission = _admit(_agent(_Db()))
    assert admission.lease.persistence is None
    admission.lease.release()


def test_a_reader_handle_beside_a_client_store_keeps_the_client_calls(monkeypatch):
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    db = _Db()
    admission = _admit(_agent(db, persistence=SimpleNamespace(read_only=True)))
    lease = admission.lease
    assert lease.persistence is None
    lease.release()
    assert db.events == [("acquire", "s1", lease.holder), ("release", "s1", lease.holder)]


def test_reload_after_a_wait_matches_the_store_read(monkeypatch):
    """A wait behind another holder reloads through the handle: the rotated tip and the repaired,
    row-id-stamped transcript the store returns for the same file, plus the carried input."""
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    db = open_persistence(profile_scope())
    try:
        seed_session(db, "s1", "cli")
        seed_messages(db, "s1", {"role": "user", "content": "one"}, {"role": "assistant", "content": "reply"})
        assert db.coordination.try_acquire_compression_lock(session_id="s1", holder="compressor")
        db.lineage.publish_compression_child(
            "s1", CompressionChild("s1-child", "cli", model="m"),
            [NewMessage(role="assistant", content="summary")], compression_lock_holder="compressor")
        seed_messages(db, "s1-child", {"role": "user", "content": "two"},
                      {"role": "user", "content": "three"})  # a durable user;user pair the reload repairs
        expected_id = db.lineage.resolve_resume_target("s1")
        expected = conversation(db, expected_id, ConversationRequest(repair_alternation=True, include_row_ids=True))
        # Another holder for long enough that the first attempt misses and the turn waits.
        assert db.coordination.try_acquire_turn_lease(session_id="s1", holder="other", ttl_seconds=1.5)
    finally:
        db.close()
    assert expected_id == "s1-child" and any("_row_id" in m for m in expected)

    carried = {"role": "user", "content": "carried", _PERSIST_AFTER_ADMISSION_INTERRUPT: True}
    handle = open_persistence(profile_scope())
    try:
        agent = _agent(_Unused(), persistence=handle)
        task_context = {"session_id": "s1", "task_id": "t", "platform": "cli"}
        from agent.turn_facade_lease import admit_durable_turn_lease
        admission = admit_durable_turn_lease(
            agent, session_id="s1", relay_turn_id="s1:t:abcd", task_context=task_context,
            conversation_history=[{"role": "user", "content": "stale"}, carried])
        assert agent.session_id == "s1-child" and task_context["session_id"] == "s1-child"
        assert agent._session_db_created is True
        assert admission.conversation_history == [*expected, carried]
        assert "Session is free; loading the latest transcript..." in agent.statuses
        admission.lease.release()
    finally:
        handle.close()
