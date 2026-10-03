"""Contract tests for turn-admission-bound transcript writes."""

from types import SimpleNamespace

from agent.session_persistence import _db_flush_write, _persistence_session_id


class RecordingDb:
    def __init__(self):
        self.calls = []

    def append_messages_batch(self, **kwargs):
        self.calls.append(kwargs)


def _agent(*, live_session, admitted_session=None):
    agent = SimpleNamespace(
        session_id=live_session,
        _session_db=RecordingDb(),
        _active_compression_lock_holder=None,
        _active_session_turn_lease_holder=None,
        _active_session_turn_lease_ttl_seconds=300.0,
    )
    if admitted_session is not None:
        agent._inflight_turn_session_id = admitted_session
    return agent


def test_remaining_turn_write_stays_with_admission_session_after_switch():
    agent = _agent(live_session="branch", admitted_session="parent")

    _db_flush_write(agent, [{"role": "assistant", "content": "remaining"}], [{}])

    assert agent._session_db.calls[0]["session_id"] == "parent"


def test_idle_flush_uses_current_session():
    agent = _agent(live_session="current")

    _db_flush_write(agent, [{"role": "assistant", "content": "idle"}], [{}])

    assert agent._session_db.calls[0]["session_id"] == "current"


def test_admission_binding_is_fail_closed_when_live_identity_changes():
    agent = _agent(live_session="child", admitted_session="parent")

    assert _persistence_session_id(agent) != agent.session_id
    assert _persistence_session_id(agent) == "parent"
