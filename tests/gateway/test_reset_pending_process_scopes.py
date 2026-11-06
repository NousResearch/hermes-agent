"""Session reset retries unfinished scopes without crossing ownership boundaries."""

import tools.process_registry as processes
from gateway.slash_commands_session import (
    _kill_processes_for_session_reset,
    _snapshot_processes_for_session_reset,
)


def test_reset_retries_pending_scopes_after_the_agent_is_evicted(monkeypatch):
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    session_key = "agent:main:telegram:dm:scope"

    def finished(sid, *, key=session_key, persisted=False):
        session = processes.ProcessSession(
            id=sid, command="daemonize", task_id="evicted-owner", session_key=key,
            systemd_unit=f"hermes-worker-{sid}.scope", persist_on_release=persisted,
        )
        session.mark_exited(0)
        registry._finished[session.id] = session
        return session

    target = finished("proc_retry")
    persisted = finished("proc_persisted", persisted=True)
    foreign = finished("proc_foreign", key="agent:beta:telegram:dm:scope")
    clean = finished("proc_clean")
    monkeypatch.setattr(processes, "_stop_systemd_unit", lambda _unit: False)
    for session in (target, persisted, foreign):
        assert registry.kill_process(session.id)["scope_stop_failed"] == session.systemd_unit

    first = _snapshot_processes_for_session_reset(session_key)
    assert first == (target.id,)
    assert _kill_processes_for_session_reset(first) == 0
    assert target._scope_stop_pending

    stopped_units = []

    def stop_scope(unit):
        stopped_units.append(unit)
        return True

    monkeypatch.setattr(processes, "_stop_systemd_unit", stop_scope)
    retry = _snapshot_processes_for_session_reset(session_key)
    assert retry == (target.id,)
    assert _kill_processes_for_session_reset(retry) == 1
    assert stopped_units == [target.systemd_unit]
    assert not target._scope_stop_pending
    assert not clean._scope_stop_pending
    assert persisted._scope_stop_pending and foreign._scope_stop_pending
    assert _snapshot_processes_for_session_reset(session_key) == ()
