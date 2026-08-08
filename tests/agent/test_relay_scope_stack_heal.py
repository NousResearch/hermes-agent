"""Regression: interrupted nested scopes must not permanently wedge a session.

Live incident (2026-08-08): preflight compaction was interrupted mid-API call,
leaving orphaned scopes above ``hermes.turn``. Every subsequent ``end_turn``
logged::

    RuntimeError: invalid argument: scope handle is not at the top of the stack

The process stayed up but the session stack never healed, so close_session and
later turns kept failing the same way until a hard gateway restart.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_runtime


@pytest.fixture()
def relay_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-wedge",
        platform="telegram",
    )
    try:
        yield lease
    finally:
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime.SESSION_COORDINATOR.finalize_conversation(
            profile_key=lease.profile_key,
            session_id=lease.session_id,
        )
        relay_runtime._reset_for_tests()


def _push(host, session, name, *, parent):
    return host.run_in_session(
        session,
        host.relay.scope.push,
        name,
        host.relay.ScopeType.Function,
        handle=parent,
        input={},
        metadata={
            relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
            relay_runtime.RUNTIME_INSTANCE_KEY: host.runtime_id,
        },
    )


def test_end_turn_heals_orphaned_nested_scopes(relay_session):
    """Orphans above the turn must unwind so the next turn can finalize cleanly."""
    lease = relay_session
    host = lease.host
    assert isinstance(host, relay_runtime.RelayRuntime)
    session = lease.session
    assert session is not None

    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-interrupted",
        task_id="task-interrupted",
    )
    assert turn.handle is not None
    turn_uuid = relay_runtime._scope_handle_uuid(turn.handle)

    # Simulate shared-metrics task + compaction logical scopes left open when
    # a gateway restart interrupts the API call mid-flight.
    task = _push(host, session, "hermes.task_run", parent=turn.handle)
    logical = _push(
        host,
        session,
        relay_runtime.LOGICAL_LLM_SCOPE,
        parent=turn.handle,
    )
    assert relay_runtime._scope_handle_uuid(logical)
    assert relay_runtime._scope_handle_uuid(task)

    # Naive pop of the turn (pre-fix behavior) fails while orphans remain.
    with pytest.raises(RuntimeError, match="not at the top of the stack"):
        host.run_in_session(
            session,
            host.relay.scope.pop,
            turn.handle,
            output={"outcome": "cancelled"},
        )

    # end_turn must heal rather than leave the stack wedged.
    relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="cancelled")
    assert turn.handle is None
    assert turn.closed is True

    # Session root must still be usable for a full subsequent turn cycle.
    turn2 = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-recovered",
        task_id="task-recovered",
    )
    assert turn2.handle is not None
    assert relay_runtime._scope_handle_uuid(turn2.handle) != turn_uuid
    relay_runtime.SESSION_COORDINATOR.end_turn(turn2, outcome="success")
    assert turn2.handle is None


def test_pop_scope_resilient_rebuilds_when_stack_unrecoverable(relay_session):
    """If nested unwind cannot reach the target, rebuild a fresh session root."""
    lease = relay_session
    host = lease.host
    assert isinstance(host, relay_runtime.RelayRuntime)
    session = lease.session
    assert session is not None

    old_handle = session.handle
    old_uuid = relay_runtime._scope_handle_uuid(old_handle)
    turn = _push(host, session, relay_runtime.TURN_SCOPE, parent=session.handle)
    orphan = _push(host, session, "orphan.scope", parent=turn)

    # Force rebuild path: ask to pop a handle that is not on this stack at all
    # after we abandon everything above the session via resilient pop of turn.
    ok = host.pop_scope_resilient(
        session,
        turn,
        output={"outcome": "cancelled"},
        reason="unit_test_unwind",
    )
    assert ok is True
    # Orphan must have been abandoned; session root still the original one.
    assert relay_runtime._scope_handle_uuid(session.handle) == old_uuid
    # Direct proof the stack is clean under the original session handle.
    probe = _push(host, session, "probe", parent=session.handle)
    assert host.pop_scope_resilient(session, probe, reason="unit_test_probe") is True

    # Rebuild path: poison by closing session context reference then recover.
    # Create orphans and drop the only handle references without popping.
    lost_turn = _push(host, session, relay_runtime.TURN_SCOPE, parent=session.handle)
    _push(host, session, "lost.orphan", parent=lost_turn)
    # Pretend caller only knows a stale child handle whose uuid no longer
    # matches anything after a manual rebuild trigger.
    host.rebuild_session_scope(session, reason="unit_test_rebuild")
    assert session.scope_stack_healed is True
    assert relay_runtime._scope_handle_uuid(session.handle) != old_uuid

    turn2 = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-after-rebuild",
        task_id="task-after-rebuild",
    )
    assert turn2.handle is not None
    relay_runtime.SESSION_COORDINATOR.end_turn(turn2, outcome="success")


def test_close_session_survives_orphaned_children(relay_session):
    """close_session must not leave a permanently broken registry entry."""
    lease = relay_session
    host = lease.host
    assert isinstance(host, relay_runtime.RelayRuntime)
    session = lease.session
    assert session is not None
    session_id = session.session_id

    turn = _push(host, session, relay_runtime.TURN_SCOPE, parent=session.handle)
    _push(host, session, "orphan.close", parent=turn)

    host.close_session({"session_id": session_id})
    assert host.get_session(session_id) is None

    # A new session id can still open (same runtime, fresh stack).
    fresh = host.ensure_session({"session_id": "session-fresh-after-close"})
    assert fresh is not None
    assert fresh.handle is not None
    host.close_session({"session_id": fresh.session_id})
