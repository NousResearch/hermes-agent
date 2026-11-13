"""Read-only parked-goal evidence and exact claim consumption.

The scanner never clears a barrier. Its fingerprint binds the whole observed goal,
including legacy rows, so a control made after observation invalidates the wake.
"""
import hashlib
import uuid
from typing import Any, cast

CONTROL_PREFIX = 'goal-control:'


def state_token(state):
    return hashlib.sha256(state.to_json().encode()).hexdigest()


def observed_token(state):
    return hashlib.sha256(getattr(state, '_persisted_raw', state.to_json()).encode()).hexdigest()


def parked(state):
    return bool(state is not None and state.status == 'active' and (
        state.waiting_until or state.waiting_on_pid is not None or state.waiting_on_session is not None))


def ready(state, now):
    """Non-consuming readiness; legacy worker counts are not proof of completion."""
    from hermes_cli.goals import _MAX_BARRIER_WAIT_S, _pid_alive, _session_waiting
    if not parked(state):
        return False
    if state.waiting_on_session is not None:
        waiting = _session_waiting(state.waiting_on_session)
    elif state.waiting_on_pid is not None:
        waiting = _pid_alive(state.waiting_on_pid)
    else:
        waiting = now < state.waiting_until
    expired = bool(state.waiting_since and not state.waiting_until
                   and now - state.waiting_since > _MAX_BARRIER_WAIT_S)
    return not waiting or expired


def read_state(conn, target):
    from hermes_cli.goals import GoalState
    # One statement observes goal bytes and their private incarnation in the same snapshot.
    row = conn.execute('SELECT g.value,c.value FROM state_meta g '
                       'LEFT JOIN state_meta c ON c.key=? WHERE g.key=?',
                       (CONTROL_PREFIX + target, 'goal:' + target)).fetchone()
    if row is None:
        return None
    state = GoalState.from_json(row[0])
    state._persisted_raw = row[0]
    # The host witness is deliberately not a serialized GoalState dataclass field.
    cast(Any, state)._control_incarnation = row[1]
    return state


def persist_state(conn, target, state):
    """Atomically replace goal bytes, advance their incarnation and retire old duty.

    Even byte-identical controls must leave evidence before a host decision exists.
    Return the committed witness to the caller only after its transaction succeeds.
    """
    incarnation = uuid.uuid4().hex
    conn.executemany('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     [('goal:' + target, state.to_json()), (CONTROL_PREFIX + target, incarnation)])
    conn.execute('DELETE FROM state_meta WHERE key=?', ('goal-decision:' + target,))
    return incarnation


def consume(conn, target, token, now):
    state = read_state(conn, target)
    if state is None or observed_token(state) != token or not ready(state, now):
        return False
    state.clear_wait()
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (state.to_json(), 'goal:' + target))
    return True
