"""Host-owned outstanding post-turn decisions, not producer-supplied wake permission."""
import json
from functools import partial

from hermes_state_runtime import RuntimeStoreError

PREFIX = 'goal-decision:'


def identity(owner, target, token, after):
    return json.dumps(['goal-continuation', owner, target, token, after], separators=(',', ':'))


def read(conn, target):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (PREFIX + target,)).fetchone()
    return json.loads(row[0]) if row is not None else None


def proven(authority, conn, duty, owner, target, route, token, *, allow_started=False):
    """A completed incarnation survives restart; an unfinished one never becomes permission."""
    from hermes_cli.goals_waits import observed_token, read_state
    if (duty is None or duty['owner'] != owner or duty['target'] != target
            or duty['route'] != route or duty['profile'] != authority.profile_id or duty['token'] != token
            or not duty.get('incarnation')):
        return False
    state = read_state(conn, target)
    if (state is None or observed_token(state) != token
            or getattr(state, '_control_incarnation', None) != duty['incarnation']):
        return False
    row = conn.execute('SELECT target_session_id,status,outcome,owner_epoch,generation '
                       'FROM session_admissions WHERE admission_id=?', (duty['after'],)).fetchone()
    if (row is None or row[0] != owner or row[3] != duty['epoch'] or row[4] != duty['generation']):
        return False
    if row[1:3] == ('terminal', 'completed'):
        return True
    live = authority.sessions.get(owner)
    return bool(allow_started and row[1] == 'started' and duty['epoch'] == authority.epoch
                and authority.executing.get(owner) == duty['generation'] and live is not None
                and live.event_stream.execution.get('admission_id') == duty['after'])


def outstanding(authority, conn, target, state, *, allow_started=False):
    from hermes_cli.goals_waits import observed_token, parked
    if state is None or state.status != 'active' or parked(state):
        return None
    duty = read(conn, target)
    if duty is not None and proven(authority, conn, duty, duty['owner'], target, duty['route'],
                                   observed_token(state), allow_started=allow_started):
        return duty
    return None


async def record(authority, entry, owner, manager, prompt):
    """Persist the fresh judge's exact CAS result before successor admission can be refused.

    This seam is called only by the production post-turn consumer. Metadata cannot create a
    record, and a crash between this write and predecessor settlement leaves unusable intent.
    """
    from gateway.session_runtime_workers import tracked_write
    from hermes_cli.goals_waits import observed_token
    from hermes_state_runtime import _epoch
    live = authority.sessions.get(owner)
    if live is None:
        raise RuntimeStoreError('admission_conflict')
    stamp = dict(live.event_stream.execution)
    duty = dict(owner=owner, target=entry.session_id, route=entry.session_key,
                profile=authority.profile_id, token=observed_token(manager.state),
                incarnation=getattr(manager.state, '_control_incarnation', None),
                after=stamp.get('admission_id'), epoch=stamp.get('authority_epoch'),
                generation=stamp.get('execution_generation'))
    duty['identity'] = identity(owner, entry.session_id, duty['token'], duty['after'])
    duty['prompt'] = prompt
    from gateway.session_results import execution_result
    captured = execution_result.get()
    if captured is not None:
        # Installed before the first fallible write. A refused/rolled-back record cannot let
        # this exact predecessor settle completed without either duty or a newer control.
        captured['goal_continuation_write'] = partial(settle, authority, duty)
    def write(conn):
        _epoch(conn, authority.epoch)
        if (not proven(authority, conn, duty, owner, entry.session_id, entry.session_key,
                       duty['token'], allow_started=True)
                or authority.executing.get(owner) != duty['generation']):
            raise RuntimeStoreError('admission_conflict')
        persist(conn, duty)
    await tracked_write(authority, partial(authority.db._execute_write,
                                          authority._admission_gate(write, queue_only=True)), ordered=True)
    return duty


def persist(conn, duty):
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                 'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (PREFIX + duty['target'], json.dumps(duty, sort_keys=True)))


def settle(authority, duty, conn, row):
    """Retain exact intent WITH successful settlement, never retry model work.

    The runtime transaction fenced this claim or its same-owner captured-result resolution.
    Unlike admission, settlement runs after the executing stamp cleared; its actual row is
    the predecessor proof. Failure rolls back the terminal transition, never model work.
    """
    from gateway.session_goal_waits import eligible
    from hermes_cli.goals_waits import observed_token, read_state
    if ((row['admission_id'], row['target_session_id'], row['owner_epoch'], row['generation'])
            != (duty['after'], duty['owner'], duty['epoch'], duty['generation'])
            or duty['profile'] != authority.profile_id or not duty.get('incarnation')):
        raise RuntimeStoreError('admission_conflict')
    state = read_state(conn, duty['target'])
    if (state is None or observed_token(state) != duty['token']
            or getattr(state, '_control_incarnation', None) != duty['incarnation']
            or not eligible(authority, duty['owner'], duty['target'], duty['route'], conn, wait=False)):
        return  # a newer control/close/reset owns the goal; never recreate superseded intent
    saved = read(conn, duty['target'])
    if saved is None:
        authority._require_admission_open(queue_only=True)
        persist(conn, duty)
    elif saved != duty:
        raise RuntimeStoreError('admission_conflict')


def check(authority, conn, evidence, owner, route, payload, *, claim=False, replaces=None):
    duty = read(conn, evidence['target'])
    if (duty is None or not proven(authority, conn, duty, owner, evidence['target'], route, evidence['token'],
                   allow_started=not claim) or duty['after'] != evidence['after']):
        raise RuntimeStoreError('admission_conflict')
    automation = payload.get('local_automation_v1') or payload['native_text_v1']['automation']
    expected = duty['identity']
    if replaces is not None:
        previous = conn.execute('SELECT request_id FROM session_admissions WHERE admission_id=? '
                                'AND target_session_id=?', (replaces, owner)).fetchone()
        if previous is None or previous[0] != expected:
            raise RuntimeStoreError('admission_conflict')
        expected = json.dumps(['goal-yield', replaces], separators=(',', ':'))
    if automation['identity'] != expected or payload['text'] != duty['prompt']:
        raise RuntimeStoreError('admission_conflict')
    if replaces is not None:
        duty['identity'] = expected
        conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                     (json.dumps(duty, sort_keys=True), PREFIX + evidence['target']))
    if claim:
        conn.execute('DELETE FROM state_meta WHERE key=?', (PREFIX + evidence['target'],))
