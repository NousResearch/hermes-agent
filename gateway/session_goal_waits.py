"""Private persisted standing-goal wake admission and FIFO fences."""
import json
import logging
from functools import partial

from gateway.config import Platform
from hermes_state_runtime import RuntimeStoreError


class GoalPriority(RuntimeStoreError):
    """An unstarted goal reservation must yield its FIFO place, not lose its intent."""

    def __init__(self, admission_id):
        super().__init__('session_busy')
        self.admission_id = admission_id


def descriptor(payload):
    automation = payload.get('local_automation_v1') or payload.get('native_text_v1', {}).get('automation', {})
    return automation.get('goal_wait')


def metadata(value):
    if value is None:
        return {}
    if (not isinstance(value, dict) or set(value) not in ({'target', 'token'}, {'target', 'token', 'after'})
            or not all(isinstance(value[k], str) and value[k] for k in ('target', 'token'))
            or ('after' in value and (not isinstance(value['after'], str) or not value['after']))):
        raise RuntimeStoreError('invalid_params')
    return {'goal_wait': dict(value)}


def eligible(authority, owner, target, route, conn, *, wait=True):
    from gateway.session_completion_notes import _closed_by_user
    from gateway.session_viewer_close import CLOSE_PREFIX, ORPHAN_REASONS, _PendingClose
    from hermes_cli import goals
    from hermes_cli.goals_waits import ready, read_state
    live = authority.sessions.get(owner)
    entry = authority.runner.session_store.lookup_by_session_key(route)
    if (live is None or entry is None or entry.suspended or entry.session_id != target
            or _closed_by_user(authority, owner)):
        return False
    close = conn.execute('SELECT value FROM state_meta WHERE key=?', (CLOSE_PREFIX + owner,)).fetchone()
    if close is not None and _PendingClose.from_record(close[0]).reason not in ORPHAN_REASONS:
        return False
    if live.source.platform == Platform.LOCAL:
        from gateway.session_local_recovery import local_adapter
        policy = local_adapter(authority).policies.get(live.source.chat_id)
        if policy is None or policy.source not in {'cli', 'tui', 'gui'}:
            return False
    state = read_state(conn, target)
    return ready(state, goals.time.time()) if wait else state is not None and state.status == 'active'


def guard(authority, payload, owner, route, *, claim=False, head=None, replaces=None):
    evidence = descriptor(payload)
    if evidence is None:
        return None
    def check(conn):
        from hermes_cli import goals
        from hermes_cli.goals_waits import consume, read_state, observed_token
        if head is not None:
            current = conn.execute("SELECT admission_id FROM session_admissions WHERE target_session_id=? "
                                   "AND status='queued' ORDER BY seq LIMIT 1", (owner,)).fetchone()
            if current is None or current[0] != head:
                return
        target, token = evidence['target'], evidence['token']
        wait = 'after' not in evidence
        state = read_state(conn, target)
        valid = (state is not None and observed_token(state) == token
                 and eligible(authority, owner, target, route, conn, wait=wait))
        if not valid:
            raise RuntimeStoreError('admission_conflict')
        if not wait:
            from gateway.session_goal_decisions import check
            check(authority, conn, evidence, owner, route, payload, claim=claim, replaces=replaces)
        from gateway.session_viewer_automation import viewer_driven
        from agent.estop import check_paused
        if viewer_driven(authority, owner) or check_paused('goal', logging.getLogger('gateway.run')):
            raise RuntimeStoreError('session_busy')
        if replaces is not None:
            previous = conn.execute('SELECT payload_json FROM session_admissions WHERE admission_id=? '
                                    'AND target_session_id=?', (replaces, owner)).fetchone()
            if previous is None or descriptor(json.loads(previous[0])) != evidence:
                raise RuntimeStoreError('admission_conflict')
            return  # the canonical replacement guard retires only this unstarted reservation
        # Autonomous work never overtakes pending human input or a real completion.
        others = conn.execute("SELECT admission_id,status FROM session_admissions WHERE target_session_id=? "
                              "AND status IN ('queued','started','unknown')", (owner,)).fetchall()
        if any(row[0] != head and not (not claim and not wait and row[1] == 'started'
                                      and row[0] == evidence['after']) for row in others):
            if head and any(row[0] != head and row[1] == 'queued' for row in others):
                raise GoalPriority(head)
            raise RuntimeStoreError('session_busy')
        if claim and wait and not consume(conn, target, token, goals.time.time()):
            raise RuntimeStoreError('admission_conflict')
    return check


async def yield_priority(authority, ref, admission_id):
    """Replace the unstarted reservation at the FIFO tail in one canonical admission write.

    The successor's identity is tied to this predecessor, so a lost ACK cannot add a second
    successor. The persisted goal/barrier is untouched; started, unknown and terminal work is
    never requeued. A crash before the replacement commit leaves the original queued instead.
    """
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import get_session_admission
    row = get_session_admission(authority.db, admission_id=admission_id)
    if row is None or row['status'] != 'queued':
        return
    evidence = descriptor(row['payload'])
    live = authority.sessions[ref.session_id]
    source = live.source
    if source.platform != Platform.LOCAL:
        from gateway.session_envelope import restore_native
        source = restore_native(row['payload'], authority.runner).source
    identity = json.dumps(['goal-yield', admission_id], separators=(',', ':'))
    event = MessageEvent(text=row['payload']['text'], source=source, internal=True,
                         message_id=identity, metadata={'gateway_session_key': live.route,
                         'gateway_session_id': evidence['target'], 'goal_wait': evidence})
    try:
        await authority.admit_automation(authority.runner._adapter_for_source(source), event, identity,
                                         replaces=admission_id)
    except RuntimeStoreError as exc:
        if exc.reason != 'stale_generation':
            raise
        # Cancellation/claim won while the replacement awaited its write; re-read normally.


async def admit_wait(authority, entry, source, state, *, after=None, prompt=None):
    from gateway.platforms.event import MessageEvent
    from gateway.session_automation import committed_route_envelope
    from gateway.session_envelope import restore_native
    from hermes_cli.goals import GoalManager
    from hermes_cli.goals_waits import observed_token
    runner = authority.runner
    owner = source.chat_id if source.platform == Platform.LOCAL else authority.logical_owner(entry.session_id)
    evidence = {'target': entry.session_id, 'token': observed_token(state)}
    if after is not None:
        evidence['after'] = after
    identity = json.dumps(['goal-wait' if after is None else 'goal-continuation', owner,
                           evidence['target'], evidence['token']], separators=(',', ':'))
    if after is not None:
        from gateway.session_goal_decisions import identity as continuation_identity
        identity = continuation_identity(owner, evidence['target'], evidence['token'], after)
    if source.platform != Platform.LOCAL:
        envelope = committed_route_envelope(authority, entry)
        if envelope is None:
            return
        source = restore_native({'text': '', 'native_text_v1': {
            k: v for k, v in envelope.items() if k != 'media'}}, runner).source
    mgr = GoalManager(entry.session_id)
    mgr._state = state
    if after is None:
        prompt = mgr.next_continuation_prompt()
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeStoreError('invalid_params')
    event = MessageEvent(text=prompt, source=source, internal=True,
                         message_id=identity, metadata={'gateway_session_key': entry.session_key,
                         'gateway_session_id': entry.session_id, 'goal_wait': evidence})
    await authority.admit_automation(runner._adapter_for_source(source), event, identity)


async def admit_continuation(runner, source, manager, prompt):
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is None:
        return False
    entry = runner.session_store.lookup_by_session_key(runner._session_key_for_source(source))
    if entry is None or entry.session_id != manager.session_id:
        return True  # the route left this decision; never retarget it
    owner = source.chat_id if source.platform == Platform.LOCAL else authority.logical_owner(entry.session_id)
    from gateway.session_goal_decisions import record
    duty = await record(authority, entry, owner, manager, prompt)
    await admit_wait(authority, entry, source, manager.state, after=duty['after'], prompt=duty['prompt'])
    return True


async def pause_route(runner, route):
    """Explicit Stop, including an idle LOCAL slash command, invalidates only this goal duty."""
    from gateway.session_authorities import active_authority
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import _epoch
    from hermes_cli.goals_waits import read_state, persist_state
    authority = active_authority(runner)
    entry = runner.session_store.lookup_by_session_key(route) if authority is not None else None
    if entry is None:
        return
    target = entry.session_id
    def write(conn):
        _epoch(conn, authority.epoch)
        state = read_state(conn, target)
        if state is not None and state.status == 'active':
            state.status, state.paused_reason = 'paused', 'stop_command'
            state.clear_wait()
            persist_state(conn, target, state)
    await tracked_write(authority, partial(authority.db._execute_write, authority._admission_gate(write)), ordered=True)


async def claim(authority, ref, live, first):
    """Commit the checked FIFO head and consume a goal barrier in that exact transaction."""
    from gateway.session_runtime_workers import tracked_write
    from gateway.session_authority import claim_session_input
    authority._require_admission_open()
    live.rescan = False
    check = guard(authority, first['payload'], ref.session_id, live.route, claim=True,
                  head=first['admission_id']) if first is not None else None
    row = await tracked_write(authority, partial(claim_session_input, authority.db, epoch=authority.epoch,
        session_id=ref.session_id, _guard=authority._admission_gate(check),
        head=first['admission_id'] if first is not None else None),
        then=partial(authority._stamp_claim, ref, live))
    return (False if row is None and live.rescan else row), first


async def preflight(authority, ref, row):
    """Fence before LOCAL recovery can reopen a user-closed conversation."""
    evidence = descriptor(row['payload'])
    if evidence is None:
        return
    from gateway.session_runtime_workers import tracked_write
    live = authority.sessions[ref.session_id]
    await tracked_write(authority, partial(authority.db._execute_write,
        authority._admission_gate(guard(authority, row['payload'], ref.session_id, live.route,
                                        head=row['admission_id']))))
