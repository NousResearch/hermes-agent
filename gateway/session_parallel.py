"""Conversation correlation over independently owned native work sessions.

The public source remains the real sender and destination. Only owner-created
bindings change the execution route; public metadata cannot name one.
"""
import asyncio
from dataclasses import dataclass
import hashlib
import json
import re

from hermes_state_runtime import RuntimeStoreError


@dataclass(frozen=True)
class WorkBinding:
    work_id: str
    scope: str
    route: str
    requester: str


def binding_of(source):
    value = getattr(source, 'parallel_work', None)
    return value if isinstance(value, WorkBinding) else None


def authority_for_source(runner, source):
    from gateway.session_authorities import active_authority, authority_for_home
    from gateway.session_identity import identity_of
    identity = identity_of(source)
    if identity is None:
        return active_authority(runner)
    return authority_for_home(runner, identity.runtime_home)


def conversation_scope(source):
    from gateway.session_identity import identity_of
    identity = identity_of(source)
    return json.dumps([str(identity.runtime_home) if identity else source.profile,
                       identity.transport_profile if identity else source.profile,
                       source.platform.value, source.scope_id, source.chat_id,
                       source.thread_id or source.prospective_thread_id], separators=(',', ':'))


def restore_binding(source, values):
    if not isinstance(values, dict) or set(values) != {'work_id', 'scope', 'route', 'requester'}:
        raise RuntimeStoreError('admission_conflict', field='parallel_work')
    binding = WorkBinding(**values)
    if (not all(isinstance(value, str) and value for value in values.values())
            or not re.fullmatch('[a-f0-9]{64}', binding.work_id)
            or not binding.route.endswith(':work:' + binding.work_id)
            or binding.scope != conversation_scope(source)):
        raise RuntimeStoreError('admission_conflict', field='parallel_work')
    source.parallel_work = binding
    return source


def restore_entry(runner, entry, source):
    from hermes_state_parallel import binding_values, by_route
    authority = authority_for_source(runner, source)
    row = by_route(authority.db, entry.session_key) if authority is not None else None
    if row is not None:
        if authority.logical_owner(entry.session_id) != row['session_id']:
            raise RuntimeStoreError('admission_conflict', field='parallel_work')
        restore_binding(source, binding_values(row))
    return source


def admission_guard(runner, source, session_id, message_id, guard):
    binding = binding_of(source)
    if binding is None:
        return guard
    from hermes_state_parallel import commit_work
    from gateway.run import _load_gateway_config
    display = _load_gateway_config().get('display') or {}
    limit = display.get('parallel_max_runs', 2)
    if type(limit) is not int or not 1 <= limit <= 8:
        raise RuntimeStoreError('invalid_params', field='display.parallel_max_runs', detail='use an integer from 1 to 8')

    def write(conn):
        if guard is not None:
            guard(conn)
        commit_work(conn, binding, session_id, message_id, limit=limit)
    return write


_APPROVAL_WORDS = frozenset({'yes', 'no', 'approve', 'deny', 'allow', 'reject'})
_SCOPED_COMMANDS = frozenset({'stop', 'approve', 'deny', 'queue', 'status', 'discard'})
_TARGET_NOTICE = 'Reply to the request or a confirmed bot response for the work you mean. No work was changed.'
_STALE_PROMPT = 'That question is no longer pending. Reply to the current question or the original request. No work was changed.'


@dataclass(frozen=True)
class PromptTarget:
    kind: str
    request_id: str
    generation: int
    route: str


def prompt_target(event):
    target = getattr(event, '_parallel_prompt', None)
    return target if isinstance(target, PromptTarget) else None


def prompt_is_current(runner, source, target):
    authority = authority_for_source(runner, source)
    sid = runner.session_store.peek_session_id(target.route)
    row = authority.db.get_session(sid) if authority is not None and sid else None
    if row is None or row['runtime_generation'] != target.generation:
        return False
    if target.kind == 'approval':
        from tools.approval import list_gateway_approvals
        return any(p['request_id'] == target.request_id for p in list_gateway_approvals(target.route))
    from tools.clarify_gateway import get_pending
    entry = get_pending(target.request_id)
    return entry is not None and entry.session_key == target.route


def pin_prompt(runner, event, row, command):
    # Work-level controls still address the work, not the old question. Text
    # and approval commands on a question receipt must retain its exact fence.
    if not row.get('prompt_kind') or command in {'stop', 'discard', 'queue', 'status'}:
        return None
    target = PromptTarget(row['prompt_kind'], row['prompt_id'], row['prompt_generation'], row['route'])
    if not prompt_is_current(runner, event.source, target):
        return _STALE_PROMPT
    event._parallel_prompt = target
    return None


def selected_approval_ids(runner, event, ids):
    target = prompt_target(event)
    if target is None:
        return ids
    return ([target.request_id] if target.kind == 'approval' and target.request_id in ids
            and prompt_is_current(runner, event.source, target) else [])


async def route_input(runner, event):
    """Pin execution ownership before either guard; return a refusal, else None."""
    from gateway.session_ingress import executing_admission
    if event.internal or executing_admission.get():
        return None
    source = event.source
    authority = authority_for_source(runner, source)
    mode = runner._effective_busy_input_mode(source)
    if authority is None:
        return 'Parallel mode needs the native session authority; no work was admitted.' if mode == 'parallel' else None
    if not runner._is_user_authorized_for_source(source, allow_adapter_delegation=source.role_authorized):
        return None  # ordinary ingress still owns the authorization refusal
    if binding_of(source) is not None:
        return None
    if source.platform.value != 'discord':
        return 'Parallel mode currently supports Discord messaging only. Set queue, steer or interrupt for this surface.' if mode == 'parallel' else None
    from hermes_state_parallel import active_work, binding_values, by_anchor
    scope = conversation_scope(source)
    anchor = event.reply_to_message_id
    target = await asyncio.to_thread(by_anchor, authority.db, scope, anchor) if anchor else None
    command = event.get_command() if event.allow_gateway_control else None
    control = command in _SCOPED_COMMANDS or (event.allow_gateway_control and event.text.strip().lower() in _APPROVAL_WORDS)
    active = await asyncio.to_thread(active_work, authority.db, scope)
    if target is None and control and active:
        if anchor:
            return _TARGET_NOTICE
        eligible = [row for row in active if row['requester'] == str(source.user_id)]
        if len(eligible) != 1:
            return _TARGET_NOTICE
        target = eligible[0]
    if target is not None:
        if target['requester'] != str(source.user_id):
            return 'Only the originating sender can continue or control this parallel work.'
        notice = pin_prompt(runner, event, target, command)
        if notice is not None:
            return notice
        if command and command not in _SCOPED_COMMANDS:
            return 'This conversation command is not supported on parallel work. Use /stop for that work or send /busy outside a work reply.'
        restore_binding(source, binding_values(target))
        return None
    if command:
        if active and command not in {'busy', 'help', 'status'}:
            return _TARGET_NOTICE
        return None
    if mode != 'parallel':
        return None
    if anchor:
        return _TARGET_NOTICE  # a missing receipt must not select some other work
    if not event.message_id or not source.user_id:
        return 'Parallel mode requires a stable message ID and sender ID. No work was admitted.'
    request = json.dumps([scope, str(source.user_id), str(event.message_id)], separators=(',', ':'))
    wid = hashlib.sha256(request.encode()).hexdigest()
    route = runner.session_store._generate_session_key(source) + ':work:' + wid
    source.parallel_work = WorkBinding(wid, scope, route, str(source.user_id))
    return None


async def route_adapter_input(adapter, event):
    runner = getattr(adapter, 'gateway_runner', None)
    if runner is None:
        return None
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_identity import identity_of
    identity = identity_of(event.source)
    home = identity.runtime_home if identity else None
    async with runner._async_scope_or_null(_async_profile_runtime_scope, home):
        return await route_input(runner, event)


async def prepare_inbound(runner, event, source, internal):
    notice = await route_input(runner, event)
    key = runner._session_key_for_source(source)
    if notice is not None:
        return key, notice
    reply = None if internal else await runner._hm_pending_reply_intercepts(event, source, key)
    return key, reply


def record_output(runner, source, result, *, prompt=None):
    binding = binding_of(source)
    if binding is None:
        return
    authority = authority_for_source(runner, source)
    if authority is None:
        raise RuntimeStoreError('profile_mismatch')
    from hermes_state_parallel import record_output as record
    record(authority.db, binding, result, prompt=prompt)


async def record_delivery(runner, source, send, *, prompt=None):
    """Capture an eventual receipt, including sends acknowledged after a timeout."""
    result = await send
    await asyncio.to_thread(record_output, runner, source, result, prompt=prompt)
    return result


def physical_claim(runner, source):
    """Persist before executor submission; a lost owner retains unknown occupancy."""
    import uuid
    binding = binding_of(source)
    if binding is None:
        return lambda: None
    authority = authority_for_source(runner, source)
    if authority is None:
        raise RuntimeStoreError('profile_mismatch')
    token = uuid.uuid4().hex
    from hermes_platform.host.facts import kernel_boot_identity
    machine, boot = kernel_boot_identity()
    authority.db._execute_write(lambda conn: conn.execute(
        'INSERT INTO parallel_work_physical(token,work_id,machine_id,boot_id) VALUES(?,?,?,?)',
        (token, binding.work_id, machine, boot)))
    return lambda: authority.db._execute_write(lambda conn: conn.execute(
        'DELETE FROM parallel_work_physical WHERE token=?', (token,)))


async def stop_work(runner, source, route):
    await runner._hm_stop_owner_turn(route)
    authority = authority_for_source(runner, source)
    if authority is None:
        return
    from gateway.session_contract import SessionRef
    from gateway.session_ancillary import _interrupt_delegations
    from gateway.session_runtime_workers import physical_workers
    from agent.interrupt_compat import request_hard_interrupt
    sid = runner.session_store.peek_session_id(route)
    if sid in authority.sessions:
        _interrupt_delegations(authority, SessionRef(authority.profile_id, sid))
    for worker, holder in list(physical_workers(authority)):
        if worker.task_id == sid and holder and holder[0] is not None:
            request_hard_interrupt(holder[0], '/stop')


async def execute_turn(runner, source, run):
    from agent.physical_work import bind_work, submit_work
    with bind_work(lambda: physical_claim(runner, source)):
        future = submit_work(runner._get_executor(), run)
    return await asyncio.wrap_future(future)


def physical_count(authority):
    db = getattr(authority, 'db', None)
    if not callable(getattr(db, '_read_ctx', None)):
        return 0
    with db._read_ctx() as conn:
        return conn.execute('SELECT COUNT(*) FROM parallel_work_physical').fetchone()[0]


def batch_key(adapter, event, key):
    runner = adapter.gateway_runner
    if (event.source.platform.value == 'discord' and runner is not None
            and (event.reply_to_message_id or runner._effective_busy_input_mode(event.source) == 'parallel')):
        return key + ':input:' + str(event.message_id)
    return key


def drain_counts(runner):
    from gateway.run_runtime import managed_turn_count
    from gateway.session_authorities import all_authorities
    return (runner._running_agent_count() + managed_turn_count(runner)
            + sum(physical_count(a) for a in all_authorities(runner)),
            runner._active_cron_job_count(), runner._active_api_run_count(),
            runner._active_deferred_agent_worker_count())
