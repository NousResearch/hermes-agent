"""Trusted dispatcher claims enter the ordinary owner's durable admission queue."""
import asyncio
import contextlib
from contextlib import closing
from dataclasses import asdict, replace
import os
import threading
import json
from pathlib import Path
import sqlite3
import time

from hermes_state_runtime import RuntimeStoreError


def build_kanban_policy(connection, params, config):
    from hermes_cli import kanban_db as kb
    from hermes_cli.profiles import resolve_profile_env
    from gateway.session_policy import build_policy
    authority, actor = connection.authority, connection.actor
    if not connection.native_owner or 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (set(params) not in ({'board', 'task_id', 'run_id', 'claim_lock'}, {'board', 'task_id', 'run_id', 'claim_lock', 'db'})
            or any(not isinstance(params[k], str) or not params[k] for k in ('board', 'task_id', 'claim_lock'))
            or type(params['run_id']) is not int):
        raise RuntimeStoreError('invalid_params')
    try:
        board = kb._require_slug(params['board'])
        path = Path(params['db']).resolve(strict=True) if 'db' in params else kb.kanban_db_path(board=board).resolve()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            task = kb.get_task(conn, params['task_id'])
            run = conn.execute('SELECT * FROM task_runs WHERE id=?', (params['run_id'],)).fetchone()
            if (task is None or run is None or task.status != 'running' or run['status'] != 'running'
                    or task.current_run_id != params['run_id'] or run['task_id'] != task.id
                    or task.claim_lock != params['claim_lock'] or run['claim_lock'] != task.claim_lock):
                raise RuntimeStoreError('stale_kanban_claim')
            if Path(resolve_profile_env(task.assignee)).resolve() != Path(authority.profile_id).resolve():
                raise RuntimeStoreError('profile_mismatch')
            bound = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
                                 (task.id, task.current_run_id)).fetchone()
            if bound and json.loads(bound[0])['request_id'] != 'kanban:' + json.dumps([board, task.id, task.current_run_id], separators=(',', ':')):
                raise RuntimeStoreError('stale_kanban_claim')
            context = kb.build_worker_context(conn, task.id)
            skills = list(task.skills or ())
            if kb._retry_status_for_run(conn, task.id, task.current_run_id) == 'review':
                # Same rule as the dispatcher's review lane: the reviewer always loads sdlc-review.
                skills = list(dict.fromkeys([*skills, 'sdlc-review']))
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise RuntimeStoreError('invalid_kanban_claim') from exc
    cwd = task.workspace_path
    params_policy = {'source': 'cli', 'cwd': cwd}
    for name, value in (('model', task.model_override), ('provider', task.provider_override), ('reasoning', task.reasoning_effort)):
        if value:
            params_policy[name] = value
    private = {}
    policy = build_policy(params_policy, config, private_secrets=private)
    if policy.model is None:
        from gateway.run import _resolve_gateway_model
        policy = replace(policy, model=_resolve_gateway_model(policy.config()))
    context = dict(params, workspace=cwd, branch=task.branch_name, tenant=task.tenant,
        profile=task.assignee, workspaces_root=str(kb.workspaces_root(board=board).resolve()),
        skills=skills, goal_mode=task.goal_mode, goal_max_turns=task.goal_max_turns,
        goal_text='\n\n'.join(p for p in (task.title, task.body) if p), context=context,
        max_runtime_seconds=task.max_runtime_seconds, accept_hooks=True)
    context['db'] = str(path)
    request = json.loads(policy.request_json)
    request.update(source='kanban', board=board, task_id=task.id, run_id=task.current_run_id)
    return replace(policy, source='kanban', platform='cli', toolsets=tuple(sorted(set(policy.toolsets) | {'kanban'})),
                   request_json=json.dumps(request, sort_keys=True), kanban_json=json.dumps(context)), private


async def run_task(connection, params):
    if not connection.native_owner or 'session:create' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from gateway.run import _load_gateway_config
    from gateway.session_local import create_local_session
    from gateway.session_contract import Submission
    from gateway.session_local_recovery import local_identity
    from hermes_state_local import local_receipt
    from gateway.session_policy import restore_policy
    authority = connection.authority
    # Resolved off-loop up front: no await separates the saved-claim check from session
    # creation. The Kanban claim check and its marker share one off-loop transaction; a
    # concurrent retry of the same attempt meets that marker (same session) or a stale claim.
    owner_db = str(await asyncio.to_thread(Path(authority.db.db_path).resolve))
    # The dispatcher attempt is the durable request identity, including after the
    # task closes. Retry checks the saved claim, never reinterprets updated cards.
    request_id = 'kanban:' + json.dumps([params.get('board'), params.get('task_id'), params.get('run_id')], separators=(',', ':'))
    sid = local_identity(authority.profile_id, connection.actor.subject, request_id)
    saved = local_receipt(authority.db, sid) if authority.db.get_session(sid) is not None else None
    if saved:
        policy = restore_policy(saved['policy'])
        context = json.loads(policy.kanban_json or '{}')
        if (not connection.native_owner or set(params) not in ({'board', 'task_id', 'run_id', 'claim_lock'}, {'board', 'task_id', 'run_id', 'claim_lock', 'db'})
                or any(params[k] != context.get(k) for k in params)):
            raise RuntimeStoreError('permission_denied')
        from gateway.session_local_recovery import restore_local_session
        ref = restore_local_session(authority, sid)
    else:
        policy, private = build_kanban_policy(connection, params, _load_gateway_config())
        ref = create_local_session(authority, connection.actor, {'request_id': request_id},
                                   trusted_policy=policy, trusted_secrets=private)
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    import psutil
    from hermes_cli.process_identity import worker_birth
    context = json.loads(policy.kanban_json)

    def admit_claim():
        # The Kanban DB is the cross-profile dispatch bus (busy_timeout up to 120 s): the claim
        # check and its marker share one write transaction, off the owner loop.
        with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
            # Either this exact attempt was already admitted to this session (retry after a
            # crash between session creation and admission), or the claim is still current
            # now; a reclaimed/closed task never reaches resume/submit.
            marker = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
                                  (params['task_id'], params['run_id'])).fetchone()
            if marker:
                if json.loads(marker[0])['session_id'] != ref.session_id:
                    raise RuntimeStoreError('stale_kanban_claim')
            else:
                task = kb.get_task(conn, params['task_id'])
                if not (task and task.status == 'running' and task.current_run_id == params['run_id'] and task.claim_lock == params['claim_lock']):
                    raise RuntimeStoreError('stale_kanban_claim')
                kb._append_event(conn, task.id, 'owner_admitted',
                    {'db': owner_db, 'session_id': ref.session_id,
                     'request_id': request_id, 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                    run_id=task.current_run_id)
    from gateway.session_runtime_workers import tracked_write
    await tracked_write(authority, admit_claim)
    await connection.resume(ref, {})
    # Finite like the `chat -q` worker it replaced: nobody can answer an approval or clarify in a
    # card (approvals.single_query_mode applies). Never unattended: that is `-z`'s auto-approval.
    # A retry of an attempt admitted before cards were finite resubmits its committed payload,
    # so the store returns the original receipt instead of admission_conflict.
    from hermes_state_runtime import find_session_admission
    committed = None if not saved else await asyncio.to_thread(find_session_admission, authority.db,
        principal_id=connection.actor.subject, session_id=ref.session_id, request_id=request_id)
    payload: dict = {'text': f'work kanban task {params["task_id"]}'}
    if committed is None or 'finite' in committed['payload']:
        payload['finite'] = True
    # The dispatcher's trusted producer marker (#134889 KB-8): nobody waits on a card attempt, so one
    # lost across an owner restart settles interrupted instead of pausing the card session. An
    # attempt main committed without it resubmits unmarked (same digest, original receipt).
    card = None
    if committed is None or 'kanban_card_v1' in committed['payload']:
        card = {'board': context.get('board'), 'task_id': params['task_id'], 'run_id': params['run_id']}
    receipt = await authority.submit(connection.actor, Submission(request_id, ref, payload, 'queue'),
                                     _kanban_card=card)
    return {'session_id': ref.session_id, 'receipt': asdict(receipt)}


def bind_worker_context(frame, adopted, *, wait_seconds=None):
    """The adopted private exec's Kanban bootstrap: bind ``OWNER`` here, or refuse.

    Only the adopted private exec calls this, before tool discovery imports, and only
    with the ``worker.adopt`` result its owner just returned — that verified adoption
    is the sole issuer precondition (:func:`_require_adopted_scope`).  The frame's
    coordinates are lookup input, never authority: the grant is built from the rows
    read here.

    The claim check and this run's ``worker_bound`` append share ONE board write
    transaction, and every denial precedes the append, so a refused binding leaves no
    board residue.  Returns the frozen candidate for :func:`managed_execution_scope`,
    or ``None`` for a frame that is not a card.
    """
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        return None
    if frame['policy']['source'] != 'kanban':
        raise ValueError('invalid_kanban_policy')
    _require_adopted_scope(frame, adopted)
    from hermes_cli.kanban_db_dispatch import _process_fingerprint
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    env = _worker_env(context)
    _await_submitter_registration(Path(context['db']), context, wait_seconds)
    with connect_closing(Path(context['db'])) as conn:
        with kb.write_txn(conn):
            candidate = _bind_claim_owner(conn, context, frame['home'],
                                          _process_fingerprint(os.getpid()))
            claimed = conn.execute("SELECT payload, created_at FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed' ORDER BY id DESC LIMIT 1",
                                   (context['task_id'], context['run_id'])).fetchone()
    # Heartbeats extend the claim by the TTL the dispatcher claimed it with (a spawned worker
    # inherited it from the dispatcher's environment); this daemon's own default is not that.
    expires = json.loads(claimed[0]).get('expires') if claimed else None
    if isinstance(expires, int) and claimed[1]:
        env['HERMES_KANBAN_CLAIM_TTL_SECONDS'] = str(max(1, expires - int(claimed[1])))
    _watch_submitter(Path(context['db']), context['run_id'])
    os.environ.update(env)
    os.chdir(context['workspace'])
    return candidate


def _worker_env(context):
    """The lookup/git/terminal environment a Kanban worker runs with (no authority)."""
    env = {'HERMES_KANBAN_TASK': context['task_id'], 'HERMES_KANBAN_BOARD': context['board'],
        'HERMES_KANBAN_DB': context['db'], 'HERMES_KANBAN_RUN_ID': str(context['run_id']),
        'HERMES_KANBAN_CLAIM_LOCK': context['claim_lock'], 'HERMES_KANBAN_WORKSPACE': context['workspace'],
        'HERMES_KANBAN_WORKSPACES_ROOT': context['workspaces_root'], 'HERMES_PROFILE': context['profile'],
        'HERMES_SESSION_SOURCE': 'kanban'}
    for key, value in (('HERMES_KANBAN_BRANCH', context['branch']), ('HERMES_TENANT', context['tenant'])):
        if value:
            env[key] = value
    if context['goal_mode']:
        env['HERMES_KANBAN_GOAL_MODE'] = '1'
    from hermes_cli.kanban_db_dispatch import _worker_terminal_timeout_env
    for key in ('TERMINAL_TIMEOUT', 'TERMINAL_MAX_FOREGROUND_TIMEOUT'):
        value = _worker_terminal_timeout_env(context['max_runtime_seconds'], os.environ.get(key))
        if value:
            env[key] = value
    return env


def _require_adopted_scope(frame, adopted):
    """The frame's private scope and adoption row must describe THIS interpreter.

    The owner verified this process's live pid and kernel birth plus an unguessable
    secret against the durable assignment before returning ``adopted``; re-checking
    the returned row and the frame's own scope here means the bootstrap cannot be
    driven by a hand-built frame, and a recycled pid cannot re-adopt an assignment.
    """
    import psutil
    from hermes_cli.process_identity import worker_birth
    scope, home = frame.get('scope'), frame.get('home')
    if (not isinstance(scope, dict) or not isinstance(adopted, dict) or not isinstance(home, str)
            or not Path(home).is_absolute() or scope.get('pid') != os.getpid()
            or scope.get('profile_id') != home
            or adopted.get('status') != 'running'
            or adopted.get('owner_epoch') != scope.get('epoch')
            or adopted.get('execution_id') != scope.get('execution_id')
            or adopted.get('generation') != scope.get('generation')):
        raise RuntimeStoreError('stale_kanban_claim')
    try:
        birth = worker_birth(psutil.Process())
    except (psutil.Error, OSError) as exc:
        raise RuntimeStoreError('stale_kanban_claim') from exc
    if birth != scope.get('birth'):
        raise RuntimeStoreError('stale_kanban_claim')


def _submitter_pending(db_path, context) -> bool:
    """True while the dispatcher has not recorded this attempt's spawned worker yet."""
    try:
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            row = conn.execute('SELECT status, current_run_id, claim_lock, worker_pid FROM tasks WHERE id=?',
                               (context['task_id'],)).fetchone()
    except sqlite3.Error:
        return False
    if row is None or row[0] != 'running' or row[1] != context['run_id'] or row[2] != context['claim_lock']:
        return False
    return row[3] is None


def _await_submitter_registration(db_path, context, wait_seconds=None):
    """Wait out the dispatcher's spawn/registration race, read-only and bounded.

    The launcher writes the spawned worker's pid and boot/start fingerprint right
    after ``Popen``; a managed run that starts faster waits here instead of relaxing
    the proof its grant is built from.
    """
    from agent.kanban_execution_identity import REGISTRATION_POLL_SECONDS, REGISTRATION_WAIT_SECONDS
    deadline = time.monotonic() + (REGISTRATION_WAIT_SECONDS if wait_seconds is None
                                   else max(0.0, float(wait_seconds)))
    while _submitter_pending(db_path, context):
        if time.monotonic() >= deadline:
            raise RuntimeStoreError('stale_kanban_claim')
        time.sleep(REGISTRATION_POLL_SECONDS)


def _bind_claim_owner(conn, context, home, fingerprint):
    """Validate this frame against the live board, append the binding, return the grant.

    Caller holds the board write transaction.  Every denial raises BEFORE the
    ``worker_bound`` append, so a refused binding leaves no trace.  The board is
    named by its canonical file — never re-resolved from this worker's environment,
    which is exactly the isolation the dispatcher's pins exist to prevent.
    """
    from agent.kanban_execution_identity import (
        PROVENANCE_MANAGED_BOUND, ROLE_OWNER, KanbanExecutionContext, connection_db_path,
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import UNVERIFIED_WORKER_FINGERPRINT
    from hermes_cli.profiles import resolve_profile_env
    now = int(time.time())
    task = conn.execute(
        'SELECT id, assignee, status, claim_lock, claim_expires, current_run_id, worker_pid, worker_started_at '
        'FROM tasks WHERE id=?', (context['task_id'],)).fetchone()
    run = conn.execute('SELECT task_id, ended_at, claim_lock, claim_expires, worker_pid, worker_started_at '
                       'FROM task_runs WHERE id=?', (int(context['run_id']),)).fetchone()
    if (task is None or task['status'] != 'running' or task['current_run_id'] != context['run_id']
            or task['claim_lock'] != context['claim_lock'] or run is None
            or run['task_id'] != task['id'] or run['ended_at'] is not None
            or run['claim_lock'] != context['claim_lock']):
        raise RuntimeStoreError('stale_kanban_claim')
    if _canonical_board_file(context['db']) != _canonical_board_file(connection_db_path(conn)):
        raise RuntimeStoreError('stale_kanban_claim')
    try:
        kb._require_slug(context['board'])
    except ValueError as exc:
        raise RuntimeStoreError('stale_kanban_claim') from exc
    if (task['assignee'] != context['profile']
            or _canonical_board_file(resolve_profile_env(task['assignee'])) != _canonical_board_file(home)):
        raise RuntimeStoreError('profile_mismatch')
    # The recorded process fields stay the SUBMITTER's: this interpreter is recorded
    # beside them, in the event the reconciliation compares against.
    submitter_pid, submitter_started_at = _submitter_registration(task, run)
    for expires in (task['claim_expires'], run['claim_expires']):
        if expires is None or int(expires) < now:
            raise RuntimeStoreError('stale_kanban_claim')
    if not fingerprint or fingerprint == UNVERIFIED_WORKER_FINGERPRINT:
        raise RuntimeStoreError('stale_kanban_claim')
    _require_unconflicted_binding(conn, task['id'], int(context['run_id']), context['claim_lock'], fingerprint)
    kb._append_event(conn, task['id'], 'worker_bound', {'pid': os.getpid(), 'claim_lock': context['claim_lock'],
                     'started_at': fingerprint}, run_id=int(context['run_id']))
    return KanbanExecutionContext(
        role=ROLE_OWNER, provenance=PROVENANCE_MANAGED_BOUND,
        db_path=_canonical_board_file(context['db']), board=context['board'], task_id=task['id'],
        run_id=int(context['run_id']), claim_lock=context['claim_lock'],
        worker_pid=os.getpid(), worker_started_at=fingerprint,
        submitter_pid=submitter_pid, submitter_started_at=submitter_started_at)


def _canonical_board_file(path):
    try:
        return str(Path(str(path)).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return ""


def _submitter_registration(task, run):
    """The dispatcher-registered submitter of this attempt: ``(pid, boot-start fingerprint)``."""
    pid, started_at = task['worker_pid'], task['worker_started_at']
    if not pid or not started_at:
        raise RuntimeStoreError('stale_kanban_claim')
    if run['worker_pid'] is not None and int(run['worker_pid']) != int(pid):
        raise RuntimeStoreError('stale_kanban_claim')
    registered = run['worker_started_at']
    if registered is not None and str(registered) != str(started_at):
        raise RuntimeStoreError('stale_kanban_claim')
    return int(pid), str(started_at)


def _require_unconflicted_binding(conn, task_id, run_id, claim_lock, fingerprint):
    """A binding already recorded for this run may only be superseded by a gone interpreter.

    This process's own binding (a resumed turn) is fine.  A binding naming a still-live
    interpreter is a conflict: another execution holds this run, and this one must not
    take it over.  A dead or recycled one is the previous attempt's, which this
    execution supersedes — that is the same liveness rule the dispatcher's crash
    recovery already applies to a gone bound interpreter, and it is why an owner
    restart's fresh interpreter can still bind.
    """
    from agent.kanban_execution_identity import _latest_worker_binding
    from hermes_cli.kanban_db_dispatch import _worker_alive
    bound, parsed_ok = _latest_worker_binding(conn, task_id, run_id)
    if not parsed_ok:
        raise RuntimeStoreError('stale_kanban_claim')
    if bound is None:
        return
    if (int(bound['pid']) == os.getpid()
            and bound['claim_lock'] == claim_lock and bound['started_at'] == fingerprint):
        return
    if _worker_alive(bound['pid'], bound['started_at']):
        raise RuntimeStoreError('stale_kanban_claim')


@contextlib.contextmanager
def managed_execution_scope(candidate):
    """Scope the adopted exec's identity around the rest of its turn.

    A Kanban card scopes the grant its bootstrap verified; every other managed turn
    scopes an explicit detached role instead of inheriting whatever an ambient slot
    held.  Context-local only: no process-wide identity is installed, so the owner
    daemon and its other runs acquire nothing.
    """
    from agent.kanban_execution_identity import (
        ROLE_FOREIGN_INSPECTOR, KanbanExecutionContext, scoped_execution_context,
    )
    ctx = candidate if candidate is not None else KanbanExecutionContext(
        role=ROLE_FOREIGN_INSPECTOR, reason='managed_non_kanban')
    with scoped_execution_context(ctx) as installed:
        yield installed


_TURNS_DONE = threading.Event()


def _watch_submitter(db, run_id):
    """The dispatcher's spawned worker is this attempt's handle: killing it (an operator, a
    timeout, an OOM kill) must end the attempt, as it did when the worker ran the agent itself.
    Exiting hard leaves no ``worker_result``, so the dead-worker sweep books this run's crash.
    The run row is re-read each tick: the dispatcher records the pid only after its spawn returns."""
    from hermes_cli.kanban_db_dispatch import _worker_alive

    def watch():
        with closing(sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            while not _TURNS_DONE.wait(1.0):
                try:
                    row = conn.execute('SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?', (run_id,)).fetchone()
                except sqlite3.Error:
                    continue
                if row and row[0] and row[0] != os.getpid() and not _worker_alive(row[0], row[1]):
                    os._exit(1)
    threading.Thread(target=watch, name='kanban-submitter-watch', daemon=True).start()


def run_worker_turns(agent, frame, history):
    from gateway.session_managed_worker import worker_turn_scope
    from gateway.session_surface import arm_surface_turn
    with worker_turn_scope(frame):
        # The agent consumes explicit one-turn attributes, not the ContextVar: arm them exactly as
        # the in-process runner does (voice route marker + surface note on the user message).
        arm_surface_turn(agent)
        return _run_worker_turns(agent, frame, history)


@contextlib.contextmanager
def _worker_workspace(frame):
    """A managed worker's chat turn in its session's workspace, as the in-process turn runner binds it
    (#140665): the session id as the terminal task id, the route as the session key (persistent Docker
    keys the workspace container from it) and the frozen cwd as the session's mount. Yields the task
    id, or None for a frame with no session scope (nothing is bound then)."""
    session_id = (frame.get('scope') or {}).get('session_id')
    if not session_id:
        yield None
        return
    from gateway.session_authorities import served_profile_name
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import bind_turn_workspace, restore_policy
    from hermes_state_keys import profile_from_session_key
    policy = restore_policy(frame['policy'])
    route = frame.get('route') or session_id
    # The profile its in-process turns bind, so persistent Docker keys the container they and the
    # session's side agents use (#140665): a served secondary's name, which its routes carry
    # (``agent:<name>:``); none for the launch home, whatever its name (``hermes -p beta gateway``
    # keys ``agent:main:``, and only the default profile runs the multiplexer).
    profile = served_profile_name(frame.get('home'))
    tokens = set_session_vars(session_key=route, session_id=session_id,
                              profile=profile if profile_from_session_key(route) == profile else '')
    try:
        bind_turn_workspace(policy, session_id)
        yield session_id
    finally:
        clear_session_vars(tokens)


def _run_worker_turns(agent, frame, history):
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        from gateway.session_display import worker_display_kwargs
        author = frame.get('turn_author')
        with _worker_workspace(frame) as task_id:
            return agent.run_conversation(frame['text'], conversation_history=history, **worker_display_kwargs(),
                                          **({'task_id': task_id} if task_id else {}),
                                          **({'turn_author': author} if author is not None else {}))
    from hermes_cli.turn_exit import credential_failure_flags, turn_exit_code
    code, last_output = 1, ''
    try:
        result = _run_task_turns(agent, frame, history, context)
        code = turn_exit_code(result, kanban_worker=True)
        last_output = str(result.get('final_response') or '')[-500:] if isinstance(result, dict) else ''
        return result
    except Exception as exc:
        code = turn_exit_code(None, kanban_worker=True, **credential_failure_flags(exc))
        raise
    finally:
        _record_worker_result(context, code, last_output)


def record_start_failure(frame, exc):
    """The worker died before any turn ran (agent construction resolves credentials): book the
    one-shot CLI's credential exit (75 quota wall / 78 re-login / 1) instead of a bare crash."""
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        return
    from hermes_cli.turn_exit import credential_failure_flags, turn_exit_code
    _record_worker_result(context, turn_exit_code(None, kanban_worker=True, **credential_failure_flags(exc)), '')


def _record_worker_result(context, code, last_output):
    _TURNS_DONE.set()
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
        if _bound_worker_matches(conn, context):
            # Closing a run clears its claim/PID and replaces metadata; keep the
            # result in the immutable attempt event stream instead.
            kb._append_event(conn, context['task_id'], 'worker_result',
                {'pid': os.getpid(), 'claim_lock': context['claim_lock'], 'exit_code': code,
                 'last_output': last_output}, run_id=context['run_id'])


def record_worker_loss(policy, reason, *, pid=None, bound=False):
    """Append a run-scoped ``worker_lost`` event for this attempt to the card's board.

    Dropped when the card has moved past the run (it says nothing about the current attempt).
    Never raises: the caller's own loss handling must not depend on the board write.
    """
    try:
        context = json.loads(policy.kanban_json or 'null')
        if not context:
            return
        from hermes_cli.kanban_db_connect import connect_closing
        from hermes_cli import kanban_db as kb
        run_id = int(context['run_id'])
        with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
            row = conn.execute('SELECT status, current_run_id FROM tasks WHERE id=?',
                               (context['task_id'],)).fetchone()
            if row is None or row[0] != 'running' or row[1] != run_id:
                return
            kb._append_event(conn, context['task_id'], 'worker_lost',
                             {'reason': str(reason)[:200], 'claim_lock': context['claim_lock'],
                              'run_id': run_id, 'pid': pid, 'bound': bool(bound)},
                             run_id=run_id)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('kanban worker_lost record failed')


def _bound_worker_matches(conn, context):
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound' ORDER BY id DESC LIMIT 1",
        (context['task_id'], context['run_id'])).fetchone()
    bound = json.loads(row[0]) if row else {}
    return (bound.get('pid'), bound.get('claim_lock')) == (os.getpid(), context['claim_lock'])


def _bound_claim_is_live(context):
    """Retry proof for a managed worker: the SAME reconciliation the authorizer uses.

    The run's ``worker_pid``/``worker_started_at`` are the dispatcher's spawned
    submitter here, so the executing interpreter is proven by the grant's
    ``worker_bound`` event.  Re-entering the model requires a live, unexpired lease,
    and an unreadable board is never proof of ownership.  There is deliberately no
    second, weaker (pid-only) grant predicate beside this one.
    """
    from agent.kanban_execution_identity import ROLE_OWNER, current_execution_context, reconcile_owner
    ctx = current_execution_context()
    if ctx is None or ctx.role != ROLE_OWNER:
        return False
    if str(ctx.task_id or '') != str(context['task_id']) or ctx.run_id != context['run_id']:
        return False
    try:
        with closing(sqlite3.connect(Path(context['db']).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            return reconcile_owner(conn, ctx, require_lease=True).ok
    except sqlite3.Error:
        return False


def worker_result(path, params):
    """Read only the dispatcher's exact attempt; never infer success from admission settlement."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        row = conn.execute("SELECT payload FROM task_events WHERE run_id=? AND task_id=? AND kind='worker_result' ORDER BY id DESC LIMIT 1",
            (params['run_id'], params['task_id'])).fetchone()
        result = json.loads(row[0]) if row else {}
        return result if result.get('claim_lock') == params['claim_lock'] else {}


def worker_exit_code(path, params):
    return worker_result(path, params).get('exit_code', 1)


# Who wrote a card attempt's prompts: the dispatcher, never the operator, so memory providers do not
# file "work kanban task t_..." and the worker's progress notes as the human's own turns (#137406).
# Not a ``bot:<profile>`` id: a Bot Mode profile named ``kanban`` would share its memory key.
KANBAN_TURN_AUTHOR = {'id': 'kanban:dispatcher', 'name': 'Kanban dispatcher', 'is_bot': True}


def _run_task_turns(agent, frame, history, context):
    from agent.skill_commands import build_preloaded_skills_prompt
    skills, loaded, missing = build_preloaded_skills_prompt(context['skills'], task_id=agent.session_id)
    if missing and not loaded:
        raise ValueError('kanban_skills_unavailable')
    prompt = '\n\n'.join(p for p in (skills, context['context'], frame['text']) if p)
    result = agent.run_conversation(prompt, conversation_history=history, turn_author=KANBAN_TURN_AUTHOR)

    def recover(nudge):
        nonlocal result
        prior = result.get('messages') if isinstance(result, dict) else None
        result = agent.run_conversation(nudge, conversation_history=prior or
                                        agent._session_db.get_messages_as_conversation(agent.session_id),
                                        turn_author=KANBAN_TURN_AUTHOR)
    # Same in-place retry as the one-shot CLI worker (agent/kanban_turn_recovery.py), but the
    # carrier is this frame's bound claim, never the owner process's environment.
    from agent.kanban_turn_recovery import recover_failed_kanban_turns
    recover_failed_kanban_turns(recover, lambda: result, task_id=context['task_id'],
                                claim_check=lambda: _bound_claim_is_live(context))
    from hermes_cli.turn_exit import turn_exit_code
    # Only a settled turn may continue the goal loop: an exhausted or denied recovery reaches
    # the honest exit path without re-entering the model under an unprovable claim.
    if not context['goal_mode'] or turn_exit_code(result, kanban_worker=True) != 0:
        return result
    from hermes_cli.goals import run_kanban_goal_loop, DEFAULT_MAX_TURNS
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    def status():
        with connect_closing() as conn:
            return kb.goal_run_status(conn, context['task_id'], context['run_id'])
    def block(reason):
        with connect_closing() as conn:
            kb.block_task(conn, context['task_id'], reason=reason, expected_run_id=context['run_id'])
    def turn(prompt):
        nonlocal result
        result = agent.run_conversation(prompt, conversation_history=agent._session_db.get_messages_as_conversation(agent.session_id),
                                        turn_author=KANBAN_TURN_AUTHOR)
        return result.get('final_response', '')
    run_kanban_goal_loop(task_id=context['task_id'], goal_text=context['goal_text'], run_turn=turn,
        task_status_fn=status, block_fn=block, max_turns=context['goal_max_turns'] or DEFAULT_MAX_TURNS,
        first_response=result.get('final_response', ''))
    return result
