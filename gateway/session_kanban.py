"""Trusted dispatcher claims enter the ordinary owner's durable admission queue."""
import asyncio
import contextlib
from contextlib import closing
from dataclasses import asdict, replace
import threading
import json
import os
from pathlib import Path
import sqlite3

from hermes_state_runtime import RuntimeStoreError


def _ownership_enforce() -> bool:
    """Cross-session ownership gate for fresh Kanban admissions (#47 Phase 2).

    C5a (#47 Phase 2, SAGE deployment condition): the gate is STAGED — it is
    OFF unless ``HERMES_KANBAN_OWNERSHIP_ENFORCE`` is explicitly set to a
    truthy value (``1``/``true``/``yes``, case-insensitive, surrounding
    whitespace ignored). Deploying the code does NOT activate the gate: the
    operator migrates the ledger, backfills already-running cards, verifies,
    and only then exports the flag. Any other value (including unset) keeps
    the gate inert, so an operator can also bypass it without a code change.
    """
    return os.environ.get('HERMES_KANBAN_OWNERSHIP_ENFORCE', '').strip().lower() in {'1', 'true', 'yes'}


_OWNERSHIP_TABLES = ('task_ownership', 'task_ownership_events')


def _ownership_schema_present(conn) -> bool:
    """True when both ownership ledger tables already exist (DDL is skipped).

    C5 (#47 Phase 2): the ledger DDL is issued OUTSIDE ``kb.write_txn`` (see
    ``admit_claim``). This read-only probe keeps steady-state admissions free of
    any schema write — ``init_schema`` runs only when a table is actually
    missing.
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
        _OWNERSHIP_TABLES,
    ).fetchall()
    return {r[0] for r in rows} >= set(_OWNERSHIP_TABLES)


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


def _release_refused_claim(conn, task, row, *, refused, scope, scope_reason):
    """Release the card claim in the SAME admission transaction as the refusal.

    P1-A defensive helper — a refused admission must never leave the card
    claimed:

    * ``row is None`` (the card row vanished between the status check and the
      read) still releases by ``task.id``.
    * a corrupt ``consecutive_failures`` (``int()`` raises) fails CLOSED to
      ``blocked`` — the card is never re-queued on an unreadable breaker count.

    P1-D: the refusal INCREMENTS ``consecutive_failures`` (SQL-side, so a
    concurrent writer cannot lose the step) and the breaker reads the NEW
    count: a card already at >=1 failure parks ``blocked`` instead of
    ``ready``. The counter is what eventually stops the dispatcher from
    re-spawning a card the ownership gate keeps refusing (the cooldown only
    spaces the retries; see ``check_respawn_guard``'s
    ``ownership_refusal_cooldown``).
    """
    from hermes_cli import kanban_db as kb
    status = 'ready'
    if row is not None:
        try:
            failures = int(row['consecutive_failures'] or 0)
        except (TypeError, ValueError):
            failures = 2  # fail closed: an unreadable breaker count parks the card
        # The UPDATE below increments the breaker; the status reads the NEW count
        # (old + 1, the only writer in this transaction).
        status = 'blocked' if failures + 1 >= 2 else 'ready'
    conn.execute(
        "UPDATE tasks SET status=?, claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, worker_started_at=NULL, "
        "consecutive_failures=COALESCE(CAST(consecutive_failures AS INTEGER),0)+1, "
        "last_failure_error=? WHERE id=?",
        (status, 'ownership_refused', task.id))
    # C1 (#47 Phase 2): the UPDATE above is the refusal's durable half. Closing
    # the run and appending the event are bookkeeping — if either raises, the
    # exception must NOT propagate out of this helper. The caller's fail-closed
    # handler would re-run the helper with the pre-increment row, re-read the
    # old counter and defeat the breaker (status='ready' at failures=2). Swallow
    # here so the transaction always commits the release.
    import logging
    log = logging.getLogger(__name__)
    try:
        kb._end_run(conn, task.id, outcome='refused', error='ownership_refused',
                    metadata={'reason': refused, 'scope': scope})
    except Exception as exc:
        log.debug('kanban refusal: _end_run failed: %s', exc)
    try:
        kb._append_event(conn, task.id, 'ownership_refused',
                         {'reason': refused, 'scope': scope, 'scope_reason': scope_reason},
                         run_id=task.current_run_id)
    except Exception as exc:
        log.debug('kanban refusal: _append_event failed: %s', exc)


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
    import os
    import psutil
    from hermes_cli.process_identity import worker_birth
    context = json.loads(policy.kanban_json)

    def admit_claim():
        # The Kanban DB is the cross-profile dispatch bus (busy_timeout up to 120 s): the claim
        # check and its marker share one write transaction, off the owner loop.
        from cross_session_ownership import (
            OwnershipError, canonical_key_from_kanban, claim_ownership, init_schema,
        )
        refused = None
        scope = None
        scope_reason = None
        task = None
        row = None
        with connect_closing(Path(context['db'])) as conn:
            # C5 (#47 Phase 2): the ownership ledger DDL is NOT issued inside
            # ``kb.write_txn``. CREATE TABLE/INDEX inside a BEGIN IMMEDIATE write
            # transaction is wrong twice over: a DDL failure mid-transaction
            # leaves the admission txn in an undefined state, and
            # ``CREATE ... IF NOT EXISTS`` silently masks an incompatible
            # pre-existing ledger table (a broken schema must fail closed, not
            # no-op). Prepare the ledger in its own committed step first; the
            # claim still JOINS the admission write below, so ledger and card
            # cannot diverge.
            if _ownership_enforce() and not _ownership_schema_present(conn):
                try:
                    init_schema(conn)
                except sqlite3.Error:
                    # Fail closed: the claim below is skipped, and the card claim
                    # is released in the same write transaction as the refusal.
                    refused = 'ownership_txn_error'
            with kb.write_txn(conn):
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
                    if _ownership_enforce():
                        # #47 Phase 2: the cross-session ownership ledger gates fresh admission.
                        # The ledger claim and the card admission commit together (join_txn=True):
                        # a second, independent lock would let the ledger and the card diverge.
                        # _record_task_failure cannot run here (it opens its own write_txn), so the
                        # refused release is RAW SQL inside this same transaction.
                        #
                        # P1-A: the WHOLE ownership decision AND the refused release run inside
                        # one try/except Exception within kb.write_txn. An escaping exception
                        # (canonical_key_from_kanban raising OwnershipError on a whitespace-bearing
                        # source_repo, a corrupt consecutive_failures counter, an assert) rolls the
                        # admission back, the release never lands, and the card stays 'running'
                        # with a stale claim_lock forever. Fail closed: the refusal is booked and
                        # released in this SAME transaction.
                        released = False
                        try:
                            row = conn.execute('SELECT * FROM tasks WHERE id=?', (task.id,)).fetchone()
                            diag: list[str] = []
                            scope = (canonical_key_from_kanban(
                                {c: row[c] for c in row.keys()}, diagnostics=diag)
                                if row is not None else None)
                            scope_reason = next((d for d in diag if d.startswith('fail_closed:')), None)
                            if not scope:
                                refused = 'ownership_unavailable'
                            elif refused is None:
                                try:
                                    # P1-C: the ledger must verify the claimed session is a REAL
                                    # session in the owner DB, not merely a well-formed id. Format
                                    # validation alone is not identity assurance
                                    # (cross_session_ownership.claim_ownership.verify_session).
                                    res = claim_ownership(
                                        conn, scope, ref.session_id, join_txn=True, ttl_seconds=900,
                                        verify_session=lambda sid: authority.db.get_session(sid) is not None)
                                except (OwnershipError, sqlite3.Error):
                                    # P0 (#47 Phase 2): a broken/unreadable ledger schema
                                    # (e.g. an incompatible pre-existing task_ownership table)
                                    # raises a plain sqlite3 error, NOT OwnershipError.
                                    # Without this branch the exception escaped the gate,
                                    # the release below never ran, and the card stayed
                                    # 'running' with a stale claim_lock forever. Fail closed:
                                    # the refusal must release the card claim in this same
                                    # transaction, exactly like the OwnershipError path.
                                    res = None
                                    refused = 'ownership_txn_error'
                                if res is not None and not res['ok']:
                                    refused = 'owned_by_other_session'
                            if refused:
                                # Release the card claim with RAW SQL in the SAME transaction.
                                # Breaker: a card already at >=2 consecutive failures parks
                                # 'blocked' instead of 'ready' (the count is read, not
                                # incremented — the resolution's rule). The helper is defensive:
                                # a missing row still releases by task.id and an unreadable
                                # counter fails closed to 'blocked'.
                                _release_refused_claim(conn, task, row, refused=refused,
                                                       scope=scope, scope_reason=scope_reason)
                                released = True
                        except Exception:
                            # Fail closed: any unexpected decision error refuses the admission.
                            refused = 'ownership_txn_error'
                            if not released:
                                try:
                                    # C1 (#47 Phase 2): re-read the row. A prior helper call may
                                    # have already incremented consecutive_failures before it
                                    # raised; the stale row would re-read the pre-increment count
                                    # and defeat the breaker (status='ready' at failures=2).
                                    row = conn.execute('SELECT * FROM tasks WHERE id=?',
                                                       (task.id,)).fetchone()
                                    _release_refused_claim(conn, task, row, refused=refused,
                                                           scope=scope, scope_reason=scope_reason)
                                except Exception:
                                    # Last resort: the release itself failed. Nothing more this
                                    # transaction can do; the reconciler sweeps the stale claim.
                                    pass
                    if not refused:
                        kb._append_event(conn, task.id, 'owner_admitted',
                            {'db': owner_db, 'session_id': ref.session_id,
                             'request_id': request_id, 'pid': os.getpid(), 'birth': worker_birth(psutil.Process()),
                             'scope': scope, 'scope_reason': scope_reason},
                             run_id=task.current_run_id)
        if refused:
            # Raised OUTSIDE the with: the release committed, then the admission is refused.
            raise RuntimeStoreError('ownership_refused')
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


def bind_worker_context(frame):
    """Only the adopted private exec calls this, before tool discovery imports."""
    import os
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        return
    if frame['policy']['source'] != 'kanban':
        raise ValueError('invalid_kanban_policy')
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
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    with connect_closing(Path(context['db'])) as conn:
        with kb.write_txn(conn):
            task = kb.get_task(conn, context['task_id'])
            if (task is None or task.status != 'running' or task.current_run_id != context['run_id']
                    or task.claim_lock != context['claim_lock']):
                raise RuntimeStoreError('stale_kanban_claim')
            # The dispatcher's spawned worker stays the run's pid (reclaim, timeout and operator
            # kills target it; ``_watch_submitter`` ties this interpreter's life to it). The bound
            # interpreter is recorded beside it so a sweep can tell when the attempt is truly gone.
            from hermes_cli.kanban_db_dispatch import UNVERIFIED_WORKER_FINGERPRINT, _process_fingerprint
            fingerprint = _process_fingerprint(os.getpid()) or UNVERIFIED_WORKER_FINGERPRINT
            kb._append_event(conn, task.id, 'worker_bound', {'pid': os.getpid(), 'claim_lock': context['claim_lock'],
                             'started_at': fingerprint}, run_id=context['run_id'])
            claimed = conn.execute("SELECT payload, created_at FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed' ORDER BY id DESC LIMIT 1",
                                   (task.id, context['run_id'])).fetchone()
    # Heartbeats extend the claim by the TTL the dispatcher claimed it with (a spawned worker
    # inherited it from the dispatcher's environment); this daemon's own default is not that.
    expires = json.loads(claimed[0]).get('expires') if claimed else None
    if isinstance(expires, int) and claimed[1]:
        env['HERMES_KANBAN_CLAIM_TTL_SECONDS'] = str(max(1, expires - int(claimed[1])))
    _watch_submitter(Path(context['db']), context['run_id'])
    os.environ.update(env)
    os.chdir(context['workspace'])


_TURNS_DONE = threading.Event()


def _watch_submitter(db, run_id):
    """The dispatcher's spawned worker is this attempt's handle: killing it (an operator, a
    timeout, an OOM kill) must end the attempt, as it did when the worker ran the agent itself.
    Exiting hard leaves no ``worker_result``, so the dead-worker sweep books this run's crash.
    The run row is re-read each tick: the dispatcher records the pid only after its spawn returns."""
    import os
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


# --- C3 (#47 Phase 2): the worker beats BOTH leases itself -------------------------

# The dispatcher claims the card and the cross-session ledger with a 900 s TTL.
# A worker blocked in ONE long turn (a render, a single huge tool call) emits no
# tool-layer activity, so nothing extends either lease: at 900 s the ledger goes
# STALE, a second session takes the card over, and two workers execute the same
# work item (P0). The worker therefore beats both leases itself, immediately at
# turn start and then on a fixed interval from a daemon thread that lives for
# the turn.
HEARTBEAT_TTL_SECONDS = 900
HEARTBEAT_INTERVAL_SECONDS = 300


def _heartbeat_interval_seconds() -> float:
    """``HERMES_KANBAN_HEARTBEAT_INTERVAL_SECONDS`` > TTL/3, bounded below by 1 s."""
    raw = (os.environ.get('HERMES_KANBAN_HEARTBEAT_INTERVAL_SECONDS') or '').strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return float(HEARTBEAT_INTERVAL_SECONDS)


def _admission_lease(conn, context):
    """``(scope, session_id)`` the admission gate claimed for this run.

    The ``owner_admitted`` event payload is the authority: it records the exact
    ledger scope and owner session the gate claimed. Re-deriving the scope from
    the card row here would target a DIFFERENT ledger row if the card's
    title/body/idempotency key changed while the worker ran, and re-using the
    worker's ambient session id would silently heartbeat as the wrong owner.
    Returns ``(None, None)`` when no marker exists (the caller falls back).
    """
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' "
        "ORDER BY id DESC LIMIT 1",
        (context.get('task_id'), context.get('run_id')),
    ).fetchone()
    if row is None or not row[0]:
        return None, None
    try:
        payload = json.loads(row[0])
    except (ValueError, TypeError):
        return None, None
    return (payload.get('scope') or None), (payload.get('session_id') or None)


def _ledger_scope_from_row(conn, context):
    """Fail-closed fallback: derive the scope from the card row (admission's rule)."""
    try:
        from cross_session_ownership import canonical_key_from_kanban
    except Exception:
        return None
    row = conn.execute('SELECT * FROM tasks WHERE id=?', (context.get('task_id'),)).fetchone()
    if row is None:
        return None
    try:
        return canonical_key_from_kanban({c: row[c] for c in row.keys()})
    except Exception:
        return None


def _release_ownership_claim(conn, context, *, reason):
    """Release the cross-session ledger claim this attempt holds; never raises.

    C4 (#47 Phase 2): the worker's completion is the end of its authority over
    the logical work item. Without this the ledger row stays ACTIVE until its
    TTL lapses, so a second session is locked out of the same work item for up
    to fifteen minutes after the first worker already finished, and a Direct
    dispatch for the same issue is refused in that window.

    The scope is the admission marker's own scope (the exact row the gate
    claimed) with the card row as the fail-closed fallback. The owner-session
    candidates are the worker's ambient ``HERMES_SESSION_ID`` and the session
    recorded on the ``owner_admitted`` marker: ``record_worker_loss`` runs in
    the OWNER process, whose ambient session id is not necessarily the card's
    session.

    ``release_ownership`` opens its own ``BEGIN IMMEDIATE`` (no ``join_txn``),
    so the caller must invoke this AFTER its ``kb.write_txn`` has committed, on
    the same connection. A missing/foreign claim returns
    ``{ok: False, reason: ...}`` instead of raising, which is the idempotent
    path: an already released or superseded claim is a no-op, not an error.
    Returns the per-candidate release results (for diagnostics/tests).
    """
    import logging
    results = []
    log = logging.getLogger(__name__)
    try:
        from cross_session_ownership import release_ownership
    except Exception as exc:  # pragma: no cover - import failure is fatal elsewhere too
        log.debug('kanban ownership release skipped: %s', exc)
        return results
    try:
        scope, admitted = _admission_lease(conn, context)
        if scope is None:
            scope = _ledger_scope_from_row(conn, context)
        if not scope:
            return results
        candidates = []
        try:
            # ContextVar-aware: the worker's session may be bound in-process,
            # not present in ``os.environ``.
            from gateway.session_context import get_session_env
            candidates.append(get_session_env('HERMES_SESSION_ID'))
        except Exception:
            candidates.append(os.environ.get('HERMES_SESSION_ID', ''))
        if admitted:
            candidates.append(admitted)
        seen = set()
        for session_id in candidates:
            session_id = str(session_id or '').strip()
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            try:
                results.append(release_ownership(conn, scope, session_id, reason=reason))
            except Exception as exc:
                # Never raise: a broken/unreadable ledger must not fail the
                # worker's own exit booking.
                log.debug('kanban ownership release failed: %s', exc)
                results.append({'ok': False, 'reason': f'error:{exc}'[:200], 'scope': scope})
    except Exception as exc:
        log.debug('kanban ownership release skipped: %s', exc)
    return results


def _heartbeat_claims_once(context, owner_session=None, *, ttl_seconds=HEARTBEAT_TTL_SECONDS):
    """Extend BOTH leases that keep this worker's run authoritative; never raises.

    Returns a dict: ``{'kanban': bool, 'ledger': bool, 'scope': str|None,
    'owner': str|None, 'error': str|None, ...}``. A board/DB failure is a
    skipped beat, not a dead worker: an unextended lease is detected by the
    reconciler, whereas a worker killed by its own heartbeat is guaranteed to
    lose the run. The owner session resolves from the admission marker first
    (the exact identity the ledger claimed), then the caller's value, then
    ``HERMES_SESSION_ID``.
    """
    import logging
    import time
    result = {'kanban': False, 'ledger': False, 'scope': None, 'owner': None, 'error': None}
    db = (context.get('db') or os.environ.get('HERMES_KANBAN_DB') or '').strip()
    task_id = (context.get('task_id') or os.environ.get('HERMES_KANBAN_TASK') or '').strip()
    claim_lock = context.get('claim_lock') or os.environ.get('HERMES_KANBAN_CLAIM_LOCK')
    if not db or not task_id:
        result['error'] = 'missing_carrier'
        return result
    try:
        from hermes_cli.kanban_db_connect import connect_closing
        from hermes_cli import kanban_db as kb
    except Exception as exc:  # pragma: no cover - import failure is fatal elsewhere too
        result['error'] = f'import:{exc}'[:200]
        return result
    try:
        with connect_closing(Path(db)) as conn:
            scope, admitted = _admission_lease(conn, context)
            if scope is None:
                scope = _ledger_scope_from_row(conn, context)
            owner = admitted or owner_session or (os.environ.get('HERMES_SESSION_ID') or '').strip()
            result['scope'] = scope
            result['owner'] = owner or None
            # 1) The Kanban claim: CAS on claim_lock, mirrors onto task_runs.
            # Fail-closed like the ledger: never resurrect a LAPSED lease. This
            # worker holds the exact claim_lock, but a lease that already expired
            # was abandonable by the reconciler; extending it here would launder a
            # stale claim into a live one and defeat the retry proof.
            try:
                now = int(time.time())
                live = conn.execute(
                    "SELECT claim_expires FROM tasks WHERE id=? AND status='running' AND claim_lock=?",
                    (task_id, claim_lock)).fetchone()
                if live is None or live[0] is None or int(live[0]) < now:
                    result['kanban_reason'] = 'claim_not_live'
                else:
                    result['kanban'] = bool(kb.heartbeat_claim(
                        conn, task_id, ttl_seconds=ttl_seconds, claimer=claim_lock))
            except Exception as exc:
                result['kanban_error'] = str(exc)[:200]
            # 2) The cross-session ownership ledger: CAS on owner_session + lease.
            if not _ownership_enforce():
                result['ledger_reason'] = 'ownership_not_enforced'
            elif not scope:
                result['ledger_reason'] = 'no_scope'
            elif not owner:
                result['ledger_reason'] = 'no_owner_session'
            else:
                try:
                    from cross_session_ownership import heartbeat_ownership
                    hb = heartbeat_ownership(conn, scope, owner, ttl_seconds=ttl_seconds)
                    result['ledger'] = bool(hb.get('ok'))
                    result['ledger_reason'] = hb.get('reason')
                except Exception as exc:
                    result['ledger_error'] = str(exc)[:200]
    except Exception as exc:
        result['error'] = str(exc)[:200]
        logging.getLogger(__name__).debug('kanban worker heartbeat skipped: %s', exc)
    return result


def _start_claim_heartbeat(context, owner_session=None, *, interval_seconds=None,
                           ttl_seconds=HEARTBEAT_TTL_SECONDS):
    """Start the per-turn daemon heartbeat thread; returns its stop Event.

    The first beat runs immediately (a turn may outlive the TTL before the first
    tick), then every ``interval_seconds``. The thread never raises out of its
    target: :func:`_heartbeat_claims_once` absorbs every failure.
    """
    stop = threading.Event()
    interval = float(interval_seconds if interval_seconds is not None else _heartbeat_interval_seconds())

    def beat():
        _heartbeat_claims_once(context, owner_session, ttl_seconds=ttl_seconds)
        while not stop.wait(interval):
            _heartbeat_claims_once(context, owner_session, ttl_seconds=ttl_seconds)

    threading.Thread(target=beat, name='kanban-claim-heartbeat', daemon=True).start()
    return stop


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
    stop_heartbeat = None
    try:
        # C3 (#47 Phase 2): beat both leases for the WHOLE turn, not only on tool
        # activity. A worker blocked in one long call emits no activity, so the
        # tool-layer auto-heartbeat never fires and the ledger lease would lapse.
        owner_session = getattr(agent, 'session_id', None) or os.environ.get('HERMES_SESSION_ID')
        stop_heartbeat = _start_claim_heartbeat(context, owner_session)
        result = _run_task_turns(agent, frame, history, context)
        code = turn_exit_code(result, kanban_worker=True)
        last_output = str(result.get('final_response') or '')[-500:] if isinstance(result, dict) else ''
        return result
    except Exception as exc:
        code = turn_exit_code(None, kanban_worker=True, **credential_failure_flags(exc))
        raise
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat.set()
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
    import os
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    with connect_closing(Path(context['db'])) as conn:
        booked = False
        with kb.write_txn(conn):
            if _bound_worker_matches(conn, context):
                # Closing a run clears its claim/PID and replaces metadata; keep the
                # result in the immutable attempt event stream instead.
                kb._append_event(conn, context['task_id'], 'worker_result',
                    {'pid': os.getpid(), 'claim_lock': context['claim_lock'], 'exit_code': code,
                     'last_output': last_output}, run_id=context['run_id'])
                booked = True
        if booked:
            # C4 (#47 Phase 2): the attempt is over, so its ledger authority ends.
            # AFTER the commit — release_ownership opens its own BEGIN IMMEDIATE.
            _release_ownership_claim(conn, context, reason=f'worker_result:{code}')


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
        with connect_closing(Path(context['db'])) as conn:
            lost = False
            with kb.write_txn(conn):
                row = conn.execute('SELECT status, current_run_id FROM tasks WHERE id=?',
                                   (context['task_id'],)).fetchone()
                if row is None or row[0] != 'running' or row[1] != run_id:
                    return
                kb._append_event(conn, context['task_id'], 'worker_lost',
                                 {'reason': str(reason)[:200], 'claim_lock': context['claim_lock'],
                                  'run_id': run_id, 'pid': pid, 'bound': bool(bound)},
                                 run_id=run_id)
                lost = True
            if lost:
                # C4 (#47 Phase 2): a lost worker no longer holds the work item.
                # AFTER the commit — release_ownership opens its own BEGIN IMMEDIATE.
                _release_ownership_claim(conn, context,
                                         reason=f'worker_lost:{str(reason)[:120]}')
    except Exception:
        import logging
        logging.getLogger(__name__).exception('kanban worker_lost record failed')


def _bound_worker_matches(conn, context):
    import os
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound' ORDER BY id DESC LIMIT 1",
        (context['task_id'], context['run_id'])).fetchone()
    bound = json.loads(row[0]) if row else {}
    return (bound.get('pid'), bound.get('claim_lock')) == (os.getpid(), context['claim_lock'])


def _bound_claim_is_live(context):
    """Retry proof for a managed worker: the frame's exact run/claim lease is live and unexpired,
    and THIS interpreter is the one ``bind_worker_context`` bound to it. The run's worker_pid is
    the dispatcher's spawned submitter here, so pid identity comes from ``worker_bound`` instead."""
    from agent.kanban_turn_recovery import worker_claim_is_live
    if not worker_claim_is_live(task_id=context['task_id'], db_path=context['db'], run_id=str(context['run_id']),
                                claim_lock=context['claim_lock'], verify_pid=False):
        return False
    try:
        with closing(sqlite3.connect(Path(context['db']).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            return _bound_worker_matches(conn, context)
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
    # C3 (#47 Phase 2): the background thread covers the long turn; these beats
    # cover every model ENTRY boundary (first turn, each retry, each goal turn).
    owner_session = getattr(agent, 'session_id', None) or os.environ.get('HERMES_SESSION_ID')

    def beat():
        _heartbeat_claims_once(context, owner_session)

    beat()
    result = agent.run_conversation(prompt, conversation_history=history, turn_author=KANBAN_TURN_AUTHOR,
                                    task_id=context['task_id'])

    def recover(nudge):
        nonlocal result
        beat()
        prior = result.get('messages') if isinstance(result, dict) else None
        result = agent.run_conversation(nudge, conversation_history=prior or
                                        agent._session_db.get_messages_as_conversation(agent.session_id),
                                        turn_author=KANBAN_TURN_AUTHOR,
                                        task_id=context['task_id'])
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
        beat()
        result = agent.run_conversation(prompt, conversation_history=agent._session_db.get_messages_as_conversation(agent.session_id),
                                        turn_author=KANBAN_TURN_AUTHOR,
                                        task_id=context['task_id'])
        return result.get('final_response', '')
    run_kanban_goal_loop(task_id=context['task_id'], goal_text=context['goal_text'], run_turn=turn,
        task_status_fn=status, block_fn=block, max_turns=context['goal_max_turns'] or DEFAULT_MAX_TURNS,
        first_response=result.get('final_response', ''))
    return result
