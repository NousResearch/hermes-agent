"""Trusted dispatcher claims enter the ordinary owner's durable admission queue."""
from contextlib import closing
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3

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
        skills=list(task.skills or ()), goal_mode=task.goal_mode, goal_max_turns=task.goal_max_turns,
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
    context = json.loads(policy.kanban_json)
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
                {'db': str(Path(authority.db.db_path).resolve()), 'session_id': ref.session_id,
                 'request_id': request_id, 'pid': os.getpid(), 'birth': psutil.Process().create_time()},
                run_id=task.current_run_id)
    await connection.resume(ref, {})
    receipt = await authority.submit(connection.actor, Submission(request_id, ref,
        {'text': f'work kanban task {params["task_id"]}'}, 'queue'))
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
            # Reclaim/timeout must track the executing interpreter, not its disposable viewer.
            conn.execute('UPDATE tasks SET worker_pid=? WHERE id=?', (os.getpid(), task.id))
            conn.execute('UPDATE task_runs SET worker_pid=? WHERE id=?', (os.getpid(), context['run_id']))
            kb._append_event(conn, task.id, 'worker_bound', {'pid': os.getpid(), 'claim_lock': context['claim_lock']}, run_id=context['run_id'])
    os.environ.update(env)
    os.chdir(context['workspace'])
    from agent.shell_hooks import register_from_config
    register_from_config(json.loads(frame['policy']['config_json']), accept_hooks=context['accept_hooks'])


def run_worker_turns(agent, frame, history):
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        return agent.run_conversation(frame['text'], conversation_history=history)
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
    code = 1
    try:
        result = _run_task_turns(agent, frame, history, context)
        code = KANBAN_RATE_LIMIT_EXIT_CODE if result.get('failed') and result.get('failure_reason') in {'rate_limit', 'billing'} else int(bool(result.get('failed') or result.get('interrupted')))
        return result
    finally:
        import os
        from hermes_cli.kanban_db_connect import connect_closing
        from hermes_cli import kanban_db as kb
        with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
            row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound' ORDER BY id DESC LIMIT 1",
                (context['task_id'], context['run_id'])).fetchone()
            if row and json.loads(row[0]) == {'pid': os.getpid(), 'claim_lock': context['claim_lock']}:
                # Closing a run clears its claim/PID and replaces metadata; keep the
                # result in the immutable attempt event stream instead.
                kb._append_event(conn, context['task_id'], 'worker_result',
                    {'pid': os.getpid(), 'claim_lock': context['claim_lock'], 'exit_code': code}, run_id=context['run_id'])


def worker_exit_code(path, params):
    """Read only the dispatcher's exact attempt; never infer success from admission settlement."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        row = conn.execute("SELECT payload FROM task_events WHERE run_id=? AND task_id=? AND kind='worker_result' ORDER BY id DESC LIMIT 1",
            (params['run_id'], params['task_id'])).fetchone()
        result = json.loads(row[0]) if row else {}
        return result.get('exit_code', 1) if result.get('claim_lock') == params['claim_lock'] else 1


def _run_task_turns(agent, frame, history, context):
    from agent.skill_commands import build_preloaded_skills_prompt
    skills, loaded, missing = build_preloaded_skills_prompt(context['skills'], task_id=agent.session_id)
    if missing and not loaded:
        raise ValueError('kanban_skills_unavailable')
    prompt = '\n\n'.join(p for p in (skills, context['context'], frame['text']) if p)
    result = agent.run_conversation(prompt, conversation_history=history)
    if not context['goal_mode']:
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
        result = agent.run_conversation(prompt, conversation_history=agent._session_db.get_messages_as_conversation(agent.session_id))
        return result.get('final_response', '')
    run_kanban_goal_loop(task_id=context['task_id'], goal_text=context['goal_text'], run_turn=turn,
        task_status_fn=status, block_fn=block, max_turns=context['goal_max_turns'] or DEFAULT_MAX_TURNS,
        first_response=result.get('final_response', ''))
    return result
