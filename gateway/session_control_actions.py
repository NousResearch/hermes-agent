"""Desktop's goal / loop / heartbeat / subgoal card buttons (``session.control``) on the owner.

``session.control.read`` was served, so the cards rendered as controllable, while every button
answered -32601. Each action is the typed slash command the button stands for, run through
``execute_command`` (the gateway's own /goal /loop /heartbeat /subgoal handlers, with their
capability, local-source, admission and busy rules), answered with the owner's control snapshot.
"""
import logging

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_ACTION_COMMANDS = {
    'goal.pause': ('goal', 'pause'), 'goal.resume': ('goal', 'resume'),
    'goal.clear': ('goal', 'clear'), 'goal.unwait': ('goal', 'unwait'),
    'loop.pause': ('loop', 'pause'), 'loop.resume': ('loop', 'resume'), 'loop.stop': ('loop', 'stop'),
    'heartbeat.pause': ('heartbeat', 'pause'), 'heartbeat.resume': ('heartbeat', 'resume'),
    'heartbeat.clear': ('heartbeat', 'clear'),
}
_SET_ACTIONS = frozenset({'goal.set', 'goal.wait_time', 'loop.set', 'heartbeat.set'})
_SUBGOAL_ACTIONS = frozenset({'subgoal.add', 'subgoal.remove', 'subgoal.clear'})
_ENVELOPE = ('type', 'output', 'notice', 'message', 'display')


def _validate(action, args):
    if not isinstance(args, dict) or set(args) - {'text', 'index'}:
        raise RuntimeStoreError('invalid_params')
    if action == 'subgoal.remove':
        index = args.get('index')
        if type(index) is not int or index < 1:
            raise RuntimeStoreError('invalid_params')
    if action == 'subgoal.add':
        if not isinstance(args.get('text'), str) or not args['text'].strip():
            raise RuntimeStoreError('invalid_params')
    if action not in _ACTION_COMMANDS and action not in _SUBGOAL_ACTIONS:
        raise RuntimeStoreError('invalid_params')


def _set_args(action, args):
    """Validate typed setter fields before a manager or its persisted state is opened."""
    if not isinstance(args, dict):
        raise RuntimeStoreError('invalid_params')
    if action == 'goal.set':
        title, maximum, contract = args.get('title'), args.get('max_turns'), args.get('contract')
        if (set(args) - {'title', 'max_turns', 'contract'} or not isinstance(title, str) or not title.strip()
                or (maximum is not None and (type(maximum) is not int or maximum < 1))
                or (contract is not None and (not isinstance(contract, dict)
                    or set(contract) - {'outcome', 'verification', 'constraints', 'boundaries', 'stop_when'}
                    or not all(isinstance(value, str) for value in contract.values())))):
            raise RuntimeStoreError('invalid_params')
        return {'title': title.strip(), 'max_turns': maximum, 'contract': contract}
    if action == 'goal.wait_time':
        seconds, reason = args.get('seconds'), args.get('reason', '')
        if set(args) - {'seconds', 'reason'} or type(seconds) is not int or seconds < 1 or not isinstance(reason, str):
            raise RuntimeStoreError('invalid_params')
        return {'seconds': seconds, 'reason': reason.strip()}
    if action == 'loop.set':
        prompt, interval, times, until = args.get('prompt'), args.get('interval_seconds'), args.get('times', 0), args.get('until', '')
        if (set(args) - {'prompt', 'interval_seconds', 'times', 'until'} or not isinstance(prompt, str) or not prompt.strip()
                or (interval is not None and type(interval) is not int) or type(times) is not int or times < 0
                or not isinstance(until, str)):
            raise RuntimeStoreError('invalid_params')
        return {'prompt': prompt.strip(), 'interval_seconds': interval, 'times': times, 'until': until.strip()}
    prompt, interval = args.get('prompt'), args.get('interval_seconds')
    if (set(args) - {'prompt', 'interval_seconds'} or not isinstance(prompt, str) or not prompt.strip()
            or type(interval) is not int or interval < 60):
        raise RuntimeStoreError('invalid_params')
    return {'prompt': prompt.strip(), 'interval_seconds': interval}


def _loop_route(source):
    platform = getattr(source, 'platform', '')
    route = {'platform': platform.value if hasattr(platform, 'value') else str(platform or '')}
    route.update({key: str(getattr(source, key, '') or '')
                  for key in ('chat_id', 'chat_type', 'thread_id', 'user_id', 'user_name', 'profile')})
    return {key: value for key, value in route.items() if value}


async def _set_control(connection, ref, params, action, args):
    """Run a setter at the owner under the same authorization and workspace scope as slash commands."""
    from pathlib import Path
    from gateway.run import _profile_runtime_scope
    from gateway.session_ancillary import physical_target
    from gateway.session_busy_controls import authorize
    from gateway.session_commands import _session_workspace
    from gateway.session_local import authorize_local_source
    from gateway.session_viewer_automation import invoke

    authority = connection.authority
    authorize(connection, ref, params, 'session:control')
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    runner, target = authority.runner, physical_target(authority, ref)
    with _profile_runtime_scope(Path(authority.profile_id)), _session_workspace(runner, live.source):
        if action == 'loop.set':
            from hermes_cli.loops import LoopManager

            await runner._warm_goals_session_db('loop manager')
            LoopManager(session_id=target).set(**args, route=_loop_route(live.source))
        elif action == 'heartbeat.set':
            reply = invoke(runner, 'heartbeat', target, 'set', [args['prompt']], {'interval_seconds': args['interval_seconds']})
            quick_key = runner._session_key_for_source(live.source)
            runner._register_heartbeat_watch(quick_key, live.source, target)
        elif action == 'goal.set':
            reply = invoke(runner, 'goal', target, 'set', [args['title']],
                           {key: value for key, value in {'max_turns': args['max_turns'], 'contract': args['contract']}.items()
                            if value is not None})
        else:
            reply = invoke(runner, 'goal', target, 'wait_for_seconds', [args['seconds']], {'reason': args['reason']})
        if action != 'loop.set' and reply.get('error'):
            raise RuntimeStoreError('invalid_params')
    return {'type': 'exec', 'output': '✓ Control set.'}


async def _subgoal(connection, ref, params, action, args):
    """Mutate one goal criterion through the session's live goal manager."""
    from pathlib import Path
    from agent.i18n import t
    from gateway.platforms.base import MessageEvent
    from gateway.run import _profile_runtime_scope
    from gateway.session_busy_controls import authorize
    from gateway.session_local import authorize_local_source

    authority = connection.authority
    authorize(connection, ref, params, 'session:control')
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    with _profile_runtime_scope(Path(authority.profile_id)):
        mgr, _entry = await authority.runner._get_goal_manager_for_event(
            MessageEvent(text='/subgoal', source=live.source))
        if mgr is None:
            raise RuntimeStoreError('command_unavailable', detail=t('gateway.goal.unavailable'))
        if not mgr.has_goal():
            raise RuntimeStoreError('goal_not_active', detail=t('gateway.subgoal.no_goal'))
        try:
            if action == 'subgoal.add':
                added = mgr.add_subgoal(args['text'].strip())
                output = t('gateway.subgoal.added', index=len(mgr.state.subgoals), text=added)
            elif action == 'subgoal.remove':
                removed = mgr.remove_subgoal(args['index'])
                output = t('gateway.subgoal.removed', index=args['index'], text=removed)
            else:
                count = mgr.clear_subgoals()
                output = (t('gateway.subgoal.cleared_one' if count == 1 else 'gateway.subgoal.cleared_other', count=count)
                          if count else t('gateway.subgoal.none_to_clear'))
        except IndexError as exc:
            raise RuntimeStoreError('invalid_params', field='index', detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise RuntimeStoreError('invalid_params', detail=str(exc)) from exc
    return {'type': 'exec', 'output': output}


def publish(authority, session_id, *, only_if_present=False):
    """Push one live session's control snapshot to every viewer as ``session.control.update``."""
    live = authority.sessions.get(session_id) if authority is not None else None
    if live is None:
        return
    from gateway.session_ancillary import control_snapshot
    from gateway.session_contract import SessionRef
    try:
        snapshot = control_snapshot(authority, SessionRef(authority.profile_id, session_id))
    except RuntimeStoreError:
        logger.debug('session.control.update skipped for %s', session_id, exc_info=True)
        return
    if not only_if_present or snapshot['revision']:
        live.event_stream.publish(session_id, {'control': snapshot}, event_type='session.control.update')


async def control(connection, ref, params):
    if set(params) - {'session_id', 'profile', 'action', 'args'} or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    action = params.get('action')
    if not isinstance(action, str):
        raise RuntimeStoreError('invalid_params')
    args = {} if params.get('args') is None else params['args']
    action = action.strip()
    if action in _SET_ACTIONS:
        args = _set_args(action, args)
    else:
        _validate(action, args)
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    scope = {key: params[key] for key in ('session_id', 'profile') if key in params}
    if action in _SET_ACTIONS:
        result = await _set_control(connection, ref, scope, action, args)
        publish(connection.authority, ref.session_id)
    elif action in _SUBGOAL_ACTIONS:
        result = await _subgoal(connection, ref, scope, action, args)
        publish(connection.authority, ref.session_id)
    else:
        from gateway.session_commands import execute_command
        name, arg = _ACTION_COMMANDS[action]
        result = await execute_command(connection, ref, {**scope, 'name': name, 'arg': arg}, dispatch=True)
    from gateway.session_ancillary import read
    snapshot = await read(connection, ref, scope, kind='control')
    return {**snapshot, 'dispatch': {key: result.get(key) for key in _ENVELOPE}}
