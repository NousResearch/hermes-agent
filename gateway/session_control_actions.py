"""Desktop's goal / loop / heartbeat / subgoal card buttons (``session.control``) on the owner.

``session.control.read`` was served, so the cards rendered as controllable, while every button
answered -32601. Each action is the typed slash command the button stands for, run through
``execute_command`` (the gateway's own /goal /loop /heartbeat /subgoal handlers, with their
capability, local-source, admission and busy rules), answered with the owner's control snapshot.
"""
from hermes_state_runtime import RuntimeStoreError

_ACTION_COMMANDS = {
    'goal.pause': ('goal', 'pause'), 'goal.resume': ('goal', 'resume'),
    'goal.clear': ('goal', 'clear'), 'goal.unwait': ('goal', 'unwait'),
    'loop.pause': ('loop', 'pause'), 'loop.resume': ('loop', 'resume'), 'loop.stop': ('loop', 'stop'),
    'heartbeat.pause': ('heartbeat', 'pause'), 'heartbeat.resume': ('heartbeat', 'resume'),
    'heartbeat.clear': ('heartbeat', 'clear'), 'subgoal.clear': ('subgoal', 'clear'),
}
_ENVELOPE = ('type', 'output', 'notice', 'message', 'display')


def _command(action, args):
    """``(name, arg)`` of the slash command for *action*, or ``('subgoal', None)`` for an add, whose
    free text must not be parsed as a /subgoal verb ("clear the cache" is a criterion, not a clear)."""
    if not isinstance(args, dict) or set(args) - {'text', 'index'}:
        raise RuntimeStoreError('invalid_params')
    if action in _ACTION_COMMANDS:
        return _ACTION_COMMANDS[action]
    if action == 'subgoal.remove':
        index = args.get('index')
        if type(index) is not int or index < 1:
            raise RuntimeStoreError('invalid_params')
        return 'subgoal', f'remove {index}'
    if action == 'subgoal.add':
        if not isinstance(args.get('text'), str) or not args['text'].strip():
            raise RuntimeStoreError('invalid_params')
        return 'subgoal', None
    raise RuntimeStoreError('invalid_params')


async def _add_subgoal(connection, ref, params, text):
    """``/subgoal <text>``'s add branch (``slash_commands_goals``) on the session's goal manager."""
    from pathlib import Path
    from agent.i18n import t
    from gateway.platforms.base import MessageEvent
    from gateway.run import _profile_runtime_scope
    from gateway.session_busy_controls import authorize
    from gateway.session_local import authorize_local_source
    from gateway.slash_commands_goals import _mgr_call
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
            return {'type': 'exec', 'output': t('gateway.goal.unavailable')}
        if not mgr.has_goal():
            return {'type': 'exec', 'output': t('gateway.subgoal.no_goal')}
        added, err = _mgr_call('/subgoal', mgr.add_subgoal, text)
        index = len(mgr.state.subgoals) if mgr.state else 0
    return {'type': 'exec', 'output': err or t('gateway.subgoal.added', index=index, text=added)}


async def control(connection, ref, params):
    if set(params) - {'session_id', 'profile', 'action', 'args'} or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    action = params.get('action')
    if not isinstance(action, str):
        raise RuntimeStoreError('invalid_params')
    args = {} if params.get('args') is None else params['args']
    name, arg = _command(action.strip(), args)
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    scope = {key: params[key] for key in ('session_id', 'profile') if key in params}
    if arg is None:
        result = await _add_subgoal(connection, ref, scope, args['text'].strip())
    else:
        from gateway.session_commands import execute_command
        result = await execute_command(connection, ref, {**scope, 'name': name, 'arg': arg}, dispatch=True)
    from gateway.session_ancillary import read
    snapshot = await read(connection, ref, scope, kind='control')
    # Other viewers (another Desktop window, Ink) repaint their cards from this event, not the reply.
    connection.authority.sessions[ref.session_id].event_stream.publish(
        ref.session_id, snapshot, event_type='session.control.update')
    return {**snapshot, 'dispatch': {key: result.get(key) for key in _ENVELOPE}}
