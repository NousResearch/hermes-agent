"""Desktop's goal / loop / heartbeat / subgoal card buttons (``session.control``) on the owner.

``session.control.read`` was served, so the cards rendered as controllable, while every button
answered -32601. Each action is the typed slash command the button stands for, run through
``execute_command`` (the gateway's own /goal /loop /heartbeat /subgoal handlers, with their
capability, local-source, admission and busy rules), answered with the owner's control snapshot.
The subgoal criteria buttons run the same manager directly: their "nothing changed" answers
(no active goal, no such criterion) are named refusals, not the typed command's error text
inside a success envelope, which Desktop would paint as "action succeeded".
"""
import logging

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_ACTION_COMMANDS = {
    'goal.pause': ('goal', 'pause'), 'goal.resume': ('goal', 'resume'),
    'goal.clear': ('goal', 'clear'), 'goal.unwait': ('goal', 'unwait'),
    'loop.pause': ('loop', 'pause'), 'loop.resume': ('loop', 'resume'), 'loop.stop': ('loop', 'stop'),
    'heartbeat.pause': ('heartbeat', 'pause'), 'heartbeat.resume': ('heartbeat', 'resume'),
    'heartbeat.clear': ('heartbeat', 'clear'), 'subgoal.clear': ('subgoal', 'clear'),
}
_ENVELOPE = ('type', 'output', 'notice', 'message', 'display')


def _command(action, args):
    """``(name, arg)`` of the slash command for *action*; the two subgoal actions return their
    ``subgoal.add`` / ``subgoal.remove`` name and run the manager directly, whose free text must
    not be parsed as a /subgoal verb ("clear the cache" is a criterion, not a clear)."""
    if not isinstance(args, dict) or set(args) - {'text', 'index'}:
        raise RuntimeStoreError('invalid_params')
    if action in _ACTION_COMMANDS:
        return _ACTION_COMMANDS[action]
    if action == 'subgoal.remove':
        index = args.get('index')
        if type(index) is not int or index < 1:
            raise RuntimeStoreError('invalid_params')
        return 'subgoal.remove', index
    if action == 'subgoal.add':
        if not isinstance(args.get('text'), str) or not args['text'].strip():
            raise RuntimeStoreError('invalid_params')
        return 'subgoal.add', args['text'].strip()
    raise RuntimeStoreError('invalid_params')


async def _run_subgoal(connection, ref, params, op, arg):
    """A subgoal card action: the typed /subgoal command's branch on the session's goal manager.
    A card action whose change didn't land is a named refusal, never the command's error text
    inside a success envelope: Desktop reads ``type: 'exec'`` as "action succeeded"."""
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
            raise RuntimeStoreError('goal_unavailable', detail=t('gateway.goal.unavailable'))
        if not mgr.has_goal():
            raise RuntimeStoreError('no_active_goal', detail=t('gateway.subgoal.no_goal'))
        if op == 'add':
            added, err = _mgr_call('/subgoal', mgr.add_subgoal, arg)
            if err:
                raise RuntimeStoreError('no_active_goal', detail=err)
            index = len(mgr.state.subgoals) if mgr.state else 0
            return {'type': 'exec', 'output': t('gateway.subgoal.added', index=index, text=added)}
        removed, err = _mgr_call('/subgoal remove', mgr.remove_subgoal, arg, errors=(IndexError, RuntimeError))
        if err:
            raise RuntimeStoreError('subgoal_not_found', detail=err)
        return {'type': 'exec', 'output': t('gateway.subgoal.removed', index=arg, text=removed)}


def publish(authority, session_id, *, only_if_present=False):
    """Push one live session's control snapshot to every viewer as ``session.control.update``.

    Desktop and Ink paint the goal/loop/heartbeat cards from this event and run the loop countdown off
    its ``next_due_at``, so every place that moves that state on the owner publishes: the slash
    commands, a fired or settled loop wakeup, a judged turn. ``only_if_present`` skips a session with
    no automation state (a plain turn's event stream stays unchanged).
    """
    live = authority.sessions.get(session_id) if authority is not None else None
    if live is None:
        return
    from gateway.session_ancillary import control_snapshot
    from gateway.session_contract import SessionRef
    try:
        snapshot = control_snapshot(authority, SessionRef(authority.profile_id, session_id))
    except RuntimeStoreError:  # unreadable row: the change itself already landed; the next publish repaints
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
    name, arg = _command(action.strip(), args)
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    scope = {key: params[key] for key in ('session_id', 'profile') if key in params}
    # Other viewers (another Desktop window, Ink) repaint their cards from session.control.update, not the
    # reply: execute_command publishes it for the automation commands, the subgoal actions here.
    if name.startswith('subgoal.'):
        result = await _run_subgoal(connection, ref, scope, name[len('subgoal.'):], arg)
        publish(connection.authority, ref.session_id)
    else:
        from gateway.session_commands import execute_command
        result = await execute_command(connection, ref, {**scope, 'name': name, 'arg': arg}, dispatch=True)
    from gateway.session_ancillary import read
    snapshot = await read(connection, ref, scope, kind='control')
    return {**snapshot, 'dispatch': {key: result.get(key) for key in _ENVELOPE}}
