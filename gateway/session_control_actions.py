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
_SUBGOAL_ACTIONS = frozenset({'subgoal.add', 'subgoal.remove', 'subgoal.clear'})
_ENVELOPE = ('type', 'output', 'notice', 'message', 'display')


def _validate(action, args):
    if not isinstance(args, dict) or set(args) - {'text', 'index'}:
        raise RuntimeStoreError('invalid_params')
    if action == 'subgoal.remove' and (type(args.get('index')) is not int or args['index'] < 1):
        raise RuntimeStoreError('invalid_params')
    if action == 'subgoal.add' and (not isinstance(args.get('text'), str) or not args['text'].strip()):
        raise RuntimeStoreError('invalid_params')
    if action not in _ACTION_COMMANDS and action not in _SUBGOAL_ACTIONS:
        raise RuntimeStoreError('invalid_params')


async def _subgoal(connection, ref, params, action, args):
    """A criterion button on the session's goal manager. A goal that is gone or an index that no longer exists is a refusal the card shows; the typed /subgoal command reports the same cases as text."""
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
    action = action.strip()
    _validate(action, args)
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    scope = {key: params[key] for key in ('session_id', 'profile') if key in params}
    # Other viewers (another Desktop window, Ink) repaint their cards from session.control.update, not the
    # reply: execute_command publishes it for the automation commands, the criterion buttons here.
    if action in _SUBGOAL_ACTIONS:
        result = await _subgoal(connection, ref, scope, action, args)
        publish(connection.authority, ref.session_id)
    else:
        from gateway.session_commands import execute_command
        name, arg = _ACTION_COMMANDS[action]
        result = await execute_command(connection, ref, {**scope, 'name': name, 'arg': arg}, dispatch=True)
    from gateway.session_ancillary import read
    snapshot = await read(connection, ref, scope, kind='control')
    return {**snapshot, 'dispatch': {key: result.get(key) for key in _ENVELOPE}}
