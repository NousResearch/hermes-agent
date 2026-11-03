"""The composer's create / edit actions for one session's goal, loop and heartbeat.

Desktop's dialog is the first place a control is *authored* rather than toggled, so these six
actions run the manager write on the owner: built for the session's physical row, inside the owning
profile's runtime scope and the session's own workspace, under the admission lock the gateway's
``/goal`` path takes. A create checks for an existing automation and writes prompt plus criteria in
one state write; an edit goes through the manager's atomic ``mutate_meta``. The scheduler keeps
owning ticks: a created loop carries the route a restart must fire back to.
"""
import asyncio
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

# Closed field sets: an action's args carry nothing a manager would silently ignore.
_FIELDS = {
    'goal': frozenset({'prompt', 'criteria', 'max_turns'}),
    'loop': frozenset({'prompt', 'interval_seconds', 'run_limit', 'stop_condition'}),
    'heartbeat': frozenset({'prompt', 'interval_seconds'}),
}
ACTIONS = frozenset(f'{kind}.{verb}' for kind in _FIELDS for verb in ('create', 'update'))
_MAX_TURNS_CEILING = 1000
_INTERVAL_CEILING = 86_400 * 7
_RUN_LIMIT_CEILING = 10_000


def _refuse(field, detail):
    raise RuntimeStoreError('invalid_params', field=field, detail=detail)


def _prompt(args):
    prompt = args.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        _refuse('prompt', 'a non-empty prompt is required')
    return prompt.strip()


def _criteria(args):
    criteria = args.get('criteria')
    if criteria is None:
        return None
    if not isinstance(criteria, list) or not all(isinstance(item, str) for item in criteria):
        _refuse('criteria', 'must be a list of strings')
    stripped = [item.strip() for item in criteria]
    if any(not item for item in stripped):
        _refuse('criteria', 'items must be non-empty')
    return stripped


def _bounded_int(args, field, *, floor, ceiling):
    value = args.get(field)
    if value is None:
        return None
    if type(value) is not int or value < floor:
        _refuse(field, f'must be an integer >= {floor}')
    if value > ceiling:
        _refuse(field, f'must be <= {ceiling}')
    return value


def _stop_condition(args):
    until = args.get('stop_condition')
    if until is None:
        return None
    if not isinstance(until, str) or not until.strip():
        _refuse('stop_condition', 'must be a non-empty string')
    return until.strip()


def _clean(action, args):
    """The action's args as manager arguments, or ``invalid_params`` naming the field."""
    kind, verb = action.split('.', 1)
    if not isinstance(args, dict) or set(args) - _FIELDS[kind]:
        _refuse('args', 'unknown argument')
    cleaned = {'prompt': _prompt(args)}
    if kind == 'goal':
        cleaned['criteria'] = _criteria(args)
        cleaned['max_turns'] = _bounded_int(args, 'max_turns', floor=1, ceiling=_MAX_TURNS_CEILING)
    elif kind == 'loop':
        from hermes_cli.loops import min_interval_seconds
        cleaned['interval_seconds'] = _bounded_int(
            args, 'interval_seconds', floor=min_interval_seconds(), ceiling=_INTERVAL_CEILING)
        if cleaned['interval_seconds'] is None:
            _refuse('interval_seconds', 'an interval is required')
        cleaned['run_limit'] = _bounded_int(
            args, 'run_limit', floor=1 if verb == 'create' else 0, ceiling=_RUN_LIMIT_CEILING)
        cleaned['stop_condition'] = _stop_condition(args)
    else:
        from hermes_cli.heartbeat import MIN_INTERVAL_SECONDS
        cleaned['interval_seconds'] = _bounded_int(
            args, 'interval_seconds', floor=MIN_INTERVAL_SECONDS, ceiling=_INTERVAL_CEILING)
        if cleaned['interval_seconds'] is None:
            _refuse('interval_seconds', 'an interval is required')
    return cleaned


def _require_absent(mgr, label):
    state = mgr.state
    if state is not None and state.status != 'cleared':
        raise RuntimeStoreError('automation_exists', field=label, detail=(
            f'a {state.status} {label} already exists for this session; clear it first'))


def _require_editable(mgr, label):
    state = mgr.state
    if state is None or state.status in {'cleared', 'done'}:
        raise RuntimeStoreError('automation_missing', field=label, detail=(
            f'no editable {label} exists for this session (status={state.status if state else "missing"})'))
    return state


def _goal(runner, target, action, args):
    from gateway.session_viewer_automation import manager
    mgr = manager(runner, 'goal', target)
    if action == 'goal.create':
        _require_absent(mgr, 'goal')
        state = mgr.set(args['prompt'], max_turns=args['max_turns'], criteria=args['criteria'])
        message = mgr.next_continuation_prompt() or state.goal
        notice = f'✓ Goal set: {state.goal}'
        return {'type': 'send', 'output': notice, 'notice': notice, 'message': message,
                'display': state.goal}
    _require_editable(mgr, 'goal')
    state = mgr.update(args['prompt'], max_turns=args['max_turns'], criteria=args['criteria'])
    notice = f'✓ Goal updated: {state.goal}'
    return {'type': 'exec', 'output': notice, 'notice': notice}


def _loop(runner, target, action, args, route):
    from gateway.session_viewer_automation import manager
    mgr = manager(runner, 'loop', target)
    if action == 'loop.create':
        _require_absent(mgr, 'loop')
        state = mgr.set(args['prompt'], interval_seconds=args['interval_seconds'],
                        times=args['run_limit'] or 0, until=args['stop_condition'] or '', route=route)
        lines = [f'✓ Loop set ({state.cadence_label()}): {state.prompt}']
        if state.times:
            lines.append(f'  Runs {state.times} time{"s" if state.times != 1 else ""}, then stops.')
        if state.until:
            lines.append(f'  Stops when: {state.until}')
        notice = '\n'.join(lines)
        return {'type': 'exec', 'output': notice, 'notice': notice}
    _require_editable(mgr, 'loop')
    state = mgr.update(args['prompt'], interval_seconds=args['interval_seconds'],
                       times=args['run_limit'], until=args['stop_condition'])
    notice = f'✓ Loop updated ({state.cadence_label()}): {state.prompt}'
    return {'type': 'exec', 'output': notice, 'notice': notice}


def _heartbeat(runner, target, action, args):
    from hermes_cli.heartbeat import format_interval
    from gateway.session_viewer_automation import manager
    mgr = manager(runner, 'heartbeat', target)
    if action == 'heartbeat.create':
        _require_absent(mgr, 'heartbeat')
        state = mgr.set(args['prompt'], interval_seconds=args['interval_seconds'])
        notice = f'✓ Heartbeat set (every {format_interval(state.interval_seconds)}): {state.prompt}'
        return {'type': 'exec', 'output': notice, 'notice': notice}
    _require_editable(mgr, 'heartbeat')
    state = mgr.update(args['prompt'], interval_seconds=args['interval_seconds'])
    notice = f'✓ Heartbeat updated (every {format_interval(state.interval_seconds)}): {state.prompt}'
    return {'type': 'exec', 'output': notice, 'notice': notice}


def apply(runner, target, action, args, route):
    """Run one validated create / edit against the session's persisted automation row."""
    kind = action.split('.', 1)[0]
    if kind == 'goal':
        return _goal(runner, target, action, args)
    if kind == 'loop':
        return _loop(runner, target, action, args, route)
    return _heartbeat(runner, target, action, args)


async def run(connection, ref, params, action, args):
    """Authorize, then write off the owner loop under the owning profile, workspace and lock."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_ancillary import physical_target
    from gateway.session_busy_controls import authorize
    from gateway.session_commands import _session_workspace, session_route
    from gateway.session_local import authorize_local_source
    authority = connection.authority
    authorize(connection, ref, params, 'session:control')
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    target = physical_target(authority, ref)
    route = session_route(live.source)
    home = Path(authority.profile_id)

    def write():
        from hermes_cli.goals import goal_admission_lock
        with _profile_runtime_scope(home), _session_workspace(authority.runner, live.source):
            # The gateway's /goal path takes this same lock: a create's existing-row check and its
            # write, and an edit's read and write, are one admission for the session.
            with goal_admission_lock(target):
                try:
                    return apply(authority.runner, target, action, _clean(action, args), route)
                except RuntimeStoreError:
                    raise
                except (RuntimeError, ValueError, IndexError) as exc:
                    # The manager's own refusals are answers, not escapes: a loop awaiting its tick,
                    # a turn cap under the turns already used, a goal or loop no longer editable. The
                    # dialog shows the field and the manager's reason (as ``/subgoal`` maps its own).
                    raise RuntimeStoreError('invalid_params', field=action.split('.', 1)[0],
                                            detail=str(exc)) from exc
    result = await asyncio.to_thread(write)
    if action == 'heartbeat.create':
        # The gateway-wide poller fires due ticks through the adapter FIFO; a heartbeat authored
        # here joins it now instead of after the next restart's restore scan. Loop ticks need no
        # registration: the wakeup watcher scans the stores and the new row carries its route.
        runner = authority.runner
        runner._register_heartbeat_watch(runner._session_key_for_source(live.source), live.source, target)
    return result
