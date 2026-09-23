"""Session-scoped busy preferences and controls of the existing execution only."""
from functools import partial
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {
        'config.get': partial(busy_config, connection),
        'config.set': partial(busy_config, connection, write=True),
        'session.steer': partial(correct, connection, verb='steer'),
        'session.redirect': partial(correct, connection, verb='redirect'),
    }


def authorize(connection, ref, params, capability):
    connection.authority.authorize(connection.actor, ref, capability)
    if 'profile' in params:
        from hermes_cli.profiles import profile_matches_home
        profile = params['profile']
        if not isinstance(profile, str):
            raise RuntimeStoreError('invalid_params')
        if profile and not profile_matches_home(profile, Path(connection.authority.profile_id)):
            raise RuntimeStoreError('profile_mismatch')


async def busy_config(connection, ref, params, *, write=False):
    allowed = {'session_id', 'profile', 'key'} | ({'value'} if write else set())
    if (set(params) - allowed or not ref.session_id or params.get('key') != 'busy'
            or (write and params.get('value') not in ('interrupt', 'steer', 'queue'))):
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    authority = connection.authority
    live = authority.sessions[ref.session_id]
    if write:
        # UI negotiation is not a settings write and must not rebuild frozen
        # launch policy, toolsets, history, or the retained AIAgent.
        live.busy_input_mode = params['value']
    value = getattr(live, 'busy_input_mode', None)
    if value is None:
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(authority.runner, live.source)
        value = (policy.config().get('display', {}).get('busy_input_mode', 'interrupt') if policy else
                 authority.runner._effective_busy_input_mode(live.source))
        if value not in ('interrupt', 'steer', 'queue'):
            value = 'interrupt'
    return {'key': 'busy', 'value': value, 'scope': 'session'}


async def correct(connection, ref, params, *, verb):
    if (set(params) - {'session_id', 'profile', 'text', 'execution_generation'}
            or not isinstance(params.get('text'), str) or not params['text'].strip()
            or type(params.get('execution_generation')) is not int):
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control')
    authority = connection.authority
    generation = params['execution_generation']
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        authority.check_approval_generation(ref.session_id, generation)
        # Managed workers have a separate control channel, not a cached owner
        # agent. Refuse explicitly instead of acknowledging undeliverable text.
        if ref.session_id in getattr(authority, '_managed_workers', {}):
            raise RuntimeStoreError('unsupported_control')
        agent = authority.agent(ref)
        method = getattr(agent, verb, None)
        if not callable(method):
            raise RuntimeStoreError('execution_not_ready')
        accepted = method(params['text'])
        return {'status': ({'steer': 'queued', 'redirect': 'redirected'}[verb] if accepted else 'rejected'),
                'text': params['text'], 'execution_generation': generation,
                'authority_epoch': authority.epoch}
