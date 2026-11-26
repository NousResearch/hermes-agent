"""Owner-reserved delegated children of a verified worker execution (#141793).

A worker never mints a child. It asks on its OWN fenced receipt stream (epoch, assignment,
generation, sequence), and the owner creates the child's session row and execution, binds it to
the parent's verified process identity and the worker's fresh secret, and enforces the frozen
policy's delegation caps. The child then persists through the ordinary ``worker.persist`` path
under its own claim. Expected refusals (caps, draining) are receipts, not errors: an error would
leave the parent's journal pending and refuse every later write of its turn.
"""
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal
from persistence.workers.delegations import WorkerDelegationPayload
from persistence.workers.vocabulary import CHILD_SEPARATOR

_DEFAULTS = {'max_concurrent_children': 10, 'max_spawn_depth': 1}


def _limit(section, key):
    try:
        return max(1, int(section.get(key, _DEFAULTS[key])))
    except (TypeError, ValueError):
        return _DEFAULTS[key]


def delegation_limits(authority, execution_id, session_id):
    """``(max_concurrent_children, max_spawn_depth)`` from the root conversation's frozen policy
    (the profile's config for a session with no local policy), never from the asking worker."""
    from gateway.session_policy import policy_for_source, profile_config
    root = execution_id.split(CHILD_SEPARATOR, 1)[0]
    if root != execution_id:
        found = authority_persistence(authority).worker_executions.execution_session(execution_id=root)
        session_id = found if found is not None else session_id
    live = authority.sessions.get(authority.logical_owner(session_id))
    policy = policy_for_source(authority.runner, live.source) if live is not None else None
    config = policy.config(authority) if policy is not None else profile_config(authority)
    section = config.get('delegation') or {}
    return _limit(section, 'max_concurrent_children'), _limit(section, 'max_spawn_depth')


async def worker_child_request(connection, ref, params):
    """worker.persist ``child.reserve``: the parent's full claim and assignment are verified first."""
    import asyncio
    from gateway.session_worker import _SCOPE, _claim, _verify
    if set(params) != _SCOPE | {'epoch', 'sequence', 'operation', 'payload'}:
        raise AuthorityRefusal('invalid_params')
    claim = _claim(connection, ref, params)
    if 'worker:adopt' not in connection.actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    _verify(connection, ref, params, claim)
    authority = connection.authority
    workers = authority_persistence(authority).worker_executions

    def reserve():
        refused, limits = None, (0, 0)
        try:
            authority._require_admission_open()
            limits = delegation_limits(authority, params['execution_id'], params['session_id'])
        except AuthorityRefusal as exc:
            refused = exc.reason
        reservation = workers.reserve_child(
            **{key: params[key] for key in ('epoch', 'execution_id', 'session_id', 'generation', 'sequence',
                                            'profile_id', 'pid', 'birth')},
            secret=WorkerDelegationPayload(params['payload']), principal=connection.actor.subject,
            max_concurrent_children=limits[0], max_spawn_depth=limits[1], refused=refused)
        child = reservation.child
        if child is None:
            return {'refused': reservation.refused}
        return {'child': {'session_id': child.session_id, 'execution_id': child.execution_id,
                          'generation': child.generation}}
    return await asyncio.to_thread(reserve)
