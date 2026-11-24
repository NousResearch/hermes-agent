"""Execution-scoped worker operations on the ordinary authenticated authority.

Only explicit local compute registrations are enabled here. A delegated child is reserved by
the owner on its parent's receipt stream (gateway/session_worker_child.py), never registered.
"""
import asyncio
import os
from dataclasses import asdict

from persistence.authority.vocabulary import admission_fingerprint
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal

_SCOPE = {'profile_id', 'session_id', 'execution_id', 'generation', 'pid', 'birth', 'secret'}


def _claim(connection, ref, params):
    import psutil
    authority = connection.authority
    if params.get('profile_id') != authority.profile_id or connection.actor.profile_id != authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    if not connection.actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    pid, birth = params.get('pid'), params.get('birth')
    secret = params.get('secret')
    if (type(pid) is not int or pid <= 0 or pid == os.getpid()
            or type(birth) not in (int, float) or not isinstance(secret, str) or not 16 <= len(secret) <= 256):
        raise AuthorityRefusal('invalid_params')
    try:
        process = psutil.Process(pid)
        from hermes_cli.process_identity import worker_birth
        if (worker_birth(process) != birth or not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
                or process.username() != psutil.Process().username()):
            raise AuthorityRefusal('permission_denied')
    except psutil.Error as exc:
        raise AuthorityRefusal('worker_not_live') from exc
    # Reuse the durable adoption digest as the producer claim. Changing any
    # assignment, principal, profile, PID birth or secret cannot adopt it.
    origin = authority_persistence(authority).worker_executions.claim_origin(
        execution_id=params['execution_id'], session_id=params['session_id'])
    return admission_fingerprint(canonical_target=origin, payload={
        key: params[key] for key in _SCOPE} | {'session_id': origin, 'principal': connection.actor.subject})


def _verify(connection, ref, params, claim):
    workers = authority_persistence(connection.authority).worker_executions
    target = workers.publication_target(
        execution_id=params['execution_id'], session_id=params['session_id'], generation=params['generation'],
        sequence=params.get('sequence', 0), operation=params.get('operation', ''), payload=params.get('payload', {}))
    workers.verify_claim(execution_id=params['execution_id'], session_id=target, generation=params['generation'],
                         adoption_secret=claim)


async def worker_request(connection, ref, params, *, operation):
    from gateway.session_authorities import require_serving
    require_serving(connection.authority)  # adopt/persist would write with the released epoch
    if operation == "persist" and isinstance(params.get("operation"), str) and params["operation"].startswith("delegation."):
        from gateway.session_worker_delegation import worker_delegation_request
        return await worker_delegation_request(connection, ref, params)
    if operation == "persist" and params.get("operation") == "child.reserve":
        from gateway.session_worker_child import worker_child_request
        return await worker_child_request(connection, ref, params)
    extra = {'register': {'kind'}, 'adopt': set(), 'persist': {'epoch', 'sequence', 'operation', 'payload'}}[operation]
    if set(params) != _SCOPE | extra:
        raise AuthorityRefusal('invalid_params')
    claim = _claim(connection, ref, params)
    authority = connection.authority
    if operation == 'register':
        authority._require_admission_open()
    scope = {key: params[key] for key in ('execution_id', 'session_id', 'generation')}
    if operation == 'register':
        # Only a worker-adoption ticket (what a launched worker carries): an interactive client
        # must not park a 'registered' row that nothing reaps (the FIFO would pause until restart).
        authority.authorize(connection.actor, ref, 'worker:adopt')
        if params['kind'] != 'compute':
            raise AuthorityRefusal('unsupported_producer')
        return asdict(await authority_persistence(authority).aio.worker_executions.register(
            epoch=authority.epoch, **scope, kind='compute', adoption_secret=claim, require_idle=True))
    if 'worker:adopt' not in connection.actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    _verify(connection, ref, params, claim)
    if operation == 'adopt':
        return asdict(await authority_persistence(authority).aio.worker_executions.adopt(
            epoch=authority.epoch, **scope, adoption_secret=claim))
    result = await asyncio.to_thread(authority_persistence(authority).worker_executions.persist_operation,
        epoch=params['epoch'], **scope, sequence=params['sequence'], operation=params['operation'],
        payload=params['payload'])
    if params['operation'] == 'execution.finish':
        # Queued input parked behind this worker; nothing else would restart its FIFO.
        authority.wake_after_worker(params['session_id'])
    return result
