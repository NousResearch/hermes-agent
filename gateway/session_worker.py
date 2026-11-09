"""Execution-scoped worker operations on the ordinary authenticated authority.

Only explicit local compute registrations are enabled here. A delegated child is reserved by
the owner on its parent's receipt stream (gateway/session_worker_child.py), never registered.
"""
import asyncio
import hmac
import logging
import os

from gateway.session_admission import admission_fingerprint
from hermes_state_runtime import (
    RuntimeStoreError, _secret_digest, adopt_worker_execution,
    mutate_worker_execution, register_worker_execution,
)

logger = logging.getLogger(__name__)

_SCOPE = {'profile_id', 'session_id', 'execution_id', 'generation', 'pid', 'birth', 'secret'}


def _claim(connection, ref, params, *, require_live=False):
    """The producer claim for one worker operation.

    ``require_live`` is for the lane that CREATES the assignment (``register``): it never reaches
    ``_verify``, so the psutil probe is its only evidence the pid is live and ours. Every other
    caller is digest-gated by ``_verify`` and keeps the permissive default.
    """
    import psutil
    authority = connection.authority
    if params.get('profile_id') != authority.profile_id or connection.actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if not connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    pid, birth = params.get('pid'), params.get('birth')
    secret = params.get('secret')
    if (type(pid) is not int or pid <= 0 or pid == os.getpid()
            or type(birth) not in (int, float) or not isinstance(secret, str) or not 16 <= len(secret) <= 256):
        raise RuntimeStoreError('invalid_params')
    try:
        process = psutil.Process(pid)
        from hermes_cli.process_identity import worker_birth
        live = (worker_birth(process) == birth and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
                and process.username() == psutil.Process().username())
    except psutil.Error as exc:
        raise RuntimeStoreError('worker_not_live') from exc
    # health: allow BLE001 -- a probe that could not be EVALUATED is no evidence against the caller
    except Exception as exc:
        # UNPROVABLE, not disproved. psutil's Windows status probe allocates the process table and
        # raises MemoryError under host memory pressure (2026-11-06, fleet-wide). Escaping here used
        # to answer the persist with JSON-RPC -32603 "internal error": the worker's store then
        # refused every later write, the child exited 1, the owner settled the attempt `unknown`
        # and the Kanban run died with "execution unknown; retained by owner" after doing its work.
        #
        # Only the digest-gated lanes may continue: the claim below binds the caller to its stored
        # assignment through _verify's hmac, so an unevaluable probe is not evidence against it.
        # `register` CREATES that assignment and never runs _verify, so the probe is its only
        # evidence the pid exists, is not a zombie and is ours: refuse cleanly (the dispatcher
        # retries) rather than park a `registered` row for a pid nothing ever verified.
        if require_live:
            raise RuntimeStoreError('worker_not_live') from exc
        logger.warning('worker liveness probe unprovable for pid %s: %r', pid, exc)
    else:
        if not live:
            raise RuntimeStoreError('permission_denied')
    # Reuse the durable adoption digest as the producer claim. Changing any
    # assignment, principal, profile, PID birth or secret cannot adopt it.
    from hermes_state_worker_compression import worker_claim_origin
    origin = worker_claim_origin(authority.db, params)
    return admission_fingerprint(canonical_target=origin, payload={
        key: params[key] for key in _SCOPE} | {'session_id': origin, 'principal': connection.actor.subject})


def _verify(connection, ref, params, claim):
    from hermes_state_worker_compression import worker_retry_target
    authority = connection.authority
    target = worker_retry_target(authority.db, params)
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                           (params['execution_id'],)).fetchone()
        if row is None:
            raise RuntimeStoreError('not_found')
        if row['session_id'] != target:
            raise RuntimeStoreError('permission_denied')
        if type(params['generation']) is not int or row['generation'] != params['generation']:
            raise RuntimeStoreError('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], _secret_digest(claim)):
            raise RuntimeStoreError('permission_denied')


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
        raise RuntimeStoreError('invalid_params')
    # _claim and _verify are blocking (a psutil process-table probe and state.db reads): run them off the
    # event loop. On it, one slow claim froze the whole gateway, chat and every other worker's receipt
    # included, for up to 12 s (py-spy, 2026-11-09).
    claim = await asyncio.to_thread(_claim, connection, ref, params, require_live=operation == 'register')
    authority = connection.authority
    if operation == 'register':
        authority._require_admission_open()
    scope = {key: params[key] for key in ('execution_id', 'session_id', 'generation')}
    if operation == 'register':
        # Only a worker-adoption ticket (what a launched worker carries): an interactive client
        # must not park a 'registered' row that nothing reaps (the FIFO would pause until restart).
        authority.authorize(connection.actor, ref, 'worker:adopt')
        if params['kind'] != 'compute':
            raise RuntimeStoreError('unsupported_producer')
        return await asyncio.to_thread(register_worker_execution, authority.db, epoch=authority.epoch,
            **scope, kind='compute', adoption_secret=claim, require_idle=True)
    if 'worker:adopt' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    await asyncio.to_thread(_verify, connection, ref, params, claim)
    if operation == 'adopt':
        return await asyncio.to_thread(adopt_worker_execution, authority.db, epoch=authority.epoch,
                                       **scope, adoption_secret=claim)
    result = await asyncio.to_thread(mutate_worker_execution, authority.db, epoch=params['epoch'],
        **scope, sequence=params['sequence'], operation=params['operation'], payload=params['payload'])
    if params['operation'] == 'execution.finish':
        # Queued input parked behind this worker; nothing else would restart its FIFO.
        authority.wake_after_worker(params['session_id'])
    return result
