"""Session-runtime lifecycle owned by the ordinary gateway bootstrap.

One SessionAuthority per reserved profile home. The launch home always has one; under
``gateway.multiplex_profiles`` every served secondary gets its own, built under that
profile's runtime scope against that profile's ``state.db``. ``runner.session_authority``
stays the launch profile's authority so single-profile behaviour is byte-identical.

The served set is not frozen at boot: ``serve_profile_runtime`` / ``unserve_profile_runtime``
grow and shrink it for the hot-serve reconcile, and a secondary whose store cannot be opened
is parked (logged, left out of ``served_profiles``) instead of aborting every other profile.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


def reserved_profile_homes(runner):
    """``(name, home)`` pairs this process reserved: launch home first, then served secondaries."""
    from hermes_constants import get_hermes_home
    launch = get_hermes_home().resolve()
    homes = [(getattr(runner, '_primary_profile_name', None) or 'default', launch)]
    if getattr(runner.config, 'multiplex_profiles', False):
        reserved = getattr(runner.config, '_runtime_profile_homes', None) or ()
        for name, home in reserved:
            canonical = Path(home).resolve()
            if canonical != launch and canonical not in {h for _, h in homes}:
                homes.append((name, canonical))
    return homes


async def _build_profile_authority(runner, name, home, *, register):
    """One profile's authority against its own ``state.db``; raises when that store is unusable."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_authority import initialize_session_authority
    registry = runner.session_authorities
    instance_id = runner.session_runtime_descriptor['instance_id']
    # Each home's store resolves through the runner's scope-following handle cache, exactly
    # the handle every later scoped read of that profile uses (one writer per state.db).
    with _profile_runtime_scope(home, hydrate_secrets=False):
        db = getattr(runner._session_db, '_db', runner._session_db)
        if db is None or Path(db.db_path).resolve().parent != home:
            raise RuntimeError(f'session authority database does not belong to the reserved profile {home}')
        registry.add(home, None, name=name)
        try:
            authority = await initialize_session_authority(
                runner, profile_id=str(home), instance_id=instance_id, db=db, register=register)
        except BaseException:
            registry.remove(home)
            raise
    registry.replace(home, authority)
    return authority


def _park_reserved_profile(runner, name, home, exc):
    """A secondary whose store is unusable is parked: logged, unreserved, left out of
    ``served_profiles`` and named under ``parked_profiles`` in the runtime descriptor. Boot
    continues for every other profile (a runtime hot-add parks the same way)."""
    logger.error("[MULTIPLEX] Profile '%s' not served: its session store is unusable (%s): %s",
                 name, home, exc)
    release_profile_home(runner, home)
    park_profile(runner, name, f'session store unusable: {exc}')


def park_profile(runner, name, reason):
    """Publish *name* under ``parked_profiles`` (name -> reason) in the runtime descriptor and the
    root's ``gateway_state.json``: a client of that profile then gets a terminal ``profile_parked``
    verdict from ``ensure`` instead of waiting its whole deadline for a service that never comes."""
    parked = parked_profile_map(runner)
    parked[name] = str(reason)[:500]
    _record_parked_profiles(parked)


def unpark_profile(runner, name):
    parked = parked_profile_map(runner)
    if parked.pop(name, None) is not None:
        _record_parked_profiles(parked)


def parked_profile_map(runner):
    descriptor = getattr(runner, 'session_runtime_descriptor', None)
    if descriptor is None:
        descriptor = runner.session_runtime_descriptor = {}
    return descriptor.setdefault('parked_profiles', {})


def _record_parked_profiles(parked):
    try:
        from gateway.status import write_runtime_status
        write_runtime_status(parked_profiles=dict(parked))
    except Exception:
        logger.debug('could not record parked_profiles', exc_info=True)


async def initialize_gateway_runtime(runner):
    from gateway.runtime_bootstrap import TicketStore
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import SessionAuthorities

    homes = reserved_profile_homes(runner)
    for _name, home in homes:
        if not process_ownership.owns(home):
            raise RuntimeError(f'session authority requires reserved profile ownership: {home}')
    instance_id = uuid.uuid4().hex
    descriptor = {
        'instance_id': instance_id, 'runtime_protocol': 1,
        'state': 'starting', 'capabilities': [],
        'served_profiles': [],
    }
    runner.session_runtime_descriptor = descriptor
    registry = SessionAuthorities(homes[0][1])
    runner.session_authorities = registry
    for index, (name, home) in enumerate(homes):
        try:
            await _build_profile_authority(runner, name, home, register=index == 0)
        except Exception as exc:
            if index == 0:
                raise  # the launch profile's store is the process's own; nothing to park it behind
            _park_reserved_profile(runner, name, home, exc)
    descriptor['authority_epoch'] = registry.launch.epoch
    descriptor['served_profiles'] = registry.served_profiles()
    # Publish this boot's verdict (an empty map clears a previous run's parked set).
    _record_parked_profiles(parked_profile_map(runner))
    runner.session_ticket_store = TicketStore(instance_id, registry.profile_ids())


def _publish_served_set(runner):
    registry = runner.session_authorities
    runner.session_runtime_descriptor['served_profiles'] = registry.served_profiles()
    runner.session_ticket_store.profile_ids = registry.profile_ids()


def reserve_profile_home(runner, name, home):
    """Grow the process reservation by one profile (hot-serve): take its ``gateway.lock`` and add it
    to the frozen boot set, so every reader of the reservation — and the next restart's
    all-or-nothing reserve — sees it. ``OwnershipConflict`` when another gateway owns the home."""
    from gateway.runtime_ownership import process_ownership
    home = Path(home).resolve()
    process_ownership.reserve([home])
    reserved = getattr(runner.config, '_runtime_profile_homes', None)
    if reserved is not None and all(Path(h).resolve() != home for _n, h in reserved):
        runner.config._runtime_profile_homes = (*reserved, (name, home))


def release_profile_home(runner, home):
    """Shrink the reservation by one profile (deleted, or parked because it cannot be served)."""
    from gateway.runtime_ownership import process_ownership
    home = Path(home).resolve()
    reserved = getattr(runner.config, '_runtime_profile_homes', None)
    if reserved is not None:
        runner.config._runtime_profile_homes = tuple(
            entry for entry in reserved if Path(entry[1]).resolve() != home)
    process_ownership.release(home)


async def serve_profile_runtime(runner, name, home):
    """Hot-serve one reserved profile's runtime: build its authority, recover its durable state and
    publish it in the descriptor/ticket store — the steps boot performs per secondary. Raises when
    the profile's store is unusable; the caller parks it (and releases the reservation)."""
    from gateway.session_authorities import owner_scope
    from gateway.session_bot import recover_bot_deliveries
    from gateway.session_hosted_service import _ensure_hosted_service, start_ready_hosted_services
    from gateway.session_local_recovery import recover_local_sessions
    from gateway.platforms.webhook_ingress import recover_webhook_finalizations
    home = Path(home).resolve()
    registry = runner.session_authorities
    if registry.for_home(home) is not None:
        return registry.for_home(home)
    authority = await _build_profile_authority(runner, name, home, register=False)
    try:
        with owner_scope(authority):
            await recover_bot_deliveries(authority)
            recover_local_sessions(authority, schedule=True)
            await recover_webhook_finalizations(authority)
        if getattr(runner, 'session_control_server', None) is not None:
            await _ensure_hosted_service(runner, authority)
    except BaseException:
        registry.remove(home)
        raise
    _publish_served_set(runner)
    start_ready_hosted_services(runner)
    return authority


async def unserve_profile_runtime(runner, home):
    """Retire one profile's authority (deleted while running) and shrink the published set.
    No-op for a profile this process never served."""
    from gateway.session_cron import unbind_owner
    home = Path(home).resolve()
    registry = runner.session_authorities
    authority = registry.remove(home) if home in registry else None
    if authority is None:
        return
    service = getattr(authority, 'hosted_room_service', None)
    if service is not None:
        await asyncio.to_thread(service.stop, timeout=5)
    tasks = [live.task for live in authority.sessions.values() if live.task is not None]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    unbind_owner(authority)
    from gateway.config import Platform
    adapter = getattr(runner, 'adapters', {}).get(Platform.API_SERVER)
    if adapter is not None:
        from gateway.platforms.api_server_runs import retire_profile_runs
        from gateway.platforms.api_server_store import release_profile_run_idempotency_store
        await retire_profile_runs(adapter, authority)
        release_profile_run_idempotency_store(adapter, authority)
    _publish_served_set(runner)


def _authorities(runner):
    from gateway.session_authorities import all_authorities
    return all_authorities(runner)


async def start_gateway_runtime_api(runner):
    from gateway.run_api import start_gateway_api
    from gateway.session_authorities import owner_scope
    runner.session_api = await start_gateway_api(runner)
    runner.session_runtime_descriptor['api_origin'] = runner.session_api.api_origin
    from gateway.session_bot import recover_bot_deliveries
    for authority in _authorities(runner):
        with owner_scope(authority):
            await recover_bot_deliveries(authority)


async def recover_gateway_native_sessions(runner):
    """Recover against the published routing index and currently connected adapters.

    Stored envelopes are input, not authority to create a route or reconnect a
    transport. The authority preflights every queued sender before any claim.
    """
    import logging
    from gateway.session_authorities import owner_scope
    authorities = _authorities(runner)
    if not authorities:
        return {}
    from gateway.session_hosted_service import ensure_hosted_service
    await ensure_hosted_service(runner)
    from gateway.session_local_recovery import recover_local_sessions
    from gateway.platforms.webhook_ingress import recover_webhook_finalizations
    logger = logging.getLogger(__name__)
    results = {}
    for authority in authorities:
        with owner_scope(authority):
            recover_local_sessions(authority, schedule=True)
            await recover_webhook_finalizations(authority)
            pending = {row['target_session_id'] for row in authority.db._read_all(
                "SELECT DISTINCT target_session_id FROM session_admissions WHERE status IN ('queued','unknown')")}
            bindings = [(owner, entry.origin, runner._adapter_for_source(entry.origin))
                        for entry in runner.session_store.list_sessions() if entry.origin is not None
                        for owner in [authority.logical_owner(entry.session_id)] if owner in pending]
            outcome = await authority.recover_native_sessions(bindings)
        for sid, verdict in outcome.items():
            logger.info('Native session startup recovery %s (%s): %s', sid, authority.profile_id, verdict)
        results.update(outcome)
    return results


def publish_gateway_runtime_ready(runner):
    descriptor = runner.session_runtime_descriptor
    if runner.session_api.task.done() or not runner._running or runner._draining:
        raise RuntimeError('gateway stopped before session API readiness')
    descriptor.update(state='ready', capabilities=[
        'session-authority-v1', 'durable-admission-v1', 'event-replay-v1'])
    from gateway.session_hosted_service import start_ready_hosted_services
    start_ready_hosted_services(runner)


async def wait_gateway_runtime(runner):
    """A vanished interactive listener is fatal, not a healthy headless runtime."""
    shutdown = asyncio.create_task(runner.wait_for_shutdown())
    listener = runner.session_api.task
    try:
        done, _ = await asyncio.wait({shutdown, listener}, return_when=asyncio.FIRST_COMPLETED)
        if listener in done and not runner._draining:
            runner.session_runtime_descriptor.update(state='failed', capabilities=[])
            error = None if listener.cancelled() else listener.exception()
            raise RuntimeError('gateway session API stopped unexpectedly') from error
        await shutdown
    finally:
        if not shutdown.done():
            shutdown.cancel()
        await asyncio.gather(shutdown, return_exceptions=True)


async def drain_gateway_runtime(runner):
    """Withdraw admission before any await; close sockets before DB teardown."""
    from gateway.run_api import stop_gateway_api
    descriptor = getattr(runner, 'session_runtime_descriptor', None)
    if descriptor is None:
        return
    runner._draining = True
    descriptor.update(state='draining', capabilities=[])
    from gateway.session_hosted_service import stop_hosted_service
    await stop_hosted_service(runner)
    # Withdraw the public ingress callback without disconnecting egress needed
    # by already admitted work. Base adapters refuse before stamping acceptance.
    for adapter in runner.adapters.values():
        adapter.set_message_handler(None)
    for adapters in getattr(runner, '_profile_adapters', {}).values():
        for adapter in adapters.values():
            adapter.set_message_handler(None)
    store = getattr(runner, 'session_ticket_store', None)
    if store is not None:
        store.revoke()
    handle = getattr(runner, 'session_api', None)
    if handle is not None:
        await stop_gateway_api(handle)


async def settle_gateway_runtime(runner):
    """Keep authority tasks alive until their last durable settlement write."""
    tasks = [live.task for authority in _authorities(runner)
             for live in authority.sessions.values() if live.task is not None]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
