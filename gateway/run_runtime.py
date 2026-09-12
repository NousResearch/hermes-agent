"""Session-runtime lifecycle owned by the ordinary gateway bootstrap.

One SessionAuthority per reserved profile home. The launch home always has one; under
``gateway.multiplex_profiles`` every served secondary gets its own, built under that
profile's runtime scope against that profile's ``state.db``. ``runner.session_authority``
stays the launch profile's authority so single-profile behaviour is byte-identical.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import uuid


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


async def initialize_gateway_runtime(runner):
    from gateway.runtime_bootstrap import TicketStore
    from gateway.runtime_ownership import process_ownership
    from gateway.run import _profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority

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
        # Each home's store resolves through the runner's scope-following handle cache, exactly
        # the handle every later scoped read of that profile uses (one writer per state.db).
        with _profile_runtime_scope(home, hydrate_secrets=False):
            db = getattr(runner._session_db, '_db', runner._session_db)
            if db is None or Path(db.db_path).resolve().parent != home:
                raise RuntimeError(f'session authority database does not belong to the reserved profile {home}')
            registry.add(home, None, name=name)
            authority = await initialize_session_authority(
                runner, profile_id=str(home), instance_id=instance_id, db=db,
                register=index == 0)
        registry.replace(home, authority)
    descriptor['authority_epoch'] = registry.launch.epoch
    descriptor['served_profiles'] = registry.served_profiles()
    runner.session_ticket_store = TicketStore(instance_id, registry.profile_ids())


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
    logger = logging.getLogger(__name__)
    results = {}
    for authority in authorities:
        with owner_scope(authority):
            recover_local_sessions(authority, schedule=True)
            pending = {row['target_session_id'] for row in authority.db._read_all(
                "SELECT DISTINCT target_session_id FROM session_admissions WHERE status IN ('queued','unknown')")}
            bindings = [(entry.session_id, entry.origin, runner._adapter_for_source(entry.origin))
                        for entry in runner.session_store.list_sessions()
                        if entry.session_id in pending and entry.origin is not None]
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
