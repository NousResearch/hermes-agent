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
        if db is None or (await asyncio.to_thread(Path(db.db_path).resolve)).parent != home:
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
    registry = SessionAuthorities(homes[0][1], multiplexed=getattr(runner.config, 'multiplex_profiles', False))
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
    release_unknown_turn_markers(runner)


def release_unknown_turn_markers(runner):
    """Before the unclean-start pass reads crash-left turn markers: a marked turn whose canonical
    admission recovered ``unknown`` (the owner died before its terminal commit) owes no reply, so
    its marker is cleared rather than ledgered as a delivery the FIFO never committed. A terminal
    admission keeps its marker: its persisted reply is ledgered and sent once, never re-run."""
    from gateway.session_authorities import owner_scope
    store = getattr(runner, 'session_store', None)
    if store is None:
        return
    for authority in _authorities(runner):
        with owner_scope(authority):
            unknown = {row['target_session_id'] for row in authority.db._read_all(
                "SELECT DISTINCT target_session_id FROM session_admissions WHERE status='unknown'")}
            if not unknown:
                continue
            for entry in store.list_sessions():
                if entry.active_turn_token and authority.logical_owner(entry.session_id) in unknown:
                    store.clear_turn_active(entry.session_key, entry.active_turn_token)


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
    registry = getattr(runner, 'session_authorities', None)
    if registry is not None and registry.for_home(home) is not None:
        raise RuntimeError('Profile authority must retire before ownership is released')
    reserved = getattr(runner.config, '_runtime_profile_homes', None)
    if reserved is not None:
        runner.config._runtime_profile_homes = tuple(
            entry for entry in reserved if Path(entry[1]).resolve() != home)
    process_ownership.release(home)


def _authority_tasks(authority):
    """Snapshot every async task family owned directly by one session authority."""
    tasks = [live.task for live in authority.sessions.values() if live.task is not None]
    tasks.extend(getattr(authority, '_bot_receipt_tasks', ()))
    from gateway.session_runtime_workers import mutation_tasks
    tasks.extend(mutation_tasks(authority))
    return list(dict.fromkeys(tasks))


# How long a Stopped turn gets to settle durably before shutdown refuses to release ownership: a managed worker
# acknowledges Stop within STOP_ACK_SECONDS (30) or is terminated, then closed (<=10).
TURN_SETTLE_SECONDS = 45.0


def managed_turn_count(runner):
    """Unmapped claims, managed workers and owner mutations across the served authorities."""
    from gateway.session_runtime_workers import uncounted_runtime_work
    return uncounted_runtime_work(runner)


def stop_authority_turns(authority, *, in_process=False):
    """Cooperative Stop for every turn *authority* is executing: the control a user's Stop sends to a
    managed worker (acknowledged, or terminated within STOP_ACK_SECONDS, and the admission settles
    through its own fenced path). ``in_process`` also hard-interrupts the running agent of each
    executing in-process turn, or latches the generation ``adopt_agent`` consumes before it exists."""
    from hermes_state_runtime import RuntimeStoreError
    signalled = 0
    workers = getattr(authority, '_managed_workers', {})
    for worker in list(workers.values()):
        if worker.closed.is_set() or (getattr(worker, 'stop', None) is not None and worker.stop.is_set()):
            continue
        try:
            worker.control({'type': 'stop'})
        except RuntimeStoreError as exc:
            # A full control queue: control() latched worker.stop first, and the owner's read
            # loop escalates on that latch alone (terminate after STOP_ACK_SECONDS).
            logger.debug('managed Stop not queued (%s); the latch escalates it', exc.reason)
        signalled += 1
    if not in_process:
        return signalled
    from agent.interrupt_compat import request_hard_interrupt
    from gateway.run import _AGENT_PENDING_SENTINEL
    running = getattr(authority.runner, '_running_agents', {})
    for sid, live in list(authority.sessions.items()):
        generation = live.event_stream.execution.get('execution_generation')
        if generation is None or sid in workers:
            continue
        adopted_generation, agent = authority.adopted.get(sid, (None, None))
        if adopted_generation != generation:
            agent = running.get(live.route)
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            authority.pending_stops[sid] = generation
        else:
            request_hard_interrupt(agent, 'Profile stopping', tool_reason='gateway shutdown')
        signalled += 1
    return signalled


def stop_managed_turns(runner):
    """``stop_authority_turns`` across every served authority (whole-runtime shutdown)."""
    return sum(stop_authority_turns(authority) for authority in _authorities(runner))


async def _retire_profile_authority(authority, timeout=None):
    """Stop profile-local services and turns before its ownership is released.

    Claims are refused from here on; every executing turn gets a cooperative Stop, and its session
    task is joined (an in-process turn's task ends only after its executor thread returned and the
    admission settled). The hosted-room stop runs alongside that join under the same deadline: it
    never interrupts accepted turns and its room threads wait for the member turns they observe, so
    those turns must be signalled first or the room stop cannot finish. False when a turn or room
    thread misses *timeout* (default TURN_SETTLE_SECONDS): it may still call tools and write
    history, so the caller must keep the profile's reservation and store handles."""
    from gateway.session_cron import unbind_owner
    from gateway.session_runtime_workers import join_authority_work, stop_authority_work
    timeout = TURN_SETTLE_SECONDS if timeout is None else timeout
    authority.retiring = True
    stop_authority_work(authority)
    service = getattr(authority, 'hosted_room_service', None)
    rooms = None if service is None else asyncio.ensure_future(asyncio.to_thread(service.stop, timeout=timeout))
    try:
        await join_authority_work(authority, timeout)
        workers_stopped = True
    except TimeoutError:
        logger.error('Profile %s workers did not stop; ownership retained', authority.profile_id)
        workers_stopped = False
    rooms_stopped = rooms is None or await rooms is not False
    if not rooms_stopped:
        logger.error('Profile %s hosted rooms did not stop; ownership retained', authority.profile_id)
    if not (workers_stopped and rooms_stopped):
        return False
    # Idle drains and receipt watchers wait on admissions this profile will no longer run.
    rest = _authority_tasks(authority)
    for task in rest:
        task.cancel()
    if rest:
        await asyncio.gather(*rest, return_exceptions=True)
    unbind_owner(authority)
    # No drain is left to answer an observer (a session_busy pause keeps API/webhook/Bot ones
    # waiting): release them all. Their rows stay queued for the next owner, so nothing is resent.
    from hermes_state_runtime import RuntimeStoreError
    authority.native_waiters.clear()
    for waiter in authority.waiters.values():
        if not waiter.done():
            waiter.set_exception(RuntimeStoreError('runtime_draining'))
    authority.waiters.clear()
    return True


class ProfileOwnershipRetained(RuntimeError):
    """A retired attempt's writer is still live: the profile keeps its reservation and authority."""


async def serve_profile_runtime(runner, name, home):
    """Hot-serve one reserved profile's runtime: build its authority, recover its durable state and
    publish it in the descriptor/ticket store — the steps boot performs per secondary. Raises when
    the profile's store is unusable; the caller parks it (and releases the reservation)."""
    from gateway.session_authorities import owner_scope
    from gateway.session_bot import recover_bot_deliveries
    from gateway.session_hosted_service import _ensure_hosted_service, start_ready_hosted_services
    from gateway.session_local_recovery import recover_local_sessions
    from gateway.platforms.webhook_ingress import recover_webhook_finalizations
    home = await asyncio.to_thread(Path(home).resolve)
    registry = runner.session_authorities
    existing = registry.for_home(home)
    if existing is not None and getattr(existing, 'retiring', False):
        # An earlier failed attempt kept this authority because its writer outlived Stop. It refuses
        # every admission, so serving it would start adapters on a dead profile: finish its
        # retirement without waiting, or refuse (the caller parks and keeps the reservation).
        if not await _retire_profile_authority(existing, timeout=0):
            raise ProfileOwnershipRetained(f'an earlier serve of {home} is still stopping')
        registry.remove(home)
    elif existing is not None:
        return existing
    authority = await _build_profile_authority(runner, name, home, register=False)
    try:
        with owner_scope(authority):
            await recover_bot_deliveries(authority)
            recover_local_sessions(authority, schedule=True)
            await recover_webhook_finalizations(authority)
        if getattr(runner, 'session_control_server', None) is not None:
            await _ensure_hosted_service(runner, authority)
    except BaseException:
        if await _retire_profile_authority(authority):
            registry.remove(home)
        raise
    _publish_served_set(runner)
    start_ready_hosted_services(runner)
    return authority


async def unserve_profile_runtime(runner, home):
    """Retire one profile's authority (deleted while running) and shrink the published set.
    False when one of its turns outlived the Stop deadline (keep the reservation); True otherwise,
    including for a profile this process never served."""
    home = await asyncio.to_thread(Path(home).resolve)
    registry = runner.session_authorities
    authority = registry.for_home(home)
    if authority is None:
        return True
    retired = await _retire_profile_authority(authority)
    if not retired:
        return False
    registry.remove(home)
    _publish_served_set(runner)
    return retired


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


async def settle_gateway_runtime(runner):
    """Keep authority tasks alive until their last durable settlement write, within a bound.

    A managed turn still running here gets its Stop first (its worker acknowledges or is
    terminated, and the admission settles interrupted or ``unknown``); a task that still misses
    TURN_SETTLE_SECONDS is logged and left running for recovery. The stop sequence continues:
    the executor quiesce skips the DB close while a writer is live, and process exit releases
    the ownership lock, so raising here would only skip teardown and disarm the watchdog."""
    authorities = _authorities(runner)
    stop_managed_turns(runner)
    tasks = [task for authority in authorities for task in _authority_tasks(authority)]
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=TURN_SETTLE_SECONDS)
        for task in done:
            if not task.cancelled():
                task.exception()
        if pending:
            logger.warning('%d authority task(s) did not settle within %.0fs; left for recovery',
                           len(pending), TURN_SETTLE_SECONDS)
    from gateway.session_runtime_workers import join_authority_work
    for authority in authorities:
        try:
            await join_authority_work(authority, 0)
        except TimeoutError:
            logger.warning('Profile %s still has live writers at shutdown; its admissions stay for recovery',
                           getattr(authority, 'profile_id', '?'))
    # Work settled above. ACP has no per-session destroy,
    # so the stop is the end of every ACP session nobody is viewing (#118216).
    from gateway.session_acp_lifecycle import end_idle_acp_sessions
    for authority in _authorities(runner):
        try:
            end_idle_acp_sessions(authority)
        except Exception:
            # Best-effort bookkeeping: a store that cannot answer must not abort the stop sequence.
            import logging
            logging.getLogger(__name__).warning('ACP sessions of %s not ended at shutdown',
                                                getattr(authority, 'profile_id', '?'), exc_info=True)
    from gateway.run_api import stop_gateway_api
    store = getattr(runner, 'session_ticket_store', None)
    if store is not None:
        store.revoke()
    handle = getattr(runner, 'session_api', None)
    if handle is not None:
        await stop_gateway_api(handle)
