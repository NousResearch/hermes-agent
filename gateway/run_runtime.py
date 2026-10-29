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
from contextlib import suppress
import logging
from pathlib import Path
import threading
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
    db = await asyncio.to_thread(open_profile_store, runner, home)
    with _profile_runtime_scope(home, hydrate_secrets=False):
        if db is None:
            raise RuntimeError(_store_open_failure(runner, home))
        if (await asyncio.to_thread(Path(db.db_path).resolve)).parent != home:
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


def open_profile_store(runner, home):
    """*home*'s SessionDB through the runner's scoped handle cache, opened in the CALLING thread
    under that profile's scope (bound here: a worker thread inherits no scope it was not given).

    Callers run it in a worker: a first open runs the schema / FTS setup, about a second per store
    on a loaded host, and run on the loop it froze every connected client for the whole
    multi-profile boot. Later opens of the same path are cache hits."""
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(home, hydrate_secrets=False):
        db = runner._session_db
        return getattr(db, '_db', db)


async def _prewarm_profile_stores(runner, homes):
    """Open every secondary's store concurrently off the loop before the authorities are built in
    order. A failed open is not reported here: the cache records it and the authority build that
    follows parks that profile with the recorded reason."""
    await asyncio.gather(*(asyncio.to_thread(open_profile_store, runner, home) for _name, home in homes),
                         return_exceptions=True)


def pin_launch_store():
    """Open the launch profile's state.db in the calling (worker) thread and hold one registry
    reference, so the synchronous ``SessionStore`` open inside ``GatewayRunner()`` on the loop is a
    registry hit instead of the schema setup. None when the open fails: the store's own open then
    reports and backs off exactly as before."""
    from hermes_state_registry import acquire
    try:
        return acquire()
    except Exception:
        logger.debug('launch state.db prewarm failed; the store open reports it', exc_info=True)
        return None


def unpin_launch_store(db):
    if db is not None:
        from hermes_state_registry import release
        release(db)


def _store_open_failure(runner, home):
    """Why *home*'s ``state.db`` has no handle: the scoped handle caches answer None for a failed open
    (and while they back off), so the park reason named an ownership mismatch over the real cause, a
    legacy store that crashed its migration on every open (#141467). The reason is the exception the
    failed open recorded; reopening here to learn it could create or migrate the store, and would skip
    the caches' backoff."""
    path = home / 'state.db'
    caches = (getattr(getattr(runner, 'session_store', None), '_db_handle_cache', None),
              getattr(runner, '_session_db_handle_cache', None))
    error = next((e for e in (c.last_error(path) for c in caches if c is not None) if e), '')
    if error:
        return f'{path} did not open: {error}'
    return f'{path} is unavailable (its last open failed; retry pending)'


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


def profile_parked(runner, name):
    """True when *name* is parked: not served now, but a rescan serves it again, so its ingress is
    temporarily unavailable (retry it), never an invalid route (refuse it for good)."""
    descriptor = getattr(runner, 'session_runtime_descriptor', None) or {}
    return bool(name) and name in (descriptor.get('parked_profiles') or {})


def refuse_parked_scope(runner):
    """An API request under a parked profile's ``/p/<profile>/`` prefix finds no authority: refuse it
    retryably (``runtime_draining``, a 503 with ``Retry-After``), not with the final
    ``profile_mismatch`` of a profile this gateway does not serve."""
    from gateway.session_authorities import served_profile_name
    from hermes_constants import get_hermes_home
    name = served_profile_name(get_hermes_home())
    if profile_parked(runner, name):
        from gateway.parked_route_refusal import ParkedProfileRefusal
        raise ParkedProfileRefusal(name)


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
    await _prewarm_profile_stores(runner, homes[1:])
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
    from gateway.session_authorities import admission_owner, owner_scope
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
                if entry.active_turn_token and admission_owner(authority, entry.session_id) in unknown:
                    store.clear_turn_active(entry.session_key, entry.active_turn_token)


def _publish_served_set(runner):
    registry = runner.session_authorities
    runner.session_runtime_descriptor['served_profiles'] = registry.served_profiles()
    runner.session_ticket_store.profile_ids = registry.profile_ids()


def reserve_profile_home(runner, name, home, cancel=None):
    """Grow the process reservation by one profile (hot-serve): take its ``gateway.lock`` and add it
    to the frozen boot set, so every reader of the reservation — and the next restart's
    all-or-nothing reserve — sees it. Waits out a transient holder within boot's window (blocking:
    the event loop calls :func:`reserve_profile_home_in_worker`). ``OwnershipConflict`` when another
    gateway owns the home; *cancel* set ends the wait with nothing reserved."""
    from gateway.run_bootstrap import _browse_reconcile_reserve_wait_s
    from gateway.runtime_ownership import process_ownership, reserve_waiting_out_transient_holders
    home = Path(home).resolve()
    reserve_waiting_out_transient_holders(process_ownership, [home], wait_s=_browse_reconcile_reserve_wait_s(),
                                          cancel=cancel)
    reserved = getattr(runner.config, '_runtime_profile_homes', None)
    if reserved is not None and all(Path(h).resolve() != home for _n, h in reserved):
        runner.config._runtime_profile_homes = (*reserved, (name, home))


async def reserve_profile_home_in_worker(runner, name, home):
    """:func:`reserve_profile_home` off the event loop, settled on cancellation. A cancelled bare
    ``to_thread`` await leaves its worker retrying, free to take the flock after the canceller's
    ownership cleanup with no coroutine left to serve or release it. So a cancel stops the worker,
    waits for it to finish (each further cancel is absorbed until then), undoes a reservation it
    completed first, and only then propagates."""
    from gateway.runtime_ownership import process_ownership
    home = await asyncio.to_thread(Path(home).resolve)
    owned_before = process_ownership.owns(home)
    cancel = threading.Event()
    worker = asyncio.ensure_future(asyncio.to_thread(reserve_profile_home, runner, name, home, cancel))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        while not worker.done():
            with suppress(asyncio.CancelledError):
                await asyncio.wait({worker})
        if not worker.cancelled() and worker.exception() is None and not owned_before:
            release_profile_home(runner, home)
        raise


def release_profile_home(runner, home):
    """Shrink the reservation by one profile (deleted, or parked because it cannot be served)."""
    from gateway.runtime_ownership import process_ownership
    home = Path(home).resolve()
    registry = getattr(runner, 'session_authorities', None)
    if registry is not None and registry.for_home(home) is not None:
        raise RuntimeError('Profile authority must retire before ownership is released')
    _withdraw_profile_home(runner, home)
    process_ownership.release(home)


def _withdraw_profile_home(runner, home):
    """Drop *home* from the served set every startup consumer reads (adapters, served status, cron
    ticks, ingress, spool recovery) without releasing its ``gateway.lock``."""
    reserved = getattr(runner.config, '_runtime_profile_homes', None)
    if reserved is not None:
        runner.config._runtime_profile_homes = tuple(
            entry for entry in reserved if Path(entry[1]).resolve() != home)


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
# What a stop keeps of its watchdog leash for the phases after the settle (adapter teardown, executor
# quiesce, SessionDB close, exit-state writes): the settle never spends it.
STOP_TAIL_RESERVE_SECONDS = 10.0


def managed_turn_count(runner):
    """Unmapped claims, managed workers and owner mutations across the served authorities."""
    from gateway.session_runtime_workers import uncounted_runtime_work
    return uncounted_runtime_work(runner)


def stop_authority_turns(authority, *, in_process=False, reason=None):
    """Cooperative Stop for every turn *authority* is executing: the control a user's Stop sends to a
    managed worker (acknowledged, or terminated within STOP_ACK_SECONDS, and the admission settles
    through its own fenced path). ``in_process`` also hard-interrupts the running agent of each
    executing in-process turn, or latches the generation ``adopt_agent`` consumes before it exists.
    ``reason`` (a gateway restart/shutdown) is recorded on each worker it signals, so the turn settles
    with that cause; a worker already stopped (the user's Stop came first) keeps none. A worker
    still being spawned gets the same Stop and cause through latches (``_stop_spawning_managed``)."""
    from hermes_state_runtime import RuntimeStoreError
    signalled = 0
    workers = getattr(authority, '_managed_workers', {})
    for worker in list(workers.values()):
        if worker.closed.is_set() or (getattr(worker, 'stop', None) is not None and worker.stop.is_set()):
            continue
        worker.stop_cause = reason
        try:
            worker.control({'type': 'stop'})
        except RuntimeStoreError as exc:
            # A full control queue: control() latched worker.stop first, and the owner's read
            # loop escalates on that latch alone (terminate after STOP_ACK_SECONDS).
            logger.debug('managed Stop not queued (%s); the latch escalates it', exc.reason)
        signalled += 1
    signalled += _stop_spawning_managed(authority, workers, reason)
    if not in_process:
        return signalled
    from agent.interrupt_compat import request_hard_interrupt
    from gateway.run import _AGENT_PENDING_SENTINEL
    running = getattr(authority.runner, '_running_agents', {})
    for sid, live in list(authority.sessions.items()):
        generation = live.event_stream.execution.get('execution_generation')
        if generation is None or sid in workers or sid in getattr(authority, '_managed_spawning', {}):
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


def _stop_spawning_managed(authority, workers, reason):
    """A managed turn whose worker is still being spawned has nothing to signal yet: latch the
    Stop for its generation (``adopt_agent`` delivers it to the worker, which is never
    bootstrapped) and, for a gateway restart/shutdown, the cause its pre-bootstrap interrupted
    result carries (#134889 GW-22). A user's Stop latched first keeps no cause."""
    stopped = 0
    for sid, generation in list(getattr(authority, '_managed_spawning', {}).items()):
        if sid in workers:
            continue
        stops = authority.pending_stops
        if reason and stops.get(sid) != generation:
            causes = getattr(authority, 'pending_stop_causes', None)
            if causes is None:
                causes = authority.pending_stop_causes = {}
            causes[sid] = (generation, reason)
        stops[sid] = generation
        stopped += 1
    return stopped


def stop_managed_turns(runner, reason=None):
    """``stop_authority_turns`` across every served authority (whole-runtime shutdown)."""
    return sum(stop_authority_turns(authority, reason=reason) for authority in _authorities(runner))


def interrupt_side_agents(runner, reason):
    """Whole-runtime shutdown: interrupt every counted side agent (``session_runtime_workers``)."""
    from gateway.session_runtime_workers import interrupt_side_agents as interrupt
    return interrupt(runner, reason)


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
    if getattr(authority.runner, 'session_ticket_store', None) is not None:
        _publish_served_set(authority.runner)  # discovery and tickets stop naming it now
    stop_authority_work(authority)
    from gateway.session_llm_oneshot import stop as stop_oneshots
    stop_oneshots(authority)
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
    # Ownership is released next; until a successor bumps the epoch, A's epoch still fences its
    # writes as current, so a stale connection to A must find it retired instead.
    authority.retired = True
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
    the profile's store is unusable; the caller parks it (and releases the reservation). Its
    queued native rows are recovered once its adapters connect (``recover_served_native_rows``)."""
    from gateway.session_authorities import owner_scope
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
            _drop_retired_local_adapter(authority)
            await recover_bot_mailbox(authority)
            recover_local_sessions(authority, schedule=True)
            await recover_webhook_finalizations(authority)
            _recover_published_api_turns(runner, authority)
        if getattr(runner, 'session_control_server', None) is not None:
            await _ensure_hosted_service(runner, authority)
    except BaseException:
        if await _retire_profile_authority(authority):
            registry.remove(home)
        raise
    _publish_served_set(runner)
    start_ready_hosted_services(runner)
    return authority


def _drop_retired_local_adapter(authority):
    """A LOCAL adapter still bound to this profile's previous authority, once that authority fully
    retired, gives way to *authority*'s own: restore and create otherwise refuse every local session
    of the served profile as ``runtime_draining``. A retained writer never gets here (the serve is
    refused while the previous authority is still retiring)."""
    from gateway.config import Platform
    from gateway.session_local_recovery import local_adapter_map
    adapters = local_adapter_map(authority)
    previous = getattr(adapters.get(Platform.LOCAL), 'authority', None)
    if previous is not None and previous is not authority and getattr(previous, 'retired', False):
        del adapters[Platform.LOCAL]


def _recover_published_api_turns(runner, authority):
    """Boot recovers queued API turns when the launch API adapter is published
    (``recover_api_turns``); a profile served after that runs the same pass on its own store."""
    from gateway.config import Platform
    api = (getattr(runner, 'adapters', None) or {}).get(Platform.API_SERVER)
    if api is not None:
        from gateway.session_api_turn import recover_authority_api_turns
        recover_authority_api_turns(api, authority)


async def recover_served_native_rows(runner):
    """Boot's native-row recovery once hot-served profiles' adapters connected (boot runs it after
    every adapter connects), in every serving ledger: a ``bot_profile`` route runs a bot's messages
    on another profile, so the new bots can carry rows queued in any ledger. A session already
    draining keeps its one drain. A store that cannot answer leaves its rows queued for the next
    start and never stops the other ledgers or the reconcile."""
    if getattr(runner, 'session_authorities', None) is None:
        return
    from gateway.session_authorities import serving_authorities
    for authority in serving_authorities(runner):
        try:
            await recover_profile_native_sessions(runner, authority)
        except Exception:  # health: allow BLE001 -- logged; the rows stay queued for the next start
            logger.error('Native session recovery failed for %s', authority.profile_id, exc_info=True)


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


async def recover_bot_mailbox(authority):
    """Owner-startup Bot Chat recovery, skipped when this token cannot use the mailbox dir.

    An elevated install/update can leave ``runtime/bot_live_delivery`` owned by Administrators with
    no entry for the user (#139485, #138872): the gateway cannot even open its ``.lock``. Bot Chat
    deliveries pause (``_locked`` already logged the dir, its owner and the remedy); every other
    client keeps working instead of the whole gateway refusing to start. Any other failure still
    aborts startup."""
    from gateway.session_bot import recover_bot_deliveries
    from tools.bot_live_delivery import MailboxUnusable
    try:
        await recover_bot_deliveries(authority)
    except MailboxUnusable as exc:
        logger.warning('Bot Chat recovery skipped for %s: %s', authority.db.db_path, exc)


async def start_gateway_runtime_api(runner):
    from gateway import run_api
    from gateway.webapp_listener import (
        STARTING, BrowserCredentialError, browser_session_token, dial_origin, listener_plan, register_control_verbs,
    )
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    # The bootstrap registered these before its control socket answered; a runner started without it
    # (an embedded gateway) gets them here, before the listener's slow start.
    register_control_verbs(runner, home)
    host, port, webapp, unusable = listener_plan(home)
    # A webapp: block that asks for the Webapp but cannot be used is reported with its reason.
    runner.webapp_status = ({"state": "unavailable", "url": None, "reason": unusable} if unusable
                            else {"state": "off", "url": None, "reason": None} if webapp is None
                            else dict(STARTING))
    try:
        # Without the Webapp the listener is main's: the same call, a private ephemeral port.
        runner.session_api = await (run_api.start_gateway_api(runner, host=host, port=port, webapp=webapp)
                                    if webapp is not None else run_api.start_gateway_api(runner))
    except (Exception, SystemExit) as exc:
        if webapp is None:
            raise
        # The Webapp's listener could not start (port taken, address unavailable, its browser credential
        # not storable in the home, or the dashboard's auth gate exiting on a wildcard bind with no
        # sign-in provider): native clients still need this gateway, so it keeps main's private
        # listener and reports why the Webapp is not served.
        if isinstance(exc, SystemExit):
            reason = str(exc.code).strip()  # the gate's own message names the fix
        elif isinstance(exc, BrowserCredentialError):
            reason = str(exc)  # names the file and why it cannot be stored
        elif isinstance(exc, OSError) and exc.filename:  # the browser credential's file, not the port
            reason = f"could not use {exc.filename} ({exc.strerror or exc})"
        else:
            cause = (exc.strerror or exc) if isinstance(exc, OSError) else exc
            reason = f"could not listen on {host}:{port} ({cause})"
        logger.error("Webapp unavailable: %s", reason)
        runner.webapp_status = {"state": "unavailable", "url": None, "reason": reason}
        # Open tabs keep sending their credential to whatever answers on the Webapp's port, and that
        # is not this gateway now. Replace it, so one captured there never opens this gateway; the
        # tabs sign in again once the Webapp is back.
        try:
            browser_session_token(home, rotate=True)
        except OSError as write_error:
            logger.warning("Could not replace the Webapp's browser credential: %s", write_error)
        runner.session_api = await run_api.start_gateway_api(runner)
    else:
        if webapp is not None:
            runner.webapp_status = {"state": "serving", "url": dial_origin(host, port), "reason": None}
    runner.session_runtime_descriptor['api_origin'] = runner.session_api.api_origin
    for authority in _authorities(runner):
        await _recover_profile_bot_mailbox(runner, authority)


async def _recover_profile_bot_mailbox(runner, authority):
    """Startup Bot Chat recovery of one served profile. A secondary whose durable leftovers cannot
    be recovered is parked like one whose store cannot be opened (retired, unreserved, named under
    ``parked_profiles``): under multiplex it would otherwise take every other profile down with it,
    on every restart (#140018). The launch profile's failure still aborts the start."""
    from gateway.session_authorities import owner_scope
    try:
        with owner_scope(authority):
            await recover_bot_mailbox(authority)
    except Exception as exc:
        registry = getattr(runner, 'session_authorities', None)
        name = None if registry is None else registry.profile_name(authority)
        if name is None:
            raise  # the launch profile's store is the process's own; nothing to park it behind
        logger.error("[MULTIPLEX] Profile '%s' not served: startup recovery failed: %s", name, exc,
                     exc_info=True)
        home = await asyncio.to_thread(Path(authority.profile_id).resolve)
        if not await _retire_profile_authority(authority):
            # A writer outlived Stop: the lock and the retiring authority stay for shutdown to
            # settle, but the profile leaves the served set, so later boot never starts it.
            _withdraw_profile_home(runner, home)
            park_profile(runner, name, f'startup recovery failed; ownership retained (a control-socket rescan, '
                                       f'`hermes -p {name} gateway run`, serves it once its writer stops): {exc}')
            return
        registry.remove(home)
        _publish_served_set(runner)
        release_profile_home(runner, home)
        park_profile(runner, name, f'startup recovery failed (a control-socket rescan, '
                                   f'`hermes -p {name} gateway run`, serves it again): {exc}')


async def recover_gateway_native_sessions(runner):
    """Recover against the published routing index and currently connected adapters.

    Stored envelopes are input, not authority to create a route or reconnect a
    transport. The authority preflights every queued sender before any claim.
    """
    from gateway.session_authorities import owner_scope, serving_authorities
    authorities = serving_authorities(runner)
    if not authorities:
        return {}
    from gateway.session_hosted_service import ensure_hosted_service
    await ensure_hosted_service(runner)
    from gateway.session_local_recovery import recover_local_sessions
    from gateway.platforms.webhook_ingress import recover_webhook_finalizations
    results = {}
    for authority in authorities:
        with owner_scope(authority):
            recover_local_sessions(authority, schedule=True)
            await recover_webhook_finalizations(authority)
        results.update(await recover_profile_native_sessions(runner, authority))
    return results


async def recover_profile_native_sessions(runner, authority):
    """Bind one authority's queued native rows to the adapters connected now."""
    from gateway.session_authorities import owner_scope
    with owner_scope(authority):
        pending = {row['target_session_id'] for row in authority.db._read_all(
            "SELECT DISTINCT target_session_id FROM session_admissions WHERE status IN ('queued','unknown')")}
        bindings = [(owner, entry.origin, runner._adapter_for_source(entry.origin))
                    for entry in runner.session_store.list_sessions() if entry.origin is not None
                    for owner in [authority.logical_owner(entry.session_id)] if owner in pending]
        outcome = await authority.recover_native_sessions(bindings)
        from gateway.session_detached import recover_detached_sessions
        outcome.update(dict.fromkeys(recover_detached_sessions(authority, pending - set(outcome)), 'detached'))
    for sid, verdict in outcome.items():
        logger.info('Native session startup recovery %s (%s): %s', sid, authority.profile_id, verdict)
    return outcome


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
    from gateway.drain_control import RESTART_WAIT_REASON
    if descriptor.get('drain_reason') == RESTART_WAIT_REASON:
        del descriptor['drain_reason']  # stopping now: no client attaches to answer anymore
    from gateway.session_hosted_service import stop_hosted_service
    await stop_hosted_service(runner)
    # Withdraw the public ingress callback without disconnecting egress needed
    # by already admitted work. Base adapters refuse before stamping acceptance.
    for adapter in runner.adapters.values():
        adapter.set_message_handler(None)
    for adapters in getattr(runner, '_profile_adapters', {}).values():
        for adapter in adapters.values():
            adapter.set_message_handler(None)


async def settle_gateway_runtime(runner, budget=None, reason=None):
    """Keep authority tasks alive until their last durable settlement write, within a bound.

    A managed turn still running here gets its Stop first (its worker acknowledges or is
    terminated, and the admission settles interrupted or ``unknown``); a task that still misses
    TURN_SETTLE_SECONDS (clamped to ``budget``, what the stop's watchdog leash has left, minus
    STOP_TAIL_RESERVE_SECONDS) is logged and left running for recovery. The stop sequence continues:
    the executor quiesce skips the DB close while a writer is live, and process exit releases
    the ownership lock, so raising here would only skip teardown and disarm the watchdog."""
    authorities = _authorities(runner)
    stop_managed_turns(runner, reason)
    from gateway.session_llm_oneshot import stop as stop_oneshots
    for authority in authorities:
        stop_oneshots(authority)  # an answer is no durable settlement: never waited for
    tasks = [task for authority in authorities for task in _authority_tasks(authority)]
    wait = TURN_SETTLE_SECONDS if budget is None else max(0.0, min(TURN_SETTLE_SECONDS,
                                                                   budget - STOP_TAIL_RESERVE_SECONDS))
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=wait) if wait > 0 else (set(), set(tasks))
        for task in done:
            if not task.cancelled():
                task.exception()
        if pending:
            logger.warning('%d authority task(s) did not settle within %.0fs; left for recovery',
                           len(pending), wait)
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
