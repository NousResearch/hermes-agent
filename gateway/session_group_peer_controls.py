"""Retry and Discard for cross-gateway members, on evidence only.

Retry requeues a deferred peer turn only when the driver proved its gateway never received it
(``hosted_room_driver.is_proven_nonadmission``) and the room's owner, roster and route are still
the ones that attempt was bound to before it was sent. Discard consumes that same durable
nonadmission proof. Accepted or uncertain work uses exact Stop and terminal reconciliation;
a missing receipt never authorizes Discard.
"""
import hashlib
import json
import time
from functools import partial

from gateway import hosted_room_driver as tasks, hosted_room_links as links, hosted_rooms as rooms
from hermes_state_runtime import RuntimeStoreError, _epoch

_OWNER = 'gateway.hosted.owner.v1:'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _member_id(task):
    return task['payload'].get('target_member_id') or task['payload']['target_profile']


def _snapshot(service, conn, binding, member_id, profile):
    """The owner, roster and route a peer attempt is bound to, read in one transaction."""
    _epoch(conn, service.authority.epoch)
    tasks._require_room_authority(conn, binding.room_id, binding.gateway_id, binding.authority_epoch)
    if binding.gateway_id != rooms.local_authority_gateway_id():
        raise RuntimeStoreError('permission_denied')
    owner = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + binding.room_id,)).fetchone()
    members = json.loads(conn.execute('SELECT members_json FROM hosted_rooms WHERE room_id=?',
                                      (binding.room_id,)).fetchone()[0])
    member = next((m for m in members if m.get('member_id') == member_id), None)
    raw = conn.execute('SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?',
                       (binding.room_id, member_id)).fetchone()
    if owner is None or member is None or raw is None:
        raise RuntimeStoreError('permission_denied')
    stored = links.StoredRoomLink.from_record(dict(raw))
    target, catalog = member.get('target') or {}, stored.catalog
    if (member.get('profile') != profile or stored.target_profile != profile
            or target.get('kind') != 'peer' or target.get('profile') != profile
            or target.get('installation_id') != catalog.installation_id
            or target.get('capability_digest') != catalog.catalog_digest
            or stored.status == 'needs_reauthorization'):
        raise RuntimeStoreError('peer_target_mismatch')
    route = {k: v for k, v in stored.as_record().items() if k not in {'status', 'updated_at'}}
    return {'owner': owner[0], 'members_digest': _digest(members), 'route_digest': _digest(route),
            'authority_gateway_id': binding.gateway_id, 'authority_epoch': binding.authority_epoch}, stored


def _stored_route(binding, stored):
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    catalog = stored.catalog
    return PeerMemberRoute(
        home_install_id=binding.gateway_id, member_id=stored.member_id,
        target_install_id=catalog.installation_id, target_profile=stored.target_profile,
        capability_digest=catalog.catalog_digest, execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id=stored.cancellation_scope_id, trace_id=stored.trace_id, grant=stored.grant)


def capture_retry_binding(service, binding, task, route, client):
    """Bind an attempt about to be sent to the current durable authority, else ``None``.

    ``None`` only means the attempt can never be retried; it is still sent as usual.
    """
    try:
        with service.authority.db._read_ctx() as conn:
            frozen, stored = _snapshot(service, conn, binding, _member_id(task), task['payload']['target_profile'])
    except (RuntimeStoreError, tasks.DriverStateError, ValueError):
        return None
    if route != _stored_route(binding, stored) or getattr(client, 'base_url', None) != stored.target_url:
        return None
    return frozen


def _validate(service, conn, task, binding):
    """The proof, and the authority it was bound to before sending, still hold.

    The task row itself is fenced by the requeue's own transaction.
    """
    if not tasks.is_proven_nonadmission(task):
        raise RuntimeStoreError('unknown_execution')
    frozen, stored = _snapshot(service, conn, binding, _member_id(task), task['payload']['target_profile'])
    if task['result']['nonadmission']['retry_binding'] != frozen:
        raise RuntimeStoreError('permission_denied')
    return stored


def _live_route(service, binding, stored):
    """The published route and client, which must still be exactly the stored one."""
    key = (binding.room_id, stored.member_id)
    route, client = service.peer_routes.get(key), service.peer_clients.get(key)
    if client is None or route != _stored_route(binding, stored):
        raise RuntimeStoreError('peer_target_mismatch')
    return route, client


def _require_owner(service, authority, runtime):
    """Act only for the room service its gateway still serves, while that gateway admits work."""
    from gateway.session_authorities import authority_for_home
    if (service.authority is not authority or getattr(authority, 'hosted_room_service', None) is not service
            or authority_for_home(authority.runner, authority.profile_id) is not authority
            or service.runtime is not runtime):
        raise RuntimeStoreError('runtime_coordination_required')
    authority._require_admission_open()
    status = runtime.status()
    if not status['running'] or status['stopping']:
        raise RuntimeStoreError('runtime_coordination_required')


def retry_available(service, task, binding):
    """Read-only: whether Retry would be accepted now (no call to the member's gateway)."""
    try:
        with service._policy_lock:
            authority, runtime = service.authority, service.runtime
            _require_owner(service, authority, runtime)
            with authority.db._read_ctx() as conn:
                _live_route(service, binding, _validate(service, conn, task, binding))
                lease = runtime._leases.get(binding.room_id)
                if lease is None:
                    return False
                tasks._require_active_lease(conn, lease, now=runtime.clock())
        return True
    except (RuntimeStoreError, ValueError):
        return False


def retry_peer(service, task, binding):
    """Requeue one proven-unreceived peer turn after a fresh check of its current grant."""
    from gateway.session_group_peers import probe_route
    with service._policy_lock:
        authority, runtime = service.authority, service.runtime
        _require_owner(service, authority, runtime)
        with authority.db._read_ctx() as conn:
            stored = _validate(service, conn, task, binding)
        _, client = _live_route(service, binding, stored)
        lease = runtime._ensure_lease(binding)
        key = (binding.room_id, stored.member_id)
    scope = {'room_id': binding.room_id, 'home_install_id': binding.gateway_id,
             'authority_gateway_id': binding.gateway_id, 'authority_epoch': binding.authority_epoch,
             'member_id': stored.member_id, 'target_profile': stored.target_profile}
    probe_error = None
    try:
        probe_route(client, stored.grant, stored.catalog, scope)
    except RuntimeStoreError as exc:
        probe_error = exc
    with service._policy_lock:
        def authorize(conn):
            # Rechecked inside the writing transaction: everything may have changed during the probe.
            # (The requeue itself is fenced on the task row and the room lease.)
            _require_owner(service, authority, runtime)
            _validate(service, conn, task, binding)

        def observe(conn, status):
            authorize(conn)
            conn.execute('UPDATE hosted_room_links SET status=?, updated_at=? WHERE room_id=? AND member_id=?',
                         (status, time.time(), *key))

        if probe_error is not None:
            if probe_error.reason in {'invalid_room_grant', 'room_reauthorization_required'}:
                authority.db._execute_write(lambda conn: observe(conn, 'needs_reauthorization'))
                service._peer_route_status[key] = 'needs_reauthorization'
            raise probe_error
        # Readiness and consuming the proof commit together, or neither does.
        operation = partial(tasks.requeue_deferred_task, authorize=lambda conn: observe(conn, 'ready'))
        result = runtime._requeue(operation, task, lease, binding.room_id)
        service._peer_route_status[key] = 'ready'
        return result


def _receipt(service, binding, task):
    """``(route, receipt scope, durable receipt)`` of a peer attempt; the receipt may be None."""
    route = service.peer_routes.get((binding.room_id, _member_id(task)))
    if route is None:
        return None, None, None
    scope = {'room_id': binding.room_id, 'home_install_id': route.home_install_id,
             'authority_gateway_id': binding.gateway_id, 'authority_epoch': binding.authority_epoch,
             'member_id': route.member_id, 'target_install_id': route.target_install_id,
             'target_profile': route.target_profile}
    return route, scope, rooms.remote_run_receipt(service.db_path, record={
        **scope, 'task_id': task['identity'].task_id, 'execution_generation': task['execution_generation']})


def discard_available(service, task, binding):
    """Only durable proven nonadmission can be discarded without remote Stop."""
    return retry_available(service, task, binding)


def discard_peer(service, task, binding):
    """Consume one exact nonadmission proof; accepted/unknown work uses Stop."""
    cancel_id = f"discard:{task['execution_generation']}"
    with service._policy_lock:
        authority, runtime = service.authority, service.runtime
        _require_owner(service, authority, runtime)
        if task['status'] == 'cancelled' and task.get('cancel_id') == cancel_id:
            return task
        if task['status'] == 'indeterminate':
            raise RuntimeStoreError('peer_stop_required' if _receipt(service, binding, task)[2] else 'unknown_execution')
        if task['status'] != 'deferred':
            raise RuntimeStoreError('stale_generation')
        def authorize(conn):
            _require_owner(service, authority, runtime)
            current = tasks._task_from_row(tasks._load_task(conn, task['identity']))
            if current['execution_generation'] != task['execution_generation'] or current['status'] != 'deferred':
                raise RuntimeStoreError('stale_generation')
            _validate(service, conn, current, binding)
        result = tasks.cancel_task(service.db_path, task['identity'], cancel_id=cancel_id,
            expected_cancel_generation=task['cancel_generation'], clock=runtime.clock, authorize=authorize)
        runtime._set_blocked(binding.room_id, False)
    service.publish_terminal(binding, result)
    runtime.wakeup()
    return result


def peer_action(service, task, binding):
    """The control a peer turn offers now: ``retry``, ``discard`` or ``None``."""
    if task['status'] == 'deferred':
        return 'retry' if retry_available(service, task, binding) else None
    return None

