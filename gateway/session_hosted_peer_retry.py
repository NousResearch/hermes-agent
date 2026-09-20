"""Canonical NEW peer retry rights from a durable, exact nonadmission proof.

The binding is captured before submit, not inferred from a late successful
probe. Status is read-only; only the explicit RPC probes the current grant.
"""
import hashlib
import json
from functools import partial

from gateway import hosted_room_driver as tasks, hosted_room_links as links, hosted_rooms as rooms
from gateway.hosted_room_route_schema import require_room_work_open
from gateway.session_group_setup import _grant_claims, verify_invited_catalog
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _snapshot(service, conn, binding, member_id, profile):
    _epoch(conn, service.authority.epoch)
    tasks._require_room_authority(conn, binding.room_id, binding.gateway_id, binding.authority_epoch)
    if binding.gateway_id != rooms.local_authority_gateway_id():
        raise RuntimeStoreError('permission_denied')
    require_room_work_open(conn, binding.room_id, error=tasks.RoomUnavailableError)
    owner = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         ('gateway.hosted.owner.v1:' + binding.room_id,)).fetchone()
    members = json.loads(conn.execute('SELECT members_json FROM hosted_rooms WHERE room_id=?',
                                      (binding.room_id,)).fetchone()[0])
    member = next((m for m in members if m.get('member_id') == member_id), None)
    raw = conn.execute('SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?',
                       (binding.room_id, member_id)).fetchone()
    if owner is None or member is None or raw is None:
        raise RuntimeStoreError('permission_denied')
    stored = links.StoredRoomLink.from_record(dict(raw))
    target = member.get('target', {})
    catalog = stored.catalog
    if (member['profile'] != profile or stored.target_profile != profile
            or target.get('kind') != 'peer' or target.get('profile') != profile
            or target.get('installation_id') != catalog.installation_id
            or target.get('capability_digest') != catalog.catalog_digest
            or catalog.execution_policy.target_profile != profile
            or not catalog.text or not catalog.persistent_process
            or stored.status == 'needs_reauthorization'):
        raise RuntimeStoreError('peer_setup_conflict')
    scope = dict(room_id=binding.room_id, home_install_id=binding.gateway_id,
        authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
        member_id=member_id, target_profile=profile)
    claims = _grant_claims(stored.grant, attachments=catalog.attachments)
    expected = dict(scope, target_install_id=catalog.installation_id,
                    execution_policy_digest=catalog.execution_policy.policy_digest)
    if (type(claims.get('authority_epoch')) is not int
            or any(claims.get(k) != v for k, v in expected.items())):
        raise RuntimeStoreError('invalid_room_grant')
    # Credentials remain private even if a driver result is later projected.
    route_record = {k: v for k, v in stored.as_record().items() if k not in {'status', 'updated_at'}}
    frozen = dict(owner=owner[0], members_digest=_digest(members), route_digest=_digest(route_record),
                  authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch)
    return frozen, stored, scope


def capture_retry_binding(service, binding, task, route, client):
    """Bind the actual transport to current durable authority BEFORE admission."""
    member = task['payload'].get('target_member_id') or task['payload']['target_profile']
    with service.authority.db._read_ctx() as conn:
        frozen, stored, scope = _snapshot(service, conn, binding, member, task['payload']['target_profile'])
    expected = PeerMemberRoute(home_install_id=binding.gateway_id, member_id=member,
        target_install_id=stored.catalog.installation_id, target_profile=stored.target_profile,
        capability_digest=stored.catalog.catalog_digest,
        execution_policy_digest=stored.catalog.execution_policy.policy_digest,
        cancellation_scope_id=stored.cancellation_scope_id, trace_id=stored.trace_id,
        grant=stored.grant, attachments=stored.catalog.attachments)
    if route != expected or client.base_url != stored.target_url:
        raise RuntimeStoreError('peer_setup_conflict')
    return frozen


def _validate(service, task, binding, conn):
    if not tasks.is_proven_nonadmission(task):
        raise RuntimeStoreError('unknown_execution')
    current = tasks._task_from_row(tasks._load_task(conn, task['identity']))
    if (not tasks.is_proven_nonadmission(current) or current['result'] != task['result']
            or current['payload_digest'] != task['payload_digest']):
        raise RuntimeStoreError('stale_generation')
    member = task['payload'].get('target_member_id') or task['payload']['target_profile']
    frozen, stored, scope = _snapshot(service, conn, binding, member, task['payload']['target_profile'])
    proof = task['result']['nonadmission']
    if (proof['authority_epoch'] != binding.authority_epoch
            or proof['retry_binding'] != frozen):
        raise RuntimeStoreError('permission_denied')
    return stored, scope


def _require_retry_owner(service, authority, runtime):
    from gateway.session_authorities import authority_for_home
    if (service.authority is not authority or getattr(authority, 'hosted_room_service', None) is not service
            or authority_for_home(authority.runner, authority.profile_id) is not authority
            or service.runtime is not runtime):
        raise RuntimeStoreError('peer_setup_conflict')
    authority._require_admission_open()
    status = runtime.status()
    if not status['running'] or status['stopping']:
        raise RuntimeStoreError('runtime_coordination_required')


def retry_available(service, task, binding):
    try:
        with service._policy_lock:
            authority, runtime = service.authority, service.runtime
            _require_retry_owner(service, authority, runtime)
            with authority.db._read_ctx() as conn:
                _validate(service, task, binding, conn)
                lease = runtime._leases.get(binding.room_id)
                if lease is None:
                    return False
                tasks._require_active_lease(conn, lease, now=runtime.clock())
        return True
    except (RuntimeStoreError, ValueError):
        return False


def retry_peer(service, task, binding):
    # Capture only current, exact authority under the policy lock. No tracked
    # renewal client, staging, recovery or submit is permitted in this operation.
    with service._policy_lock:
        authority, runtime = service.authority, service.runtime
        epoch = authority.epoch
        _require_retry_owner(service, authority, runtime)
        with authority.db._read_ctx() as conn:
            stored, scope = _validate(service, task, binding, conn)
        lease = runtime._ensure_lease(binding)
        tasks.require_active_lease(service.db_path, lease, clock=runtime.clock)
        hydrated = service._hydrate_persisted_peer_route(binding.room_id, stored.member_id)
        if hydrated is None:
            raise RuntimeStoreError('peer_setup_conflict')
        route, client = hydrated
        if capture_retry_binding(service, binding, task, route, client) != task['result']['nonadmission']['retry_binding']:
            raise RuntimeStoreError('peer_setup_conflict')

    # Network I/O must not stall healthy rooms' planning or publication.
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    probe_error = None
    try:
        verify_invited_catalog(client, stored.grant, stored.catalog, scope)
    except PeerRunsHTTPError as exc:
        probe_error = exc

    with service._policy_lock:
        def authorize(conn):
            # Recheck the installed owner AND drain gate inside the actual SQL
            # writer, not just before its potentially blocking transaction begin.
            _require_retry_owner(service, authority, runtime)
            _epoch(conn, epoch)
            _validate(service, task, binding, conn)
            tasks._require_active_lease(conn, lease, now=runtime.clock())

        def observe(conn, status):
            authorize(conn)
            conn.execute("UPDATE hosted_room_links SET status=?, updated_at=? "
                         "WHERE room_id=? AND member_id=?",
                         (status, runtime.clock(), binding.room_id, stored.member_id))

        if probe_error is not None:
            if probe_error.needs_reauthorization:
                # Even a negative observation belongs to this exact live owner;
                # never let a late failure mutate a withdrawn/replaced route.
                authority.db._execute_write(lambda conn: observe(conn, 'needs_reauthorization'))
                service._peer_route_status[(binding.room_id, stored.member_id)] = 'needs_reauthorization'
            raise RuntimeStoreError('member_unavailable') from probe_error

        # Ready and proof consumption remain one SQL transaction. Any lost
        # task/lease/work-open fence rolls both back before memory or wakeup.
        operation = partial(tasks.requeue_deferred_task, authorize=lambda conn: observe(conn, 'ready'))
        result = runtime._requeue(operation, task, lease, binding.room_id)
        service._peer_route_status[(binding.room_id, stored.member_id)] = 'ready'
        return result
