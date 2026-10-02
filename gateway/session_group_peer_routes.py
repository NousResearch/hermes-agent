"""Canonical peer routes: one current grant per member, renewed before it expires, and
replaced grants retired at once.

A route is persisted, then published. A registration that replaces a member's grant first
revokes the old grant exactly on the member's gateway; if that cannot be confirmed, nothing
changes and the operator can retry. A renewal retires the grant it replaced as soon as the
renewed one is live; the replaced grant was about to expire anyway.

During one attempt, ``CanonicalPeerClient`` keeps the attempt on that route:

- new work (dispatch, recovery replay, probe) is sent only with the route's current grant, so
  a transport resolved before a re-registration is refused before anything is sent;
- reads, Stop and approvals of accepted work follow the current grant of the same route, since
  the grant the attempt started with may have been retired meanwhile;
- a health report applies only while the grant it was made with is still current.

``maintain_peer_grants`` renews each room's grants inside that room's own driver cycle, within
a small per-cycle budget, so a member keeps working until the horizon its gateway's operator
chose at invitation (``status_ttl_seconds``), without being invited again.
"""
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
import hashlib
import logging

from gateway import hosted_room_links as links
from gateway import session_group_peer_cleanup as cleanup
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_GRANT_ERRORS = frozenset({'invalid_room_grant', 'room_reauthorization_required'})
_RENEWAL_BUDGET_SECONDS = 2.0  # per room cycle, all of its requests together
_LEASE_HEADROOM_SECONDS = 5.0  # keep the room lease and Stop ahead of renewal
_SCAN_SECONDS = 5.0
_ATTEMPT_SECONDS = 60.0  # at most one renewal attempt per route per minute
_RETRY_SECONDS, _MAX_RETRY_SECONDS = 30.0, 120.0
_RENEWED_TTL_SECONDS = 3600.0
_NEW_WORK = frozenset({'dispatch', 'recover_dispatch', 'probe'})
_OBSERVATION = frozenset({'history', 'status', 'stop', 'stop_receipt', 'output_request', 'recover_output_consent'})


def _retire(client, grant):
    """Revoke one replaced grant exactly; an already unusable grant counts as retired."""
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    retire = getattr(client, 'revoke_grant_exact', None)
    if not callable(retire):
        raise RuntimeStoreError('peer_unreachable')
    try:
        retire(grant=grant)
    except PeerRunsHTTPError as exc:
        if exc.status_code not in {401, 403} or exc.error_code not in _GRANT_ERRORS:
            raise RuntimeStoreError('peer_unreachable') from exc


def publish_route(service, *, room_id, member_id, route, client, target_url, catalog,
                  expected_grant=None, authorize=None, issuance_id=None, renewal_horizon=None):
    """Persist, then publish, one route, retiring the grant it replaces (module docstring).

    ``expected_grant`` makes it a renewal, conditional on the stored grant; ``authorize`` runs
    inside the write transaction (a renewal's lease fence).
    """
    if target_url is None or catalog is None:
        raise ValueError('a canonical peer route is always persisted')
    if not route.execution_policy_digest:
        route = replace(route, execution_policy_digest=catalog.execution_policy.policy_digest)
    if (route.capability_digest != catalog.catalog_digest
            or route.execution_policy_digest != catalog.execution_policy.policy_digest):
        raise ValueError('peer route does not match its target catalog')
    bind_store = getattr(client, 'bind_receipt_store', None)
    if callable(bind_store):
        bind_store(service.db_path)
    renewal = expected_grant is not None
    with service.peer_route_lock:
        if callable(getattr(service, 'is_retiring', None)) and service.is_retiring(room_id):
            raise RuntimeStoreError('room_retiring')
        previous = links.load_room_link(service.db_path, room_id=room_id, member_id=member_id)
        if renewal and (previous is None or previous.grant != expected_grant):
            raise RuntimeStoreError('peer_target_mismatch')
        replaced = previous.grant if previous is not None and previous.grant != route.grant else None
        if replaced is not None and not renewal:
            cleanup.retain(service.db_path, previous)
            _retire(client, replaced)
        def publication_fence(conn):
            if authorize is not None:
                authorize(conn)
            if replaced is not None:
                cleanup.retain(service.db_path, previous, conn=conn)
            if renewal_horizon is not None:
                renewed = links.make_stored_link(room_id=room_id, member_id=member_id,
                    target_url=target_url, target_profile=route.target_profile, grant=route.grant, catalog=catalog,
                    cancellation_scope_id=route.cancellation_scope_id, trace_id=route.trace_id)
                links.remember_verified_renewal(conn, previous, renewed, horizon=renewal_horizon)
            cleanup.release(conn, route.grant)
            if expected_grant is not None and issuance_id is not None:
                cleanup.release_issuance(conn, expected_grant, issuance_id)
        service._save_link(
            room_id=room_id, member_id=member_id, target_url=target_url, target_profile=route.target_profile,
            grant=route.grant, catalog=catalog, cancellation_scope_id=route.cancellation_scope_id,
            trace_id=route.trace_id, authorize=publication_fence)
        service._publish_route((room_id, member_id), route, client)
        if not renewal:  # an operator's new grant is scheduled afresh, not after the old one's
            getattr(service, '_peer_renewals', {}).pop((room_id, member_id), None)
            getattr(service, '_peer_renewal_scans', {}).pop(room_id, None)
        if replaced is not None and renewal:
            try:
                _retire(client, replaced)
            except Exception:
                logger.warning('A renewed peer grant replaced one that could not be retired: room=%s member=%s',
                               room_id, member_id)
    service.runtime.wakeup()


def set_route_status(service, key, status, grant):
    """Record a route's health, only while ``grant`` is still its stored grant."""
    with service._policy_lock:
        if service._peer_route_status.get(key) == status:
            return
        if links.mark_room_link_status(service.db_path, room_id=key[0], member_id=key[1],
                                       status=status, grant=grant):
            service._peer_route_status[key] = status


@contextmanager
def before_sending(method):
    """A failure here happened before the request left: never a non-admission.

    For a dispatch it is phase evidence (``dispatch_not_attempted``), which the driver trusts
    only for a generation it has just allocated; a recovery replay stays ambiguous.
    """
    try:
        yield
    except Exception as exc:
        if method not in {'dispatch', 'recover_dispatch'}:
            raise
        from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
        dispatch = method == 'dispatch'
        try:
            failure = copy(exc)
            if failure is exc:
                raise TypeError('an exception that cannot carry phase evidence')
            failure.not_admitted = False
            failure.dispatch_not_attempted = dispatch
            if not dispatch:
                failure.ambiguous = True
        except Exception:
            failure = PeerRunsHTTPError(
                'peer admission preflight failed', not_admitted=False,
                ambiguous=not dispatch or bool(getattr(exc, 'ambiguous', False)),
                retryable=bool(getattr(exc, 'retryable', False)),
                status_code=getattr(exc, 'status_code', None), error_code=getattr(exc, 'error_code', None))
            failure.dispatch_not_attempted = dispatch
        raise failure from exc


class CanonicalPeerClient:
    """One attempt's client for its peer route (see the module docstring)."""

    def __init__(self, service, binding, key, route, client, *, renewal_lease=None):
        self._service, self._binding, self._key = service, binding, key
        self._route, self._client = route, client
        self._renewal_lease = renewal_lease  # a maintenance renewal publishes only under this lease
        self._members = service._room(binding.room_id)['members']
        self._grant = route.grant  # this attempt's grant: its own renewal, or an adopted one
        self._stored_link = links.load_room_link(service.db_path, room_id=key[0], member_id=key[1])
        self._issuance_id = None

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value):
            return value
        if name == 'recover_dispatch' and callable(getattr(self._client, 'recover_accepted_dispatch', None)):
            return lambda **kwargs: self._recover_existing(value, kwargs)
        if name in _NEW_WORK:
            return lambda **kwargs: self._new_work(name, value, kwargs)
        if name in _OBSERVATION:
            return lambda **kwargs: self._observe(name, value, kwargs)
        return value  # revoke_grant_exact included: exact cleanup never swaps the bearer

    def _recover_existing(self, call, kwargs):
        # End must still observe accepted work. This helper cannot POST a Run;
        # a missing receipt retains the ordinary retiring/new-admission fence.
        observed = {key: kwargs[key] for key in ('dispatch', 'grant')}
        original_grant = self._grant
        accepted = self._observe('recover_accepted_dispatch', self._client.recover_accepted_dispatch, observed)
        if accepted is not None:
            return accepted
        if kwargs.get('observation_only'):
            return self._observe('recover_dispatch', call, {**observed, 'observation_only': True})
        self._grant = original_grant  # no accepted receipt: never promote an observer grant into new-work authority
        return self._new_work('recover_dispatch', call, kwargs)

    def _status(self, status, grant):
        set_route_status(self._service, self._key, status, grant)

    def _report(self, call, kwargs):
        grant = kwargs.get('grant')
        try:
            result = call(**kwargs)
        except Exception as exc:
            if getattr(exc, 'needs_reauthorization', False):
                self._status('needs_reauthorization', grant)
            elif getattr(exc, 'not_admitted', False):
                self._status('unavailable', grant)
            raise
        self._status('ready', grant)
        return result

    def _same_route(self, current, client):
        return (current is not None and replace(current, grant=self._route.grant) == self._route
                and getattr(client, 'base_url', None) == getattr(self._client, 'base_url', None))

    def _new_work(self, name, call, kwargs):
        with before_sending(name):
            if callable(getattr(self._service, 'is_retiring', None)) and self._service.is_retiring(self._key[0]):
                raise RuntimeStoreError('room_retiring')
            grant = self._grant if kwargs.get('grant') == self._route.grant else kwargs.get('grant')
            grant = self._refresh_if_due(name, grant, kwargs)
            with self._service._policy_lock:
                current = self._service.peer_routes.get(self._key)
                if current is None or current.grant != grant or not self._same_route(
                        current, self._service.peer_clients.get(self._key)):
                    raise RuntimeError('peer room route changed before admission')
        if name == 'recover_dispatch' and 'preparation_cancel_generation' in kwargs:
            kwargs = dict(kwargs)
            cancel_generation = kwargs.pop('preparation_cancel_generation')
            kwargs['before_preparation_resume'] = lambda: self._check_preparation_resume(
                kwargs['dispatch'], cancel_generation, grant)
        return self._report(call, {**kwargs, 'grant': grant})

    def _check_preparation_resume(self, dispatch, cancel_generation, grant):
        from gateway import hosted_room_driver as tasks
        from tui_gateway.hosted_room_peer_output import stored_consent
        service = self._service
        with service._policy_lock:
            from gateway.session_group_peer_controls import _require_owner
            _require_owner(service, service.authority, service.runtime)
            if self._observer_grant() != grant:
                raise RuntimeStoreError('permission_denied')
            if service.is_retiring(self._key[0]):
                raise RuntimeStoreError('room_retiring')
            current_route = service.peer_routes.get(self._key)
            if (not self._same_route(current_route, service.peer_clients.get(self._key))
                    or current_route.grant != grant):
                raise RuntimeStoreError('permission_denied')
            current = next((task for task in tasks.list_tasks(service.db_path, room_id=self._key[0])
                            if task['identity'].task_id == dispatch['task_id']), None)
            saved = stored_consent(service.db_path, dispatch)
            if (current is None or current['status'] != 'indeterminate'
                    or current['execution_generation'] != dispatch['execution_generation']
                    or current['cancel_generation'] != cancel_generation
                    or not dispatch.get('document_inputs') or saved is None
                    or saved.get('dispatched') is not True or saved.get('dispatch') != dispatch
                    or current['payload'].get('prompt') != dispatch['prompt']
                    or current['payload'].get('source_event_seq') != dispatch['source_event_seq']):
                raise RuntimeStoreError('unknown_execution')
            from tui_gateway.hosted_room_peer_documents import task_documents
            if task_documents(service.db_path, self._binding, current) != dispatch['document_inputs']:
                raise RuntimeStoreError('unknown_execution')

    def _observe(self, name, call, kwargs):
        from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
        if kwargs.get('grant') in {self._route.grant, self._grant}:
            kwargs = {**kwargs, 'grant': self._observer_grant()}
        try:
            return self._report(call, kwargs)
        except PeerRunsHTTPError as exc:
            if name not in {'history', 'status'} or not exc.needs_reauthorization:
                raise
            replacement = self._observer_grant()
            if replacement == kwargs['grant']:
                raise
            # One read-only retry closes a grant replacement that raced this read.
            return self._report(call, {**kwargs, 'grant': replacement})

    def _observer_grant(self):
        service, binding = self._service, self._binding
        with service._policy_lock:
            room = service._room(binding.room_id)
            current = service.peer_routes.get(self._key)
            if ((room['authority_gateway_id'], room['authority_epoch']) != (binding.gateway_id, binding.authority_epoch)
                    or room['members'] != self._members
                    or not self._same_route(current, service.peer_clients.get(self._key))):
                raise RuntimeError('peer room observer authority or membership changed')
            if current.grant != self._grant and service._peer_route_status.get(self._key) == 'needs_reauthorization':
                raise RuntimeError('peer room observer replacement needs reauthorization')
            self._grant = current.grant
            return current.grant

    def _refresh_if_due(self, name, grant, kwargs):
        """Renew an expiring grant before new work, then publish the renewal."""
        from gateway.hosted_room_peer import HostedMemberDispatch, room_grant_needs_dispatch_refresh
        refresh = getattr(self._client, 'refresh_grant', None)
        if not callable(refresh) or not room_grant_needs_dispatch_refresh(grant):
            return grant
        # Every canonical path uses one frozen issuance body per old grant, so a
        # lost maintenance reply and a foreground dispatch cannot mint siblings.
        extra = {'ttl_seconds': _RENEWED_TTL_SECONDS}
        if name == 'probe':
            digests = (self._route.capability_digest, self._route.execution_policy_digest)
        else:
            checked = HostedMemberDispatch.from_mapping(kwargs['dispatch'])
            digests = (checked.capability_digest, checked.execution_policy_digest)
        pending = []
        with self._service.peer_route_lock:
            if self._stored_link is None:
                raise RuntimeStoreError('peer_target_mismatch')
            issuance, self._issuance_id, extra = cleanup.prepare_issuance(
                self._service.db_path, replace(self._stored_link, grant=grant), extra)
            if not hasattr(self._service, '_peer_cleanup_inflight'):
                self._service._peer_cleanup_inflight = set()
            self._service._peer_cleanup_inflight.add(issuance)
            pending.append(issuance)
        def retain_returned(replacement):
            with self._service.peer_route_lock:
                if self._stored_link is None:
                    raise RuntimeStoreError('peer_target_mismatch')
                key = cleanup.retain(self._service.db_path, replace(self._stored_link, grant=replacement))
                inflight = getattr(self._service, '_peer_cleanup_inflight', None)
                if inflight is None:
                    inflight = self._service._peer_cleanup_inflight = set()
                inflight.add(key)
                pending.append(key)
        extra['on_issued'] = retain_returned
        try:
            try:
                refreshed = refresh(grant=grant, capability_digest=digests[0], execution_policy_digest=digests[1],
                                    **extra)
            except Exception as exc:
                if (getattr(exc, 'needs_reauthorization', False)
                        or room_grant_needs_dispatch_refresh(grant, leeway_seconds=0)):
                    self._status('needs_reauthorization', grant)
                    raise
                return grant  # still valid for now; a later attempt renews it
            return self._publish_renewal(grant, refreshed)
        finally:
            with self._service.peer_route_lock:
                for key in pending:
                    self._service._peer_cleanup_inflight.discard(key)


    def _publish_renewal(self, grant, refreshed):
        # Cleanup and publication share this lock: no worker may retire our provisional
        # grant between retaining it and atomically making it the current route.
        with self._service.peer_route_lock:
            return self._publish_renewal_locked(grant, refreshed)

    def _publish_renewal_locked(self, grant, refreshed):
        """Make a renewal current, or retire it: a renewal is never left both live and unpublished."""
        replacement = str(refreshed.get('grant') or '')
        if not replacement:
            raise RuntimeError('peer returned no refreshed room grant')
        try:
            stored = self._stored_link
            if stored is None:
                raise RuntimeStoreError('peer_target_mismatch')
            # The route may have been deleted during refresh; its captured target still owns cleanup.
            cleanup.retain(self._service.db_path, replace(stored, grant=replacement))
            horizon = self._verify_renewal(grant, replacement)
            if refreshed.get('catalog') is not None:
                from gateway.hosted_room_peer import GatewayRoomCatalog
                from tui_gateway.hosted_room_peer_http import digest_reauthorization_error
                drift = digest_reauthorization_error(
                    GatewayRoomCatalog.from_mapping(refreshed['catalog']),
                    capability_digest=self._route.capability_digest,
                    execution_policy_digest=self._route.execution_policy_digest)
                if drift is not None:
                    self._status('needs_reauthorization', grant)
                    raise drift
            publish_route(self._service, room_id=self._key[0], member_id=self._key[1],
                          route=replace(self._route, grant=replacement), client=self._client,
                          target_url=stored.target_url, catalog=stored.catalog,
                          expected_grant=grant, authorize=self._lease_fence(), issuance_id=self._issuance_id,
                          renewal_horizon=horizon)
        except Exception:
            try:
                _retire(self._client, replacement)
            except Exception:
                pass  # the durable obligation retries after restart as well
            raise
        self._grant = replacement
        return replacement

    def _verify_renewal(self, grant, replacement):
        """A renewal may move only its own lifetime: the same scope, rights and horizon."""
        from gateway.hosted_room_peer import unverified_room_grant_claims
        from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
        old, new = unverified_room_grant_claims(grant), unverified_room_grant_claims(replacement)
        moving = {'grant_id', 'issued_at', 'expires_at'}
        try:
            unchanged = ({k: v for k, v in old.items() if k not in moving}
                         == {k: v for k, v in new.items() if k not in moving})
            ordered = old['issued_at'] <= new['issued_at'] < new['expires_at'] <= new['status_expires_at']
        except (KeyError, TypeError):
            unchanged = ordered = False
        if not (unchanged and ordered):
            self._status('needs_reauthorization', grant)
            raise PeerRunsHTTPError('peer room renewal changed the grant', status_code=403,
                                    error_code='room_capability_catalog_changed', not_admitted=True)
        return old['status_expires_at']

    def _lease_fence(self):
        if self._renewal_lease is None:
            return None
        from gateway import hosted_room_driver as driver
        lease, clock = self._renewal_lease, self._service.runtime.clock
        return lambda conn: driver._require_active_lease(conn, lease, now=clock())


def maintain_peer_grants(service, binding, lease):
    """Renew a room's peer grants before they expire, within one small budget per cycle.

    Called by the room's own driver cycle once Stop and new work are done, and between polls of
    an active turn: never on a timer of its own, never ahead of a pending Stop.
    """
    from tui_gateway.hosted_room_peer_http import room_grant_request_budget
    clock = service.runtime.clock
    now = clock()
    if now < service._peer_renewal_scans.get(binding.room_id, 0.0):
        return
    budget = min(_RENEWAL_BUDGET_SECONDS, lease.expires_at - now - _LEASE_HEADROOM_SECONDS)
    if budget <= 0:
        return
    service._peer_renewal_scans[binding.room_id] = now + _SCAN_SECONDS
    with room_grant_request_budget(budget, clock=clock):
        _renew_room(service, binding, lease, now)


def _renew_room(service, binding, lease, now):
    from gateway import hosted_room_driver as driver
    from gateway.hosted_room_peer import room_grant_needs_dispatch_refresh
    from tui_gateway.hosted_room_peer_http import room_grant_request_budget_remaining
    stored = [link for link in links.load_room_links_tolerant(service.db_path)[0]
              if link.room_id == binding.room_id and link.status == 'ready']
    current = {(link.room_id, link.member_id) for link in stored}
    for key in [k for k in service._peer_renewals if k[0] == binding.room_id and k not in current]:
        service._peer_renewals.pop(key, None)
    for link in stored:
        remaining = room_grant_request_budget_remaining()
        if remaining is not None and remaining <= 0:
            break  # routes not reached keep their schedule for the next scan
        key = (link.room_id, link.member_id)
        fingerprint = hashlib.sha256(link.grant.encode()).hexdigest()
        seen, next_at, delay = service._peer_renewals.get(key, (fingerprint, 0.0, _RETRY_SECONDS))
        if seen != fingerprint:
            delay = _RETRY_SECONDS  # a new grant restarts the backoff, not the schedule
        if now < next_at:
            continue
        service._peer_renewals[key] = (fingerprint, now + _ATTEMPT_SECONDS, _RETRY_SECONDS)
        if not room_grant_needs_dispatch_refresh(link.grant, now=now):
            continue
        route, client = service.peer_routes.get(key), service.peer_clients.get(key)
        if route is None or client is None or route.grant != link.grant:
            continue
        try:
            CanonicalPeerClient(service, binding, key, route, client, renewal_lease=lease).probe(grant=route.grant)
        except (driver.StaleLeaseError, driver.RoomUnavailableError):
            raise
        except Exception:
            service._peer_renewals[key] = (fingerprint, now + delay, min(_MAX_RETRY_SECONDS, delay * 2))
            logger.warning('Peer grant renewal pending: room=%s member=%s', *key)
    due = [service._peer_renewals.get((link.room_id, link.member_id), ('', now + _SCAN_SECONDS, 0.0))[1]
           for link in stored]
    service._peer_renewal_scans[binding.room_id] = max(now + _SCAN_SECONDS, min(due, default=now + _ATTEMPT_SECONDS))
