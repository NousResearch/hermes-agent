"""Operation-local canonical renewal: authenticated catalog and exact SQL preimage."""
import hashlib
from dataclasses import replace

from gateway import hosted_room_links as links
from gateway.hosted_room_peer import GatewayRoomCatalog, PROTOCOL_VERSION, _catalog_digest
from gateway.session_group_setup import _grant_claims, _route_record
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


class CanonicalPeerRenewal:
    """No readiness cache: created only for one automatic refresh attempt."""

    def __init__(self, service, room_id, member_id, route, client, grant):
        self.service, self.room_id, self.member_id = service, room_id, member_id
        self.route, self.client = replace(route, grant=grant), client
        self.authority = service.authority
        self.epoch = self.authority.epoch
        self.verified = None
        with self.authority.db.live_read_connection() as conn:
            if conn is None:
                raise RuntimeStoreError('peer_setup_conflict')
            self.binding = self._binding(conn)
            record = _route_record(conn, room_id, member_id)
        self.before = links.StoredRoomLink.from_record(record) if record else None
        if (self.before is None or self.before.grant != grant
                or self.before.target_url != client.base_url
                or self.before.target_profile != route.target_profile
                or self.before.catalog.catalog_digest != route.capability_digest
                or self.before.catalog.attachments != route.attachments
                or self.before.status != 'ready'):
            raise RuntimeStoreError('peer_setup_conflict')

    def _binding(self, conn):
        from gateway.session_authorities import authority_for_home
        authority, service = self.authority, self.service
        if (service.authority is not authority or authority.hosted_room_service is not service
                or authority_for_home(authority.runner, authority.profile_id) is not authority):
            raise RuntimeStoreError('peer_setup_conflict')
        authority._require_admission_open()
        _epoch(conn, self.epoch)
        row = conn.execute('SELECT authority_gateway_id,authority_epoch,members_json,disbanded_at '
                           'FROM hosted_rooms WHERE room_id=?', (self.room_id,)).fetchone()
        owner = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             ('gateway.hosted.owner.v1:' + self.room_id,)).fetchone()
        if row is None or owner is None or row['disbanded_at'] is not None:
            raise RuntimeStoreError('peer_setup_conflict')
        return tuple(row), owner[0]

    def verify(self, grant, replacement, probe):
        """Called only after the target authenticates the exact replacement probe."""
        catalog = self.before.catalog
        try:
            old = _grant_claims(grant, attachments=catalog.attachments)
            new = _grant_claims(replacement, attachments=catalog.attachments)
        except RuntimeStoreError as exc:
            raise PeerRunsHTTPError('peer room grant needs reauthorization', status_code=403,
                                    error_code='room_capability_catalog_changed', not_admitted=True) from exc
        scope = dict(room_id=self.room_id, member_id=self.member_id,
                     home_install_id=self.route.home_install_id,
                     authority_gateway_id=self.binding[0][0], authority_epoch=self.binding[0][1],
                     target_profile=self.route.target_profile)
        expected = dict(scope, version=PROTOCOL_VERSION, target_install_id=catalog.installation_id,
                        execution_policy_digest=catalog.execution_policy.policy_digest)
        mutable = {'grant_id', 'issued_at', 'expires_at'}
        current = GatewayRoomCatalog.from_mapping(probe.get('catalog')).as_mapping()
        if (grant != self.before.grant or replacement == grant
                or any(old.get(k) != v or new.get(k) != v for k, v in expected.items())
                or {k: v for k, v in old.items() if k not in mutable}
                   != {k: v for k, v in new.items() if k not in mutable}
                or not old['issued_at'] <= new['issued_at'] < new['expires_at'] <= old.get('status_expires_at', old['expires_at'])
                or type(probe.get('authority_epoch')) is not int
                or any(probe.get(k) != v for k, v in scope.items())
                or _catalog_digest(dict(current, attachments=catalog.attachments)) != catalog.catalog_digest):
            raise PeerRunsHTTPError('peer room renewal needs reauthorization', status_code=403,
                                    error_code='room_capability_catalog_changed', not_admitted=True)
        self.verified = replacement
        # Preserve the invitation commitment, not the target's passive readiness bit.
        return catalog

    def publish(self, replacement, catalog):
        if replacement != self.verified or catalog != self.before.catalog:
            raise RuntimeStoreError('peer_setup_conflict')
        written = []

        def guard(conn, record):
            if (self._binding(conn) != self.binding
                    or _route_record(conn, self.room_id, self.member_id) != self.before.as_record()):
                raise RuntimeStoreError('peer_setup_conflict')
            if record is not None:
                written.append(links.StoredRoomLink.from_record(record))
            return self.before

        self.service.register_peer_route(room_id=self.room_id, member_id=self.member_id,
            route=replace(self.route, grant=replacement), client=self.client,
            target_url=self.before.target_url, catalog=catalog,
            expected_grant_sha256=hashlib.sha256(self.before.grant.encode()).hexdigest(), setup_guard=guard)
        # The exact record passed to the successful SQL writer, not a later read
        # that could accidentally adopt a different owner's update.
        if len(written) != 1:
            raise RuntimeStoreError('peer_setup_conflict')
        return self.before, written[0]
