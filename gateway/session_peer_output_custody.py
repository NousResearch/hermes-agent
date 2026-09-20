"""Verified Home custody for a retained root peer Run (a9577feb adaptation)."""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import urllib.parse

from gateway.hosted_room_artifacts import RoomArtifactError
from gateway.hosted_room_output_fence import require_output_task, require_peer_output_receipt
from tui_gateway.hosted_room_peer_artifacts import (
    read_artifact, acknowledge_artifacts, discard_artifacts, require_discard_receipt)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, peer_result_digest
from hermes_state_runtime import _epoch


@dataclass
class PeerOutputCustody:
    service: object
    scope: object
    manifest: dict
    result: dict
    cancel_generation: int

    def __post_init__(self):
        self.manifest, self.result = copy.deepcopy(self.manifest), copy.deepcopy(self.result)
        self.authority = self.service.authority
        self.epoch, self.instance = self.authority.epoch, self.authority.instance_id
        self.owner = self.service._owner(self.scope.room_id)
        self.registry = self.authority.runner.session_authorities
        with self.authority.db._read_ctx() as conn:
            self.check_current(conn, self.scope)
            task = require_output_task(conn, self.scope, self.cancel_generation, cleanup=True)
            self.recipients = json.loads(task['payload_json']).get('recipient_member_ids')
            self.link = dict(conn.execute("SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?",
                                         (self.scope.room_id, self.scope.member_id)).fetchone())
        if not self.recipients:
            raise RoomArtifactError('Group Chat output recipients are unavailable')

    def check_current(self, conn, scope):
        from gateway.session_authorities import authority_for_home, served_profile_name
        from gateway.runtime_ownership import process_ownership
        from gateway.session_hosted_service import _OWNER
        from gateway import hosted_rooms
        authority = self.authority
        home = Path(authority.profile_id)
        if (scope != self.scope or scope.target_profile != 'default' or served_profile_name(home) != 'default'
                or self.service.authority is not authority or authority.hosted_room_service is not self.service
                or authority.runner.session_authority is not authority
                or authority.runner.session_authorities is not self.registry
                or authority_for_home(authority.runner, home) is not authority
                or (authority.epoch, authority.instance_id) != (self.epoch, self.instance)
                or authority.db._db_file_was_replaced()
                or not process_ownership.owns(home) or scope.home_install_id != hosted_rooms.local_authority_gateway_id()):
            raise RoomArtifactError('Group Chat output owner changed')
        authority._require_admission_open()
        _epoch(conn, self.epoch)
        owner = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + scope.room_id,)).fetchone()
        task = require_output_task(conn, scope, self.cancel_generation, cleanup=True)
        if (owner is None or owner[0] != self.owner or json.loads(task['result_json']) != self.result
                or self.result.get('artifact_scope') != scope.as_mapping()
                or self.result.get('artifacts') != self.manifest
                or (hasattr(self, 'recipients') and json.loads(task['payload_json']).get('recipient_member_ids') != self.recipients)):
            raise RoomArtifactError('Group Chat output receipt changed')
        link = conn.execute("SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?",
                            (scope.room_id, scope.member_id)).fetchone()
        if link is None or (hasattr(self, "link") and dict(link) != self.link):
            raise RoomArtifactError("Group Chat output link changed")
        return require_peer_output_receipt(conn, scope, self.result)

    def _route(self, scope):
        from gateway import hosted_room_links
        with self.authority.db._read_ctx() as conn:
            receipt = self.check_current(conn, scope)
        link = hosted_room_links.load_room_link(self.service.db_path, room_id=scope.room_id, member_id=scope.member_id)
        if (link is None or link.status != 'ready' or link.target_profile != scope.target_profile
                or link.catalog.installation_id != scope.target_install_id):
            raise RoomArtifactError('Group Chat output peer route changed')
        route = self.service.peer_routes.get((scope.room_id, scope.member_id))
        if (route is None or route.grant != link.grant or route.home_install_id != scope.home_install_id
                or route.target_install_id != scope.target_install_id
                or route.capability_digest != link.catalog.catalog_digest
                or route.execution_policy_digest != link.catalog.execution_policy.policy_digest):
            raise RoomArtifactError('Group Chat output peer route changed')
        client = PeerRunsHTTPClient(base_url=link.target_url, api_key='', target_profile=link.target_profile)
        client.prepare(room_id=scope.room_id, profile=scope.target_profile, source='bot_room',
                       grant=link.grant, create=False, expected_session_id=receipt['session_id'])
        return client, link, receipt

    def _verify_remote(self, client, link, receipt):
        status = client._request('/v1/runs/' + urllib.parse.quote(receipt['run_id'], safe=''),
                                 room_grant=link.grant, reject_redirects=True)
        if (status.get('status') != 'completed' or status.get('run_id') != receipt['run_id']
                or status.get('admission_id') != self.result.get('peer_admission_id')
                or status.get('execution_generation') != self.result.get('peer_execution_generation')
                or status.get('artifacts') != self.manifest or status.get('room_artifact_scope') != self.scope.as_mapping()
                or peer_result_digest(status) != self.result.get('peer_result_digest')):
            raise RoomArtifactError('Group Chat output remote result changed')

    def read(self, scope, artifact_id):
        item = next((x for x in self.manifest['items'] if x['artifact_id'] == artifact_id), None)
        if item is None:
            raise RoomArtifactError('Group Chat output file is unavailable')
        client, link, receipt = self._route(scope)
        self._verify_remote(client, link, receipt)
        data = read_artifact(client, run_id=receipt['run_id'], artifact_id=artifact_id, grant=link.grant)
        current_client, current_link, current_receipt = self._route(scope)
        if current_link != link or current_receipt != receipt:
            raise RoomArtifactError('Group Chat output route changed during read')
        self._verify_remote(current_client, current_link, current_receipt)
        return item, data

    def retirement_complete(self, scope):
        return False  # Only the authenticated target can prove exact retirement.

    def acknowledge(self, scope, artifact_ids, *, message_event_id):
        client, link, receipt = self._route(scope)
        self._verify_remote(client, link, receipt)
        if (artifact_ids != [x['artifact_id'] for x in self.manifest['items']]
                or message_event_id != 'dmessage:' + scope.task_id.removeprefix('dtask:')):
            raise RoomArtifactError('Group Chat output acknowledgement changed')
        from tui_gateway.hosted_room_artifact_service import acknowledge_published
        def transmit(checked, ids, *, message_event_id):
            _, current, current_receipt = self._route(checked)
            if current != link or current_receipt != receipt:
                raise RoomArtifactError('Group Chat output route changed before ACK')
            result = acknowledge_artifacts(client, run_id=receipt['run_id'], artifact_ids=ids,
                manifest_digest=self.manifest['manifest_digest'], message_event_id=message_event_id, grant=link.grant)
            if result.get('acknowledged') is not True:
                raise RoomArtifactError('Group Chat output ACK was not confirmed')
            return result
        # Re-prove Home's journal and bytes AFTER remote observation I/O.
        from gateway.session_hosted_output_retirement import RetainedPublication
        return acknowledge_published(RetainedPublication(self.service, scope), scope=scope, manifest=self.manifest, acknowledge=transmit)

    def discard_durably(self, scope):
        client, link, receipt = self._route(scope)
        self._verify_remote(client, link, receipt)
        _, current, current_receipt = self._route(scope)
        if current != link or current_receipt != receipt:
            raise RoomArtifactError('Group Chat output route changed before discard')
        result = require_discard_receipt(discard_artifacts(client, run_id=receipt['run_id'],
            result_digest=self.result['peer_result_digest'], grant=link.grant))
        if result['removed'] != len(self.manifest['items']):
            raise RoomArtifactError('Group Chat output retirement count changed')
        # A lost reply is uncertain, never confirmation. Every replay visits
        # the target and every successful reply rechecks current Home custody.
        current_client, current, current_receipt = self._route(scope)
        if current != link or current_receipt != receipt:
            raise RoomArtifactError('Group Chat output route changed during discard')
        self._verify_remote(current_client, current, current_receipt)
        self._route(scope)
        return result['removed']
