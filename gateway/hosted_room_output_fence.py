"""Final Files write/publication checks inside the owning SQLite transaction."""

import json

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactScope


def require_output_task(conn, scope: RoomArtifactScope, cancel_generation, *, status="settled", cleanup=False):
    if type(cancel_generation) is not int or cancel_generation < 0:
        raise RoomArtifactError("Group Chat output cancellation generation is invalid")
    room = conn.execute(
        "SELECT authority_gateway_id, authority_epoch, disbanded_at, members_json FROM hosted_rooms WHERE room_id=?",
        (scope.room_id,),
    ).fetchone()
    if (room is None or room["disbanded_at"] is not None
            or room["authority_gateway_id"] != scope.authority_gateway_id
            or room["authority_epoch"] != scope.authority_epoch):
        raise RoomArtifactError("Group Chat output authority changed")
    def matches(member):
        if member['member_id'] != scope.member_id or member['profile'] != scope.target_profile:
            return False
        target = member.get('target', {})
        if scope.target_install_id == scope.home_install_id:
            return target.get('kind', 'local') == 'local'
        return (target.get('kind') == 'peer' and target.get('installation_id') == scope.target_install_id
                and target.get('profile') == scope.target_profile)
    if not any(matches(member) for member in json.loads(room['members_json'])):
        raise RoomArtifactError("Group Chat output participant changed")
    if cleanup:
        # Closing withdraws viewers/NEW, not the original terminal custody.
        from gateway.hosted_rooms import room_safety
        room_safety._raise_if_quarantined(conn, scope.room_id)
        fence = conn.execute('SELECT authority_gateway_id,authority_epoch FROM hosted_room_disband_fences WHERE room_id=?',
                             (scope.room_id,)).fetchone()
        if fence is not None and tuple(fence) != (scope.authority_gateway_id, scope.authority_epoch):
            raise RoomArtifactError("Group Chat cleanup fence changed")
    else:
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore
        HostedRoomAttachmentStore._require_viewer_room(conn, room_id=scope.room_id,
            authority_gateway_id=scope.authority_gateway_id, authority_epoch=scope.authority_epoch)
    task = conn.execute("SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?",
                        (scope.room_id, scope.task_id)).fetchone()
    payload = json.loads(task["payload_json"]) if task else {}
    if (task is None or task["status"] != status or task["execution_generation"] != scope.execution_generation
            or task["cancel_generation"] != cancel_generation
            or payload.get("target_member_id", payload.get("target_profile")) != scope.member_id
            or payload.get("target_profile") != scope.target_profile):
        raise RoomArtifactError("Group Chat output attempt changed")
    return task


def require_peer_output_receipt(conn, scope, result):
    from gateway.hosted_rooms import _REMOTE_RUN_IDENTITY_COLUMNS, _SELECT_REMOTE_RUN
    run_id = result.get('peer_run_id')
    if not isinstance(run_id, str) or not run_id:
        raise RoomArtifactError('Group Chat output Run is unavailable')
    receipt = conn.execute(_SELECT_REMOTE_RUN,
        tuple(getattr(scope, key) for key in _REMOTE_RUN_IDENTITY_COLUMNS)).fetchone()
    if receipt is None or receipt['run_id'] != run_id or result.get('message_id') != 'peer-run:' + run_id:
        raise RoomArtifactError('Group Chat output Run changed')
    return dict(receipt)


def require_output_publication(conn, room_id, expected, *, kind, actor, payload):
    scope = RoomArtifactScope.from_mapping(expected["scope"])
    if scope.room_id != room_id:
        raise RoomArtifactError("Group Chat output room changed")
    task = require_output_task(conn, scope, expected["cancel_generation"], cleanup=kind != "message.member")
    if (payload.get("task_id") != scope.task_id or payload.get("member_id") != scope.member_id
            or payload.get("thread_id") != task["thread_id"] or payload.get("turn_id") != task["turn_id"]):
        raise RoomArtifactError("Group Chat output event coordinates changed")
    result = json.loads(task["result_json"])
    if scope.target_install_id != scope.home_install_id:
        from gateway.session_peer_output_custody import PeerOutputCustody
        custody = expected.get("peer_custody")
        if type(custody) is not PeerOutputCustody:
            raise RoomArtifactError("Group Chat output custody is unavailable")
        custody.check_current(conn, scope)
        require_peer_output_receipt(conn, scope, result)
    if result.get("artifact_scope") != scope.as_mapping() or result.get("artifacts") != expected["manifest"]:
        raise RoomArtifactError("Group Chat output receipt changed")
    if kind == "message.member":
        frozen = json.loads(task["payload_json"]).get("recipient_member_ids")
        if not frozen or payload.get("recipient_member_ids") != frozen:
            raise RoomArtifactError("Group Chat output recipients changed")
        if actor.get("id") != scope.member_id or actor.get("profile") != scope.target_profile:
            raise RoomArtifactError("Group Chat output event author changed")
        from tui_gateway.hosted_room_artifact_service import _upload_id
        items = expected["manifest"]["items"]
        attachments = payload.get("attachments", [])
        if len(attachments) != len(items):
            raise RoomArtifactError("Group Chat output event manifest changed")
        for item, attachment in zip(items, attachments):
            row = conn.execute("SELECT upload_id FROM hosted_room_attachments WHERE room_id=? AND attachment_id=?",
                               (room_id, attachment["attachment_id"])).fetchone()
            if row is None or row[0] != _upload_id(scope, item):
                raise RoomArtifactError("Group Chat output event source changed")
