"""Files remain local even when the group chat seats a bot on another gateway."""
import asyncio
import time

import pytest

from gateway import hosted_room_driver as tasks
from gateway.hosted_room_attachment_catalog import list_published
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from tests.gateway.fixtures.hosted_room_owner import MEMBERS, events, owner, run_turn
from tests.gateway.test_hosted_room_shared_files import share, sharing, write_file
from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport, PeerMemberRoute


@pytest.mark.asyncio
async def test_local_bot_file_is_downloadable_here_and_visibly_refused_by_peer(tmp_path, monkeypatch):
    peer = {**MEMBERS[1], "target": {"kind": "peer", "peer_id": "elsewhere", "installation_id": "other",
                                   "profile": "reviewer", "capability_digest": "a" * 64}}
    async with owner(tmp_path, monkeypatch, members=(MEMBERS[0], peer)) as (authority, service, runner):
        runner._handle_message = sharing([write_file(tmp_path)], "@reviewer Review the report.")
        turn = await run_turn(authority, service)
        service.prepare_room(turn.binding)
        task, = tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert "cannot share files with this group chat" in task["payload"]["prompt"]
        assert "call share_group_file" not in task["payload"]["prompt"]
        route = PeerMemberRoute(home_install_id="home", member_id="reviewer", target_install_id="other",
                                target_profile="reviewer", capability_digest="a" * 64,
                                execution_policy_digest="b" * 64, cancellation_scope_id="cancel",
                                trace_id="trace", grant="grant")
        # A real peer transport, with no network client: no dispatch may reach another machine.
        transport = PeerHostedRoomTransport(binding=turn.binding, route=route, client=None,
                                            source_event_seq=task["payload"]["source_event_seq"])
        service.runtime.transport_resolver = lambda *_: transport
        lease = service.runtime._ensure_lease(turn.binding)
        attempt = tasks.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0,
                                   clock=time.time)
        await asyncio.to_thread(service.runtime._execute_attempt, turn.binding, task, attempt)
        service.prepare_room(turn.binding)
        failure, = events(service, "turn.failed")
        assert failure["payload"]["error"].startswith("attachments_unsupported_on_route")
        gateway, epoch = service._owned_authority("room")
        item, = list_published(service.db_path, room_id="room", authority_gateway_id=gateway,
                               authority_epoch=epoch)["items"]
        assert item.get("available", True)
        saved = HostedRoomAttachmentStore(service.db_path).read_viewer(
            room_id="room", event_id=item["event_id"], attachment_id=item["attachment_id"],
            authority_gateway_id=gateway, authority_epoch=epoch)
        assert saved.data == b"explicit result bytes\n"
        # A peer API turn has no local hosted-turn binding; a share cannot pretend to succeed.
        assert "gateway that owns this group chat" in share(write_file(tmp_path))["error"]
