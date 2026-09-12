"""Root peer output: real owned stores and registered routes, no services."""
import pytest
from tests.gateway.test_canonical_peer_target_setup import target, invite  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401


def register_routes(target):
    from aiohttp import web
    target.adapter._app = web.Application()
    for method, path, handler in target.adapter._http_route_table():
        target.adapter._app.router.add_route(method, path, handler)
        target.adapter._app.router.add_route(method, "/p/{profile}" + path, handler)


@pytest.mark.asyncio
async def test_readiness_is_read_only_and_native_consent_requires_initialized_output(files_target):
    from gateway.session_peer_output import peer_output_permissions, initialize_peer_output
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    target = files_target
    register_routes(target)
    _, catalog = _local_room_catalog(target.adapter, 'default', hosted_rooms.local_authority_gateway_id())
    schema = list(target.db._conn.execute('SELECT name,sql FROM sqlite_master'))
    changes = target.db._conn.total_changes
    assert peer_output_permissions(target.adapter, profile='default', catalog=catalog, connection=target.db._conn) == ()
    assert list(target.db._conn.execute('SELECT name,sql FROM sqlite_master')) == schema
    assert target.db._conn.total_changes == changes
    assert not (target.home / 'hosted-room-artifact-outbox').exists()
    initialize_peer_output(target.adapter)
    issued = await invite(target)
    claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='artifact.ack')
    assert set(claims['permissions']) == {'approve', 'dispatch', 'status', 'stop', 'attachment.stage', 'artifact.read', 'artifact.ack'}
    assert await invite(target) == issued
    assert peer_output_permissions(target.adapter, profile='default', catalog=catalog, connection=target.db._conn) == ('artifact.read', 'artifact.ack')


@pytest.mark.asyncio
async def test_real_peer_producer_to_home_files_exact_ack_and_replay(files_target, monkeypatch):
    import asyncio
    import base64
    import json
    from pathlib import Path
    from tests.gateway.peer_output_fixtures import peer_case
    from gateway.run import _profile_runtime_scope
    from gateway.hosted_room_artifacts import RoomArtifactScope
    from gateway.session_group_files import dispatch_group_files
    from gateway.session_contract import Principal
    from gateway.session_results import admission_result
    from gateway.platforms.api_server_authority_runs import run_projection
    async with peer_case(files_target, monkeypatch) as c:
        with _profile_runtime_scope(c.target.home, hydrate_secrets=False):
            projected = run_projection(c.target.adapter, c.accepted['run_id'])
            assert projected['status'] == 'completed' and projected['artifacts'] == c.stored['result']['artifacts']
        scope = RoomArtifactScope.from_mapping(c.stored['result']['artifact_scope'])
        refs = c.row['payload']['api_turn_v1']['settings']['room_input_media']['media']
        assert [Path(x['path']).read_bytes() for x in refs] == c.raw
        assert c.issued['grant'] not in json.dumps(c.row)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            assert await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            events = [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
            assert len(events) == 1 and events[0]['payload']['recipient_member_ids'] == ['writer', 'reader']
            actor = Principal('alice', str(c.home), frozenset({'session:read'}), 'viewer')
            page = dispatch_group_files(c.service, actor, 'groups.attachment.list', {'room_id': 'room-one'})
            item = next(x for x in page['items'] if x['event_id'] == events[0]['event_id'])
            download = dispatch_group_files(c.service, actor, 'groups.attachment.download',
                dict(room_id='room-one', event_id=item['event_id'], attachment_id=item['attachment_id']))
            assert base64.b64decode(download['data_base64']) == c.output.read_bytes()
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
        assert c.target.adapter._peer_output_outbox.retirement_complete(scope)
        saved = admission_result(c.target.db, c.row['admission_id'])
        assert saved['peer_output_ack']['manifest_digest'] == projected['artifacts']['manifest_digest']
        assert [Path(x['path']).read_bytes() for x in refs] == c.raw
        assert len(c.executions) == len(c.outputs) == 1
        assert len(c.launched) == 1
        assert c.target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1
