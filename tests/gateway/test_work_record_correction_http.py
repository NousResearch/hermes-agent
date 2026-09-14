"""Invalid v2 pending evidence is rejected before real HTTP transmission."""
import asyncio
import sqlite3

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms as rooms
from gateway import hosted_room_peer as peer
from gateway import hosted_room_work_records as work
from tests.gateway.test_api_server_room_replicas import setup, TARGET  # noqa: F401
from tests.gateway.test_api_server_room_work_records import publisher
from tests.gateway.test_work_record_v2_admission_closure import enrolled_work
from tests.tui_gateway.test_replication_lineage_http import successor_publisher


@pytest.mark.asyncio
@pytest.mark.parametrize('revoked', [False, True], ids=['real-ack', 'real-revocation'])
async def test_v2_invalid_pending_cannot_bypass_delivery_validation(setup, monkeypatch, revoked):
    source, target, app = setup
    responses = []

    @web.middleware
    async def observe(request, handler):
        response = await handler(request)
        if request.path.endswith('/work-records'):
            responses.append(response.status)
        return response

    app.middlewares.append(observe)
    async with TestClient(TestServer(app)) as http:
        _, grant, current = await enrolled_work(http, source)
        assert current['version'] == 2
        catalog = peer.catalog_mapping(installation_id=TARGET, target_profile='default', persistent_process=True)
        publisher(source, http, {'grant': grant, 'catalog': catalog}, monkeypatch)
        pub = successor_publisher(source, monkeypatch)
        await asyncio.to_thread(pub._publish_one, ('room', 'reviewer'))
        assert responses == []
        with sqlite3.connect(source) as raw:
            raw.execute(f"UPDATE {work.PENDING_TABLE} SET digest='wrong' WHERE producer_epoch=2")
        if revoked:
            claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), grant, permission=work.PERMISSION)
            rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims['status_expires_at'])
        for _ in range(3):
            await asyncio.to_thread(pub._publish_one, ('room', 'reviewer'))
        state = pub.status('room')
        print('V2 HTTP RESPONSES', responses, 'ROUTE', state['routes'][0]['work_record_status'],
              'PENDING', state['work_records'][0]['status'], 'DISPOSITION', state['work_records'][0]['disposition'])
        assert responses == [], 'invalid stored v2 evidence must not cross the HTTP boundary'
