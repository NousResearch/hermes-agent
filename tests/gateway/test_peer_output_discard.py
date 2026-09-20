"""Canonical Output retirement; real HTTP/signing/Run/producer, inert agent only."""
import asyncio
import json
from pathlib import Path

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_fences import read, ack, unretired
from gateway.run import _profile_runtime_scope
from gateway.hosted_room_artifacts import RoomArtifactError
from gateway.session_results import admission_result
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def custody(c):
    with _profile_runtime_scope(c.home, hydrate_secrets=False):
        return c.service._output_source(c.service._room('room-one'), c.stored)


async def discard(c, source=None):
    scope, _, source = source or custody(c)
    with _profile_runtime_scope(c.home, hydrate_secrets=False):
        return await asyncio.to_thread(source.discard_durably, scope)


async def wire_discard(c, *, body=None, grant=None, run_id=None):
    return await asyncio.to_thread(c.client._request,
        '/v1/runs/' + (run_id or c.accepted['run_id']) + '/artifacts/discard', method='POST',
        body=body or dict(reason='verification_failed', result_digest=c.stored['result']['peer_result_digest']),
        room_grant=grant or c.issued['grant'], reject_redirects=True)


@pytest.mark.asyncio
async def test_home_discard_retires_exact_target_and_replays_durable_count(files_target, monkeypatch):
    from dataclasses import replace
    async with peer_case(files_target, monkeypatch) as c:
        scope, manifest, source = custody(c)
        assert await read(c) == c.output.read_bytes()
        outbox = c.target.adapter._peer_output_outbox
        other = replace(scope, member_id='other-writer')
        other_item = outbox.put_bytes(scope=other, source_name='other.txt', data=b'other member survives')
        assert await discard(c, (scope, manifest, source)) == 1
        assert outbox.retirement_complete(scope)
        assert outbox.read(other, other_item['artifact_id'])[1] == b'other member survives'
        future = replace(scope, execution_generation=scope.execution_generation + 1)
        future_item = outbox.put_bytes(scope=future, source_name='future.txt', data=b'next generation survives')
        assert await discard(c) == 1  # Same exact durable receipt, not absence-as-zero.
        assert outbox.read(future, future_item['artifact_id'])[1] == b'next generation survives'
        replies = [json.loads(x[3]) for x in c.wire.replies if x[1].endswith('/artifacts/discard')]
        assert replies == [{'discarded': True, 'removed': 1}] * 2
        saved = admission_result(c.target.db, c.row['admission_id'])
        assert saved['peer_output_discard']['receipt'] == replies[0]
        with pytest.raises(PeerRunsHTTPError):
            await read(c)
        with pytest.raises(PeerRunsHTTPError):
            await ack(c)
        with pytest.raises(RoomArtifactError):
            outbox.put_bytes(scope=scope, source_name='late.txt', data=b'cannot reopen retired producer')
        assert [Path(x['path']).read_bytes() for x in c.row['payload']['api_turn_v1']['settings']['room_input_media']['media']] == c.raw
        assert len(c.launched) == len(c.executions) == 1


@pytest.mark.asyncio
async def test_custody_rejects_malformed_retirement_receipt(files_target, monkeypatch):
    from gateway import session_peer_output_custody as module
    receipts = [None, {}, {'discarded': 1, 'removed': 1}, {'discarded': True, 'removed': True},
        {'discarded': True, 'removed': -1}, {'discarded': True, 'removed': 999}, {'discarded': True, 'removed': '1'},
        {'discarded': False, 'removed': 0}, {'discarded': True, 'removed': 0},
        {'discarded': True, 'removed': 1, 'extra': 0}]
    async with peer_case(files_target, monkeypatch) as c:
        for receipt in receipts:
            monkeypatch.setattr(module, 'discard_artifacts', lambda *a, **kw: receipt)
            with pytest.raises((RoomArtifactError, PeerRunsHTTPError)):
                await discard(c)
            unretired(c)
