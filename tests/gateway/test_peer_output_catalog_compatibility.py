"""Accepted Output tolerates only Files readiness, never stable commitments."""
import pytest

from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_fences import read, ack, unretired
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['endpoint', 'payload-digest'])
async def test_accepted_output_refuses_non_readiness_drift(files_target, monkeypatch, change):
    from gateway.hosted_room_peer import local_room_link_endpoint
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    from gateway import hosted_rooms
    from gateway.run import _profile_runtime_scope
    async with peer_case(files_target, monkeypatch) as c:
        assert await read(c) == c.output.read_bytes()
        with _profile_runtime_scope(c.target.home, hydrate_secrets=False):
            passive = _local_room_catalog(c.target.adapter, 'default', hosted_rooms.local_authority_gateway_id())[1]
            assert passive['attachments'] is False
        if change == 'endpoint':
            monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'https://changed-output.example.test')
            with _profile_runtime_scope(c.target.home, hydrate_secrets=False):
                assert local_room_link_endpoint()['url'] == 'https://changed-output.example.test'
        else:
            c.target.db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET payload_digest=? WHERE admission_id=?", ('0' * 64, c.row['admission_id'])))
        with pytest.raises(PeerRunsHTTPError):
            await read(c)
        with pytest.raises(PeerRunsHTTPError):
            await ack(c)
        unretired(c)
