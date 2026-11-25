"""Recovery registry and watermark must describe the same authenticated stream read."""
import pytest
from gateway.session_controls import AuthorityConnection
from tests.gateway.fixtures.canonical_batch_owner import Peer, make_owner, rpc


@pytest.mark.asyncio
async def test_resume_batch_snapshot_uses_same_stream_watermark_after_attach_yields(tmp_path, monkeypatch):
    owner = await make_owner(tmp_path, monkeypatch, lambda e: None)
    connection = AuthorityConnection(owner, Peer(), {'user_id': 'owner'})
    original = owner.attach

    async def delayed(*args, **kwargs):
        snapshot = await original(*args, **kwargs)
        # Resume awaits info projection after attach; registry is read later.
        stream = owner.sessions['s'].event_stream
        stream.publish('s', {}, event_type='review.after_attach')
        return snapshot

    monkeypatch.setattr(owner, 'attach', delayed)
    try:
        await rpc(connection, 'client.capabilities', server_requests=True)
        await rpc(connection, 'client.capabilities', canonical_clarify_projection_v1=True)
        result = (await rpc(connection, 'session.resume', session_id='s'))['result']
        stream = owner.sessions['s'].event_stream
        assert result['clarify_batches'] == []
        assert result['replay_epoch'] == stream.epoch
        assert result['clarify_batch_watermark'] == {
            'replay_epoch': stream.epoch, 'last_sequence': stream.sequence
        }
        # Do not advance the independent transcript/attach cursor.
        assert result['last_sequence'] == 0
    finally:
        await connection.close()
