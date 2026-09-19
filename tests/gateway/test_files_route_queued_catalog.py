"""The source's real queued transport must keep its signed Files commitment."""
import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_fences import read, ack


@pytest.mark.asyncio
async def test_real_queued_source_probe_preserves_invited_files_rights(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch, resolve_queued=True) as c:
        assert await read(c) == c.output.read_bytes()
        assert (await ack(c))['acknowledged'] is True
        assert len(c.launched) == len(c.executions) == 1
