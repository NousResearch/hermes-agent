"""Unexpected v3 link owners defer collection without changing legacy pairs."""
import os

import pytest

from gateway.hosted_room_input_preparation import prepare_hosted_input
from gateway.hosted_room_input_reclamation import collect_working_copies
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, expire, v3_path


@pytest.mark.asyncio
async def test_untracked_link_defers_v3_until_its_owner_releases_it(tmp_path, monkeypatch):
    db, authority = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, authority)
        prepared = prepare_hosted_input(rpc, request_id='hosted:links', prompt='read',
            attachments=[item for item, _ in bound])
        working = v3_path(prepared)
        another = tmp_path / 'retained-by-another-owner.txt'
        os.link(working, another)
        expire(db, prepared.handle)
        assert collect_working_copies(db, epoch=authority.epoch)['removed'] == 0
        assert working.read_bytes() == another.read_bytes() == bound[0][1]
        another.unlink()
        assert collect_working_copies(db, epoch=authority.epoch)['removed'] == 1
        assert not working.exists()
    finally:
        close(db, tmp_path)
