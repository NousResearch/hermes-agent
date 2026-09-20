"""Recheck current authority after I/O and at actual retirement writes."""


import sqlite3
from pathlib import Path

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_discard import discard, wire_discard
from tests.gateway.test_peer_output_fences import unretired

from gateway.session_results import admission_result
from gateway.hosted_room_artifacts import RoomArtifactOutbox
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['body', 'write'])
@pytest.mark.parametrize('change', ['owner', 'epoch', 'revoked'])
async def test_discard_body_await_and_commit_fences(files_target, monkeypatch, boundary, change):
    from gateway import hosted_room_output_discard as primitive
    async with peer_case(files_target, monkeypatch) as c:
        def mutate(conn=None):
            if change == 'owner':
                c.target.runner.session_authority = None
            else:
                sql = ('UPDATE runtime_epoch SET epoch=epoch+1' if change == 'epoch'
                       else 'UPDATE hosted_room_peer_reservations SET revoked_at=1')
                if conn is not None:
                    conn.execute(sql)
                else:
                    c.target.db._execute_write(lambda held: held.execute(sql))
        if boundary == 'body':
            def after_body():
                assert not c.target.db._conn.in_transaction
                # An awaited body must not retain the independent shared writer.
                with sqlite3.connect(c.target.home/'shared-state.db', timeout=0) as shared:
                    shared.execute('BEGIN IMMEDIATE')
                    shared.rollback()
                mutate()
            c.wire.faults.after_body = after_body
        else:
            original = primitive.retire_exact
            def before(outbox, conn, *args, **kwargs):
                assert conn.in_transaction
                mutate(conn)
                return original(outbox, conn, *args, **kwargs)
            monkeypatch.setattr(primitive, 'retire_exact', before)
        with pytest.raises(PeerRunsHTTPError):
            await wire_discard(c)
        unretired(c)
        assert 'peer_output_discard' not in admission_result(c.target.db, c.row['admission_id'])


@pytest.mark.asyncio
async def test_discard_keeps_shared_then_owner_order_and_never_initializes(files_target, monkeypatch):
    from gateway.platforms import api_server_room_artifacts as api
    async with peer_case(files_target, monkeypatch) as c:
        checked = []
        require = api.require_current_grant
        def inspect(conn, claims):
            assert conn.in_transaction
            checked.append(Path(conn.execute('PRAGMA database_list').fetchone()[2]).name)
            return require(conn, claims)
        monkeypatch.setattr(api, 'require_current_grant', inspect)
        def forbidden(*a, **kw):
            raise AssertionError('cold artifact operation attempted initialization')
        monkeypatch.setattr(RoomArtifactOutbox, '_initialize', forbidden)
        assert await discard(c) == 1
        assert await discard(c) == 1
        assert checked and all(checked[i:i+2] == ['shared-state.db', 'state.db'] for i in range(0,len(checked),2))
        # No cached success after quarantine or current owner loss.
        c.target.runner._draining = True
        with pytest.raises(PeerRunsHTTPError):
            await wire_discard(c)
