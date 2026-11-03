"""Standalone ``hermes serve``/``dashboard`` (no gateway authority) keeps session edits.

The edit runs the gateway's receipt transaction on the process's single writer under the
exact owner lock: revision/retry/idle guards hold, an owner discovery cannot reach answers 503
and one that wins the lock after discovery refuses with 409.
"""
import httpx
import pytest
from fastapi import FastAPI

from gateway.runtime_ownership import ProfileOwnership
from hermes_cli.gateway_runtime import GatewayDiscovery
from hermes_cli.web_routers.sessions import manage_router
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch

AUTH = {'Authorization': 'Bearer standalone-token'}


def _standalone_app(tmp_path, monkeypatch):
    from hermes_cli import web_server
    monkeypatch.setattr(web_server, '_SESSION_TOKEN', 'standalone-token')
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    return app


@pytest.mark.asyncio
async def test_standalone_edits_commit_with_owner_guards(tmp_path, monkeypatch):
    app = _standalone_app(tmp_path, monkeypatch)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('edit-me', source='cli')
        db.create_session('busy', source='cli')
        epoch = begin_runtime_epoch(db, instance_id='earlier-gateway')
        admit_session_input(db, epoch=epoch, principal_id='p', session_id='busy', request_id='q',
                            payload={'text': 'hi'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost',
                                     headers=AUTH) as client:
            snap = await client.get('/api/sessions/edit-me/mutation-snapshot')
            assert snap.status_code == 200, snap.text
            assert snap.json()['runtime_revision'] == 0
            body = {'request_id': 'r1', 'expected_revision': 0, 'title': 'Renamed', 'pinned': True}
            first = await client.patch('/api/sessions/edit-me', json=body)
            assert first.status_code == 200, first.text
            assert first.json()['revision'] == 1
            assert (await client.patch('/api/sessions/edit-me', json=body)).json() == first.json()
            stale = await client.patch('/api/sessions/edit-me', json=body | {'request_id': 'r2', 'title': 'Stale'})
            assert (stale.status_code, stale.json()['detail']) == (409, 'revision_conflict')
            busy = await client.delete('/api/sessions/busy', params={
                'request_id': 'd0', 'expected_revision': 0, 'expected_generation': 0})
            assert (busy.status_code, busy.json()['detail']) == (409, 'session_busy')
            gone = await client.delete('/api/sessions/edit-me', params={
                'request_id': 'd1', 'expected_revision': 1, 'expected_generation': 0})
            assert gone.status_code == 200, gone.text
            assert gone.json()['deleted_ids'] == ['edit-me']
        assert db.get_session('edit-me') is None
        assert db.get_session('busy') is not None
        # The existing epoch was reused, not advanced: a standalone edit never fences an owner.
        assert db._conn.execute('SELECT epoch FROM runtime_epoch').fetchone()[0] == epoch


@pytest.mark.asyncio
async def test_standalone_edit_refuses_while_a_gateway_owns_the_store(tmp_path, monkeypatch, caplog):
    app = _standalone_app(tmp_path, monkeypatch)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('owned', source='cli')
        owner = ProfileOwnership()
        owner.reserve([tmp_path])
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url='http://localhost', headers=AUTH) as client:
                # The held lock without a control socket reads as a starting owner: retryable, never offline.
                starting = await client.patch('/api/sessions/owned',
                                              json={'request_id': 'r1', 'expected_revision': 0, 'title': 'X'})
                assert starting.status_code == 503, starting.text
                assert starting.json()['detail'].startswith('owner_unavailable: starting: Exclusive maintenance refused')
                assert 'gateway owner not reachable (starting)' in caplog.text
                # An owner that reserves after discovery read absent still wins the offline reservation.
                monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                                    lambda home, timeout=2.0: GatewayDiscovery('absent'))
                refused = await client.patch('/api/sessions/owned',
                                             json={'request_id': 'r1', 'expected_revision': 0, 'title': 'X'})
                assert refused.status_code == 409, refused.text
                # uvicorn logs no handled 4xx: the refusal itself must leave a server-side trace (#138915).
                assert 'session write refused' in caplog.text
                # A composed gateway whose authority is not up yet never falls back to a writer.
                app.state.gateway_runner = object()
                pending = await client.patch('/api/sessions/owned',
                                             json={'request_id': 'r1', 'expected_revision': 0, 'title': 'X'})
                assert (pending.status_code, pending.json()['detail']) == (503, 'session_authority_unavailable')
        finally:
            owner.close()
        assert db.get_session('owned')['title'] is None
        assert db.get_session('owned')['runtime_revision'] == 0


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_a_stale_socket_without_a_gateway_still_edits_offline(tmp_path, monkeypatch):
    """A crashed gateway can leave a ``gateway.sock`` discovery refuses on its mode before it reads
    ``gateway.lock``. No process owns the store, so the offline reservation wins and the edit lands."""
    import socket
    app = _standalone_app(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)  # a relative bind keeps the socket path under AF_UNIX's length limit
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind('gateway.sock')
    stale.close()
    (tmp_path / 'gateway.sock').chmod(0o755)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('stale', source='cli')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://localhost', headers=AUTH) as client:
            pinned = await client.patch('/api/sessions/stale',
                                        json={'request_id': 'r1', 'expected_revision': 0, 'pinned': True})
        assert pinned.status_code == 200, pinned.text
        assert db.get_session('stale')['pinned'] == 1
