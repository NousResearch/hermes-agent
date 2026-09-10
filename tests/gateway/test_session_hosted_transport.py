"""Real private owner sockets; no public identity or foreign database access."""
import asyncio
import json
import threading
from dataclasses import asdict

import pytest

from tests.gateway.test_session_hosted_rpc import owner  # noqa: F401


def test_authenticated_owner_transport_rechecks_source_and_cold_binding(owner, tmp_path):
    from gateway.control_socket import GatewayControlServer
    from gateway.session_hosted_transport import (
        HostedRoomOwnerRPC, install_hosted_transport, check_remote_hosted_admission,
        owner_request,
    )
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import list_session_admissions, RuntimeStoreError
    authority, loop, _, _ = owner
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    allowed = [True]
    task = TaskIdentity('room', 'task', 'thread', 'turn')
    def attest(selector, operation, params):
        if not allowed[0] or selector != dict(room_id='room', member_id='member', profile='default'):
            raise RuntimeStoreError('permission_denied')
        if operation in {'submit', 'execute'}:
            assert params['task'] == asdict(task)
            assert params['execution_generation'] == 1
            if params['prompt'] != 'input':
                raise RuntimeStoreError('permission_denied')
        return {'owner': 'room-owner'}
    servers = [GatewayControlServer(source), GatewayControlServer(target)]
    install_hosted_transport(servers[0], authority, loop, attest=attest)
    install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
    for server in servers:
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
    try:
        rpc = HostedRoomOwnerRPC(home=target, source_home=source, room_id='room', member_id='member', profile='default')
        coords = dict(profile='default', source='bot_room')
        sid = rpc.create(**coords, title='Group: room')['session_id']
        args = dict(**coords, session_id=sid, prompt='input', task=task, execution_generation=1, on_terminal=lambda r: None)
        receipt = rpc.submit(**args)
        assert rpc.submit(**args)['admission_id'] == receipt['admission_id']
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert len(rows) == 1
        from gateway.session_contract import SessionRef
        ref = SessionRef(authority.profile_id, sid)
        assert check_remote_hosted_admission(authority, ref, rows[0]) is True
        # Recreate server routing to lose all process-local producer caches.
        install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
        assert check_remote_hosted_admission(authority, ref, rows[0]) is True
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            rpc.submit(**{**args, 'prompt': 'forged'})
        raw = json.dumps({'protocol': 1, 'verb': 'hosted-producer', 'params': {}}).encode()
        assert not json.loads(servers[1].handle_request_line(raw))['ok']
        allowed[0] = False
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            check_remote_hosted_admission(authority, ref, rows[0])
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            rpc.history(**coords, session_id=sid)
        assert rows == list_session_admissions(authority.db, session_id=sid, pending_only=False)
    finally:
        for server in servers:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()
