"""Cross-profile hosted observation stays bounded when the member transcript is large.

The private owner socket caps one response line at 512 KiB. A long member session must not
stop the driver-side terminal watcher (callbacks retained, completed turn never observed),
and an oversized response must surface as its own structured error, not ``runtime_draining``.
"""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest


def _server(home):
    from gateway.control_socket import GatewayControlServer
    descriptor = {'runtime_protocol': 1, 'state': 'ready', 'instance_id': 'test',
                  'authority_epoch': 1, 'served_profiles': [{'home': str(home), 'profile_id': str(home)}],
                  'capabilities': ['session-authority-v1'], 'api_origin': 'http://127.0.0.1:1',
                  'supervisor': 'none'}
    return GatewayControlServer(home, verb_handlers={'identify': lambda: descriptor})


def _owner_pair(hosted_owner, tmp_path):
    from gateway.session_hosted_transport import install_hosted_transport
    authority, loop, _, _ = hosted_owner
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    authority.profile_id = str(target)
    def attest(selector, operation, params):
        return {'owner': 'room-owner', 'target_home': authority.profile_id, 'prompt': 'input', 'attachments': []}
    servers = [_server(source), _server(target)]
    install_hosted_transport(servers[0], SimpleNamespace(profile_id=str(source)), loop, attest=attest)
    install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
    for server in servers:
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
    return source, target, servers


def test_watcher_delivers_terminal_once_with_bounded_polls_past_the_response_cap(hosted_owner, tmp_path, monkeypatch):
    from gateway import session_hosted_transport as transport
    from gateway.control_socket import _MAX_RESPONSE_BYTES
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import claim_session_input, settle_session_input
    from tui_gateway.hosted_room_driver import _bounded_terminal_result
    authority, loop, _, _ = hosted_owner
    polls, real_request = [], transport.owner_request
    def recording(home, verb, params, **kwargs):
        result = real_request(home, verb, params, **kwargs)
        polls.append((params['operation'], len(json.dumps(result, default=str).encode())))
        return result
    monkeypatch.setattr(transport, 'owner_request', recording)
    source, target, servers = _owner_pair(hosted_owner, tmp_path)
    rpc = transport.HostedRoomOwnerRPC(home=target, source_home=source, room_id='room', member_id='member', profile='default')
    try:
        coords = dict(profile='default', source='bot_room')
        sid = rpc.create(**coords, title='Group: room')['session_id']
        # A long member conversation: the transcript alone exceeds the response line cap.
        for _ in range(3):
            authority.db.append_message(sid, 'assistant', 'transcript ' * 25_000)
        received, done = [], threading.Event()
        def terminal(receipt):
            received.append(receipt)
            done.set()
        receipt = rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room', 'task', 'thread', 'turn'),
                             execution_generation=1, on_terminal=terminal)
        final = '\u20ac' * 100_000  # 300 KB UTF-8, 600 KB as an escaped JSON string
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'],
                             outcome='completed', result={'result': {'final_response': final}, 'usage': {}})
        assert done.wait(10), 'a completed turn must reach its callback despite the transcript size'
        rpc._monitor.join(5)
        assert not rpc._monitor.is_alive() and not rpc.callbacks
        assert received[0]['settlement_id'] == receipt['admission_id'] and received[0]['status'] == 'settled'
        # The driver persists the same bounded reply it would have built from the full text.
        assert _bounded_terminal_result(received[0]) == _bounded_terminal_result(
            {'message_id': receipt['admission_id'], 'text': final})
        watched = [size for operation, size in polls if operation not in {'create', 'submit'}]
        assert watched and 'history' not in [operation for operation, _ in polls]
        assert max(watched) < _MAX_RESPONSE_BYTES
        # Explicit recovery after the callback identity is lost still finds the receipt, once.
        history = rpc.history(**coords, session_id=sid)
        assert [m['settlement_id'] for m in history if m.get('task_id') == 'task'] == [receipt['admission_id']]
        assert len(received) == 1
    finally:
        with rpc._lock:
            rpc.callbacks.clear()
        for server in servers:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()


def test_oversized_owner_response_is_a_distinct_structured_error(hosted_owner, tmp_path):
    from gateway.control_socket import _MAX_RESPONSE_BYTES
    from gateway.session_hosted_transport import HostedRoomOwnerRPC
    from hermes_state_runtime import RuntimeStoreError
    _, loop, _, _ = hosted_owner
    source, target, servers = _owner_pair(hosted_owner, tmp_path)
    try:
        servers[1].private_handlers['hosted-producer'] = lambda params, peer: 'x' * _MAX_RESPONSE_BYTES
        raw = json.dumps({'protocol': 1, 'id': 7, 'verb': 'hosted-producer', 'params': {}}).encode()
        response = json.loads(servers[1].handle_request_line(raw, 'peer'))
        assert response['ok'] is False and response['id'] == 7
        assert response['error'] == response['code'] == 'response_too_large'
        assert response['limit'] == _MAX_RESPONSE_BYTES
        rpc = HostedRoomOwnerRPC(home=target, source_home=source, room_id='room', member_id='member', profile='default')
        with pytest.raises(RuntimeStoreError) as exc:
            rpc.info(profile='default', session_id='s', source='bot_room')
        assert exc.value.reason == 'response_too_large'
    finally:
        for server in servers:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()
