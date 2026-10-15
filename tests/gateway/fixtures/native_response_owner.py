"""Known-request-ID resolver cases, not a public delivery/leak demonstration."""
import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace


async def owner_cases(root):
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_sudo_prompt import _lock, _open, resolve, sudo_prompt_scope
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    class Peer:
        def __init__(self):
            self.asked = threading.Event()
            self.request = None

        def write(self, frame):
            if frame.get('method') in ('sudo', 'preview.read'):
                self.request = frame
                self.asked.set()
            return True

        def close(self):
            pass

    runner = SimpleNamespace(_draining=False)
    authorities = []
    for name in ('alpha', 'beta'):
        home = root / 'profiles' / name
        home.mkdir(parents=True)
        db = SessionDB(db_path=home / 'state.db')
        authority = SessionAuthority(runner, profile_id=str(home), instance_id='fixture', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='fixture'))
        db.create_session('same', source='test')
        authority.sessions['same'] = LiveSession(None, 'fixture-' + name)
        admit_session_input(db, epoch=authority.epoch, principal_id='fixture', session_id='same',
                            request_id='fixture', payload={'text': 'fixture'})
        claim_session_input(db, epoch=authority.epoch, session_id='same')
        authorities.append(authority)
    alpha, beta = authorities
    runner.session_authorities = authorities
    results = []
    try:
        for mode, expected in (('host', True), ('restricted', False), ('no_capability', False),
                               ('retired', False), ('reserved_again', False), ('stale_generation', False),
                               ('replaced_session', False), ('rotated_epoch', False),
                               ('cancelled', False), ('expired', False), ('missing_owner', False),
                               ('unknown', False), ('error', True)):
            peer = Peer()
            beta.sessions['same'].event_stream.fanout.attach(peer)
            identity = {'user_id': 'fixture', 'profile_id': alpha.profile_id}
            if mode != 'restricted':
                identity['profile_scope'] = 'host'
            if mode == 'no_capability':
                identity['capabilities'] = ['session:read']
            connection = AuthorityConnection(alpha, object(), identity)
            agent = SimpleNamespace(_interrupt_requested=False)
            turn = SimpleNamespace(_approval_owner=(beta, 'same', 1),
                                   _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))

            def prompt():
                if mode in ('missing_owner', 'expired'):
                    from gateway.session_sudo_prompt import ask_sudo
                    return ask_sudo(beta.sessions['same'].event_stream, 'same', '',
                                    owner=None if mode == 'missing_owner' else turn._approval_owner,
                                    timeout=0 if mode == 'expired' else 30)
                if mode == 'error':
                    from gateway.local_desktop_ui import _ask_result
                    return _ask_result(turn, agent, 'preview.read', {}, 30)
                from tools.terminal_tool import _get_sudo_password_callback
                with sudo_prompt_scope(turn, agent):
                    return _get_sudo_password_callback()()

            task = asyncio.create_task(asyncio.to_thread(prompt))
            assert await asyncio.to_thread(peer.asked.wait, 2), 'callback did not emit request'
            frame = {'jsonrpc': '2.0', 'id': peer.request['id'], 'result': {'value': ''}}
            with _lock:
                pending = _open.get(frame['id'])
            cleanup = dict(frame)
            original_live, original_epoch = beta.sessions['same'], beta.epoch
            if mode == 'unknown':
                frame['id'] = 'srq-unknown-fixture'
            if mode == 'error':
                frame = {'jsonrpc': '2.0', 'id': peer.request['id'],
                         'error': {'code': -32601, 'message': 'unsupported'}}
            if mode == 'retired':
                beta.retired = True
            if mode == 'reserved_again':
                runner.session_authorities = [alpha, SimpleNamespace(profile_id=beta.profile_id)]
            if mode == 'stale_generation':
                beta.db._execute_write(lambda conn: conn.execute(
                    'UPDATE sessions SET runtime_generation=2 WHERE id=?', ('same',)))
            if mode == 'replaced_session':
                beta.sessions['same'] = LiveSession(None, 'replacement-fixture')
            if mode == 'rotated_epoch':
                beta.epoch += 1
            if mode == 'cancelled':
                agent._interrupt_requested = True
            await connection.dispatch(frame)
            settled = pending is not None and pending.event.is_set()
            results.append({'case': mode, 'settled': settled, 'expected': expected})
            resolve(cleanup)  # internal cleanup only, not an authenticated settlement
            await task
            with _lock:
                assert cleanup['id'] not in _open
            assert resolve(cleanup) is False, 'duplicate consumed a second waiter'
            beta.retired = False
            beta.sessions['same'], beta.epoch = original_live, original_epoch
            runner.session_authorities = authorities
            beta.db._execute_write(lambda conn: conn.execute(
                'UPDATE sessions SET runtime_generation=1 WHERE id=?', ('same',)))
            await connection.close()
        results.append({'case': 'simultaneous_equal_native_ids',
                        'settled': await _simultaneous(alpha, beta, Peer), 'expected': True})
    finally:
        for authority in authorities:
            authority.db.close()
    return results


async def _simultaneous(alpha, beta, peer_type):
    from gateway.config import Platform
    from gateway.session_controls import AuthorityConnection
    from gateway.session_sudo_prompt import _lock, _open, resolve, sudo_prompt_scope
    connection = AuthorityConnection(alpha, object(), {'user_id': 'fixture', 'profile_scope': 'host'})
    peers, tasks, requests, pending = [], [], [], []
    try:
        for authority in (alpha, beta):
            peer = peer_type()
            peers.append(peer)
            authority.sessions['same'].event_stream.fanout.attach(peer)
            turn = SimpleNamespace(_approval_owner=(authority, 'same', 1),
                                   _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))
            def prompt(captured=turn):
                from tools.terminal_tool import _get_sudo_password_callback
                with sudo_prompt_scope(captured, SimpleNamespace(_interrupt_requested=False)):
                    return _get_sudo_password_callback()()
            tasks.append(asyncio.create_task(asyncio.to_thread(prompt)))
            assert await asyncio.to_thread(peer.asked.wait, 2)
            requests.append({'jsonrpc': '2.0', 'id': peer.request['id'], 'result': {'value': ''}})
            with _lock:
                pending.append(_open[peer.request['id']])
        assert requests[0]['id'] != requests[1]['id']
        await connection.dispatch(requests[0])
        assert pending[0].event.is_set() and not pending[1].event.is_set()
        await connection.dispatch(requests[1])
        assert pending[1].event.is_set()
        assert not resolve(requests[0]) and not resolve(requests[1])
        return True
    finally:
        for request in requests:
            resolve(request)  # teardown only, if an assertion above failed
        await asyncio.gather(*tasks)
        await connection.close()


if __name__ == '__main__':
    import json
    import tempfile
    with tempfile.TemporaryDirectory(prefix='native-owner-fixture-') as temporary:
        result = asyncio.run(owner_cases(Path(temporary)))
        print(json.dumps(result, indent=2))
        assert all(row['settled'] == row['expected'] for row in result), result
