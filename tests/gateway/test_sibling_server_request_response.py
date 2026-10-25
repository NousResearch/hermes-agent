"""#142849: a viewer's answer to a sibling profile's server request settles it over the shared socket."""
import asyncio
import threading
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


class _Transport:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


@pytest.mark.asyncio
async def test_host_socket_answers_a_sibling_profiles_sudo_request(tmp_path, monkeypatch):
    """The Desktop's shared-primary socket chats on profile beta and answers beta's ``sudo`` request with
    a response frame (which carries no profile): the waiting tool gets the password, not a timeout."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_controls import AuthorityConnection
    from gateway.session_server_requests import ask_viewer
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    connection, runner = None, None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        store = runner.session_ticket_store
        ticket = store.mint(profile_id=str(root.resolve()), subject='uid:desk', purpose='interactive', scope='host')
        grant = store.redeem(ticket, profile_id=None, purpose='interactive')
        launch = authority_for_profile_id(runner, grant['profile_id'])
        identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                    'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                    'native_bootstrap': True, 'profile_scope': 'host'}
        transport = _Transport()
        with owner_scope(launch):
            connection = AuthorityConnection(launch, transport, identity, operator=True)
            created = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
                'profile': 'beta', 'request_id': 'c0', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'm',
                'toolsets': []}})
        session_id = created['result']['session_id']
        beta = runner.session_authorities.for_home(homes[2][1])
        assert beta is not launch and session_id in beta.sessions
        events = beta.sessions[session_id].event_stream
        out = {}
        worker = threading.Thread(target=lambda: out.update(value=ask_viewer(
            events, session_id, 'sudo', {'command': 'sudo true'}, timeout=5)), daemon=True)
        worker.start()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while not any(f.get('method') == 'sudo' for f in transport.frames):
            assert loop.time() < deadline, transport.frames
            await asyncio.sleep(0.01)
        rid = next(f['id'] for f in transport.frames if f.get('method') == 'sudo')
        with owner_scope(launch):
            await connection.dispatch({'jsonrpc': '2.0', 'id': rid, 'result': {'value': 'hunter2'}})
        await asyncio.to_thread(worker.join, 10)
        assert out == {'value': 'hunter2'}, out
    finally:
        if connection is not None:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in getattr(runner, 'session_authorities', None) or []:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))
