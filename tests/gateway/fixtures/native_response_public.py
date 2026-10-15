"""Public dashboard native-response fixture; no model, credentials or installed runtime.

The auth provider is in-memory and the admitted turn handler is inert. HTTP middleware,
ticket mint/consume, WebSocket gates, authority dispatch, FIFO and sudo callback are real.
"""
import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch


def public_native_cases(root):
    from fastapi.testclient import TestClient
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_sudo_prompt import sudo_prompt_scope
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.base import Session
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    class Provider(StubAuthProvider):
        name = 'native-fixture'

        def verify_session(self, *, access_token):
            if access_token != 'public-synthetic-bearer':
                return None
            return Session('fixture-operator', '', 'Fixture', '', self.name,
                           9999999999, '', '')

    agent = SimpleNamespace(_interrupt_requested=False)
    returned = threading.Event()
    answers, seen = [], []
    authorities = []

    async def handle(event):
        seen.append(event.text)
        if event.text == 'ask-native':
            generation = beta.executing['beta-only']
            turn = SimpleNamespace(_approval_owner=(beta, 'beta-only', generation),
                                   _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))

            def prompt():
                from tools.terminal_tool import _get_sudo_password_callback
                with sudo_prompt_scope(turn, agent):
                    return _get_sudo_password_callback()()

            answers.append(await asyncio.to_thread(prompt))
            returned.set()
        return 'inert-completed'

    runner = SimpleNamespace(_draining=False, _handle_message=handle,
                             _adapter_for_source=lambda source: None)
    for name in ('alpha', 'beta'):
        home = root / 'profiles' / name
        home.mkdir(parents=True)
        db = SessionDB(db_path=home / 'state.db')
        authority = SessionAuthority(runner, profile_id=str(home), instance_id='fixture-runtime', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='fixture-runtime'))
        authorities.append(authority)
    alpha, beta = authorities
    runner.session_authorities = authorities
    beta.db.create_session('beta-only', source='telegram')
    beta.sessions['beta-only'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='fixture'),
                                             'fixture-beta')
    clear_providers()
    register_provider(Provider())
    client = TestClient(web_server.app, base_url='https://fixture.example.test')
    results = []
    rid = 0

    def rpc(ws, method, *, expect_error=False, **params):
        nonlocal rid
        rid += 1
        ws.send_json({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params})
        while True:
            frame = ws.receive_json()
            if frame.get('id') == rid:
                if expect_error:
                    assert 'error' in frame, frame
                    return frame['error']
                assert 'result' in frame, frame
                return frame['result']

    def ticket():
        refused = client.post('/api/auth/ws-ticket')
        assert refused.status_code == 401, refused.status_code
        response = client.post('/api/auth/ws-ticket', headers={
            'Authorization': 'Bearer public-synthetic-bearer'})
        assert response.status_code == 200, response.status_code
        return response.json()['ticket']

    state = web_server.app.state
    try:
        with patch.object(state, 'auth_required', True, create=True), \
                patch.object(state, 'bound_host', 'fixture.example.test', create=True), \
                patch.object(state, 'bound_port', 443, create=True), \
                patch.object(state, 'session_authority', alpha, create=True), \
                patch.object(web_server, '_DASHBOARD_EMBEDDED_CHAT_ENABLED', True), \
                patch('tui_gateway.ws._start_backend_liveness'):
            for number in (1, 2):
                returned.clear()
                agent._interrupt_requested = False
                with client.websocket_connect('wss://fixture.example.test/api/ws?ticket=' + ticket(), headers={
                        'origin': 'https://fixture.example.test'}) as ws:
                    ready = ws.receive_json()
                    assert ready['params']['type'] == 'gateway.ready', ready
                    assert ready['params']['payload']['session_authority'] is True
                    description = rpc(ws, 'runtime.describe', profile='beta')
                    assert description['profile_id'] == beta.profile_id, description
                    assert rpc(ws, 'runtime.describe')['profile_id'] == alpha.profile_id
                    refused = rpc(ws, 'runtime.describe', profile='beta', unexpected=True, expect_error=True)
                    assert refused['data']['reason'] == 'invalid_params', refused
                    assert refused['data']['fields'] == ['unexpected'], refused
                    rpc(ws, 'session.resume', profile='beta', session_id='beta-only')
                    first = rpc(ws, 'prompt.submit', profile='beta', session_id='beta-only',
                                submission_id=f'ask-{number}', text='ask-native')
                    assert first['status'] == 'queued', first
                    while True:
                        request = ws.receive_json()
                        if request.get('method') == 'sudo':
                            break
                    assert set(request['params']) == {'session_id', 'command'}, request
                    ws.send_json({'jsonrpc': '2.0', 'id': request['id'], 'result': {'value': ''}})
                    # Queue a correlated barrier after the one-way answer: send alone is not acceptance.
                    rpc(ws, 'ping')
                    settled = returned.wait(2)
                    terminal, next_once, snapshot_clear, next_terminal = False, False, False, False
                    first_receipt, next_receipt = None, None
                    if settled:
                        async def drain():
                            await beta.sessions['beta-only'].task
                        ws.portal.call(drain)
                        first_receipt = rpc(ws, 'prompt.receipt', profile='beta', session_id='beta-only',
                                            admission_id=first['admission_id'])
                        terminal = first_receipt['status'] == 'terminal'
                        snapshot = rpc(ws, 'session.resume', profile='beta', session_id='beta-only')
                        snapshot_clear = not snapshot['pending']
                        following = rpc(ws, 'prompt.submit', profile='beta', session_id='beta-only',
                                        submission_id=f'next-{number}', text=f'next-{number}')
                        ws.portal.call(drain)
                        next_receipt = rpc(ws, 'prompt.receipt', profile='beta', session_id='beta-only',
                                           admission_id=following['admission_id'])
                        next_terminal = next_receipt['status'] == 'terminal'
                        final = rpc(ws, 'session.resume', profile='beta', session_id='beta-only')
                        snapshot_clear = snapshot_clear and not final['pending'] and not final['running']
                        next_once = seen.count(f'next-{number}') == 1
                    else:
                        # Interruption is cleanup only, never counted as public response settlement.
                        agent._interrupt_requested = True
                        async def unwind():
                            await beta.sessions['beta-only'].task
                        ws.portal.call(unwind)
                    results.append({'socket': number, 'public_reply_settled': settled,
                                    'terminal_admission': terminal, 'snapshot_clear': snapshot_clear,
                                    'next_send_once': next_once, 'next_terminal': next_terminal,
                                    'first_receipt': first_receipt, 'next_receipt': next_receipt})
    finally:
        clear_providers()
        client.close()
        for authority in authorities:
            authority.db.close()
    return {'public_http_ticket': True, 'gated_websocket_route': True,
            'actual_admission_worker': True, 'real_model_or_tool_execution': False,
            'answers': answers, 'seen': seen, 'cases': results}


if __name__ == '__main__':
    import os
    import tempfile
    with tempfile.TemporaryDirectory(prefix='public-native-fixture-') as temporary:
        os.environ['HERMES_HOME'] = temporary
        result = public_native_cases(Path(temporary))
        print(json.dumps(result, indent=2))
        assert all(all(case[key] for key in ('public_reply_settled', 'terminal_admission',
                                            'snapshot_clear', 'next_send_once', 'next_terminal'))
                   for case in result['cases']), result
