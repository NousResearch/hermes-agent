"""A local viewer's turn asks for browser-vault input (a site's one-time code, a login to save, a password
manager's master password) through the Desktop/Ink cards, as main's sidecar did
(``tui_gateway/agent_callbacks.py``). The canonical owner registered no vault callback, so ``browser_vault``
reported ``prompt_unavailable`` in every Desktop chat and an emailed sign-in code could not be entered."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.waiting import wait_until


class _Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


def _events():
    from gateway.session_events import SessionEvents
    events, peer = SessionEvents(), _Peer()
    events.fanout.attach(peer)
    return events, peer


def _turn(events, platform=Platform.LOCAL, session_id='s1'):
    authority = SimpleNamespace(sessions={session_id: SimpleNamespace(event_stream=events)})
    return SimpleNamespace(_approval_owner=(authority, session_id, 1),
                           _ctx=SimpleNamespace(source=SimpleNamespace(platform=platform)))


def _answer(peer, method, value):
    from gateway.session_server_requests import is_response_frame, resolve
    wait_until(lambda: any(f.get('method') == method for f in peer.frames), what=f'{method} request')
    request = next(f for f in peer.frames if f.get('method') == method)
    from tui_gateway.contracts.registry import SERVER_REQUESTS, validate_params
    assert validate_params(SERVER_REQUESTS[method], request['params'])[1] is None
    frame = {'jsonrpc': '2.0', 'id': request['id'], 'result': {'value': value}}
    assert is_response_frame(frame) and resolve(frame)
    return request


@pytest.mark.parametrize(('platform', 'installed'), [(Platform.LOCAL, True), (Platform.TELEGRAM, False)])
def test_scope_installs_the_callbacks_only_for_local_viewer_turns_and_restores_them(platform, installed):
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import vault_prompt_scope
    events, _ = _events()
    before = (unlock.get_unlock_prompt_callback(), unlock.get_save_login_prompt_callback(),
              unlock.get_code_prompt_callback(), getattr(unlock._current_session_tls, 'sid', None))
    with vault_prompt_scope(_turn(events, platform), SimpleNamespace()):
        got = (unlock.get_unlock_prompt_callback(), unlock.get_save_login_prompt_callback(),
               unlock.get_code_prompt_callback())
        assert all(cb is not None for cb in got) is installed
        if installed:
            assert unlock.can_prompt_here()
            assert unlock._current_session_tls.sid == 's1'
    assert (unlock.get_unlock_prompt_callback(), unlock.get_save_login_prompt_callback(),
            unlock.get_code_prompt_callback(), getattr(unlock._current_session_tls, 'sid', None)) == before


def test_code_request_reaches_the_viewer_and_the_code_never_rides_the_event_stream():
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import vault_prompt_scope
    events, peer = _events()
    out = {}

    def run():
        with vault_prompt_scope(_turn(events), SimpleNamespace()):
            out['code'] = unlock.get_code_prompt_callback()('www.epicgames.com', '')
    worker = threading.Thread(target=run)
    worker.start()
    request = _answer(peer, 'vault.code', '954145')
    worker.join(5)
    assert request['params'] == {'session_id': 's1', 'site': 'www.epicgames.com', 'hint': ''}
    assert out['code'] == '954145'
    assert not any('954145' in repr(f) for f in peer.frames)


def test_tool_worker_threads_inherit_the_code_prompt():
    """browser_vault runs on a tool worker when calls run in parallel; the callbacks must reach it."""
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    events, _ = _events()
    seen = {}
    with vault_prompt_scope(_turn(events), SimpleNamespace()):
        worker = threading.Thread(target=propagate_context_to_thread(
            lambda: seen.update(cb=unlock.get_code_prompt_callback(), can=unlock.can_prompt_here())))
        worker.start()
        worker.join(5)
    assert seen['cb'] is not None and seen['can'] is True


@pytest.mark.parametrize(('value', 'expected'), [
    (json.dumps({'identifier': 'me@example.com', 'password': 'pw'}), {'identifier': 'me@example.com', 'password': 'pw'}),
    ('not json', None),
    (json.dumps({'identifier': 'me@example.com'}), None),
    ('', None),
])
def test_save_login_parses_the_cards_json_answer(value, expected):
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import vault_prompt_scope
    events, peer = _events()
    out = {}

    def run():
        with vault_prompt_scope(_turn(events), SimpleNamespace()):
            out['answer'] = unlock.get_save_login_prompt_callback()('https://example.com', 'example.com')
    worker = threading.Thread(target=run)
    worker.start()
    _answer(peer, 'vault.save_login', value)
    worker.join(5)
    assert out['answer'] == expected


def test_unlock_prompt_returns_the_master_password():
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import vault_prompt_scope
    events, peer = _events()
    out = {}

    def run():
        with vault_prompt_scope(_turn(events), SimpleNamespace()):
            out['master'] = unlock.get_unlock_prompt_callback()('bitwarden', 'Bitwarden')
    worker = threading.Thread(target=run)
    worker.start()
    request = _answer(peer, 'vault.unlock_prompt', 'correct horse')
    worker.join(5)
    assert request['params']['display_name'] == 'Bitwarden'
    assert out['master'] == 'correct horse'


def test_interrupt_withdraws_the_code_card():
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import vault_prompt_scope
    events, peer = _events()
    agent = SimpleNamespace(_interrupt_requested=False)
    out = {}

    def run():
        with vault_prompt_scope(_turn(events), agent):
            out['code'] = unlock.get_code_prompt_callback()('example.com', '')
    worker = threading.Thread(target=run)
    worker.start()
    wait_until(lambda: any(f.get('method') == 'vault.code' for f in peer.frames), what='vault.code request')
    agent._interrupt_requested = True
    worker.join(5)
    assert out['code'] == ''
    wait_until(lambda: any(f.get('params', {}).get('type') == 'request.cancel' for f in peer.frames),
               what='request.cancel')


def test_no_attached_viewer_returns_at_once():
    from agent.vault_backends import unlock
    from gateway.session_events import SessionEvents
    from gateway.session_vault_prompt import vault_prompt_scope
    started = time.monotonic()
    with vault_prompt_scope(_turn(SessionEvents()), SimpleNamespace()):
        assert unlock.get_code_prompt_callback()('example.com', '') == ''
    assert time.monotonic() - started < 2


def _enter_code_with_a_fake_page():
    """browser_vault_enter_code against a page with one code field; returns (tool result, filled JS)."""
    from unittest.mock import patch
    from tools import browser_vault_tool
    controls = [{'index': 0, 'type': 'text', 'name': 'otp', 'label': 'Security code', 'autocomplete': 'one-time-code'}]
    seen = {}

    def fake_eval(task_id, expr):
        return {'success': True, 'result': json.dumps(controls) if 'querySelectorAll' in expr
                else 'https://www.epicgames.com/id/login/mfa'}

    def fake_secret(task_id, expr):
        seen['expr'] = expr
        return {'success': True, 'result': json.dumps({'filled': 1})}

    with patch.object(browser_vault_tool, '_focus_bound_origin', lambda *a, **k: None), \
         patch.object(browser_vault_tool, '_eval_js', side_effect=fake_eval), \
         patch.object(browser_vault_tool, '_eval_js_secret', side_effect=fake_secret):
        raw = browser_vault_tool.browser_vault_enter_code(task_id='t')
    return json.loads(raw), raw, seen.get('expr', '')


def test_enter_code_in_a_desktop_chat_asks_the_viewer_and_fills_the_page():
    """The reported bug: in a Desktop chat browser_vault said "this session cannot ask the user" and an
    emailed sign-in code could not be entered. Without the scope that is still the answer; with it the
    code card opens, and the typed code reaches the page but not the tool result or the event stream."""
    from gateway.session_vault_prompt import vault_prompt_scope
    out, _, _ = _enter_code_with_a_fake_page()
    assert out['error_type'] == 'prompt_unavailable'

    events, peer = _events()
    result = {}

    def run():
        with vault_prompt_scope(_turn(events), SimpleNamespace()):
            result['out'], result['raw'], result['expr'] = _enter_code_with_a_fake_page()
    worker = threading.Thread(target=run)
    worker.start()
    _answer(peer, 'vault.code', '954145')
    worker.join(5)
    assert result['out']['success'] and result['out']['source'] == 'user'
    assert '954145' in result['expr']
    assert '954145' not in result['raw']
    assert not any('954145' in repr(f) for f in peer.frames)


def test_closing_the_session_locks_only_the_managers_it_unlocked():
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import release_unlocks, vault_prompt_scope
    events, _ = _events()
    try:
        with vault_prompt_scope(_turn(events, session_id='s1'), SimpleNamespace()):
            assert unlock.store_session_token('test-backend-a', 'token-a')
        with vault_prompt_scope(_turn(events, session_id='s2'), SimpleNamespace()):
            assert unlock.store_session_token('test-backend-b', 'token-b')
        release_unlocks('s1')
        assert not unlock.is_unlocked('test-backend-a')
        assert unlock.is_unlocked('test-backend-b')
    finally:
        unlock.lock('test-backend-a')
        unlock.lock('test-backend-b')


def test_a_manager_unlocked_on_a_tool_worker_is_released_with_its_session():
    """browser_vault often runs on a tool worker (parallel calls): the unlock owner must cross that boundary
    with the callbacks, or closing the chat leaves the manager unlocked until its idle timeout."""
    from agent.vault_backends import unlock
    from gateway.session_vault_prompt import release_unlocks, vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    events, _ = _events()
    try:
        with vault_prompt_scope(_turn(events, session_id='s1'), SimpleNamespace()):
            worker = threading.Thread(target=propagate_context_to_thread(
                lambda: unlock.store_session_token('test-backend-w', 'token-w')))
            worker.start()
            worker.join(5)
        assert unlock.is_unlocked('test-backend-w')
        release_unlocks('s1')
        assert not unlock.is_unlocked('test-backend-w')
    finally:
        unlock.lock('test-backend-w')


def test_deleting_a_chat_locks_the_managers_it_unlocked(monkeypatch):
    """Delete, the owner-run prune and onboarding cleanup retire the live session without the
    viewer-close settle, so retiring it locks what the chat unlocked. Other chats keep theirs."""
    from agent.vault_backends import unlock
    from gateway import session_local_mcp, session_policy, session_worker_construct
    from gateway.session_mutations import retire_live_sessions
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools import async_delegation
    for module, name in ((session_policy, 'release_launch_secrets'), (session_policy, 'release_session_workspaces'),
                         (session_local_mcp, 'release_editor_mcp'),
                         (session_worker_construct, 'release_session_prefill'), (async_delegation, 'allow_for_session')):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    authority = SimpleNamespace(runner=SimpleNamespace(), profile_id='default', sessions={})
    events, _ = _events()
    try:
        with vault_prompt_scope(_turn(events, session_id='s-del'), SimpleNamespace()):
            assert unlock.store_session_token('test-backend-d', 'token-d')
        with vault_prompt_scope(_turn(events, session_id='s-keep'), SimpleNamespace()):
            assert unlock.store_session_token('test-backend-k', 'token-k')
        retire_live_sessions(authority, ['s-del'])
        assert not unlock.is_unlocked('test-backend-d')
        assert unlock.is_unlocked('test-backend-k')
    finally:
        unlock.lock('test-backend-d')
        unlock.lock('test-backend-k')
