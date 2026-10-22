"""A local viewer's turn asks for the sudo password through the TUI's masked prompt, as main's sidecar
did (``tui_gateway/agent_callbacks.py`` ``set_sudo_password_callback``): FINDINGS row 9, where the
canonical owner registered no callback and ``sudo`` failed headless in 0.1 s."""
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform


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


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_sudo_request_reaches_the_viewer_and_its_answer_returns():
    from gateway.session_sudo_prompt import ask_sudo, is_response_frame, resolve
    events, peer = _events()
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault('value', ask_sudo(events, 's1', 'sudo true')))
    worker.start()
    _wait_for(lambda: any(f.get('method') == 'sudo' for f in peer.frames))
    request = next(f for f in peer.frames if f.get('method') == 'sudo')
    assert request['params'] == {'session_id': 's1', 'command': 'sudo true'}
    from tui_gateway.contracts.registry import SERVER_REQUESTS, validate_params
    assert validate_params(SERVER_REQUESTS['sudo'], request['params'])[1] is None
    answer = {'jsonrpc': '2.0', 'id': request['id'], 'result': {'value': 'hunter2'}}
    assert is_response_frame(answer) and resolve(answer)
    worker.join(5)
    assert out['value'] == 'hunter2'
    # The password never rides the event stream.
    assert not any('hunter2' in repr(f) for f in peer.frames)


def test_interrupt_withdraws_the_card():
    from gateway.session_sudo_prompt import ask_sudo
    events, peer = _events()
    flag = threading.Event()
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault(
        'value', ask_sudo(events, 's1', 'sudo true', cancelled=flag.is_set)))
    worker.start()
    _wait_for(lambda: any(f.get('method') == 'sudo' for f in peer.frames))
    flag.set()
    worker.join(5)
    assert out['value'] == ''
    _wait_for(lambda: any(f.get('params', {}).get('type') == 'request.cancel' for f in peer.frames))


def test_no_attached_viewer_is_headless():
    from gateway.session_events import SessionEvents
    from gateway.session_sudo_prompt import ask_sudo
    assert ask_sudo(SessionEvents(), 's1', 'sudo true', timeout=0.1) == ''


@pytest.mark.parametrize(('platform', 'installed'), [(Platform.LOCAL, True), (Platform.TELEGRAM, False)])
def test_scope_installs_the_callback_only_for_local_viewer_turns(platform, installed):
    from gateway.session_sudo_prompt import sudo_prompt_scope
    from tools.terminal_tool import _get_sudo_password_callback
    events, _ = _events()
    authority = SimpleNamespace(sessions={'s1': SimpleNamespace(event_stream=events)})
    turn = SimpleNamespace(_approval_owner=(authority, 's1', 1),
                           _ctx=SimpleNamespace(source=SimpleNamespace(platform=platform)))
    before = _get_sudo_password_callback()
    with sudo_prompt_scope(turn, SimpleNamespace()):
        assert (_get_sudo_password_callback() is not None) is installed
    assert _get_sudo_password_callback() is before


@pytest.mark.asyncio
async def test_the_viewer_response_frame_is_routed_by_the_authority_connection(tmp_path, monkeypatch):
    from gateway import session_sudo_prompt
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    pending = session_sudo_prompt._Pending('s')
    session_sudo_prompt._open['srq-abc'] = pending
    try:
        reply = await connection.dispatch({'jsonrpc': '2.0', 'id': 'srq-abc', 'result': {'value': 'pw'}})
        assert reply is None
        assert pending.event.is_set() and pending.value == 'pw'
    finally:
        session_sudo_prompt._open.pop('srq-abc', None)
        await connection.close()
