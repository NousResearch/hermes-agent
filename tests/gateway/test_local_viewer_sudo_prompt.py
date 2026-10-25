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
    from gateway.session_server_requests import is_response_frame, resolve
    from gateway.session_sudo_prompt import ask_sudo
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


def _abstain(request_id):
    from gateway.session_server_requests import ABSTAIN_CODE
    return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': ABSTAIN_CODE, 'message': 'not mine'}}


def _ask_in_background(events, out):
    from gateway.session_sudo_prompt import ask_sudo
    worker = threading.Thread(target=lambda: out.setdefault('value', ask_sudo(events, 's1', 'sudo true')),
                              daemon=True)  # the default 120 s timeout: a hang must not hold the run
    worker.start()
    return worker


def test_a_sole_viewer_that_abstains_declines_the_request_at_once():
    """A ``-Q`` client queued behind the running turn is the only viewer (#143565): its abstention declines
    the prompt now, as with no viewer attached, instead of after the 120 s timeout."""
    from gateway.session_server_requests import resolve
    events, peer = _events()
    out = {}
    worker = _ask_in_background(events, out)
    _wait_for(lambda: any(f.get('method') == 'sudo' for f in peer.frames))
    request = next(f for f in peer.frames if f.get('method') == 'sudo')
    assert resolve(_abstain(request['id']), transport=peer)
    worker.join(5)
    assert not worker.is_alive() and out['value'] == ''
    _wait_for(lambda: any(f.get('params', {}).get('type') == 'request.cancel' for f in peer.frames))


def test_an_abstention_leaves_the_request_to_the_other_viewers():
    """The Desktop window that shows the card still answers it after the ``-Q`` client abstained."""
    from gateway.session_server_requests import resolve
    events, finite = _events()
    desktop = _Peer()
    events.fanout.attach(desktop)
    out = {}
    worker = _ask_in_background(events, out)
    _wait_for(lambda: any(f.get('method') == 'sudo' for f in desktop.frames))
    request = next(f for f in desktop.frames if f.get('method') == 'sudo')
    assert resolve(_abstain(request['id']), transport=finite)
    worker.join(0.75)  # past a few polls
    assert worker.is_alive()
    assert resolve({'jsonrpc': '2.0', 'id': request['id'], 'result': {'value': 'hunter2'}})
    worker.join(5)
    assert out['value'] == 'hunter2'


def test_a_declined_batch_keeps_its_locked_answers():
    from gateway.session_server_requests import ask_viewer_result, lock_answer, resolve
    events, peer = _events()
    out = {}
    questions = [{'qid': 'q0', 'question': 'colour?'}, {'qid': 'q1', 'question': 'size?'}]
    worker = threading.Thread(target=lambda: out.setdefault('result', ask_viewer_result(
        events, 's1', 'clarify', {'questions': questions}, qids=['q0', 'q1'])), daemon=True)
    worker.start()
    _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
    request = next(f for f in peer.frames if f.get('method') == 'clarify')
    assert lock_answer(request['id'], 'q0', 'red', session_ok=lambda session_id, events: True) == ['q1']
    assert resolve(_abstain(request['id']), transport=peer)
    worker.join(5)
    assert out['result'] == {'answers': {'q0': 'red'}, 'outcome': 'cancelled'}


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
    from gateway import session_server_requests
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    # Only a viewer attached to the session gets the request frame, and only it may answer.
    await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
    pending = session_server_requests._Pending('s', events=authority.sessions['s'].event_stream)
    session_server_requests._open['srq-abc'] = pending
    try:
        reply = await connection.dispatch({'jsonrpc': '2.0', 'id': 'srq-abc', 'result': {'value': 'pw'}})
        assert reply is None
        assert pending.event.is_set() and pending.value == 'pw'
    finally:
        session_server_requests._open.pop('srq-abc', None)
        await connection.close()


@pytest.mark.asyncio
async def test_an_abstention_names_the_transport_the_session_attached(tmp_path, monkeypatch):
    """``_ask`` compares abstentions with the session's fan-out peers, so the connection must record the
    very transport ``session.resume`` attached."""
    from gateway import session_server_requests
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    transport = object()
    connection = AuthorityConnection(authority, transport, {'user_id': 'owner'})
    await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
    events = authority.sessions['s'].event_stream
    pending = session_server_requests._Pending('s', events=events)
    session_server_requests._open['srq-abc'] = pending
    try:
        assert await connection.dispatch(_abstain('srq-abc')) is None
        assert not pending.event.is_set() and pending.abstained == {transport}
        assert transport in events.fanout.transports()
    finally:
        session_server_requests._open.pop('srq-abc', None)
        await connection.close()
