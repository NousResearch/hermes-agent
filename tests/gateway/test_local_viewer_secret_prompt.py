"""A local viewer's turn asks for a skill's missing env var through main's masked secret panel
(``secret`` server→client request), and the gateway saves the answer in the profile ``.env``. Before,
a gateway surface only returned the "secure secret entry is not available" hint."""
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


def _turn(platform):
    from gateway.session_events import SessionEvents
    events, peer = SessionEvents(), _Peer()
    events.fanout.attach(peer)
    authority = SimpleNamespace(sessions={'s1': SimpleNamespace(event_stream=events)})
    turn = SimpleNamespace(_approval_owner=(authority, 's1', 1),
                           _ctx=SimpleNamespace(source=SimpleNamespace(platform=platform)))
    return turn, peer


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _answer_when_asked(peer, value):
    from gateway.session_sudo_prompt import resolve
    _wait_for(lambda: any(f.get('method') == 'secret' for f in peer.frames))
    request = next(f for f in peer.frames if f.get('method') == 'secret')
    resolve({'jsonrpc': '2.0', 'id': request['id'], 'result': {'value': value}})
    return request


def test_local_viewer_turn_prompts_and_saves_the_secret(monkeypatch):
    from gateway.session_secret_prompt import secret_prompt_scope
    from tools.skills_tool_setup import _capture_required_environment_variables
    saved = []
    from hermes_cli import config
    monkeypatch.setattr(config, 'save_env_value_secure',
                        lambda key, value: saved.append((key, value)) or {'success': True, 'stored_as': key,
                                                                          'validated': False})
    monkeypatch.setenv('HERMES_GATEWAY_SESSION', '1')  # a gateway surface: the hint path before this fix
    monkeypatch.delenv('HERMES_INTERACTIVE', raising=False)
    turn, peer = _turn(Platform.LOCAL)
    out = {}
    with secret_prompt_scope(turn, SimpleNamespace()):
        # A tool worker thread, started the way the agent starts them (context propagated).
        from tools.thread_context import propagate_context_to_thread
        worker = threading.Thread(target=propagate_context_to_thread(lambda: out.update(
            _capture_required_environment_variables('hab-skill', [{'name': 'HAB_KEY', 'prompt': 'Enter the key'}]))))
        worker.start()
        request = _answer_when_asked(peer, 'sekrit')
        worker.join(5)
    assert request['params'] == {'session_id': 's1', 'env_var': 'HAB_KEY', 'prompt': 'Enter the key',
                                 'metadata': {'skill_name': 'hab-skill'}}
    from tui_gateway.contracts.registry import SERVER_REQUESTS, validate_params
    assert validate_params(SERVER_REQUESTS['secret'], request['params'])[1] is None
    assert saved == [('HAB_KEY', 'sekrit')]
    assert out['missing_names'] == [] and out['gateway_setup_hint'] is None
    assert not any('sekrit' in repr(f) for f in peer.frames)


def test_skipped_secret_stays_missing(monkeypatch):
    from gateway.session_secret_prompt import secret_prompt_scope
    from tools.skills_tool_setup import _capture_required_environment_variables
    turn, peer = _turn(Platform.LOCAL)
    out = {}
    with secret_prompt_scope(turn, SimpleNamespace()):
        # A tool worker thread, started the way the agent starts them (context propagated).
        from tools.thread_context import propagate_context_to_thread
        worker = threading.Thread(target=propagate_context_to_thread(lambda: out.update(
            _capture_required_environment_variables('hab-skill', [{'name': 'HAB_KEY', 'prompt': 'Enter the key'}]))))
        worker.start()
        _answer_when_asked(peer, '')
        worker.join(5)
    assert out['missing_names'] == ['HAB_KEY'] and out['setup_skipped'] is True


@pytest.mark.parametrize('platform', [Platform.TELEGRAM])
def test_messaging_turn_keeps_the_unsupported_hint(monkeypatch, platform):
    from gateway.session_secret_prompt import secret_prompt_scope
    from tools.skills_tool_setup import _capture_required_environment_variables
    monkeypatch.setenv('HERMES_GATEWAY_SESSION', '1')
    monkeypatch.delenv('HERMES_INTERACTIVE', raising=False)
    turn, peer = _turn(platform)
    with secret_prompt_scope(turn, SimpleNamespace()):
        out = _capture_required_environment_variables('hab-skill', [{'name': 'HAB_KEY', 'prompt': 'p'}])
    assert out['gateway_setup_hint'] and peer.frames == []
