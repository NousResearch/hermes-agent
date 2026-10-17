"""Per-session setting commands run their real gateway handler for a LOCAL session.

``slash.exec`` dispatches the same ``_handle_<name>_command`` a messaging chat runs; for a local
session the setting lands where main's CLI/TUI put it (the session, or the profile's config.yaml
exactly where main wrote it), and ``config.get`` reads it back.
"""
import asyncio
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


def _run(tmp_path, steps, extra_config=None):
    """``steps``: list of (label, method, params) run in order on one local session; returns
    ({label: result-or-error}, home)."""
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False},
              'model': {'provider': 'custom', 'default': 'settings-model', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}}, **(extra_config or {})}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='settings', source='cli', toolsets=[], cwd=str(work))
            sid = created['result']['session_id']
            out['sid'] = sid
            for label, method, params in steps:
                if method == 'turn':
                    await rpc(ws, 'prompt.submit', session_id=sid, input_id=label, text=params['text'])
                    deadline = time.monotonic() + 45
                    while True:
                        snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
                        if not snap['running'] and not snap['pending'] and (
                                'RECOVERY_ACK_' + params['text']) in json.dumps(snap['messages']):
                            break
                        assert time.monotonic() < deadline, ('turn never settled', snap)
                        await asyncio.sleep(.1)
                    out[label] = [r for r in peer.requests if params['text'] in json.dumps(r.get('messages'))][-1]
                    continue
                try:
                    reply = await rpc(ws, method, session_id=sid, **params)
                except TimeoutError:
                    raise AssertionError(f'{method} {params!r} did not answer within the rpc bound') from None
                result = reply.get('result')
                out[label] = ((result.get('output') if method == 'slash.exec' else result)
                              if result is not None else 'ERR:' + reply.get('error', {}).get('message', ''))
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
    return out, home, config


def _exec(label, command):
    return (label, 'slash.exec', {'command': command})


def _turn(label, text):
    return (label, 'turn', {'text': text})


def _get(label, key):
    return (label, 'config.get', {'key': key})


def _yaml(home):
    from hermes_yaml import safe_load
    return safe_load((home / 'config.yaml').read_text())


@pytest.mark.platforms("linux")
def test_local_session_settings_run_the_messaging_handlers_and_read_back_through_config_get(tmp_path):
    steps = [
        _exec('reasoning', 'reasoning'), _exec('reasoning high', 'reasoning high'), _get('get high', 'reasoning'),
        _exec('reasoning show', 'reasoning show'), _get('get show', 'show_reasoning'),
        _exec('reasoning status', 'reasoning'),
        _turn('turn high', 'TURN_HIGH'),
        _exec('reasoning --global low', 'reasoning --global low'), _get('get low', 'reasoning'),
        _turn('turn low', 'TURN_LOW'),
        _exec('reasoning reset', 'reasoning reset'), _get('get reset', 'reasoning'),
        _exec('busy queue', 'busy queue'), _get('get busy', 'busy'), _exec('busy', 'busy'),
        _get('verbose before', 'verbose'), _exec('verbose', 'verbose'), _get('get verbose', 'verbose'),
        _exec('verbose 2', 'verbose'), _get('get verbose 2', 'verbose'),
        _exec('yolo', 'yolo'), _get('get yolo', 'yolo'), _exec('yolo off', 'yolo'), _get('get yolo off', 'yolo'),
        _exec('footer status', 'footer status'), _exec('footer on', 'footer on'), _exec('footer status 2', 'footer status'),
        _exec('voice on', 'voice on'), _exec('voice status', 'voice status'), _exec('voice off', 'voice off'),
        _exec('approvals smart', 'approvals smart'), _get('get approvals', 'approvals.mode'),
        _exec('personality', 'personality'), _exec('personality concise', 'personality concise'),
        _exec('codex-runtime', 'codex-runtime'), _exec('fast', 'fast'),
    ]
    out, home, config = _run(tmp_path, steps)
    saved = _yaml(home)
    # /reasoning: a session level is this session's next turn, config.yaml untouched; --global saves
    # AND applies to this session (which never re-reads config.yaml); reset falls back to its policy.
    assert 'medium' in out['reasoning'] and 'Scope' in out['reasoning'], out['reasoning']
    assert 'session only' in out['reasoning high'] and out['get high']['value'] == 'high', out
    assert out['get show']['value'] == '1' and 'Display:** on' in out['reasoning status'], out
    assert 'high' in out['reasoning status'] and 'session override' in out['reasoning status'], out['reasoning status']
    # ...and the next turn's request carries it (the frozen session never re-reads config.yaml).
    assert out['turn high'].get('reasoning_effort') == 'high', out['turn high']
    assert out['turn low'].get('reasoning_effort') == 'low', out['turn low']
    assert 'saved to config' in out['reasoning --global low'] and out['get low']['value'] == 'low', out
    assert saved['agent']['reasoning_effort'] == 'low'
    assert out['get reset']['value'] == 'medium', out['get reset']
    # /busy: saved like the messaging handler and applied to this session.
    assert '`queue`' in out['busy queue'] and out['get busy']['value'] == 'queue', out
    assert '`queue`' in out['busy'] and saved['display']['busy_input_mode'] == 'queue', out['busy']
    # /verbose: no messaging opt-in gate; cycles this session's mode (all -> verbose -> off).
    assert out['verbose before']['value'] == 'all'
    assert 'VERBOSE' in out['verbose'] and out['get verbose']['value'] == 'verbose', out
    assert out['get verbose 2']['value'] == 'off', out
    # /yolo: this session's bypass, read back by config.get.
    assert 'ON' in out['yolo'] and out['get yolo']['value'] == '1', out
    assert 'OFF' in out['yolo off'] and out['get yolo off']['value'] == '0', out
    # /footer: `status` reports without toggling (the handler parsed its argument from a missing field).
    assert 'OFF' in out['footer status'] and 'ON' in out['footer on'] and 'ON' in out['footer status 2'], out
    assert saved['display']['runtime_footer']['enabled'] is True
    # /voice: the LOCAL chat's voice state (playback stays in the client).
    assert 'On' in out['voice status'] and out['voice off'], out
    # /approvals and /personality: profile settings, saved to config.yaml as on every surface.
    assert 'smart' in out['approvals smart'] and out['get approvals']['value'] == 'smart', out
    assert saved['approvals']['mode'] == 'smart' and saved['display']['personality'] == 'concise'
    assert 'concise' in out['personality'] and 'concise' in out['personality concise'], out
    assert 'openai_runtime' in out['codex-runtime'], out['codex-runtime']
    # /fast on a custom endpoint: the handler's own refusal, not unsupported_command.
    assert '/fast' in out['fast'] and not out['fast'].startswith('ERR:'), out['fast']
    assert saved['model'] == config['model']


def _set(label, key, value, **extra):
    return (label, 'config.set', {'key': key, 'value': value, **extra})


@pytest.mark.platforms("linux")
def test_config_set_serves_the_settings_desktop_and_ink_write_through_the_same_handlers(tmp_path):
    steps = [
        _set('global high', 'reasoning', 'high', scope='global'), _get('get global', 'reasoning'),
        _turn('turn global', 'TURN_GLOBAL'),
        _set('hide', 'reasoning', 'hide'), _get('get hide', 'show_reasoning'),
        _set('approvals', 'approvals.mode', 'smart'), _get('get approvals', 'approvals.mode'),
        _set('bad approvals', 'approvals.mode', 'nope'),
        _set('zap on', 'yolo', '1', scope='global'), _get('get approvals zap', 'approvals.mode'),
        _set('zap off', 'yolo', '0', scope='global'),
        _set('tips', 'display.in_app_tips', 'false'), _set('engine', 'voice.voice_chat_mode', 'gpt-live'),
        _set('bad engine', 'voice.voice_chat_mode', 'loud'),
    ]
    out, home, _ = _run(tmp_path, steps)
    saved = _yaml(home)
    assert out['global high'] == {'key': 'reasoning', 'value': 'high', 'scope': 'global'}, out['global high']
    assert out['get global']['value'] == 'high' and saved['agent']['reasoning_effort'] == 'high', out
    assert out['turn global'].get('reasoning_effort') == 'high', out['turn global']
    assert out['hide']['value'] == 'hide' and out['get hide']['value'] == '0', out
    assert out['get approvals']['value'] == 'smart' and out['bad approvals'] == 'ERR:invalid_params', out
    assert isinstance(out['zap on'], dict) and out['zap on']['value'] == '1', out['zap on']
    assert out['get approvals zap']['value'] == 'off', out['get approvals zap']
    assert out['zap off']['value'] == '0' and saved['approvals']['mode'] == 'manual', out
    assert saved['display']['in_app_tips'] is False and saved['voice']['voice_chat_mode'] == 'gpt-live'
    assert out['bad engine'] == 'ERR:invalid_params', out['bad engine']
