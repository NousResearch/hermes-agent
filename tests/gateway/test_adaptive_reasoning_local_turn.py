"""``agent.adaptive_reasoning`` on a LOCAL session turn (classic CLI, Ink, Desktop): the owner's TurnRunner
and a managed worker both run it, from the session's policy config, and neither adjusts a level the session
chose.

The classic CLI no longer builds the agent that runs its turn (``hermes_cli/gateway_agent.py`` is a proxy), so
the shell-side provenance never reaches it: the owner reads the session's own policy. One ordinary daemon,
an in-process and a ``gateway.managed_workers`` session from the same profile; the loopback provider records
the ``reasoning_effort`` each turn sent.
"""
import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc
from tests.gateway.test_managed_worker_construct import run_pair
from tests.waiting import async_wait_until

DEBUG = 'Why does the gateway keep failing after I restart it?'  # lone code signal: high
PLAIN = 'An ambiguous follow-up'


def _adaptive_profile(home):
    return {'agent': {'reasoning_effort': 'medium', 'adaptive_reasoning': {'enabled': True, 'max_effort': 'high'}}}


async def _settle(ws, sid, request_id, text, **submit):
    """Run one turn and return the result the owner committed for it.

    Waits on the turn's receipt, not its admission row: until the turn's completion is published the
    owner reports a terminal row as its started claim, with no result (``receipt`` in
    gateway/session_authority.py). ``run_pair``'s own ``settle`` reads the receipt once, as soon as the
    row is terminal, so under load it can miss the result; the turns here use this instead."""
    submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=request_id, text=text, **submit)
    assert 'result' in submitted, submitted
    admission = submitted['result']['admission_id']

    async def terminal():
        receipt = await rpc(ws, 'prompt.receipt', session_id=sid, admission_id=admission, include_result=True)
        assert 'result' in receipt, receipt
        return receipt['result'] if receipt['result']['status'] == 'terminal' else {}
    settled = await async_wait_until(terminal, timeout=120, interval=.05, what=f'admission {request_id} receipt terminal')
    assert 'result' in settled, settled  # stored in the same write as the terminal status
    return settled['result']


async def _two_turns(ws, sid, name, _row_settle, servers):
    efforts = []
    for index, text in enumerate((DEBUG, PLAIN)):
        start = len(servers.primary.requests)
        assert (await _settle(ws, sid, f'{name}-{index}', text))['final_response'] == 'PRIMARY_OK'
        sent = [r for r in servers.primary.requests[start:]
                if any(text in str(m.get('content')) for m in r['messages'] if m.get('role') == 'user')]
        efforts.append([r.get('reasoning_effort') for r in sent])
    return {'efforts': efforts}


@pytest.mark.platforms("linux")
def test_local_session_turn_adapts_then_restores_the_baseline(tmp_path):
    results = run_pair(tmp_path, _adaptive_profile, _two_turns)
    for name in ('inproc', 'managed'):
        assert results[name]['efforts'] == [['high'], ['medium']], (name, results)
    assert (results['inproc']['workers'], results['managed']['workers']) == (0, 2), results


@pytest.mark.platforms("linux")
def test_local_session_reasoning_pick_stops_adjusting_and_reset_resumes(tmp_path):
    """A committed session ``/reasoning <level>`` (``config.set reasoning``, what Ink, Desktop and the classic
    CLI send) pins the session: the owner clears the typed override once the pick is the policy's, so only
    the policy's own record says so. ``/reasoning reset`` gives the level, and adaptation, back."""
    async def turn(ws, sid, name, _row_settle, servers):
        async def debug_turn(tag):
            start = len(servers.primary.requests)
            assert (await _settle(ws, sid, f'{name}-{tag}', DEBUG))['final_response'] == 'PRIMARY_OK'
            return [r.get('reasoning_effort') for r in servers.primary.requests[start:]]

        efforts = [await debug_turn('adapt')]
        for value, tag in (('low', 'pinned'), ('reset', 'reset')):
            changed = await rpc(ws, 'config.set', session_id=sid, key='reasoning', value=value)
            assert 'result' in changed, changed
            efforts.append(await debug_turn(tag))
        return {'efforts': efforts}
    results = run_pair(tmp_path, _adaptive_profile, turn)
    for name in ('inproc', 'managed'):
        assert results[name]['efforts'] == [['high'], ['low'], ['high']], (name, results)


@pytest.mark.platforms("linux")
def test_local_one_turn_reasoning_pick_is_never_adjusted(tmp_path):
    """``/model <model> --once --reasoning low`` (the admission's ``model_override``, what the classic CLI sends)
    is that turn's own level: the owner sends it as is on a debugging turn, and the next turn adapts the
    session's level again. In process only: a managed worker's bootstrap carries no one-turn route
    (``_bootstrap`` in gateway/session_managed_worker.py, the same before adaptation), so that session runs no turn."""
    async def turn(ws, sid, name, _row_settle, servers):
        if name == 'managed':
            return {}

        def sent(start):
            return [r.get('reasoning_effort') for r in servers.primary.requests[start:]
                    if any(DEBUG in str(m.get('content')) for m in r['messages'] if m.get('role') == 'user')]
        start = len(servers.primary.requests)
        once = await _settle(ws, sid, f'{name}-once', DEBUG, model_override={'model': 'primary-model', 'reasoning': 'low'})
        assert once['final_response'] == 'PRIMARY_OK', once
        efforts = [sent(start)]
        start = len(servers.primary.requests)
        assert (await _settle(ws, sid, f'{name}-next', DEBUG))['final_response'] == 'PRIMARY_OK'
        return {'efforts': efforts + [sent(start)]}
    results = run_pair(tmp_path, _adaptive_profile, turn)
    assert results['inproc'] == {'efforts': [['low'], ['high']], 'workers': 0}, results


@pytest.mark.platforms("linux")
def test_api_request_level_is_never_adjusted(tmp_path):
    """An API request (``POST /api/sessions/<id>/chat``) runs its own level, like cron/batch: on the same
    adaptive profile the debugging prompt goes out at the session's medium, never adjusted to high
    (``api is not None`` in ``reasoning_explicit``, gateway/run_turn_runner.py)."""
    import asyncio
    import json
    import os
    import socket
    from pathlib import Path

    import aiohttp

    from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
    from tests.gateway.test_managed_worker_construct import Primary, _serve
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    primary = _serve(Primary)
    purl = f'http://127.0.0.1:{primary.server_port}/v1'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        api_port = sock.getsockname()[1]
    config = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': purl},
              'gateway': {'multiplex_profiles': False,
                          'platforms': {'api_server': {'enabled': True, 'port': api_port, 'host': '127.0.0.1'}}},
              'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
              'platform_toolsets': {'api_server': []}, **_adaptive_profile(home)}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = {**child_env(), **{k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=purl, API_SERVER_KEY='adaptive-api-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(api_port), PYTHONUNBUFFERED='1')

    async def exercise():
        headers = {'Authorization': 'Bearer adaptive-api-secret'}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as client:
            base = f'http://127.0.0.1:{api_port}/api/sessions'
            async with client.post(base, json={'id': 'api', 'title': 'API'}, headers=headers) as response:
                assert response.status in (200, 201), await response.text()
            async with client.post(f'{base}/api/chat', headers=headers, json={'message': DEBUG}) as response:
                return response.status, await response.text()
    try:
        with daemon(root, home, env, barrier=False):
            status, text = asyncio.run(exercise())
    finally:
        primary.shutdown()
        primary.server_close()
    assert status == 200 and 'PRIMARY_OK' in text, (status, text)
    sent = [r.get('reasoning_effort') for r in primary.requests
            if any(DEBUG in str(m.get('content')) for m in r['messages'] if m.get('role') == 'user')]
    assert sent == ['medium'], sent


@pytest.mark.platforms("linux")
def test_local_session_launch_level_is_never_adjusted(tmp_path):
    """``--reasoning low`` / a Desktop pick distinct from the default (``session.create reasoning``) is the
    session's own choice: the debugging turn stays at it."""
    results = run_pair(tmp_path, _adaptive_profile, _two_turns, reasoning='low')
    for name in ('inproc', 'managed'):
        assert results[name]['efforts'] == [['low'], ['low']], (name, results)
