"""``/moa <prompt>`` on a local session runs that one turn through the default MoA preset.

Regression for #147510: Desktop, the Ink TUI and ACP send ``/moa`` through ``slash.exec`` and submit the
``send`` directive it returns; ``hermes chat -q "/moa ..."`` submits the raw text. Both reached the
gateway's ``/moa`` handler, which armed the runner's per-session override that a local session's frozen
launch route never reads, so the chat's own model answered and nothing said so.

Real runner, session authority, local launch policy, admission FIFO and turn path; the agent records the
route it was built on instead of calling a model.
"""
import json
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tests.waiting import async_wait_until

PROMPT = 'What is 17 x 23? Reply with just the number.'
SESSION_ROUTE = ('custom', 'session-model')
MOA_ROUTE = ('moa', 'review')  # the profile's default preset, not the seeded ``default``


class _Agent:
    """Records the route each turn's agent was built on and the message it was asked to answer."""

    turns: list = []
    started = threading.Event()
    gate: threading.Event | None = None

    def __init__(self, **kwargs):
        self.model, self.provider = kwargs.get('model'), kwargs.get('provider')
        self.session_id = kwargs.get('session_id')
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
        self.session_prompt_tokens = self.session_completion_tokens = 0

    def run_conversation(self, message, **_kwargs):
        type(self).turns.append((message, self.provider, self.model))
        type(self).started.set()
        if type(self).gate is not None:
            type(self).gate.wait(30)
        return {'final_response': '391', 'messages': [], 'api_calls': 1}


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    import run_agent
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB

    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))  # a single-profile owner: this IS the launch home
    (tmp_path / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'session-model', 'base_url': 'http://127.0.0.1:9/v1',
                  'api_key': 'loopback-only'},
        'moa': {'default_preset': 'review', 'presets': {'review': {
            'reference_models': [{'provider': 'custom', 'model': 'ref-model'}],
            'aggregator': {'provider': 'custom', 'model': 'agg-model'}}}},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    _Agent.turns, _Agent.gate = [], None
    _Agent.started.clear()
    monkeypatch.setattr(run_agent, 'AIAgent', _Agent)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner', db=db)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        yield authority, connection, tmp_path
    finally:
        if _Agent.gate is not None:
            _Agent.gate.set()
        await connection.close()
        db.close()


def _create(authority, connection, cwd, source):
    from gateway.session_local import create_local_session
    ref = create_local_session(authority, connection.actor, {
        'request_id': source, 'source': source, 'cwd': str(cwd), 'toolsets': []})
    return ref.session_id


async def _call(connection, method, **params):
    reply = await connection.dispatch({'id': method, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


async def _submit(connection, sid, input_id, text, **extra):
    submitted = await _call(connection, 'prompt.submit', session_id=sid, submission_id=input_id, text=text, **extra)
    assert submitted.get('status') == 'queued', submitted


def _rows(authority, sid):
    from hermes_state_runtime import list_session_admissions
    return list_session_admissions(authority.db, session_id=sid, pending_only=False)


async def _settled(authority, sid, count):
    def done():
        rows = _rows(authority, sid)
        return len(rows) >= count and all(row['status'] == 'terminal' for row in rows)
    await async_wait_until(done, timeout=30, what=f'{count} settled admissions')


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['desktop-slash-exec', 'chat-q-raw-submit'])
async def test_moa_runs_one_turn_on_the_default_preset_then_the_sessions_own_route(owner, surface):
    authority, connection, cwd = owner
    sid = _create(authority, connection, cwd, 'gui' if surface == 'desktop-slash-exec' else 'oneshot')
    await _call(connection, 'session.resume', session_id=sid)
    typed, extra = f'/moa {PROMPT}', {}
    if surface == 'desktop-slash-exec':
        directive = await _call(connection, 'slash.exec', session_id=sid, command=typed)
        assert directive['type'] == 'send' and directive['display'] == typed, directive
        typed = directive['message']
    else:
        extra = {'finite': True}
    await _submit(connection, sid, 'moa', typed, **extra)
    await _settled(authority, sid, 1)
    await _submit(connection, sid, 'next', 'and again', **extra)
    await _settled(authority, sid, 2)

    assert _Agent.turns == [(PROMPT, *MOA_ROUTE), ('and again', *SESSION_ROUTE)]


@pytest.mark.asyncio
async def test_a_busy_session_still_refuses_slash_moa_and_runs_a_queued_one_on_the_preset(owner):
    """Mid-run, ``slash.exec`` keeps refusing ``/moa`` and admits nothing; a ``/moa`` submitted behind the
    running turn waits its place in the FIFO and runs the preset when it drains, once."""
    authority, connection, cwd = owner
    sid = _create(authority, connection, cwd, 'gui')
    await _call(connection, 'session.resume', session_id=sid)
    _Agent.gate = threading.Event()
    await _submit(connection, sid, 'first', 'first turn')
    await async_wait_until(_Agent.started.is_set, timeout=30, what='the first turn running')

    refused = await _call(connection, 'slash.exec', session_id=sid, command=f'/moa {PROMPT}')
    await _submit(connection, sid, 'moa', f'/moa {PROMPT}')
    assert [row['request_id'] for row in _rows(authority, sid)] == ['first', 'moa']
    _Agent.gate.set()
    await _settled(authority, sid, 2)
    await _submit(connection, sid, 'next', 'and again')
    await _settled(authority, sid, 3)

    assert refused['type'] == 'exec' and '/moa' in refused['output'], refused
    assert _Agent.turns == [('first turn', *SESSION_ROUTE), (PROMPT, *MOA_ROUTE), ('and again', *SESSION_ROUTE)]
