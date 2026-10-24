"""Regression for #143587: gateway /compress must not rewrite a plugin engine's history."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent.context_engine import ContextEngine
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from hermes_cli.gateway_mutations import slash_mutation
import hermes_state_runtime as rt


class SentinelEngine(ContextEngine):
    name = 'sentinel'

    def __init__(self):
        self.calls = []

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, **kwargs):
        self.calls.append('compress')
        raise AssertionError('plugin compression cannot be rolled back by the gateway')

    def on_session_start(self, session_id, **kwargs):
        self.calls.append('notify')


@asynccontextmanager
async def engine_session(tmp_path, monkeypatch, *, frozen='compressor', live='compressor',
                         resident=None, managed=False, discovered=True, installed=False):
    """Real gateway mutation with isolated live profile config and a peek-only plugin registry."""
    from agent.context_compressor import ContextCompressor
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_constants import get_hermes_home
    from hermes_cli import plugins
    import plugins.context_engine as engines

    config = {'context': {'engine': frozen}}
    if managed:
        config['gateway'] = {'managed_workers': True}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: config)
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(f'context:\n  engine: {live!r}\n')
    engine, side_effects = SentinelEngine(), []
    manager = SimpleNamespace(_context_engine=engine if installed else None, _discovered=discovered)
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)

    def no_load(*args, **kwargs):
        raise AssertionError('engine decision must not load, construct or clone a plugin')

    monkeypatch.setattr(engines, 'load_context_engine', no_load)
    monkeypatch.setattr(plugins, 'get_plugin_context_engine', no_load)
    monkeypatch.setattr(engine, 'clone_for_agent', no_load)
    if installed:
        plugin = home / 'plugins' / 'sentinel'
        plugin.mkdir(parents=True)
        (plugin / '__init__.py').write_text('from agent.context_engine import ContextEngine\n'
                                          'raise AssertionError("plugin must not be imported")\n')

    def native_compress(self, messages, **kwargs):
        side_effects.append('native')
        if resident == 'plugin' or (resident is None and (frozen if managed else live) == 'sentinel' and installed):
            raise AssertionError('native compression must never replace the configured engine')
        return messages  # exercise the native no-change path without a provider

    def memory_hook(*args, **kwargs):
        side_effects.append('memory')
        return ''

    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', native_compress)
    agent = None if resident is None else SimpleNamespace(
        context_compressor=engine if resident == 'plugin' else object.__new__(ContextCompressor),
        _memory_manager=SimpleNamespace(on_pre_compress=memory_hook))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda route: side_effects.append('evict'),
        _cached_agent_for=lambda route: agent,
        _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='engine', source='acp',
        cwd=str(tmp_path), model='frozen', toolsets=[]))
    for i in range(4):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}')
    before = db.get_session(ref.session_id)
    history = db.get_messages(ref.session_id)
    stream = authority.sessions[ref.session_id].event_stream
    watermark = stream.watermark()

    async def compress(raw='', request_id='compress'):
        operation, payload = slash_mutation('/compress', raw)
        return await owner.dispatch({'id': request_id, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': request_id,
            'expected_revision': before['runtime_revision'],
            'expected_generation': before['runtime_generation'], 'operation': operation, 'payload': payload}})

    try:
        yield compress, side_effects
        assert engine.calls == []
        after = db.get_session(ref.session_id)
        assert (after['runtime_revision'], after['runtime_generation']) == (
            before['runtime_revision'], before['runtime_generation'])
        assert db.get_messages(ref.session_id) == history
        assert not db.list_meta_prefix('gateway.mutation.v1.')
        assert stream.since(*watermark)['events'] == []
    finally:
        await owner.close()
        store.close_all_db_handles()


def assert_refusal(reply):
    result = reply.get('result', {})
    assert result.get('status') == 'noop', reply
    token_line = result['summary']['token_line']
    assert isinstance(token_line, str) and token_line
    assert token_line.startswith('Approx request size: ~')
    assert token_line.endswith(' tokens (unchanged)')
    assert result['summary'] == {
        'noop': True, 'aborted': False, 'refused_would_grow': False, 'fallback_used': False,
        'headline': 'Manual /compress is unavailable for the configured context engine.',
        'token_line': token_line,
        'note': 'Gateway compression cannot yet commit plugin state transactionally. History is unchanged.'}


@pytest.mark.asyncio
@pytest.mark.parametrize('resident', [None, 'plugin'])
@pytest.mark.parametrize('args', ['', 'here 2'])
async def test_plugin_compress_refuses_before_side_effects(tmp_path, monkeypatch, resident, args):
    async with engine_session(tmp_path, monkeypatch, frozen='compressor', live='sentinel',
                              resident=resident, installed=True) as (compress, side_effects):
        for raw, rid in ((args + ' --preview', 'preview'), (args, 'compress'), (args, 'retry')):
            assert_refusal(await compress(raw.strip(), rid))
        assert side_effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize('resident', [None, 'native'])
@pytest.mark.parametrize('live', ['compressor', 'Compressor', ' compressor', 'not-installed'])
async def test_live_native_or_unresolved_engine_uses_native_fallback(tmp_path, monkeypatch, resident, live):
    async with engine_session(tmp_path, monkeypatch, frozen='sentinel', live=live,
                              resident=resident) as (compress, side_effects):
        reply = await compress()
        assert reply['result']['summary']['noop'] is True
        assert 'unavailable' not in reply['result']['summary']['headline']
        assert 'native' in side_effects


@pytest.mark.asyncio
@pytest.mark.parametrize('frozen, live, refused', [('sentinel', 'compressor', True),
                                                ('compressor', 'sentinel', False)])
async def test_managed_worker_uses_frozen_engine(tmp_path, monkeypatch, frozen, live, refused):
    async with engine_session(tmp_path, monkeypatch, frozen=frozen, live=live,
                              managed=True, installed=True) as (compress, side_effects):
        reply = await compress()
        if refused:
            assert_refusal(reply)
            assert side_effects == []
        else:
            assert 'native' in side_effects


@pytest.mark.asyncio
async def test_resident_engine_wins_over_live_config(tmp_path, monkeypatch):
    async with engine_session(tmp_path, monkeypatch, live='compressor', resident='plugin') as (compress, effects):
        assert_refusal(await compress())
        assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize('live', ['sentinel', 'compressor'])
async def test_cold_unknown_plugin_load_refuses_but_plain_native_does_not(tmp_path, monkeypatch, live):
    async with engine_session(tmp_path, monkeypatch, live=live, discovered=False) as (compress, effects):
        reply = await compress()
        if live == 'sentinel':
            assert_refusal(reply)
            assert effects == []
        else:
            assert 'native' in effects
