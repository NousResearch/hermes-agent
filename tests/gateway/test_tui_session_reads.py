"""A TUI session's /status, /context and /cron on the shared owner read as main's TUI sidecar did.

Main's TUI answered these itself (``session.status``: ``Hermes TUI Status``; ``/context``:
``_format_live_context_output`` with the category table; ``/cron``: the classic CLI's overview in
the slash worker). Through the owner a TUI session got the messaging gateway's markdown instead
(FINDINGS rows 17 / 19 / 21). Other surfaces keep the gateway's handlers.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'model: {default: fake-model, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._adapters_for_profile = lambda profile: runner.adapters
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']
    yield SimpleNamespace(call=call, home=home, authority=authority, workdir=tmp_path)


async def _exec(owner, sid, command):
    return (await owner.call('slash.exec', session_id=sid, command=command))['output']


@pytest.mark.asyncio
async def test_tui_status_is_main_tui_status(owner):
    sid = (await owner.call('session.create', request_id='t', source='tui', cwd=str(owner.workdir)))['session_id']
    out = await _exec(owner, sid, 'status')
    lines = out.splitlines()
    assert lines[0] == 'Hermes TUI Status' and lines[1] == '', out
    assert lines[2].startswith('Session ID: ') and lines[3].startswith('Path: ')
    labels = [line.split(':')[0] for line in lines[4:]]
    assert labels == ['Model', 'Created', 'Last Activity', 'Tokens', 'Agent Running'], out
    assert 'Model: fake-model (custom)' in out and 'Agent Running: No' in out
    assert '**' not in out and 'Gateway Status' not in out


@pytest.mark.asyncio
async def test_tui_context_is_main_live_context(owner):
    sid = (await owner.call('session.create', request_id='t', source='tui', cwd=str(owner.workdir)))['session_id']
    out = await _exec(owner, sid, 'context')
    assert out.splitlines()[:4] == ['Conversation is empty (no messages yet).',
                                    '  user: 0, assistant: 0, tool: 0, system: 0',
                                    'Model: fake-model', 'Provider: auto'], out
    assert 'Context Window' not in out


@pytest.mark.asyncio
async def test_tui_context_with_a_live_agent_renders_the_category_table(owner, monkeypatch):
    sid = (await owner.call('session.create', request_id='t', source='tui', cwd=str(owner.workdir)))['session_id']
    from agent import context_breakdown
    agent = SimpleNamespace(model='fake-model', provider='custom')
    owner.authority.runner._cached_agent_for = lambda route: agent
    monkeypatch.setattr(context_breakdown, 'compute_session_context_breakdown', lambda a, m: {
        'categories': [{'id': 'system', 'label': 'System prompt', 'tokens': 3000, 'color': '#fff'}],
        'context_max': 128000, 'context_used': 120, 'context_percent': 0, 'estimated_total': 3000,
        'context_estimated': False, 'context_source': 'provider_usage', 'model': 'fake-model'})
    out = await _exec(owner, sid, 'context')
    assert 'Estimated usage by category' in out and 'System prompt' in out, out
    assert out.splitlines()[0] == 'Conversation is empty (no messages yet).'


@pytest.mark.asyncio
async def test_tui_cron_is_the_classic_overview_with_the_profile_jobs(owner):
    from cron.jobs import create_job
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(owner.home))
    try:
        create_job('Summarize the news', '0 9 * * *', name='daily-digest')
    finally:
        reset_hermes_home_override(token)
    sid = (await owner.call('session.create', request_id='t', source='tui', cwd=str(owner.workdir)))['session_id']
    out = await _exec(owner, sid, 'cron')
    assert '(^_^) Scheduled Tasks' in out and '/cron list' in out, out
    assert '0 9 * * *' in out and 'Summarize the news' in out, out
    listed = await _exec(owner, sid, 'cron list')
    assert 'Scheduled Jobs:' in listed and 'daily-digest' in listed, listed


@pytest.mark.asyncio
async def test_tui_cron_writes_stay_refused(owner):
    sid = (await owner.call('session.create', request_id='t', source='tui', cwd=str(owner.workdir)))['session_id']
    from gateway.session_tui_reads import _cron_read
    assert _cron_read('pause abc') is None and _cron_read('add "every 2h" x') is None
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(owner.authority, object(), {'user_id': 'human'})
    reply = await connection.dispatch({'id': 9, 'method': 'slash.exec',
                                       'params': {'session_id': sid, 'command': 'cron remove abc'}})
    assert reply.get('error', {}).get('message') == 'unsupported_command', reply


@pytest.mark.asyncio
async def test_other_surfaces_keep_the_gateway_handlers(owner):
    from gateway.session_tui_reads import tui_read
    for source in ('cli', 'gui', 'acp'):
        policy = SimpleNamespace(source=source)
        assert tui_read(policy, 'status', '') is None and tui_read(policy, 'context', '') is None
    assert tui_read(None, 'status', '') is None


@pytest.mark.asyncio
async def test_context_breakdown_verb_serves_the_owner_agents_categories(owner, monkeypatch):
    """The classic CLI's /context (and Desktop's popover) read the breakdown of the gateway's own
    agent through ``session.context_breakdown``; before the agent exists it is the empty shape."""
    sid = (await owner.call('session.create', request_id='c', source='cli', cwd=str(owner.workdir)))['session_id']
    empty = await owner.call('session.context_breakdown', session_id=sid)
    assert empty['categories'] == [] and empty['estimated_total'] == 0
    from agent import context_breakdown, context_file_sources
    agent = SimpleNamespace(model='fake-model', provider='custom')
    owner.authority.runner._cached_agent_for = lambda route: agent
    monkeypatch.setattr(context_breakdown, 'compute_session_context_breakdown', lambda a, m: {
        'categories': [{'id': 'system', 'label': 'System prompt', 'tokens': 3000, 'color': '#fff'}],
        'context_max': 128000, 'context_used': 120, 'context_percent': 0, 'estimated_total': 3000,
        'context_estimated': False, 'context_source': 'provider_usage', 'model': 'fake-model'} if a is agent else {})
    monkeypatch.setattr(context_file_sources, 'context_file_sources_for_agent', lambda a: [])
    live = await owner.call('session.context_breakdown', session_id=sid)
    assert live['categories'][0]['tokens'] == 3000 and live['model'] == 'fake-model', live


@pytest.mark.asyncio
async def test_context_breakdown_is_reused_until_the_transcript_moves(owner, monkeypatch):
    """#140667: Desktop re-polls ``session.context_breakdown`` on every idle mount and reconnect,
    and each poll rebuilt the prompt over the whole stored transcript (seconds on a 9,501-row
    session). The owner reuses the payload while the transcript revision is unchanged and
    recomputes once a row lands."""
    sid = (await owner.call('session.create', request_id='r', source='cli', cwd=str(owner.workdir)))['session_id']
    from agent import context_breakdown, context_file_sources
    agent = SimpleNamespace(model='fake-model', provider='custom')
    owner.authority.runner._cached_agent_for = lambda route: agent
    computed = []

    def compute(a, messages):
        computed.append(len(messages))
        return {'categories': [{'id': 'conversation', 'label': 'Conversation', 'tokens': len(messages),
                                'color': '#fff'}], 'context_max': 128000, 'context_used': 1,
                'context_percent': 0, 'estimated_total': len(messages), 'context_estimated': True,
                'context_source': 'local_estimate', 'model': 'fake-model'}
    monkeypatch.setattr(context_breakdown, 'compute_session_context_breakdown', compute)
    monkeypatch.setattr(context_file_sources, 'context_file_sources_for_agent', lambda a: [])
    first = await owner.call('session.context_breakdown', session_id=sid)
    again = await owner.call('session.context_breakdown', session_id=sid)
    assert computed == [0] and again == first
    from gateway.session_tui_reads import _stored_id
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    owner.authority.db.append_message(stored, 'user', 'hello')
    moved = await owner.call('session.context_breakdown', session_id=sid)
    assert computed == [0, 1] and moved['estimated_total'] == 1
    await owner.call('session.context_breakdown', session_id=sid)
    assert computed == [0, 1]


@pytest.mark.asyncio
async def test_context_breakdown_follows_an_in_place_row_rewrite(owner, monkeypatch):
    """The turn prologue rewrites an already-written user row in place (``set_user_message_content``:
    an @-file expansion, multimodal context), which moves neither the row count nor the highest id.
    A breakdown cached before the rewrite must not keep the old size (probe by @dokterdok)."""
    sid = (await owner.call('session.create', request_id='r', source='cli', cwd=str(owner.workdir)))['session_id']
    from agent import context_breakdown, context_file_sources
    agent = SimpleNamespace(model='fake-model', provider='custom')
    owner.authority.runner._cached_agent_for = lambda route: agent

    def compute(a, messages):
        chars = sum(len(str(m.get('content') or '')) for m in messages)
        return {'categories': [], 'context_max': 128000, 'context_used': chars, 'context_percent': 0,
                'estimated_total': chars, 'context_estimated': True, 'context_source': 'local_estimate',
                'model': 'fake-model'}
    monkeypatch.setattr(context_breakdown, 'compute_session_context_breakdown', compute)
    monkeypatch.setattr(context_file_sources, 'context_file_sources_for_agent', lambda a: [])
    from gateway.session_tui_reads import _stored_id
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    db = owner.authority.db
    row_id = db.append_message(stored, 'user', 'explain @file:big.py')
    before = await owner.call('session.context_breakdown', session_id=sid)
    expanded = 'explain @file:big.py\n' + 'x' * 20000
    assert db.set_user_message_content(stored, row_id, expanded) == 1
    after = await owner.call('session.context_breakdown', session_id=sid)
    assert after['estimated_total'] == len(expanded) > before['estimated_total']
