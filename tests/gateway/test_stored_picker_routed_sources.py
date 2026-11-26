"""Regression for TUI-8 (#134889): the native picker's stored feed lists messaging and cron sessions.

Before the shared gateway the picker listed every source except the internal ones (kanban, tool,
oneshot); the stored feed listed only local labels, so a Telegram conversation or a finished cron run
could not be picked from Ink's ``/resume``. Resuming such a row is the cold mount
(``gateway/session_route_recovery.py``); this pins that the listing offers it.
"""
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _restored_source=lambda entry: entry.origin)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    authority._schedule = lambda ref: None
    runner.session_authority = authority
    return authority, store.persistence


async def _finished_cron_run(authority, db, tmp_path):
    """A cron fire through the owner's own path, then settled the way a finished run books it."""
    from cron import jobs
    from gateway import session_cron
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='m')
        receipt = await session_cron.operation(authority, 'submit', {'job_id': job['id'], 'request_id': 'fire',
                                                                     'extra_prompt': None})
    db.admissions.cancel(epoch=authority.epoch, admission_id=receipt['admission_id'])
    db.sessions.end(receipt['session_id'], 'cron_complete')
    return receipt['session_id']


def _messaging_row(db, sid, chat_id, *, profile_name=None, platform='telegram'):
    route = f'agent:main:{platform}:dm:{chat_id}'
    seed_session(db, sid, platform, model='m', user_id='111', profile_name=profile_name, session_key=route,
                 chat_id=chat_id, chat_type='dm',
                 origin={'platform': platform, 'chat_id': chat_id, 'chat_type': 'dm', 'user_id': '111'})
    seed_messages(db, sid, {'role': 'user', 'content': sid + ' question'},
                  {'role': 'assistant', 'content': sid + ' answer'})
    db.sessions.end(sid, 'session_reset')


@pytest.mark.asyncio
async def test_the_stored_feed_lists_messaging_and_cron_sessions_and_hides_internal_ones(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    authority, db = await _owner(tmp_path, monkeypatch)
    _messaging_row(db, 'tg-chat', '111')
    _messaging_row(db, 'tg-other-profile', '112', profile_name='other')
    cron_run = await _finished_cron_run(authority, db, tmp_path)
    for sid, source in [('worker', 'kanban'), ('integration', 'tool'), ('once', 'oneshot')]:
        seed_session(db, sid, source, model='m')
        seed_messages(db, sid, {'role': 'user', 'content': sid + ' question'})
        db.sessions.end(sid, 'completed')
    # The native local transport, as `gateway/run_api.py` admits it (a verified local operator).
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
                                                       'profile_id': str(tmp_path), 'instance_id': 'owner'},
                                 operator=True)

    listed = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {'scope': 'stored'}})
    assert 'result' in listed, listed
    ids = {row['id'] for row in listed['result']['sessions']}
    assert {'tg-chat', cron_run} <= ids
    assert not ids & {'worker', 'integration', 'once', 'tg-other-profile'}

    resumed = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 'tg-chat'}})
    assert 'result' in resumed, resumed
    assert [m['content'] for m in resumed['result']['messages']] == ['tg-chat question', 'tg-chat answer']


@pytest.mark.asyncio
async def test_a_compressed_messaging_conversation_is_listed_under_its_lineage_root(tmp_path, monkeypatch):
    """The listing projects a compressed lineage onto its tip; the cold mount is the root."""
    from gateway.session_controls import AuthorityConnection

    authority, db = await _owner(tmp_path, monkeypatch)
    _messaging_row(db, 'tg-root', '113')
    root = session_row(db, 'tg-root')
    seed_session(db, 'tg-tip', 'telegram', model='m', user_id='111', parent_session_id='tg-root',
                 session_key=root['session_key'], chat_id='113', chat_type='dm', origin_json=root['origin_json'])
    raw_holder(db)._execute_write(
        lambda conn: conn.execute("UPDATE sessions SET end_reason='compression' WHERE id='tg-root'"))
    seed_messages(db, 'tg-tip', {'role': 'user', 'content': 'after compression'})
    assert db.lineage.tip('tg-root') == 'tg-tip'
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
                                                       'profile_id': str(tmp_path), 'instance_id': 'owner'},
                                 operator=True)

    listed = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {'scope': 'stored'}})
    rows = [row for row in listed['result']['sessions'] if row['id'] in {'tg-root', 'tg-tip'}]
    assert [(row['id'], row['session_id']) for row in rows] == [('tg-root', 'tg-root')]
    assert rows[0]['message_count'] == 1  # the tip's segment, as every picker row reads after compression


@pytest.mark.asyncio
@pytest.mark.parametrize('platform', ['irc', 'line', 'google_chat'])
async def test_a_bundled_plugin_platform_conversation_is_listed_and_resumes(tmp_path, monkeypatch, platform):
    """Plugin platforms are ``Platform`` pseudo-members that enum iteration never yields; the cold mount
    resolves them by label, so the listing must offer the same rows."""
    from gateway.session_controls import AuthorityConnection

    authority, db = await _owner(tmp_path, monkeypatch)
    _messaging_row(db, 'plugin-chat', '114', platform=platform)
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
                                                       'profile_id': str(tmp_path), 'instance_id': 'owner'},
                                 operator=True)

    listed = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {'scope': 'stored'}})
    assert 'plugin-chat' in {row['id'] for row in listed['result']['sessions']}, listed
    resumed = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 'plugin-chat'}})
    assert 'result' in resumed, resumed
