"""Which registry holders fence a Bot Chat DM, and what the refusal says (HERMES 37.12).

An idle, attached authority-era Desktop pane holds its lease on a still-legacy Bot Chat for as long
as the window stays open; it re-checks the canonical ledger every turn and serializes on the durable
turn lease, so the DM adopts the row under it exactly as on an adopted chat. Any other live holder
still refuses, by name. A registry that cannot prove the holder refuses as such on both the adoption
and the legacy-ticket path, never as a held chat and never as an unknown outcome."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import pytest_asyncio

from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until

CHAT = 'old-bot-chat'


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    """The real authority, session_bot and Bot DM tool against a legacy Bot Chat row. Stand-ins as in
    test_bot_dm_held_by_session_writer.py: the RPC hop, endpoint discovery and the LLM turn."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import gateway_runtime
    from hermes_cli.active_sessions import release_active_session
    from hermes_cli.gateway_client import rpc_error
    from hermes_state import SessionDB
    from tools import bot_live_delivery, bot_mode_dm

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    db = store._db
    db.create_session(CHAT, source='desktop', model='old', cwd=str(tmp_path))
    db.set_session_title(CHAT, 'Bot Chat')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'uid:1000', 'provider': 'local', 'profile_id': 'default', 'instance_id': 'fixture',
        'capabilities': ['session:read', 'session:submit', 'session:create']})
    monkeypatch.setattr(connection, '_sibling_for', lambda profile: None)
    authority._native_legacy_transports = {connection.actor.transport_id: connection.actor}
    executed = []

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    loop = asyncio.get_running_loop()

    def rpc(method):
        def call(home, params):
            frame = asyncio.run_coroutine_threadsafe(connection.dispatch(
                {'id': 1, 'method': method, 'params': params}), loop).result()
            if 'error' in frame:
                raise rpc_error(frame['error'])
            return frame['result']
        return call

    monkeypatch.setattr(bot_live_delivery, 'authority_delivery', rpc('bot_relay.deliver'))
    monkeypatch.setattr(bot_live_delivery, 'authority_settle', rpc('bot_relay.settle'))
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint', lambda home, timeout: SimpleNamespace(
        state='ready', endpoint=SimpleNamespace(instance_id='fixture')))
    dm_dir = tmp_path / 'dm'
    dm_dir.mkdir(mode=0o700)
    monkeypatch.setattr(bot_mode_dm, '_dm_dir', lambda: dm_dir)
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery', lambda command, label, **kwargs: '{}')
    leases = []
    try:
        yield SimpleNamespace(authority=authority, db=db, home=tmp_path, executed=executed, leases=leases)
    finally:
        for lease in leases:
            release_active_session(lease)
        db.close()


async def _send(owner, text):
    from tools import bot_mode_dm
    raw = await asyncio.to_thread(bot_mode_dm._start_delivery, ['hermes', '-p', 'default'], text, '@default',
                                  stdin_file=False, task_id=None, agent=None, profile_home=owner.home)
    return json.loads(raw)


def _admissions(owner):
    return [row['request_id'] for row in list_session_admissions(owner.db, session_id=CHAT, pending_only=False)]


def _adopted(owner):
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    row = owner.db.get_session(CHAT)
    key = row.get('chat_id') or CHAT
    with owner.db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)',
                            (POLICY_PREFIX + key, LEGACY_PREFIX + key)).fetchone() is not None


def _plant_unreadable_entry(home):
    """One entry the strict schema rejects, for a session other than the Bot Chat."""
    path = home / 'runtime' / 'active_sessions.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text())['entries'] if path.exists() else []
    entries.append({'lease_id': 'f' * 32, 'session_id': 'some-other-chat', 'surface': 'cli',
                    'pid': os.getpid(), 'process_start_time': 'nan', 'started_at': 1, 'updated_at': 1})
    path.write_text(json.dumps({'entries': entries}))
    return path


def _remove_planted_entry(path):
    entries = [e for e in json.loads(path.read_text())['entries'] if e['lease_id'] != 'f' * 32]
    path.write_text(json.dumps({'entries': entries}))


def _legacy_ticket(home, chat):
    """A pre-authority mailbox ticket: an owner pin and no admission id."""
    root = home / 'runtime' / 'bot_live_delivery'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = 'a' * 64
    record = dict(id=key, delivery_id=key, status='queued', created_at=1, sequence=1, message='old queued DM',
                  owner=dict(profile_home=str(home.resolve()), session_id=chat, lease_id='0' * 32,
                             live_session_id='old-pane'))
    (root / f'{key}.json').write_text(json.dumps(record))


@pytest.mark.asyncio
async def test_an_idle_authority_desktop_pane_does_not_fence_a_legacy_bot_chat(owner):
    from tui_gateway import server
    lease, refusal = server._claim_active_session_slot(
        CHAT, live_session_id='pane-1', surface='desktop', profile_home=owner.home)
    assert lease is not None, refusal
    owner.leases.append(lease)

    sent = await _send(owner, 'ping')
    assert sent.get('status') in {'queued', 'claimed', 'settled'}, sent
    assert _adopted(owner)
    assert _admissions(owner) == ['bot:' + sent['delivery_id']]
    await async_wait_until(lambda: owner.executed == ['ping'], what='the adopted DM ran')
    assert not lease.released, 'the pane keeps its lease beside canonical delivery'


@pytest.mark.asyncio
async def test_a_dm_during_that_panes_turn_is_a_definite_no_send_then_adopts(owner):
    """The pane's running turn holds the durable turn lease: adoption is refused inside its write
    transaction, nothing is bound, and the answer names no session to close."""
    from tui_gateway import server
    lease, refusal = server._claim_active_session_slot(
        CHAT, live_session_id='pane-1', surface='desktop', profile_home=owner.home)
    assert lease is not None, refusal
    owner.leases.append(lease)
    holder = f'pid={os.getpid()}:turn=pane-1:platform=desktop'
    assert owner.db.try_acquire_session_turn_lease(CHAT, holder)

    refused = await _send(owner, 'ping')
    assert refused.get('reason') == 'runtime_coordination_required' and 'status' not in refused, refused
    assert refused['error'].endswith('the message was NOT sent; it is safe to resend later.'), refused
    assert not _adopted(owner) and _admissions(owner) == []

    owner.db.release_session_turn_lease(CHAT, holder)
    sent = await _send(owner, 'ping again')
    assert sent.get('status') in {'queued', 'claimed', 'settled'} and _adopted(owner), sent
    await async_wait_until(lambda: owner.executed == ['ping again'], what='the adopted DM ran')


@pytest.mark.asyncio
async def test_a_live_classic_cli_lease_refuses_and_is_named(owner):
    from hermes_cli.active_sessions import try_acquire_active_session
    # cli.py HermesCLI._claim_active_session: untracked, no authority marker.
    lease, refusal = try_acquire_active_session(session_id=CHAT, surface='cli', config={},
        registry_home=owner.home, metadata={'live_session_id': CHAT})
    assert lease is not None, refusal
    owner.leases.append(lease)

    refused = await _send(owner, 'ping')
    assert refused.get('reason') == 'runtime_coordination_required' and 'status' not in refused, refused
    error = refused['error']
    assert 'NOT sent' in error and f'a cli session, pid {os.getpid()}, open 0 min' in error, error
    assert error.endswith('safe to resend after that cli session closes.'), error
    assert 'desktop' not in error and 'registry' not in error, error
    assert not _adopted(owner) and _admissions(owner) == [] and owner.executed == []


@pytest.mark.asyncio
async def test_an_unprovable_registry_refuses_adoption_and_legacy_tickets_alike(owner):
    path = _plant_unreadable_entry(owner.home)
    on_adoption = await _send(owner, 'ping')
    _remove_planted_entry(path)
    bound = await _send(owner, 'bind')
    assert bound.get('status') in {'queued', 'claimed', 'settled'} and _adopted(owner), bound

    _plant_unreadable_entry(owner.home)
    _legacy_ticket(owner.home, CHAT)
    on_legacy_ticket = await _send(owner, 'pong')

    for answer in (on_adoption, on_legacy_ticket):
        assert answer.get('reason') == 'coordination_unavailable' and 'status' not in answer, answer
        assert 'NOT sent' in answer['error'] and 'registry' in answer['error'], answer
        assert 'held by another session writer' not in answer['error'], answer
        assert 'Do not resend' not in answer['error'] and 'session closes' not in answer['error'], answer
    assert on_adoption['error'] == on_legacy_ticket['error']
    assert _admissions(owner) == ['bot:' + bound['delivery_id']]
