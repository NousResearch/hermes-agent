"""A teammate's ``message_agent`` to a bot its user paused gets a definite, typed no-send
(``recipient_paused``) from the receiving owner instead of queueing a turn the bot will not take;
nothing is admitted and nothing replays it. Traffic that is not a teammate's (a trusted producer
such as cron's report) is still admitted, and the resume admits teammates again."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

import hermes_yaml as yaml
from hermes_state_runtime import list_session_admissions

CHAT = 'bot-chat'
TEAMMATE = {'id': 'bot:coder', 'name': 'coder', 'is_bot': True}


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import gateway_runtime
    from hermes_cli.gateway_client import rpc_error
    from hermes_state import SessionDB
    from tools import bot_live_delivery, bot_mode_dm

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    db = store._db
    db.create_session(CHAT, source='desktop', model='m', cwd=str(tmp_path))
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
    try:
        yield SimpleNamespace(db=db, home=tmp_path, dm_dir=dm_dir, executed=executed)
    finally:
        db.close()


def _pause(owner, paused):
    meta = {'ui_meta': {'hermes-bots': {'title': 'Scribe', 'paused': paused}}}
    (owner.home / 'profile.yaml').write_text(yaml.safe_dump(meta), encoding='utf-8')


async def _send(owner, text, author=TEAMMATE):
    from tools import bot_mode_dm
    raw = await asyncio.to_thread(bot_mode_dm._start_delivery, ['hermes', '-p', 'default'], text, '@scribe',
                                  stdin_file=False, task_id=None, agent=None, profile_home=owner.home,
                                  author=author)
    return json.loads(raw)


def _admissions(owner):
    return [row['request_id'] for row in list_session_admissions(owner.db, session_id=CHAT, pending_only=False)]


@pytest.mark.asyncio
async def test_a_teammate_dm_to_a_paused_bot_is_a_typed_definite_no_send(owner):
    from tools.bot_live_delivery import read_delivery_result

    _pause(owner, True)
    refused = await _send(owner, 'Message from 🤖 Coder (@coder): review my PR')

    assert refused.get('reason') == 'recipient_paused', refused
    assert 'NOT sent' in refused['error'] and 'resumes' in refused['error'], refused
    assert _admissions(owner) == [] and owner.executed == []
    # Settled on both sides: the owner's receipt is the terminal refusal, and nothing local can replay it.
    receipt = read_delivery_result(str(owner.home), refused['delivery_id'])
    assert receipt['refused'] is True and receipt['reason'] == 'recipient_paused'
    assert list(owner.dm_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_a_paused_bot_still_takes_producer_traffic_and_resume_admits_teammates(owner):
    _pause(owner, True)
    produced = await _send(owner, 'cron report', author=None)
    assert produced.get('reason') is None and produced['status'] in {'queued', 'claimed', 'settled'}, produced

    _pause(owner, False)
    resumed = await _send(owner, 'Message from 🤖 Coder (@coder): back?')
    assert resumed.get('reason') is None and resumed['status'] in {'queued', 'claimed', 'settled'}, resumed
    assert len(_admissions(owner)) == 2


@pytest.fixture
def fresh(monkeypatch):
    """A profile with a real config.yaml and no Bot Chat yet: a first delivery mints it (the
    follow-profile suite's owner seam)."""
    from gateway import run
    from gateway.session_authorities import served_profile_name
    from hermes_constants import get_hermes_home
    from tests.gateway.test_follow_profile_followups import _profile
    from tests.gateway.test_session_follow_profile_config import _owner

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    _profile(home, 'fixture-model', 1, 'fixture-key')
    store, _runner, _authority, connection = _owner(home)
    try:
        yield SimpleNamespace(db=store._db, home=home, connection=connection, profile=served_profile_name(home))
    finally:
        store._db.close()


@pytest.mark.asyncio
async def test_a_refusal_before_the_bot_chat_exists_is_saved_and_replays_as_refused(fresh):
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import read_delivery_result

    assert fresh.db.get_session_by_title('Bot Chat') is None
    key = 'e' * 32
    params = dict(id=key, profile=fresh.profile, message='Message from 🤖 Coder (@coder): ship it?', author=TEAMMATE)
    _pause(fresh, True)
    refused = await deliver(fresh.connection, params)
    _pause(fresh, False)
    # The sender lost that answer and resends the same id after the resume: still the refusal.
    replayed = await deliver(fresh.connection, params)

    receipt = read_delivery_result(str(fresh.home), key)
    assert receipt and receipt['refused'] is True and receipt['reason'] == 'recipient_paused', receipt
    assert receipt['session_id'] == refused['session_id'] and fresh.db.get_session(receipt['session_id'])
    assert replayed.get('refused') is True and replayed['reason'] == 'recipient_paused', replayed
    assert list_session_admissions(fresh.db, session_id=receipt['session_id'], pending_only=False) == []
