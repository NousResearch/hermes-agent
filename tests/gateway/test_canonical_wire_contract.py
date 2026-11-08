"""The canonical dispatcher speaks only the declared wire contract (``tui_gateway/contracts``)."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_group_controls import GROUP_METHODS
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.contracts.registry import CANONICAL_METHODS, METHODS


@pytest.fixture
def owner(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        yield AuthorityConnection(authority, object(), {'user_id': 'owner'}), db


def test_every_canonical_verb_has_a_declared_contract(owner):
    """Each name the authority answers is a ``METHODS`` (shared with the sidecar) or
    ``CANONICAL_METHODS`` declaration, so it is rendered into the generated TypeScript and
    OpenRPC; and no canonical declaration outlives its handler."""
    connection, _ = owner
    served = set(connection.handlers()) | set(GROUP_METHODS) | {'profiles.list'}
    assert sorted(served - set(METHODS) - set(CANONICAL_METHODS)) == []
    assert sorted(set(CANONICAL_METHODS) - served) == []


@pytest.mark.asyncio
async def test_declared_params_gate_canonical_verbs_before_their_handlers(owner):
    """An unknown or missing key is ``4001 invalid_params`` naming the field, including verbs
    whose handlers only ``.get`` their keys; a well-formed call is unaffected."""
    connection, db = owner
    revision = db.get_session('s')['runtime_revision']
    mutate = {'session_id': 's', 'request_id': 'r1', 'expected_revision': revision,
              'operation': 'rename', 'payload': {'title': 'Declared'}}
    try:
        for method, params, field in (
                ('prompt.receipt', {'session_id': 's', 'admission_id': 'a', 'bogus': 1}, 'bogus'),
                ('prompt.cancel', {'session_id': 's'}, 'admission_id'),
                ('session.mutate', {**mutate, 'expected_revison': revision}, 'expected_revison')):
            response = await connection.dispatch({'id': 1, 'method': method, 'params': params})
            assert response['error']['code'] == 4001, (method, response)
            assert response['error']['data']['reason'] == 'invalid_params'
            assert field in response['error']['data']['fields'], (method, response)
        applied = await connection.dispatch({'id': 2, 'method': 'session.mutate', 'params': mutate})
        CANONICAL_METHODS['session.mutate'].result.model_validate(applied['result'])
        assert db.get_session('s')['title'] == 'Declared'
    finally:
        await connection.close()


def test_handoff_methods_queue_poll_and_keep_timeout_cas_on_owner_store(monkeypatch, owner):
    """Desktop handoff uses the authority's profile store; a timeout cannot fail a claimed row."""
    import asyncio

    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig

    connection, db = owner
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id='telegram-home', name='Telegram home'))
    monkeypatch.setattr('gateway.config.load_gateway_config', lambda: config)

    async def exercise():
        try:
            request = await connection.dispatch({'id': 3, 'method': 'handoff.request',
                                                 'params': {'session_id': 's', 'platform': 'telegram'}})
            assert request.get('result') == {'queued': True, 'session_key': 's',
                                             'platform': 'telegram', 'home_name': 'Telegram home'}, request
            assert db.get_handoff_state('s')['state'] == 'pending'

            state = await connection.dispatch({'id': 4, 'method': 'handoff.state',
                                               'params': {'session_id': 's'}})
            assert state.get('result') == {'state': 'pending', 'platform': 'telegram', 'error': ''}, state

            assert db.claim_handoff('s')
            timeout = await connection.dispatch({'id': 5, 'method': 'handoff.fail',
                                                 'params': {'session_id': 's', 'error': 'timed out'}})
            assert timeout.get('result') == {'failed': False, 'state': 'running'}, timeout
            assert db.get_handoff_state('s')['state'] == 'running'

            db.complete_handoff('s')
            complete = await connection.dispatch({'id': 6, 'method': 'handoff.state',
                                                  'params': {'session_id': 's'}})
            assert complete.get('result') == {'state': 'completed', 'platform': 'telegram', 'error': ''}, complete
        finally:
            await connection.close()

    asyncio.run(exercise())


def test_handoff_request_is_serialized_against_admission_after_idle_check(monkeypatch, owner):
    """An input committed after the handler's idle snapshot wins the DB race; it cannot leave
    both a queued turn and a pending handoff for the watcher to dispatch over."""
    import asyncio

    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input

    connection, db = owner
    authority = connection.authority
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id='telegram-home', name='Telegram home'))
    monkeypatch.setattr('gateway.config.load_gateway_config', lambda: config)
    real_request_handoff = db.request_handoff

    def admission_wins_after_idle_snapshot(session_id, platform):
        # The handler has already read no pending rows; this is the competing SQLite writer.
        admit_session_input(db, epoch=authority.epoch, principal_id='race', session_id=session_id,
                            request_id='racing-input', payload={'text': 'arrived concurrently'})
        return real_request_handoff(session_id, platform)

    monkeypatch.setattr(db, 'request_handoff', admission_wins_after_idle_snapshot)

    async def exercise():
        try:
            response = await connection.dispatch({'id': 7, 'method': 'handoff.request',
                                                  'params': {'session_id': 's', 'platform': 'telegram'}})
            assert response.get('error', {}).get('data', {}).get('reason') == 'handoff_in_flight', response
            assert db.get_handoff_state('s')['state'] is None
            pending = authority._pending_rows(SessionRef(authority.profile_id, 's'))
            assert [row['request_id'] for row in pending] == ['racing-input']
        finally:
            await connection.close()

    asyncio.run(exercise())


def test_pending_handoff_does_not_claim_a_legacy_queued_turn(owner):
    """A row left by the old check/write race cannot start while its handoff is pending."""
    from hermes_state_runtime import RuntimeStoreError, admit_session_input, claim_session_input

    connection, db = owner
    epoch = connection.authority.epoch
    admit_session_input(db, epoch=epoch, principal_id='race', session_id='s',
                        request_id='legacy-race', payload={'text': 'already queued'})
    db._write_sql("UPDATE sessions SET handoff_state='pending' WHERE id='s'")

    with pytest.raises(RuntimeStoreError, match='handoff_in_flight'):
        claim_session_input(db, epoch=epoch, session_id='s')

    assert db.get_handoff_state('s')['state'] == 'pending'
    assert db._read_one("SELECT status FROM session_admissions WHERE request_id='legacy-race'")['status'] == 'queued'


@pytest.mark.parametrize('dispatch_fails', [False, True], ids=['complete', 'fail'])
def test_real_handoff_watcher_holds_admission_barrier_until_terminal_state(monkeypatch, owner, tmp_path, dispatch_fails):
    """The production watcher claims the real owner row; prompts stay refused for pending/running
    and resume only after that dispatch completes or fails."""
    import asyncio
    import sys
    import types

    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from gateway.run_adapters import GatewayAdapterLifecycleMixin
    from gateway.session_contract import SessionRef, Submission
    from hermes_state import AsyncSessionDB
    from hermes_state_runtime import RuntimeStoreError

    connection, db = owner
    authority = connection.authority
    ref = SessionRef(authority.profile_id, 's')
    authority._schedule_admitted = lambda *_args: None
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id='telegram-home', name='Telegram home'))
    monkeypatch.setattr('gateway.config.load_gateway_config', lambda: config)
    real_sleep = asyncio.sleep

    async def yielding_sleep(_seconds):
        await real_sleep(0)

    # Let the dispatch task run while keeping this self-contained (the watcher normally waits 5s).
    monkeypatch.setattr(asyncio, 'sleep', yielding_sleep)

    gateway_package = sys.modules['gateway']
    had_run_attribute = hasattr(gateway_package, 'run')
    previous_run_attribute = getattr(gateway_package, 'run', None)

    @asynccontextmanager
    async def profile_scope(_home):
        yield

    run_helpers = types.ModuleType('gateway.run')
    run_helpers._async_profile_runtime_scope = profile_scope

    async def reclaim_stale(_runner):
        return None

    run_helpers._reclaim_stale = reclaim_stale

    async def watch_scopes(_runner):
        return [(None, tmp_path)]

    run_helpers._resolve_handoff_watch_scopes = watch_scopes
    monkeypatch.setitem(sys.modules, 'gateway.run', run_helpers)

    async def submit(request_id):
        return await authority.submit(connection.actor,
                                      Submission(request_id, ref, {'text': request_id}, 'queue'))

    async def assert_blocked(request_id):
        with pytest.raises(RuntimeStoreError, match='handoff_in_flight'):
            await submit(request_id)

    async def exercise():
        try:
            queued = await connection.dispatch({'id': 8, 'method': 'handoff.request',
                                                'params': {'session_id': 's', 'platform': 'telegram'}})
            assert queued.get('result', {}).get('queued') is True, queued
            await assert_blocked('during-pending')

            watcher = SimpleNamespace(_session_db=AsyncSessionDB(db), _running=True, config=None)

            async def process_handoff(row, _profile_name=None):
                assert row['id'] == 's'
                assert (await watcher._session_db.get_handoff_state('s'))['state'] == 'running'
                await assert_blocked('during-running')
                watcher._running = False
                if dispatch_fails:
                    raise RuntimeError('synthetic handoff failed')

            watcher._process_handoff = process_handoff
            await asyncio.wait_for(GatewayAdapterLifecycleMixin._handoff_watcher(
                watcher, interval=0.0, drain_timeout=2), 5)

            terminal = db.get_handoff_state('s')['state']
            assert terminal == ('failed' if dispatch_fails else 'completed')
            accepted = await submit('after-terminal')
            assert accepted.status == 'queued'
            assert db.get_handoff_state('s')['state'] == terminal
        finally:
            await connection.close()

    try:
        asyncio.run(exercise())
    finally:
        if had_run_attribute:
            gateway_package.run = previous_run_attribute
        elif getattr(gateway_package, 'run', None) is run_helpers:
            del gateway_package.run
