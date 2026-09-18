"""Inert authority construction keeps Files reclamation inside the owning scope.

No gateway bootstrap, listener, adapter, service runtime or model is started.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import run_input_reclamation, run_runtime
from gateway.run import _profile_runtime_scope
from gateway.run_runtime import _build_profile_authority
from gateway.runtime_ownership import process_ownership
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_cron import owner_for_home, unbind_owner
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError


@pytest.fixture
def inert_runtime(tmp_path, monkeypatch):
    from agent import secret_scope

    homes = [tmp_path / 'launch', tmp_path / 'secondary']
    for home in homes:
        home.mkdir()
        (home / 'config.yaml').write_text(
            f'terminal:\n  backend: local\n  cwd: {home}\n')
        (home / '.env').write_text(f'BACKEND_REFRESH_MARKER={home.name}\n')
    monkeypatch.setenv('HERMES_HOME', str(homes[0]))
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', True)
    databases = {home: SessionDB(home / 'state.db') for home in homes}

    class Runner:
        @property
        def _session_db(self):
            return databases[get_hermes_home()]

    runner = Runner()
    runner.config = SimpleNamespace(
        multiplex_profiles=True, _runtime_profile_homes=tuple((h.name, h) for h in homes))
    runner.session_store = SimpleNamespace()
    runner.session_authorities = SessionAuthorities(homes[0])
    runner.session_runtime_descriptor = {'instance_id': 'synthetic-owner', 'state': 'starting'}
    runner._running = runner._draining = False
    runner.adapters, runner._profile_adapters = {}, {}
    runner.session_api = runner.session_control_server = None
    statuses = []

    def forbidden(*args, **kwargs):
        raise AssertionError('execution is outside this inert construction test')

    monkeypatch.setattr(SessionAuthority, '_schedule', forbidden)
    monkeypatch.setattr(run_runtime, '_record_parked_profiles', lambda parked: statuses.append(dict(parked)))
    process_ownership.reserve(homes)
    try:
        yield runner, homes, databases, statuses
    finally:
        # Hygiene only: every failure assertion runs before these bindings are removed.
        for home, database in databases.items():
            bound = owner_for_home(home)
            if bound is not None:
                unbind_owner(bound[0])
            database.close()
            process_ownership.release(home)


@pytest.mark.asyncio
@pytest.mark.parametrize('ingress_ready', [False, True], ids=['pre-ingress', 'hot-serve-refused'])
async def test_scoped_authority_reclaims_before_publication(inert_runtime, monkeypatch, ingress_ready):
    from agent import secret_scope
    from gateway.hosted_room_input_reclamation import READY_KEY
    from tools.terminal_scope import terminal_env

    runner, homes, databases, _ = inert_runtime
    runner._running = ingress_ready
    runner.session_runtime_descriptor['state'] = 'ready' if ingress_ready else 'starting'
    observations, authorities = [], []
    original = run_input_reclamation.collect_legacy_copies_before_ingress

    def observe(current, authority):
        home = get_hermes_home()
        assert current is runner and authority.db is databases[home]
        assert current.session_authorities.for_home(home) is None
        assert secret_scope.get_secret('BACKEND_REFRESH_MARKER') == home.name
        assert terminal_env('TERMINAL_CWD') == str(home)
        original(current, authority)
        with authority.db._read_ctx() as conn:
            ready = conn.execute('SELECT value FROM state_meta WHERE key=?', (READY_KEY,)).fetchone()
        assert bool(ready) is not ingress_ready
        observations.append(home)

    monkeypatch.setattr(run_input_reclamation, 'collect_legacy_copies_before_ingress', observe)
    with _profile_runtime_scope(homes[0], hydrate_secrets=False):
        for index, home in enumerate(homes):
            authority = await _build_profile_authority(
                runner, home.name, home, register=index == 0)
            authorities.append(authority)
            assert runner.session_authorities.for_home(home) is authority
            assert get_hermes_home() == homes[0]
            assert secret_scope.get_secret('BACKEND_REFRESH_MARKER') == homes[0].name
            assert terminal_env('TERMINAL_CWD') == str(homes[0])
    assert observations == homes
    assert runner.session_authority is authorities[0]
    assert runner.session_store._local_authority_epochs == {
        home / 'state.db': authority.epoch for home, authority in zip(homes, authorities)}


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_index', [1, 0], ids=['secondary-parking', 'primary-fatal'])
async def test_invalid_reclamation_marker_withdraws_failed_owner(inert_runtime, monkeypatch, failed_index):
    from gateway.hosted_room_input_reclamation import READY_KEY

    runner, homes, databases, statuses = inert_runtime
    failed_home = homes[failed_index]
    databases[failed_home]._execute_write(lambda conn: conn.execute(
        'INSERT INTO state_meta(key,value) VALUES(?,?)', (READY_KEY, 'invalid-v3-home')))
    at_release = []
    original_release = run_runtime.release_profile_home

    def release(current, home):
        at_release.append((owner_for_home(home), dict(current.session_store._local_authority_epochs)))
        original_release(current, home)

    monkeypatch.setattr(run_runtime, 'release_profile_home', release)
    with _profile_runtime_scope(homes[0], hydrate_secrets=False):
        if failed_index == 0:
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                await run_runtime.initialize_gateway_runtime(runner)
            assert statuses == [] and at_release == []
            assert process_ownership.owns(failed_home)  # fatal boot caller still owns teardown
            assert getattr(runner, 'session_authority', None) is None
            assert getattr(runner.session_store, '_local_authority_epoch', None) is None
        else:
            await run_runtime.initialize_gateway_runtime(runner)
            launch = runner.session_authorities.require(homes[0])
            launch._require_admission_open()  # no admission or execution
            assert runner.session_authority is launch
            assert owner_for_home(homes[0])[0] is launch
            assert runner.session_store._local_authority_epoch == launch.epoch
            assert runner.session_store._local_authority_epochs[homes[0] / 'state.db'] == launch.epoch
            assert process_ownership.owns(homes[0])
            assert failed_home not in runner.session_authorities
            assert runner.session_runtime_descriptor['served_profiles'] == [
                {'profile_id': str(homes[0]), 'home': str(homes[0])}]
            assert runner.session_ticket_store.profile_ids == frozenset({str(homes[0])})
            assert 'storage_unavailable' in statuses[-1]['secondary']
            assert not process_ownership.owns(failed_home)
            assert runner.config._runtime_profile_homes == (('launch', homes[0]),)
        assert owner_for_home(failed_home) is None
        assert failed_home / 'state.db' not in runner.session_store._local_authority_epochs
        if failed_index == 1:
            assert at_release == [(None, {homes[0] / 'state.db': launch.epoch})]
        with databases[failed_home]._read_ctx() as conn:
            assert conn.execute('SELECT value FROM state_meta WHERE key=?', (READY_KEY,)).fetchone()[0] == 'invalid-v3-home'


class CollectionAborted(BaseException):
    """Exercise cleanup without converting process/task exits into parked profiles."""


@pytest.mark.asyncio
@pytest.mark.parametrize('register', [False, True], ids=['secondary', 'primary'])
@pytest.mark.parametrize('replacement', ['owner', 'epoch-map', 'store'])
async def test_failed_collection_only_withdraws_its_registrations(inert_runtime, monkeypatch, register, replacement):
    from gateway.session_authority import initialize_session_authority

    runner, homes, databases, _ = inert_runtime
    if not register:
        await _build_profile_authority(runner, homes[0].name, homes[0], register=True)
    home = homes[0] if register else homes[1]
    db_path = home / 'state.db'
    launch = getattr(runner, 'session_authority', None)
    store = runner.session_store
    started, finished = asyncio.Event(), asyncio.Event()
    proceed = threading.Event()
    loop = asyncio.get_running_loop()
    captured = []
    error = CollectionAborted('inert collection aborted')

    def fail(current, authority):
        captured.append(authority)
        loop.call_soon_threadsafe(started.set)
        try:
            assert proceed.wait(10)
            raise error
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(run_input_reclamation, 'collect_legacy_copies_before_ingress', fail)
    task = asyncio.create_task(_build_profile_authority(runner, home.name, home, register=register))
    try:
        await asyncio.wait_for(started.wait(), 5)
        failed = captured[0]
        epochs = store._local_authority_epochs
        assert owner_for_home(home)[0] is failed
        assert epochs[db_path] == failed.epoch
        if replacement == 'store':
            # A substituted store may coincidentally carry the same numeric epoch.
            # It must not be used as the target of this construction's rollback.
            runner.session_store = SimpleNamespace(
                _local_authority_epochs=dict(epochs), _local_authority_epoch=store._local_authority_epoch)
            successor = None
        else:
            if replacement == 'epoch-map':
                store._local_authority_epochs = dict(epochs)
            with _profile_runtime_scope(home, hydrate_secrets=False):
                successor = await initialize_session_authority(
                    runner, profile_id=str(home), instance_id='replacement-owner',
                    db=databases[home], register=register)
            assert successor.epoch != failed.epoch
            runner.session_authorities.replace(home, successor)
        replacement_epochs = dict(runner.session_store._local_authority_epochs)
        replacement_scalar = runner.session_store._local_authority_epoch
        proceed.set()
        with pytest.raises(CollectionAborted) as raised:
            await task
        assert raised.value is error
        assert runner.session_store._local_authority_epochs == replacement_epochs
        assert runner.session_store._local_authority_epoch == replacement_scalar
        if successor is None:
            assert owner_for_home(home) is None
            assert db_path not in epochs
            assert runner.session_authorities.for_home(home) is None
            if register:
                assert runner.session_authority is None
                assert store._local_authority_epoch is None
        else:
            assert owner_for_home(home)[0] is successor
            assert runner.session_authorities.require(home) is successor
            if register:
                assert runner.session_authority is successor
            if replacement == 'epoch-map':
                assert db_path not in epochs
        if not register:
            assert runner.session_authority is launch
            assert owner_for_home(homes[0])[0] is launch
            assert store._local_authority_epoch == launch.epoch
            assert epochs[homes[0] / 'state.db'] == launch.epoch
        assert process_ownership.owns(home)
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(finished.wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('register', [False, True], ids=['secondary', 'primary'])
async def test_cancelled_collection_withdraws_owner_without_releasing_worker_home(inert_runtime, monkeypatch, register):
    runner, homes, _, statuses = inert_runtime
    if not register:
        await _build_profile_authority(runner, homes[0].name, homes[0], register=True)
    home = homes[0] if register else homes[1]
    started, finished = asyncio.Event(), asyncio.Event()
    proceed = threading.Event()
    loop = asyncio.get_running_loop()
    original = run_input_reclamation.collect_legacy_copies_before_ingress
    observations = []

    def block(current, authority):
        loop.call_soon_threadsafe(started.set)
        try:
            assert proceed.wait(10)
            owned = (get_hermes_home(), process_ownership.owns(home))
            original(current, authority)
            observations.append(owned)
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(run_input_reclamation, 'collect_legacy_copies_before_ingress', block)
    task = asyncio.create_task(_build_profile_authority(runner, home.name, home, register=register))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert owner_for_home(home) is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()  # cancellation did NOT stop the to_thread worker
        assert process_ownership.owns(home)
        assert owner_for_home(home) is None
        assert home / 'state.db' not in runner.session_store._local_authority_epochs
        assert runner.session_authorities.for_home(home) is None
        assert statuses == []
        if register:
            assert runner.session_authority is None
            assert runner.session_store._local_authority_epoch is None
        else:
            launch = runner.session_authorities.require(homes[0])
            assert owner_for_home(homes[0])[0] is launch
            assert runner.session_store._local_authority_epoch == launch.epoch
        proceed.set()
        await asyncio.wait_for(finished.wait(), 5)
        assert observations == [(home, True)]
        assert owner_for_home(home) is None  # worker completion cannot rebind the owner
        assert home / 'state.db' not in runner.session_store._local_authority_epochs
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(finished.wait(), 5)
