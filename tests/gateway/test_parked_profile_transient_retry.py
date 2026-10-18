"""A secondary parked because its startup failed on a condition that clears by itself (a busy or
I/O-failing store, a CLI holding the Bot Chat owner session) is served again by the profile
watcher, with backoff and without a rescan. A permanent failure still parks at once, and only a
rescan retries it. Neither adds a wait to the boot."""
import asyncio
import errno
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from hermes_state_runtime import RuntimeStoreError

pytest_plugins = ['tests.gateway.test_parked_profile_rescan']  # the ``multiplexed`` runner


def _busy(tmp_path):
    """A real SQLITE_BUSY: another connection holds the write lock."""
    path = tmp_path / 'busy.db'
    with closing(sqlite3.connect(path, isolation_level=None)) as holder, \
            closing(sqlite3.connect(path, timeout=0)) as writer:
        holder.execute('CREATE TABLE IF NOT EXISTS t (x)')
        holder.execute('BEGIN EXCLUSIVE')
        try:
            writer.execute('INSERT INTO t VALUES (1)')
        except sqlite3.OperationalError as exc:
            return exc
    raise AssertionError('the write lock did not refuse the second writer')


def _locked(tmp_path):
    """The store's own write retry gave up: a codeless re-raise naming the real busy error."""
    try:
        raise sqlite3.OperationalError('database is locked (another Hermes process held the '
                                       'state.db write lock for over 30s)') from _busy(tmp_path)
    except sqlite3.OperationalError as exc:
        return exc


def _not_a_database(tmp_path):
    path = tmp_path / 'damaged.db'
    path.write_bytes(b'not a database' * 100)
    with closing(sqlite3.connect(path)) as conn:
        try:
            conn.execute('SELECT * FROM sqlite_master').fetchall()
        except sqlite3.DatabaseError as exc:
            return exc
    raise AssertionError('the damaged file opened')


def _chained(cause, exc):
    try:
        raise exc from cause
    except type(exc) as raised:
        return raised


def _malformed_registry(tmp_path):
    """The Bot Chat liveness registry's strict read of a file that isn't JSON."""
    from hermes_cli.active_sessions import ActiveSessionRegistryError, _read_entries
    path = tmp_path / 'active_sessions.json'
    path.write_text('{not json')
    try:
        _read_entries(path, strict=True)
    except ActiveSessionRegistryError as exc:
        return exc
    raise AssertionError('the malformed registry was read')


def _registry_error(cause):
    from hermes_cli.active_sessions import ActiveSessionRegistryError
    return _chained(cause, ActiveSessionRegistryError('active session registry unreadable'))


@contextmanager
def _heartbeat():
    """Gaps between 10 ms ticks of the event loop, which every served profile shares."""
    loop, gaps = asyncio.get_running_loop(), []
    last, handle = [time.monotonic()], [None]

    def beat():
        now = time.monotonic()
        gaps.append(now - last[0])
        last[0] = now
        handle[0] = loop.call_later(0.01, beat)

    handle[0] = loop.call_later(0.01, beat)
    try:
        yield gaps
    finally:
        handle[0].cancel()
        gaps.append(time.monotonic() - last[0])


def _hold_beta(runner, beta, site, monkeypatch):
    """Hold a condition that clears by itself until the returned ``release`` runs: another
    connection's write lock on beta's store (``store``: the open fails busy), or a CLI holding
    beta's Bot Chat owner session (``recovery``: recovery refuses ``runtime_coordination_required``)."""
    if site == 'store':
        import hermes_state
        hermes_state.SessionDB(beta / 'state.db').close()  # an existing profile store
        monkeypatch.setattr(hermes_state.SessionDB, '_WRITE_PATIENCE_S', 0.2)  # the real one is 20 s
        for cache in (runner.session_store._db_handle_cache, runner._session_db_handle_cache):
            monkeypatch.setattr(cache, '_initial_retry_delay', 0.0)  # every attempt opens again
        holder = sqlite3.connect(beta / 'state.db', isolation_level=None)
        holder.execute('BEGIN EXCLUSIVE')

        def release():
            holder.execute('ROLLBACK')
            holder.close()

        return release
    from gateway import session_bot
    real, held = session_bot.recover_bot_deliveries, {'cli': True}

    async def recover(authority):
        if held['cli'] and Path(authority.profile_id).resolve() == beta:
            raise RuntimeStoreError('runtime_coordination_required')
        return await real(authority)

    monkeypatch.setattr(session_bot, 'recover_bot_deliveries', recover)
    return lambda: held.update(cli=False)


def _fail_recovery(monkeypatch, beta, exc):
    """Beta's startup Bot recovery always raises *exc*; returns the attempts."""
    from gateway import session_bot
    attempts = []

    async def recover(authority):
        if Path(authority.profile_id).resolve() == beta:
            attempts.append(authority.profile_id)
            raise exc

    monkeypatch.setattr(session_bot, 'recover_bot_deliveries', recover)
    return attempts


async def _boot(runner, monkeypatch):
    """Boot's runtime steps; returns how long they took."""
    async def api(_runner):
        return SimpleNamespace(api_origin='http://127.0.0.1:1')

    monkeypatch.setattr('gateway.run_api.start_gateway_api', api)
    started = time.monotonic()
    await run_runtime.initialize_gateway_runtime(runner)
    await run_runtime.start_gateway_runtime_api(runner)
    return time.monotonic() - started


def _watch(runner, beta, monkeypatch):
    """The running gateway's profile watcher, with beta's adapters connecting without a network."""
    root = runner.session_authorities.launch.profile_id
    runner._running = True
    runner._note_served_profiles([('default', Path(root))])

    async def start_adapters(name, home, claimed):
        return 0

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(runner, '_start_one_profile_adapters', start_adapters)
    monkeypatch.setattr(runner, '_after_profiles_added', nothing)
    monkeypatch.setattr('hermes_cli.profiles.profiles_to_serve',
                        lambda **kw: [('default', Path(root)), ('beta', beta)])
    monkeypatch.setattr('gateway.status.live_gateway_pid_for_home', lambda home: None)
    clock = time.monotonic

    async def cycle(after):
        """One watcher cycle *after* seconds from now."""
        monkeypatch.setattr(run_runtime, '_monotonic', lambda: clock() + after, raising=False)
        return await runner.reconcile_served_profiles(reason='watcher')

    return cycle


@pytest.mark.asyncio
@pytest.mark.parametrize('site', ['store', 'recovery'])
async def test_a_transient_startup_failure_is_served_again_by_the_watcher(multiplexed, monkeypatch, site):
    runner, beta = multiplexed
    release = _hold_beta(runner, beta, site, monkeypatch)
    monkeypatch.setattr(run_runtime, 'PARK_RETRY_FIRST_SECONDS', 10.0, raising=False)
    monkeypatch.setattr(run_runtime, 'PARK_RETRY_MAX_SECONDS', 40.0, raising=False)
    assert await _boot(runner, monkeypatch) < 5.0, 'the boot waited for the transient failure'
    parked = run_runtime.parked_profile_map(runner)
    assert 'beta' in parked and runner.session_authorities.for_home(beta) is None

    cycle = _watch(runner, beta, monkeypatch)
    assert (await cycle(0.0))['parked'] == [], 'retried before its backoff lapsed'
    # The first retry meets the condition still held: parked again, with a doubled wait (20 s).
    # A locked store's open waits out its write patience; the loop the other profiles are served
    # on keeps running meanwhile.
    import hermes_state
    monkeypatch.setattr(hermes_state.SessionDB, '_WRITE_PATIENCE_S', 4.0)
    with _heartbeat() as gaps:
        assert (await cycle(10.0))['parked'] == ['beta'], 'the watcher never retried the transient park'
    assert max(gaps) < 2.0, f'the retry held the event loop for {max(gaps):.1f}s'
    release()
    assert (await cycle(20.0))['parked'] == [], 'the backoff did not grow'
    result = await cycle(30.0)
    assert result['added'] == ['beta'], result
    assert 'beta' not in parked
    served = runner.session_authorities.for_home(beta)
    assert served is not None and not served.retiring
    assert str(beta) in {p['profile_id'] for p in runner.session_runtime_descriptor['served_profiles']}


@pytest.mark.asyncio
async def test_a_permanent_startup_failure_parks_at_once_and_waits_for_a_rescan(multiplexed, tmp_path,
                                                                                  monkeypatch):
    runner, beta = multiplexed
    attempts = _fail_recovery(monkeypatch, beta, _not_a_database(tmp_path))
    assert await _boot(runner, monkeypatch) < 5.0, 'the boot waited before parking a permanent failure'
    assert 'retried automatically' not in run_runtime.parked_profile_map(runner)['beta']
    result = await _watch(runner, beta, monkeypatch)(3600.0)
    assert result['added'] == result['parked'] == [] and len(attempts) == 1
    assert 'beta' in run_runtime.parked_profile_map(runner)


@pytest.mark.parametrize('make, transient', [
    (_busy, True),
    (_locked, True),
    (lambda tmp: _chained(_busy(tmp), RuntimeStoreError('storage_unavailable')), True),
    (lambda tmp: RuntimeStoreError('runtime_coordination_required'), True),
    (lambda tmp: OSError(errno.EIO, 'Input/output error'), True),
    (lambda tmp: _chained(_busy(tmp), run_runtime.ProfileStoreUnavailable('not opened')), True),
    (lambda tmp: _registry_error(OSError(errno.EIO, 'Input/output error')), True),
    (_not_a_database, False),
    (lambda tmp: run_runtime.ProfileStoreUnavailable('not opened'), False),
    (_malformed_registry, False),
    (lambda tmp: _registry_error(PermissionError(errno.EACCES, 'Permission denied')), False),
    (lambda tmp: RuntimeStoreError('storage_unavailable'), False),
    (lambda tmp: _chained(_busy(tmp), RuntimeError('a bug while handling it')), False),
    (lambda tmp: OSError('mailbox write failed'), False),
], ids=['busy', 'locked-reraise', 'wrapped-busy', 'cli-holds-bot-chat', 'eio', 'open-failed-busy',
        'registry-eio', 'not-a-database', 'open-failed-unknown', 'registry-malformed', 'registry-denied',
        'store-refusal', 'bug', 'other-oserror'])
def test_transient_failures_are_told_apart_by_type_and_result_code(tmp_path, make, transient):
    assert run_runtime.transient_serve_failure(make(tmp_path)) is transient
