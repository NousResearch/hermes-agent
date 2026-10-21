"""A secondary parked at boot (its startup Bot recovery failed) and served again by the
control-socket rescan is fully served: its local sessions work, and the durable work queued
before the park is recovered the way a cold boot recovers it (local, API turns, native rows).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.runtime_ownership import process_ownership


@pytest.fixture
def multiplexed(tmp_path, monkeypatch):
    """``(runner, beta_home)``: a multiplexed runner serving the default profile and ``beta``."""
    from gateway import run
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {
        'platform_toolsets': {'cli': [], 'tui': []},
        'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'api_key': 'test-key'}})
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True, mode=0o700)
    (beta / 'config.yaml').write_text('{}')
    homes = [('default', root), ('beta', beta)]
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    yield runner, beta.resolve()
    for authority in list(getattr(runner, 'session_authorities', None) or []):
        if authority is not None:
            authority.db.close()
    for _, home in homes:
        process_ownership.release(home)


async def _boot_with_beta_parked(runner, beta, monkeypatch):
    """Boot's API start with beta's startup Bot recovery failing once: beta is parked."""
    from gateway import session_bot
    from gateway.run_runtime import parked_profile_map, start_gateway_runtime_api
    real, failed = session_bot.recover_bot_deliveries, []

    async def recover_once_failing(authority):
        if Path(authority.profile_id).resolve() == beta and not failed:
            failed.append(authority)
            raise OSError('mailbox write failed')
        return await real(authority)

    async def api(_runner):
        return SimpleNamespace(api_origin='http://127.0.0.1:1')

    monkeypatch.setattr(session_bot, 'recover_bot_deliveries', recover_once_failing)
    monkeypatch.setattr('gateway.run_api.start_gateway_api', api)
    await start_gateway_runtime_api(runner)
    assert 'beta' in parked_profile_map(runner)


async def _rescan(runner, beta, monkeypatch, connected=None):
    """The control-socket rescan serves beta again (its adapters connect without a network)."""
    async def start_adapters(name, home, claimed):
        if connected is not None:
            connected.append(name)
        return 0

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(runner, '_start_one_profile_adapters', start_adapters)
    monkeypatch.setattr(runner, '_after_profiles_added', nothing)
    result = await runner._apply_profile_changes({}, ['beta'], [], [], reason='control-socket',
                                                 live={'beta': beta})
    assert result['added'] == ['beta'], result
    return runner.session_authorities.for_home(beta)


def _actor(authority):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, object(), {'user_id': 'human'}).actor


@pytest.mark.asyncio
async def test_local_sessions_work_after_the_rescan(multiplexed, tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.session_authorities import owner_scope
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    runner, beta = multiplexed
    await initialize_gateway_runtime(runner)
    first = runner.session_authorities.for_home(beta)
    params = dict(source='tui', cwd=str(tmp_path), model='m', toolsets=[])
    with owner_scope(first):
        existing = create_local_session(first, _actor(first), dict(params, request_id='before-park'))
    await _boot_with_beta_parked(runner, beta, monkeypatch)
    served = await _rescan(runner, beta, monkeypatch)
    assert served is not first
    with owner_scope(served):
        restore_local_session(served, existing.session_id)
        create_local_session(served, _actor(served), dict(params, request_id='after-rescan'))


@pytest.mark.asyncio
async def test_api_work_queued_before_the_park_is_recovered_by_the_rescan(multiplexed, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.session_api import bind_api_session
    from gateway.session_authority import SessionAuthority
    from hermes_state_runtime import admit_session_input
    runner, beta = multiplexed
    api = SimpleNamespace(gateway_runner=runner)
    runner.adapters[Platform.API_SERVER] = api  # the launch API adapter is already published
    monkeypatch.setattr(runner, '_adapter_for_source', lambda source: api)
    scheduled = []
    monkeypatch.setattr(SessionAuthority, '_schedule',
                        lambda self, ref: scheduled.append((Path(self.profile_id).name, ref.session_id)))
    await initialize_gateway_runtime(runner)
    first = runner.session_authorities.for_home(beta)
    ref = bind_api_session(first, 'api-queued')
    admit_session_input(first.db, epoch=first.epoch, principal_id='api', session_id=ref.session_id,
                        request_id='chat:scope:queued',
                        payload={'text': 'do the queued work', 'api_turn_v1': {'history': [], 'settings': {}}})
    await _boot_with_beta_parked(runner, beta, monkeypatch)
    await _rescan(runner, beta, monkeypatch)
    # A cold boot schedules it when the API adapter is published (recover_api_turns).
    assert ('beta', 'api-queued') in scheduled, f'not scheduled after the rescan: {scheduled}'


@pytest.mark.asyncio
async def test_native_recovery_runs_once_the_rescanned_adapters_connect(multiplexed, monkeypatch):
    """Boot recovers queued native rows after the adapters connect
    (``recover_gateway_native_sessions``); the rescan runs the same step once the served profile's
    bots connect, in every serving ledger: a ``bot_profile`` route runs a bot's messages on another
    profile, so the rows those bots carry can be queued in the default profile's ledger."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.session_authority import SessionAuthority
    runner, beta = multiplexed
    connected, recovered = [], []
    real = SessionAuthority.recover_native_sessions

    async def record(self, bindings):
        recovered.append((Path(self.profile_id).name, list(connected)))
        return await real(self, bindings)

    monkeypatch.setattr(SessionAuthority, 'recover_native_sessions', record)
    await initialize_gateway_runtime(runner)
    await _boot_with_beta_parked(runner, beta, monkeypatch)
    await _rescan(runner, beta, monkeypatch, connected)
    launch = Path(runner.session_authorities.launch.profile_id).name
    assert sorted(recovered) == [(launch, ['beta']), ('beta', ['beta'])], f'native recovery after the rescan: {recovered}'
