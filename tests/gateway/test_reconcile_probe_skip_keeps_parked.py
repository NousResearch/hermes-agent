"""A profile skipped by the own-gateway probe this cycle keeps its parked verdict (N114)."""
from unittest.mock import patch

import pytest

from tests.gateway.test_multiplex_hot_serve import _mkprofile, _runner


@pytest.mark.asyncio
async def test_probe_skip_keeps_a_parked_profile_parked(tmp_path, monkeypatch):
    from gateway import run_profile_reconcile as reconcile_mod
    from gateway.run import MultiplexConfigError

    runner, home = _runner(tmp_path, monkeypatch)
    _mkprofile(home, 'alpha', 'DISCORD_BOT_TOKEN=alpha-token\n')
    with patch('hermes_cli.profiles.get_active_profile_name', return_value='default'):
        await runner._start_secondary_profile_adapters()
        _mkprofile(home, 'delta', 'DISCORD_BOT_TOKEN=delta-token\n')

        async def park(name, profile_home, claimed):
            raise MultiplexConfigError('open dm_policy')

        runner._start_one_profile_adapters = park
        assert (await runner.reconcile_served_profiles())['parked'] == ['delta']

        # The next cycle's own-gateway probe stalls for delta: unprovable, skipped this cycle.
        monkeypatch.setattr('gateway.status.live_gateway_pid_for_home', lambda profile_home: None)
        monkeypatch.setattr(reconcile_mod, '_OWN_GATEWAY_PROBE_TIMEOUT_SECS', 0.0)
        await runner.reconcile_served_profiles(reason='watcher')
        assert 'delta' in runner.session_runtime_descriptor['parked_profiles']
