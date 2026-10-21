"""The messaging gateway starts the host toolchain probe at boot, before its first config verdict.

Harness: a named profile with ``multiplex_profiles`` makes ``start_gateway`` exit EX_CONFIG right
after the probe call (see test_multiplex_requires_default_profile.py), so nothing binds or forks.
"""
import sys

import pytest

from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from tools import env_probe


async def _start_until_config_verdict(tmp_path, monkeypatch):
    from gateway import run_bootstrap
    from gateway.config import GatewayConfig

    root = tmp_path / '.hermes'
    (root / 'profiles' / 'foo').mkdir(parents=True)
    (root / 'config.yaml').write_text('model: {default: m}\n')
    (root / 'profiles' / 'foo' / 'config.yaml').write_text('gateway:\n  multiplex_profiles: true\n')
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles' / 'foo'))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, '_default_hermes_root_memo', None)
    monkeypatch.setattr('hermes_cli.resource_limits.apply_nofile_soft_limit', lambda: None)
    monkeypatch.setattr('gateway.code_skew.record_boot_fingerprint', lambda: None)
    monkeypatch.setattr('gateway.run._write_runtime_status_quiet', lambda **kw: None)
    config = GatewayConfig()
    config.multiplex_profiles = True
    with pytest.raises(SystemExit) as exc:
        await run_bootstrap.start_gateway(config=config)
    assert exc.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE


@pytest.mark.asyncio
async def test_start_gateway_warms_the_probe_at_boot(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(env_probe, 'warm_environment_probe_at_boot', lambda: calls.append(1))

    await _start_until_config_verdict(tmp_path, monkeypatch)

    assert calls == [1]


@pytest.mark.asyncio
async def test_start_gateway_boots_past_an_unimportable_probe(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'tools.env_probe', None)  # the lazy import raises ImportError

    await _start_until_config_verdict(tmp_path, monkeypatch)
