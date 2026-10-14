"""Gateway startup wiring honors explicit browser profile homes."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

@pytest.mark.asyncio
async def test_browser_startup_wiring(tmp_path, monkeypatch):
    from gateway import run
    from tools import browser_tab_lifecycle as life
    from tools.process_registry import process_registry
    from gateway.run_startup import GatewayStartupMixin
    seen = []
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    second = tmp_path / 'second'
    second.mkdir()
    monkeypatch.setattr(run, '_multiplex_profile_homes', lambda config: [('second', second)])
    monkeypatch.setattr(life, 'startup', lambda home: seen.append(home))
    monkeypatch.setattr(process_registry, 'recover_from_checkpoint', lambda: 0)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True),
        _start_register_plugins_relay_hooks=lambda: None,
        _subscribe_plugin_rewire=lambda manager: None,
        hooks=SimpleNamespace(discover_and_load=lambda: None),
        _recover_secondary_process_checkpoints=lambda registry: 0,
        _recover_unclean_sessions=AsyncMock(return_value=(0, 0)),
        _suspend_stuck_loop_sessions=lambda: 0)
    await GatewayStartupMixin._start_recover_previous_run(runner)
    assert seen == [tmp_path, second]
