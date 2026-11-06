"""Gateway start reloads its platform config after the left-core migration installs a platform plugin.

``start_gateway`` loads the gateway config before the migration runs (config verdicts come before the
duplicate-instance guard), so a home whose platform left core (Yuanbao, Home Assistant) got the plugin
installed by this very start but served it only on the next one.
"""

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.mark.asyncio
async def test_a_platform_installed_by_this_start_is_in_the_runners_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin_installed = []
    seen = {}

    def load_config():
        platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
        if plugin_installed:
            platforms[Platform.WEIXIN] = PlatformConfig(enabled=True)  # the platform the plugin ships
        return GatewayConfig(platforms=platforms)

    def recover_at_startup(*, say=None, gateway_start=False):
        plugin_installed.append("homeassistant")
        return ["homeassistant"]

    class Runner:
        def __init__(self, config):
            seen["platforms"] = set(config.platforms)
            raise SystemExit(0)

    monkeypatch.setattr("gateway.run.load_gateway_config_for_runner", load_config)
    monkeypatch.setattr("hermes_cli.left_core_migration.recover_at_startup", recover_at_startup)
    monkeypatch.setattr("gateway.run._host_attach_or_none", lambda replace, force: _none())
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", Runner)

    with pytest.raises(SystemExit):
        await gateway_run.start_gateway(replace=False, verbosity=None)

    assert seen["platforms"] == {Platform.TELEGRAM, Platform.WEIXIN}


async def _none():
    return None
