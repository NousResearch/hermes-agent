"""Gateway start reloads its platform config after the left-core migration installs a platform plugin.

``start_gateway`` loads the gateway config before the migration runs (config verdicts come before the
duplicate-instance guard), so a home whose platform left core (Yuanbao, Home Assistant) got the plugin
installed by this very start but served it only on the next one.
"""

import pytest

import hermes_agent.gateway.run as gateway_run
from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig


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

    monkeypatch.setattr("hermes_agent.gateway.run.load_gateway_config_for_runner", load_config)
    monkeypatch.setattr("hermes_agent.gateway.config.load_gateway_config", load_config)
    monkeypatch.setattr("hermes_agent.hermes_cli.left_core_migration.recover_at_startup", recover_at_startup)
    monkeypatch.setattr("hermes_agent.gateway.run._host_attach_or_none", lambda replace, force: _none())
    monkeypatch.setattr("hermes_agent.gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("hermes_agent.gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("hermes_agent.hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr("hermes_agent.gateway.run.GatewayRunner", Runner)

    with pytest.raises(SystemExit):
        await gateway_run.start_gateway(replace=False, verbosity=None)

    assert seen["platforms"] == {Platform.TELEGRAM, Platform.WEIXIN}


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [["yuanbao"], []])
async def test_a_named_launch_stays_standalone_across_the_reload(tmp_path, monkeypatch, installed):
    """``hermes -p foo gateway run`` on a host with default + foo downgrades the implicit multiplex
    default to standalone and reserves only foo. The reload after the migration installed a platform
    must keep that verdict: a fresh decision re-enables multiplex and serves default's adapters,
    which this process never reserved. With or without an install, the runner gets foo's own platforms."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "foo").mkdir(parents=True)
    (root / "config.yaml").write_text("model: {default: m}\nplatforms:\n  slack: {enabled: true, token: xoxb-default}\n")
    (root / "profiles" / "foo" / "config.yaml").write_text("model: {default: m}\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "foo"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    from hermes_agent import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    seen = {}
    decisions = []

    import hermes_agent.hermes_cli.gateway_multiplex_mode as mux
    monkeypatch.setattr(mux, "log_multiplex_decision", decisions.append)

    class Runner(gateway_run.GatewayRunner):  # the multiplex preflight reads its class attributes
        def __init__(self, config):
            seen["multiplex"] = config.multiplex_profiles
            seen["served"] = [name for name, _home in gateway_run._multiplex_profile_homes(config)] \
                if config.multiplex_profiles else ["foo"]
            seen["default_slack"] = Platform.SLACK in config.platforms  # default's platform, not foo's
            raise SystemExit(0)

    from hermes_agent.agent.secret_scope import is_multiplex_active, set_multiplex_active
    monkeypatch.setattr("hermes_agent.agent.secret_scope._MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr("hermes_agent.agent.secret_scope._AUTO_PINNED_HOME", None)

    def recover_at_startup(*, say=None, gateway_start=False):
        # The install reads this profile's env unscoped (manifest requires_env): it must not fail closed.
        seen["multiplex_active_at_migration"] = is_multiplex_active()
        return installed

    monkeypatch.setattr("hermes_agent.hermes_cli.left_core_migration.recover_at_startup", recover_at_startup)
    monkeypatch.setattr("hermes_agent.gateway.run._host_attach_or_none", lambda replace, force: _none())
    monkeypatch.setattr("hermes_agent.gateway.status.get_running_pid", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_agent.gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("hermes_agent.gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.run._write_runtime_status_quiet", lambda **kw: None)
    monkeypatch.setattr("hermes_agent.hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr("hermes_agent.gateway.run.GatewayRunner", Runner)

    try:
        with pytest.raises(SystemExit):
            await gateway_run.start_gateway(replace=False, verbosity=None)
    finally:
        set_multiplex_active(False)

    assert seen == {"multiplex": False, "served": ["foo"], "multiplex_active_at_migration": False,
                    "default_slack": False}
    assert [(d.enabled, d.source) for d in decisions] == [(True, "default"), (False, "guard")]  # no re-decision


@pytest.mark.asyncio
async def test_a_default_launch_keeps_its_reserved_homes_across_the_reload(tmp_path, monkeypatch):
    """The multiplexing half of the reload: the boot reserved default + foo, and the reloaded config
    keeps that reservation instead of rescanning profiles the process never reserved."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "foo").mkdir(parents=True)
    (root / "config.yaml").write_text("model: {default: m}\n")
    (root / "profiles" / "foo" / "config.yaml").write_text("model: {default: m}\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    from hermes_agent import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    seen = {}

    class Runner(gateway_run.GatewayRunner):
        def __init__(self, config):
            seen["multiplex"] = config.multiplex_profiles
            seen["reserved"] = [name for name, _home in getattr(config, "_runtime_profile_homes", None) or ()]
            raise SystemExit(0)

    from hermes_agent.agent.secret_scope import set_multiplex_active
    monkeypatch.setattr("hermes_agent.agent.secret_scope._MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr("hermes_agent.agent.secret_scope._AUTO_PINNED_HOME", None)
    monkeypatch.setattr("hermes_agent.hermes_cli.left_core_migration.recover_at_startup",
                        lambda *, say=None, gateway_start=False: ["yuanbao"])
    monkeypatch.setattr("hermes_agent.gateway.run._host_attach_or_none", lambda replace, force: _none())
    monkeypatch.setattr("hermes_agent.gateway.status.get_running_pid", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_agent.gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("hermes_agent.gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("hermes_agent.gateway.run._write_runtime_status_quiet", lambda **kw: None)
    monkeypatch.setattr("hermes_agent.hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr("hermes_agent.gateway.run.GatewayRunner", Runner)

    try:
        with pytest.raises(SystemExit):
            await gateway_run.start_gateway(replace=False, verbosity=None)
    finally:
        set_multiplex_active(False)

    assert seen == {"multiplex": True, "reserved": ["default", "foo"]}


async def _none():
    return None


def test_the_launch_homes_migration_reads_its_own_secrets_on_a_multiplexing_start(tmp_path, monkeypatch):
    """A default-profile gateway that multiplexes armed fail-closed secret reads while loading its
    config, before the migration runs. The launch home's install (its required-env check) must read
    the launch profile's own ``.env`` through its scope, not raise ``UnscopedSecretError``."""
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / ".env").write_text("YUANBAO_APP_ID=launch-app\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("YUANBAO_APP_ID", raising=False)
    from hermes_agent import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from hermes_agent.agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
    read = []

    def recover_at_startup(*, say=None, gateway_start=False):
        read.append((str(hermes_constants.get_hermes_home()), get_secret("YUANBAO_APP_ID")))
        return []

    monkeypatch.setattr("hermes_agent.hermes_cli.left_core_migration.recover_at_startup", recover_at_startup)
    from hermes_agent.gateway.run_startup import recover_left_core_at_gateway_start
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        recover_left_core_at_gateway_start()
    finally:
        set_multiplex_active(previous)

    assert read == [(str(root), "launch-app")]  # once: the launch home is the root
