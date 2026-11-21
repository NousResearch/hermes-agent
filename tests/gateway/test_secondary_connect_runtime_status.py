"""A secondary profile's clean first-try connect must publish its runtime status (#141781).

The initial-connect loop used to only log on success, so an adapter that never stamps itself
(Email) left no ``<profile>:<platform>`` entry in ``gateway_state.json`` — the Messaging card
then fell back to ``pending_restart`` ("Restart needed") forever, surviving every gateway
restart because each clean restart recreated the same no-entry state. The primary cold start
(``run_startup``) and the reconnect install (``_install_reconnected_adapter``) both publish;
the secondary initial connect must match them, including the degraded-receive-path guard and
clearing retry flags an earlier failure left behind.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


def _mkprofile(home, name, env=""):
    d = home / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    (d / ".env").write_text(env, encoding="utf-8")
    return d


def _runner(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._failed_platforms = {}
    # Heavy lifecycle seams stay off the connect path under test (same stub set as the
    # real-path test in test_multiplex_hot_serve.py).
    runner._register_config_hooks = lambda *a, **kw: None
    runner._subscribe_plugin_rewire = lambda *a, **kw: None
    runner._configure_profile_adapter = lambda *a: None
    runner._sync_voice_mode_state_to_adapter = lambda *a: None
    runner._adapter_credential_claim = lambda *a: None
    runner._adapter_listener_claim = lambda *a: None
    writes = []
    runner._update_platform_runtime_status = lambda platform, **kwargs: writes.append(
        (platform, kwargs)
    )
    return runner, home, writes


async def _connect_ok(adapter, platform):
    return True


@pytest.mark.asyncio
async def test_clean_first_try_connect_publishes_connected(tmp_path, monkeypatch):
    runner, home, writes = _runner(tmp_path, monkeypatch)
    gamma = _mkprofile(home, "gamma", "DISCORD_BOT_TOKEN=gamma-token\n")
    runner._create_adapter = lambda platform, config: SimpleNamespace(
        platform=platform, send_path_degraded=False
    )
    runner._connect_initial_adapter_with_timeout = _connect_ok

    connected = await runner._start_one_profile_adapters("gamma", gamma, {})

    assert connected == 1
    assert (
        "gamma:discord",
        {
            "platform_state": "connected",
            "error_code": None,
            "error_message": None,
            "needs_attention": False,
            "retrying_since": None,
        },
    ) in writes, "a clean first-try connect must stamp <profile>:<platform> connected"


@pytest.mark.asyncio
async def test_degraded_first_try_connect_publishes_retrying(tmp_path, monkeypatch):
    """Same guard as the primary cold start / reconnect install: connect() returning True on an
    unconfirmed receive path publishes ``retrying`` with the adapter's degraded message."""
    runner, home, writes = _runner(tmp_path, monkeypatch)
    gamma = _mkprofile(home, "gamma", "DISCORD_BOT_TOKEN=gamma-token\n")

    def _create(platform, config):
        return SimpleNamespace(
            platform=platform,
            send_path_degraded=True,
            DEGRADED_STATUS_MESSAGE="connected but not yet confirmed active; recovering in background",
        )

    runner._create_adapter = _create
    runner._connect_initial_adapter_with_timeout = _connect_ok

    connected = await runner._start_one_profile_adapters("gamma", gamma, {})

    assert connected == 1
    assert (
        "gamma:discord",
        {
            "platform_state": "retrying",
            "error_code": None,
            "error_message": "connected but not yet confirmed active; recovering in background",
            "needs_attention": False,
            "retrying_since": None,
        },
    ) in writes, "a degraded first-try connect must not claim connected"
