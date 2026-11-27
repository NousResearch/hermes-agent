"""A multiplexed secondary's port-binding adapter is served only at ``/p/<profile>/`` on the default
profile's api_server/webhook listener. With neither running it must not start and read as
connected (it binds nothing); it is stamped as not served, with the remedy."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


class _PortBinder:
    """Stands in for a plugin callback adapter (WeCom callback): binds a port when standalone."""

    def __init__(self):
        self.platform = Platform("line")
        self.config = PlatformConfig(enabled=True, extra={"port": 47125})
        self.connected = False

    async def connect(self, **_kw):
        self.connected = True
        return True

    def __getattr__(self, name):  # every set_* wiring hook the runner installs
        if name.startswith("set_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def _runner(default_adapters):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = False
    runner.adapters = dict(default_adapters)
    runner._failed_platforms = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._background_tasks = set()
    runner.session_store = object()
    runner._busy_text_mode = "queue"
    runner._snapshot_profile_busy_modes = lambda *a, **k: None
    runner._make_adapter_auth_check = lambda platform, profile_name=None: object()
    runner._sync_voice_mode_state_to_adapter = lambda adapter: None
    runner._bind_voice_input_callback = lambda adapter: None
    runner._extend_systemd_boot_for = lambda platform: None
    return runner


@pytest.fixture
def secondary(monkeypatch):
    @contextmanager
    def fake_scope(profile_home, *, hydrate_secrets=True):
        yield

    adapter = _PortBinder()
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: {})
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: Path("/profiles") / name)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig(
        multiplex_profiles=True, platforms={Platform("line"): adapter.config}))
    return adapter


def _wire(monkeypatch, runner, adapter, stamps):
    async def connect(a, platform):
        return await a.connect()

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: adapter)
    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", connect)
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: stamps.append((key, kw)))


@pytest.mark.asyncio
async def test_a_secondary_port_binder_without_a_default_listener_is_not_reported_connected(
        monkeypatch, secondary):
    runner, stamps = _runner({}), []
    _wire(monkeypatch, runner, secondary, stamps)

    assert await runner._start_one_profile_adapters("b", Path("/profiles/b"), {}) == 0

    assert secondary.connected is False
    assert Platform("line") not in runner._profile_adapters.get("b", {})
    assert [(k, kw["platform_state"], kw["error_code"]) for k, kw in stamps] == [
        ("b:line", "disabled", "multiplex_no_shared_listener")]


@pytest.mark.asyncio
async def test_a_secondary_port_binder_is_served_when_the_default_runs_a_listener(monkeypatch, secondary):
    runner, stamps = _runner({Platform.API_SERVER: object()}), []
    _wire(monkeypatch, runner, secondary, stamps)

    assert await runner._start_one_profile_adapters("b", Path("/profiles/b"), {}) == 1

    assert secondary.connected is True and secondary._shared_listener_profile == "b"
    assert runner._profile_adapters["b"][Platform("line")] is secondary
