"""Multiplex secondaries that enable shared-ingress Relay must not be
silently unserved: one INFO per skipped platform, a stamped ``<profile>:<platform>`` status
entry, and the loud "not being served" WARNING when no profile (default included) runs it."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner.adapters = {}
    runner._failed_platforms = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._background_tasks = set()
    runner._snapshot_profile_busy_modes = lambda *a, **k: None
    runner._platform_lock_takeover_on_start = True
    runner._startup_fail_fatal_config = lambda reason: None
    return runner


def _install_secondary(monkeypatch, runner, stamps):
    @contextmanager
    def fake_scope(profile_home, *, hydrate_secrets=True):
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: {})
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={Platform.RELAY: PlatformConfig(enabled=True)},
        ),
    )
    monkeypatch.setattr(
        runner, "_update_platform_runtime_status",
        lambda key, **kw: stamps.append((key, kw)),
    )
    runner._create_adapter = lambda platform, config: pytest.fail("shared ingress must never build a secondary adapter")


@pytest.mark.asyncio
async def test_secondary_relay_without_primary_is_reported_unserved(monkeypatch, caplog):
    runner = _runner()
    stamps = []
    _install_secondary(monkeypatch, runner, stamps)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        assert await runner._start_one_profile_adapters("work", Path("/profiles/work"), {}) == 0
        # The secondary was skipped: the startup phase folds it into the loud warning.
        aborted, _count = await runner._start_secondary_profiles(0, [])

    assert aborted is False
    infos = [r for r in caplog.records if r.levelno == logging.INFO and "not served" in r.getMessage()]
    assert infos and "'work'" in infos[0].getMessage() and "default profile" in infos[0].getMessage()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("relay" in w and "not being served" in w and "work" in w for w in warnings), warnings
    # Stamped so `hermes gateway status --profile work` / the dashboard can show the reason.
    assert ("work:relay", {
        "platform_state": "disabled", "error_code": "multiplex_shared_ingress",
        "error_message": "not served under multiplex (shared ingress owned by default)",
    }) in stamps
    # ...and the CLI reader turns that stamp into the status line.
    from gateway.status import write_runtime_status
    from hermes_cli import gateway_multiplex_served as served
    key, kw = next(s for s in stamps if s[0] == "work:relay")
    write_runtime_status(platform=key, **kw)
    monkeypatch.setattr(served, "live_default_gateway_pid", lambda: 4242)
    assert served.served_profile_unserved_platforms("work") == {
        "relay": "not served under multiplex (shared ingress owned by default)"}
    assert served.served_profile_unserved_platforms("other") == {}


@pytest.mark.asyncio
async def test_secondary_relay_served_by_default_is_not_warned(monkeypatch, caplog):
    """The default owns Relay: the secondary IS served through the shared adapter — no WARNING,
    the INFO line still explains where the channel lives."""
    runner = _runner()
    runner.adapters[Platform.RELAY] = object()
    _install_secondary(monkeypatch, runner, [])

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._start_one_profile_adapters("work", Path("/profiles/work"), {})
        await runner._start_secondary_profiles(1, [])

    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "not being served" in r.getMessage()]
    assert [r for r in caplog.records if r.levelno == logging.INFO and "not served" in r.getMessage()]
