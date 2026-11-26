"""Per-profile startup-watchdog progress lease and INFO logging during multiplex MCP discovery (#160479)."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import pytest

import hermes_startup_watchdog as sw
from gateway.config import GatewayConfig
from hermes_constants import get_hermes_home
from hermes_startup_watchdog import StartupWatchdogHandle, arm_startup_watchdog, disarm_startup_watchdog
from tools import mcp_tool as _mcp_core


def _expected_per_profile_lease_s() -> float:
    return float(
        _mcp_core._MCP_DISCOVERY_PASS_MAX_SEC
        + _mcp_core._MCP_DISCOVERY_LOCK_MAX_RETRIES * _mcp_core._MCP_DISCOVERY_LOCK_RETRY_DELAY_S
    )


@pytest.fixture(autouse=True)
def _isolate_watchdog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    yield
    sw._reset_for_tests()


@pytest.mark.asyncio
async def test_multiplex_mcp_discovery_reports_per_profile_startup_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every multiplexed profile renews the startup-watchdog lease before its discovery runs,
    using the honest bound derived from mcp_tool's pass ceiling + lock-wait budget."""
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery

    homes = [
        ("default", tmp_path / "default"),
        ("alpha", tmp_path / "alpha"),
        ("beta", tmp_path / "beta"),
    ]
    for _name, home in homes:
        home.mkdir()

    events: list[tuple] = []

    def fake_report(expected_s: float, phase: str = "") -> None:
        events.append(("lease", expected_s, phase))

    def fake_discover() -> list[str]:
        events.append(("discover", get_hermes_home().name))
        return []

    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: homes)
    monkeypatch.setattr(sw, "report_startup_progress", fake_report)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", fake_discover)

    await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True))

    expected_lease = _expected_per_profile_lease_s()
    assert 0 < expected_lease <= sw._MAX_LEASE_S
    assert events == [
        ("lease", expected_lease, "mcp_discovery:default"),
        ("discover", "default"),
        ("lease", expected_lease, "mcp_discovery:alpha"),
        ("discover", "alpha"),
        ("lease", expected_lease, "mcp_discovery:beta"),
        ("discover", "beta"),
    ]


@pytest.mark.asyncio
async def test_multiplex_mcp_discovery_logs_start_and_finish_with_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Each multiplexed profile emits INFO start and finish lines (with elapsed seconds)."""
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery

    homes = [("default", tmp_path / "default"), ("ops", tmp_path / "ops")]
    for _name, home in homes:
        home.mkdir()

    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: homes)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", list)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True))

    info_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO]
    assert len(info_messages) == 4
    assert "starting" in info_messages[0].lower() and "default" in info_messages[0]
    assert "finished" in info_messages[1].lower() and "default" in info_messages[1]
    assert re.search(r"\d+\.\d+s", info_messages[1])
    assert "starting" in info_messages[2].lower() and "ops" in info_messages[2]
    assert "finished" in info_messages[3].lower() and "ops" in info_messages[3]
    assert re.search(r"\d+\.\d+s", info_messages[3])


@pytest.mark.asyncio
async def test_multiplex_mcp_discovery_failure_warns_and_continues_subsequent_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing profile logs a WARNING (no finish INFO) and does not block subsequent profiles
    from receiving their startup-watchdog lease and running discovery."""
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery

    homes = [("broken", tmp_path / "broken"), ("healthy", tmp_path / "healthy")]
    for _name, home in homes:
        home.mkdir()

    leases: list[tuple[float, str]] = []
    discovered: list[str] = []

    def fake_report(expected_s: float, phase: str = "") -> None:
        leases.append((expected_s, phase))

    def fake_discover() -> list[str]:
        name = get_hermes_home().name
        if name == "broken":
            raise RuntimeError("mcp server unreachable")
        discovered.append(name)
        return []

    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: homes)
    monkeypatch.setattr(sw, "report_startup_progress", fake_report)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", fake_discover)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True))

    expected_lease = _expected_per_profile_lease_s()
    assert leases == [
        (expected_lease, "mcp_discovery:broken"),
        (expected_lease, "mcp_discovery:healthy"),
    ]
    assert discovered == ["healthy"]

    warn_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    info_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO]
    assert any("broken" in msg and "failed" in msg.lower() for msg in warn_messages)
    assert not any("finished" in msg.lower() and "broken" in msg for msg in info_messages)
    assert any("starting" in msg.lower() and "healthy" in msg for msg in info_messages)
    assert any("finished" in msg.lower() and "healthy" in msg and re.search(r"\d+\.\d+s", msg) for msg in info_messages)


@pytest.mark.asyncio
async def test_multiplex_mcp_discovery_keeps_armed_watchdog_alive_across_slow_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces #160479: with zero CPU progress and N serial I/O-bound profiles exceeding the
    watchdog deadline, per-profile leases keep the armed watchdog from firing."""
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery

    homes = [(f"p{i}", tmp_path / f"p{i}") for i in range(3)]
    for _name, home in homes:
        home.mkdir()

    fired = threading.Event()
    monkeypatch.setattr(StartupWatchdogHandle, "_exit", staticmethod(lambda _code: fired.set()))
    monkeypatch.setattr(StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(lambda: 0.0))
    monkeypatch.setattr(sw, "_MAX_CPU_EXTENSIONS", 0)
    monkeypatch.setattr(sw, "_POLL_SLICE_S", 0.02)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: homes)
    def slow_discover() -> list[str]:
        time.sleep(0.12)  # health: allow HX013 -- simulates I/O-bound discovery outlasting watchdog timeout
        return []

    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", slow_discover)

    handle = arm_startup_watchdog(timeout_s=0.15)
    assert handle is not None
    try:
        await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True))
        assert not fired.is_set(), "watchdog fired during active multiplex MCP discovery"
        with handle._state_lock:
            assert handle._lease_count == 3
            assert handle._lease_phase == "mcp_discovery:p2"
    finally:
        disarm_startup_watchdog()


@pytest.mark.asyncio
async def test_multiplex_mcp_discovery_empty_config_runs_cleanly_without_stacking_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profiles with no configured mcp_servers return immediately from discover_mcp_tools without
    acquiring the discovery lock and without cumulatively stacking the watchdog lease."""
    import gateway.run as gateway_run
    from tools import mcp_tool_config as _mcp_config
    from tools import mcp_tool_discovery as _mcp_discovery

    homes = [("empty_a", tmp_path / "empty_a"), ("empty_b", tmp_path / "empty_b")]
    for _name, home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("model: default\n", encoding="utf-8")

    lock_calls = 0

    def fail_if_locked():
        nonlocal lock_calls
        lock_calls += 1
        raise AssertionError("discovery lock should not be acquired when mcp_servers is empty")

    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: homes)
    monkeypatch.setattr(_mcp_config, "_portable_mcp_servers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_mcp_discovery, "_acquire_discovery_lock_with_retry", fail_if_locked)

    handle = arm_startup_watchdog(timeout_s=60.0)
    assert handle is not None
    try:
        await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True))
        assert lock_calls == 0
        with handle._state_lock:
            remaining = handle._lease_until - time.monotonic()
            assert handle._lease_count == 2
            assert handle._lease_phase == "mcp_discovery:empty_b"
        assert remaining <= _expected_per_profile_lease_s() + 1.0
    finally:
        disarm_startup_watchdog()


@pytest.mark.asyncio
async def test_single_profile_mcp_discovery_preserves_existing_behavior(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Single-profile mode keeps the direct unscoped call with no multiplex loop or profile lease."""
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery

    discovered: list[str] = []
    leases: list[tuple[float, str]] = []

    monkeypatch.setattr(
        gateway_run,
        "_multiplex_profile_homes",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("should not enumerate multiplex profiles")),
    )
    monkeypatch.setattr(sw, "report_startup_progress", lambda expected_s, phase="": leases.append((expected_s, phase)))
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda: discovered.append("called") or [])

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=False))

    assert discovered == ["called"]
    assert leases == []
    assert [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO] == []
