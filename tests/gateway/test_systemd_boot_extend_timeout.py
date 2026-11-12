"""A Type=notify multiplex boot asks systemd for more start time before each serial secondary connect."""

import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform
from hermes_agent.gateway.run import GatewayRunner

pytestmark = pytest.mark.platforms("linux")  # abstract AF_UNIX names are a Linux kernel feature


def _drain(receiver: socket.socket) -> list[str]:
    out = []
    while True:
        try:
            out.append(receiver.recv(4096).decode())
        except BlockingIOError:
            return out


@pytest.mark.asyncio
@pytest.mark.parametrize("watchdog_seconds,notify_socket,connect_timeout",
                         [(30, True, None), (30, False, None), (0, True, None), (30, True, "inf")])
async def test_secondary_boot_extends_systemd_start_timeout(tmp_path, monkeypatch, watchdog_seconds, notify_socket,
                                                           connect_timeout):
    home = tmp_path / ".hermes"
    for name, var in (("alpha", "DISCORD_BOT_TOKEN"), ("beta", "SLACK_BOT_TOKEN")):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("model: {default: test}\n", encoding="utf-8")
        extra = "SLACK_APP_TOKEN=xapp-test\n" if name == "beta" else ""
        (profile / ".env").write_text(f"{var}={name}-token\n{extra}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_agent.hermes_cli.profiles.get_active_profile_name", lambda: "default")
    if connect_timeout is None:
        monkeypatch.delenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", raising=False)
    else:  # an unbounded connect budget has no finite extension; every secondary must still connect
        monkeypatch.setenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", connect_timeout)
    address = f"hermes-test-extend-{os.getpid()}-{watchdog_seconds}-{notify_socket}-{connect_timeout}"
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind("\0" + address)
    receiver.setblocking(False)
    if notify_socket:
        monkeypatch.setenv("NOTIFY_SOCKET", "@" + address)
    else:
        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, systemd_watchdog_seconds=watchdog_seconds)
    runner._running = False
    runner._primary_profile_name = "default"
    runner.adapters, runner._profile_adapters = {}, {}
    runner._failed_platforms, runner._profile_failed_platforms = {}, {}
    runner.pairing_store, runner.pairing_stores = MagicMock(), {}
    runner._busy_text_modes_by_profile, runner._busy_input_modes_by_profile = {}, {}
    runner._register_config_hooks = lambda *a, **kw: None
    runner._configure_profile_adapter = lambda *a: None
    runner._sync_voice_mode_state_to_adapter = lambda *a: None
    runner._restore_secondary_completion_ledgers = lambda *a: None
    runner._adapter_credential_claim = lambda *a: None
    runner._adapter_listener_claim = lambda *a: None
    runner._create_adapter = lambda platform, config: SimpleNamespace(platform=platform)
    seen_before_connect: list[list[str]] = []

    async def connect(adapter, platform):
        seen_before_connect.append(_drain(receiver))
        return True

    runner._connect_initial_adapter_with_timeout = connect
    try:
        assert await runner._start_secondary_profile_adapters() == 2
    finally:
        receiver.close()

    assert len(seen_before_connect) == 2
    if watchdog_seconds and notify_socket and connect_timeout is None:
        # one fresh extension right before EACH serial secondary connect, sized past its connect budget
        budget = runner._platform_connect_timeout_secs(Platform.DISCORD, initial=True)
        for datagrams in seen_before_connect:
            assert len(datagrams) == 1 and datagrams[0].startswith("EXTEND_TIMEOUT_USEC=")
            assert int(datagrams[0].split("=", 1)[1]) > budget * 1_000_000
    else:
        assert seen_before_connect == [[], []]
