"""After a backend migration, the old command never starts beside the gateway on the gateway's port.

macOS lets ``hermes dashboard --port P`` bind ``127.0.0.1:P`` while the gateway listens on ``*:P`` (the more
specific address wins for loopback clients), so the bind probe alone let the retired backend come back as a
second session owner. Another address of the same port shows the same thing on Linux: the gateway's Webapp
on ``127.0.0.1:P`` and the old command asked for ``127.0.0.2:P``. With the migration's manifest saying the
old backend is retired, the command names the gateway and exits 78; without one nothing changes.
"""
import json

import pytest

from tests.hermes_cli.live_gateway import live_gateway

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    with live_gateway(tmp_path_factory.mktemp("gateway-after-migration")) as live:
        dist = live.user / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>old dashboard</title>", encoding="utf-8")
        live.env["HERMES_WEB_DIST"] = str(dist)  # the old dashboard's bundle, so it gets to its bind
        yield live


def _manifest(gateway, state):
    path = gateway.home / "backend_migration.json"
    if state is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(json.dumps({"state": state}), encoding="utf-8")


def test_the_old_command_on_another_address_of_the_gateways_port_stops_after_a_migration(gateway):
    _manifest(gateway, "done")
    try:
        # 127.0.0.2 is not the loopback the auth gate exempts: sign-in (throwaway) lets it reach its bind.
        result = gateway.cli("dashboard", "--no-open", "--host", "127.0.0.2", "--port", str(gateway.port),
                             extra_env={"HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "t2-user",
                                        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "t2-throwaway-pw"})
    finally:
        _manifest(gateway, None)
    out = result.stdout
    assert result.returncode == 78, out
    assert f"BACKEND_PORT_IN_USE port={gateway.port}" in out
    assert "held by this profile's Hermes gateway" in out
    assert "HERMES_DASHBOARD_READY" not in out


@pytest.mark.parametrize("state", [None, "rolled_back", "restored"])
def test_without_a_retired_backend_the_bind_probe_alone_decides(gateway, state, monkeypatch):
    """As on main: another address of the gateway's port is free to bind when no migration retired the
    old backend here (no manifest, or one that put the old backend back)."""
    from hermes_cli.web_server_lifecycle import _port_bind_conflict
    _manifest(gateway, state)
    try:
        monkeypatch.setenv("HERMES_HOME", str(gateway.home))
        assert _port_bind_conflict("127.0.0.2", gateway.port) is False
    finally:
        _manifest(gateway, None)


def test_with_a_retired_backend_the_gateway_on_that_port_is_named(gateway, monkeypatch):
    from hermes_cli.web_server_port_holder import gateway_port_after_migration
    _manifest(gateway, "done")
    try:
        monkeypatch.setenv("HERMES_HOME", str(gateway.home))
        holder = gateway_port_after_migration(gateway.port, gateway.home)
        assert holder is not None and holder.control_home == gateway.home
        assert gateway_port_after_migration(gateway.port + 1, gateway.home) is None
        assert gateway_port_after_migration(gateway.port, gateway.home.parent / "another-home") is None
    finally:
        _manifest(gateway, None)
