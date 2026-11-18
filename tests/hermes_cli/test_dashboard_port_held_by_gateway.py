"""``hermes dashboard`` / ``hermes serve`` on the port this home's gateway serves the Webapp on.

A host moved to the gateway keeps the Webapp on the old backend's address; a launcher left behind used to
print the generic port-in-use hint and exit 75 (EX_TEMPFAIL: try again), although no retry can succeed.
When the gateway's own control socket proves its listener holds the port, the command says so and exits
78 (EX_CONFIG), keeping the machine-readable ``BACKEND_PORT_IN_USE`` line first. A port held by anything
else keeps exactly the old answer (``test_dashboard_port_free_parity.py``).
"""
import secrets
import socket

import pytest

from tests.hermes_cli.live_gateway import live_gateway

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    with live_gateway(tmp_path_factory.mktemp("gateway-holds-port")) as live:
        dist = live.user / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>old dashboard</title>", encoding="utf-8")
        live.env["HERMES_WEB_DIST"] = str(dist)  # the old dashboard's bundle, so it gets to its bind
        yield live


def _first_line(out: str) -> str:
    return next(line for line in out.splitlines() if line.strip())


@pytest.mark.parametrize("command, name", [(["dashboard", "--no-open"], "hermes dashboard"),
                                           (["serve"], "hermes serve")])
def test_the_old_command_on_the_gateways_port_names_the_gateway_and_stops_retrying(gateway, command, name):
    result = gateway.cli(*command, "--port", str(gateway.port))
    out = result.stdout
    assert result.returncode == 78, out
    assert f"BACKEND_PORT_IN_USE port={gateway.port}" in out
    assert out.count("BACKEND_PORT_IN_USE") == 1
    assert "held by this profile's Hermes gateway" in out
    assert "serves the Webapp" in out
    assert f"`{name}` is no longer needed here" in out
    assert "hermes webapp" in out
    assert "HERMES_BACKEND_READY" not in out


def test_the_sentinel_is_the_first_line_on_stdout(gateway):
    result = gateway.cli("serve", "--port", str(gateway.port))
    assert _first_line(result.stdout).startswith(f"BACKEND_PORT_IN_USE port={gateway.port}"), result.stdout


def _ipv6_loopback_listener(port: int) -> socket.socket | None:
    try:
        holder = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return None
    try:
        holder.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        holder.bind(("::1", port))
        holder.listen(1)
    except OSError:
        holder.close()
        return None
    return holder


@pytest.mark.parametrize("host", ["::1", "::"])
def test_another_address_on_the_gateways_port_held_by_something_else_keeps_the_old_answer(gateway, host):
    """The gateway listens on 127.0.0.1:P; another process holds [::1]:P (IPv6 only). ``--host ::1``
    collides with that process, not the gateway. ``--host ::`` collides with both, but nothing proves the
    gateway's IPv4 listener is what the IPv6 bind hit. Either way it gets main's hint and exit 75 (78
    would park a supervisor for good)."""
    holder = _ipv6_loopback_listener(gateway.port)
    if holder is None:
        pytest.skip("no IPv6 loopback on this host")
    with holder:
        # A wildcard bind needs a sign-in before it gets to the port: a throwaway deployment token.
        result = gateway.cli("serve", "--host", host, "--port", str(gateway.port),
                             extra_env={"HERMES_DASHBOARD_SESSION_TOKEN": secrets.token_urlsafe(24)})
    out = result.stdout
    assert result.returncode == 75, out
    assert out.splitlines() == [
        f"BACKEND_PORT_IN_USE port={gateway.port}",
        f"  Port {gateway.port} on {host} is already in use — likely another 'hermes serve' / 'hermes dashboard' "
        "backend or the Hermes gateway. Stop the other process, or pass --port <other> "
        "(--port 0 picks a free ephemeral port).",
    ], out


def test_the_old_command_points_a_windows_user_at_task_scheduler_and_the_startup_folder():
    from hermes_cli.web_server_port_holder import launcher_examples
    assert launcher_examples("win32") == "Task Scheduler task or Startup-folder entry"
    assert launcher_examples("linux") == launcher_examples("darwin") == "systemd unit or launchd job"


@pytest.mark.parametrize("state, drain_reason, counts", [
    ("draining", "restart_wait", False), ("draining", "idle_exit", False), ("ready", None, True)])
def test_only_a_ready_gateway_is_named_as_the_ports_holder(tmp_path, state, drain_reason, counts):
    """A gateway in its restart wait still answers ``identify`` with an endpoint (a client may finish a
    parked turn), but it is going away: the old command keeps exit 75, which a supervisor retries,
    rather than 78, which ``RestartPreventExitStatus=78`` parks for good."""
    import asyncio

    from hermes_cli.web_server_port_holder import gateway_holding_port
    from tests.hermes_cli.test_gateway_runtime import control_peer

    home = tmp_path / "profile"
    identity = str(home.resolve())
    payload = {"protocol": 1, "runtime_protocol": 1, "instance_id": "owner", "authority_epoch": 1,
               "hermes_home": identity, "state": state, "api_origin": "http://127.0.0.1:41817",
               "supervisor": "none", "served_profiles": [{"profile_id": identity, "home": identity}],
               "capabilities": ["session-authority-v1"], **({"drain_reason": drain_reason} if drain_reason else {})}

    async def probe():
        async with control_peer(home, payload):
            return await asyncio.to_thread(gateway_holding_port, 41817, home)

    assert (asyncio.run(probe()) is not None) is counts


def test_a_wildcard_request_on_a_loopback_gateway_promises_no_other_device(gateway):
    """The gateway listens on 127.0.0.1 only. ``--host 0.0.0.0`` collides with it (exit 78), but nothing
    on another device can reach that listener, so the hint names the listener and points at
    ``hermes webapp`` instead of telling the user to come in from another machine."""
    result = gateway.cli("serve", "--host", "0.0.0.0", "--port", str(gateway.port),
                         extra_env={"HERMES_DASHBOARD_SESSION_TOKEN": secrets.token_urlsafe(24)})
    out = result.stdout
    assert result.returncode == 78, out
    assert "another device" not in out, out
    assert f"Gateway listener: http://127.0.0.1:{gateway.port}." in out, out
    assert "hermes webapp" in out, out


def test_the_hint_names_the_live_listener_and_writes_nothing_after_a_config_edit(tmp_path):
    """``webapp.public_url`` edited after the gateway started is not live yet: the hint names what the
    gateway really serves, and asking it reads no config (so no config backup is written)."""
    import json

    from tests.hermes_cli.live_gateway import live_gateway as boot

    with boot(tmp_path) as live:
        config = json.loads((live.home / "config.yaml").read_text(encoding="utf-8"))
        config["webapp"]["public_url"] = "https://pending.example.test"
        (live.home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        before = sorted(str(p.relative_to(live.home)) for p in live.home.rglob("*") if "backups" in p.parts)
        result = live.cli("serve", "--port", str(live.port))
        after = sorted(str(p.relative_to(live.home)) for p in live.home.rglob("*") if "backups" in p.parts)
    out = result.stdout
    assert result.returncode == 78, out
    assert "pending.example.test" not in out, out
    assert f"Gateway listener: http://127.0.0.1:{live.port}." in out, out
    assert after == before, after
