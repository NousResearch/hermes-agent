"""Which port Web Access serves a gateway's Webapp on (``hermes_cli.webapp_access._port_to_serve``).

The default profile's gateway serves the documented address, 9120 unless ``webapp.port`` says otherwise.
A standalone profile has a gateway of its own; turning Web Access on there without a ``webapp.port`` used
to bind 9120 too, which the default gateway's Webapp holds, so nothing was served. Start now gives such a
profile the first port above 9120 that no other Hermes home claims and nothing listens on, and writes it to
that profile's ``webapp.port``, so the gateway binds it, the launcher prints and opens it, the sign-in
callback names it, and it stays the same across restarts. A ``webapp.port`` the user set always stays.
No test binds a real port: what listens is faked, and the probe itself is checked on OS-chosen ports.
"""
import socket
from pathlib import Path

import pytest

from hermes_cli import webapp_access


def _write(path: Path, text: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A Hermes root with a standalone profile ``solo`` and a sibling ``other`` that claims 9122; 9121 is
    in use by something else. The process runs as ``solo`` (Web Access scopes to its gateway's home)."""
    root = _write(tmp_path / ".hermes", "")
    solo = _write(root / "profiles" / "solo", "gateway:\n  standalone: true\n")
    _write(root / "profiles" / "other", "gateway:\n  standalone: true\nwebapp:\n  port: 9122\n")
    listening = {9121}
    probed: list[tuple[str, int]] = []

    def port_free(host, port):
        probed.append((host, port))
        return port not in listening

    monkeypatch.setattr(webapp_access, "_port_free", port_free, raising=False)
    monkeypatch.setattr(webapp_access, "nous_signed_in", lambda: True)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.50")
    monkeypatch.setenv("HERMES_HOME", str(solo))
    return {"root": root, "solo": solo, "listening": listening, "probed": probed, "monkeypatch": monkeypatch}


def _webapp_block(home: Path) -> dict:
    from hermes_cli.config import read_user_config_raw

    return read_user_config_raw(home / "config.yaml").get("webapp") or {}


def test_a_standalone_profile_without_a_port_gets_a_free_one_that_every_surface_uses(machine):
    from gateway.webapp_listener import dial_origin, listener_plan

    assert webapp_access.gateway_home() == machine["solo"]
    planned = webapp_access.plan("lan")
    # 9120 is the default profile's, 9121 is in use, 9122 is the other profile's.
    assert planned[-1] == "http://192.168.1.50:9123/auth/callback"
    host, port, public_url, _callback = planned
    assert (host, port, public_url) == ("0.0.0.0", 9123, "")
    webapp_access.serve(host, public_url, port)
    assert _webapp_block(machine["solo"]) == {"enabled": True, "host": "0.0.0.0", "port": 9123}
    # The gateway binds that port, and the address the launcher prints and opens is built on it.
    bind_host, bind_port, mount, _reason = listener_plan(machine["solo"])
    assert (bind_host, bind_port) == ("0.0.0.0", 9123) and mount is not None
    assert dial_origin(bind_host, bind_port) == "http://127.0.0.1:9123"
    assert webapp_access.access_status()["port"] == 9123
    assert _webapp_block(machine["root"]) == {}


def test_the_chosen_port_stays_for_that_profile(machine):
    host, port, public_url, _ = webapp_access.plan("lan")
    webapp_access.serve(host, public_url, port)
    webapp_access.turn_off()
    machine["probed"].clear()
    machine["listening"].add(9123)  # its own gateway may hold it now
    for mode, url, expected in (("lan", None, "http://192.168.1.50:9123/auth/callback"),
                                ("public", "https://solo.example.test", "https://solo.example.test/auth/callback")):
        host, port, public_url, callback = webapp_access.plan(mode, url)
        assert port is None  # nothing to write: the config already has it
        assert callback == expected
        webapp_access.serve(host, public_url, port)
        assert _webapp_block(machine["solo"])["port"] == 9123
    assert machine["probed"] == []


def test_the_own_url_mode_also_gets_a_free_port(machine):
    host, port, _public_url, callback = webapp_access.plan("public", "https://solo.example.test")
    assert (host, port, callback) == ("127.0.0.1", 9123, "https://solo.example.test/auth/callback")
    assert machine["probed"][-1] == ("127.0.0.1", 9123)


def test_the_default_profile_keeps_the_documented_port_and_writes_none(machine):
    machine["monkeypatch"].setenv("HERMES_HOME", str(machine["root"]))
    host, port, public_url, callback = webapp_access.plan("lan")
    assert port is None and callback == "http://192.168.1.50:9120/auth/callback"
    webapp_access.serve(host, public_url, port)
    assert "port" not in _webapp_block(machine["root"])
    assert machine["probed"] == []


@pytest.mark.parametrize("configured", [9555, 9120, "not-a-port"])
def test_a_port_the_user_set_is_never_replaced(machine, configured):
    """Even one another gateway holds, or an invalid one: the gateway then says why by the setting's name."""
    _write(machine["solo"], f"gateway:\n  standalone: true\nwebapp:\n  port: {configured}\n")
    host, port, public_url, _callback = webapp_access.plan("lan")
    assert port is None and machine["probed"] == []
    webapp_access.serve(host, public_url, port)
    assert _webapp_block(machine["solo"])["port"] == configured


def test_a_port_the_managed_scope_sets_is_the_one_the_callback_names(machine, tmp_path):
    """An administrator's ``webapp.port`` wins over the profile's config when the gateway loads it, and
    saving never writes over it: Start registers the callback on that port and writes none."""
    from gateway.webapp_listener import listener_plan
    from hermes_cli import managed_scope

    managed = _write(tmp_path / "managed", "webapp:\n  port: 9555\n")
    machine["monkeypatch"].setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    host, port, public_url, callback = webapp_access.plan("lan")
    assert (port, callback) == (None, "http://192.168.1.50:9555/auth/callback")
    assert machine["probed"] == []
    webapp_access.serve(host, public_url, port)
    assert "port" not in _webapp_block(machine["solo"])
    bind_host, bind_port, mount, _reason = listener_plan(machine["solo"])
    assert (bind_host, bind_port) == ("0.0.0.0", 9555) and mount is not None


def test_with_no_free_port_start_refuses_and_names_the_setting(machine):
    machine["listening"].update(range(9121, 9221))
    with pytest.raises(webapp_access.WebappAccessError) as refused:
        webapp_access.plan("lan")
    assert "webapp.port" in str(refused.value) and str(machine["solo"] / "config.yaml") in str(refused.value)
    assert "port" not in _webapp_block(machine["solo"])


def test_two_profiles_starting_at_once_never_save_the_same_port(machine):
    """Each standalone profile's gateway has its own change lock, so two first Starts can overlap. The
    probes wait for each other: without a lock both gateways share, both would see the same free port
    and save it. Each runs as its own profile (a context-local home, as its gateway's scope does)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    twin = _write(machine["root"] / "profiles" / "twin", "gateway:\n  standalone: true\n")
    both_probe = threading.Barrier(2, timeout=2)

    def port_free(_host, port):
        try:
            both_probe.wait()  # until the other probes too, or (holding the lock) it never can
        except threading.BrokenBarrierError:
            pass
        return port not in machine["listening"]

    machine["monkeypatch"].setattr(webapp_access, "_port_free", port_free)

    def start(home: Path) -> int:
        token = set_hermes_home_override(home)
        try:
            host, port, public_url, _callback = webapp_access.plan("lan")
            webapp_access.serve(host, public_url, port)
            return _webapp_block(home)["port"]
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        starts = [pool.submit(start, home) for home in (machine["solo"], twin)]
        chosen = sorted(started.result(timeout=30) for started in starts)
    # 9120 is the default profile's, 9121 is in use, 9122 is the other profile's.
    assert chosen == [9123, 9124]


def test_a_start_that_fails_after_choosing_keeps_the_port(machine):
    """The port is saved when it is chosen, before the bundle build and the registration, so another
    profile never takes it; a Start that then fails leaves the profile that port and nothing else."""
    webapp_access.plan("lan")
    assert _webapp_block(machine["solo"]) == {"port": 9123}
    machine["probed"].clear()
    _host, port, _public_url, callback = webapp_access.plan("lan")
    assert (port, callback, machine["probed"]) == (None, "http://192.168.1.50:9123/auth/callback", [])


def test_the_probe_sees_a_port_a_listener_holds():
    """On a port the OS picked, not a real service's."""
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        assert webapp_access._port_free("127.0.0.1", port) is False
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert webapp_access._port_free("127.0.0.1", free) is True
