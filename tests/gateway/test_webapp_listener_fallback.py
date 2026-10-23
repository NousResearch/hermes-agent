"""The gateway keeps a listener for native clients whatever happens to the Webapp's.

When the Webapp's listener cannot start, even after its dashboard mount is up (another process won the
race to listen on the port), the gateway comes up on main's private loopback listener instead: native
Desktop, CLI and TUI clients still connect, the mount carries none of the Webapp's state, and
``hermes webapp --status`` says why the Webapp is not served. Each case runs a real gateway listener
in a child process (``fixtures/webapp_fallback_peer.py``) with its own ``HERMES_HOME``.
"""
import sys

import pytest

from tests.gateway.fixtures.webapp_fallback_peer import free_port as _free_port, run as _run


@pytest.fixture(scope="module")
def lost_listen_race(tmp_path_factory):
    port = _free_port()
    spec = {"webapp": {"enabled": True, "port": port}, "fail_after_mount": True}
    return port, _run(tmp_path_factory.mktemp("lost-listen-race"), spec)


def test_a_webapp_listener_failing_after_its_mount_leaves_native_clients_a_listener(lost_listen_race):
    port, report = lost_listen_race
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["api_origin"].startswith("http://127.0.0.1:")
    assert report["api_origin"] != f"http://127.0.0.1:{port}"
    assert report["native_session_list"] is True
    assert report["status"]["state"] == "unavailable"
    assert f"127.0.0.1:{port}" in report["status"]["reason"]


def test_the_private_listener_after_a_failed_webapp_mount_is_mains(lost_listen_race):
    # No Webapp surface, no Webapp bundle and not the browser's credential: what main's listener has.
    _, report = lost_listen_race
    assert report["state"]["mounted"] is True
    assert report["state"]["surface"] == "<absent>"
    assert report["state"]["web_dist_set"] is False
    assert report["state"]["session_token_is_browser_credential"] is False
    assert "WEBAPP_BUNDLE" not in report["index"]["body"]


def test_a_webapp_that_loses_its_port_replaces_the_browser_credential(lost_listen_race, tmp_path):
    # Open tabs keep sending their credential to whatever answers on the Webapp's port. When that is
    # not this gateway, the credential may have been captured, so it must not open this gateway later.
    _, report = lost_listen_race
    assert report["token_rotated"] is True
    held = _run(tmp_path, {"webapp": {"enabled": True, "port": _free_port()}, "hold_port": True})
    assert held["start_error"] is None, held.get("start_traceback")
    assert held["status"]["state"] == "unavailable"
    assert held["native_session_list"] is True
    assert held["token_rotated"] is True


def test_a_webapp_that_starts_keeps_the_browser_credential(tmp_path):
    report = _run(tmp_path, {"webapp": {"enabled": True, "port": _free_port()}})
    assert report["status"]["state"] == "serving"
    assert report["state"]["session_token_is_browser_credential"] is True
    assert report["token_rotated"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux lets two SO_REUSEADDR sockets bind one address")
def test_the_webapp_owns_its_port_before_mounting_the_dashboard(tmp_path):
    # Two processes may both bind the Webapp's port under SO_REUSEADDR; the first to listen owns it.
    # The gateway listens as soon as it binds, so a rival that tries while the dashboard mounts loses.
    port = _free_port()
    spec = {"webapp": {"enabled": True, "port": port}, "rival_listens_during_mount": True}
    report = _run(tmp_path, spec)
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["rival_listen"] == "EADDRINUSE"
    assert report["status"] == {"state": "serving", "url": f"http://127.0.0.1:{port}", "reason": None}


def test_the_webapp_port_accepts_connections_before_the_dashboard_mounts(tmp_path):
    # Listening right after the bind settles who owns the port before the slow mount starts.
    port = _free_port()
    report = _run(tmp_path, {"webapp": {"enabled": True, "port": port}, "connect_during_mount": True})
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["connect_during_mount"] == "connected"
    assert report["status"]["state"] == "serving"


@pytest.mark.skipif(sys.platform != "win32", reason="SO_EXCLUSIVEADDRUSE is a Windows socket option")
def test_on_windows_the_webapps_fixed_port_is_held_exclusively():
    # Without exclusive use another process may bind the Webapp's port with SO_REUSEADDR (a more
    # specific address then takes those connections). Main's private ephemeral listener is unchanged.
    import socket

    from gateway import run_api
    with run_api._listener_socket(socket.AF_INET, socket.SOCK_STREAM, 0, _free_port()) as fixed, \
            run_api._listener_socket(socket.AF_INET, socket.SOCK_STREAM, 0, 0) as ephemeral:
        assert fixed.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE) == 1
        assert ephemeral.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE) == 0
        fixed.bind(("127.0.0.1", 0))
        fixed.listen()
        with socket.socket() as rival:
            rival.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                rival.bind(fixed.getsockname())


def test_a_webapp_that_needs_sign_in_it_cannot_have_falls_back_to_the_private_listener(tmp_path):
    # The Webapp's own URL needs a sign-in provider and none is registered: the Webapp is unavailable,
    # and native clients keep main's private listener.
    spec = {"webapp": {"enabled": True, "port": _free_port(), "public_url": "https://webapp.example.net"}}
    report = _run(tmp_path, spec)
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["native_session_list"] is True
    assert report["status"]["state"] == "unavailable"
    assert report["state"]["auth_required"] is False
