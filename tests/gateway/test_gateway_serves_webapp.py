"""The gateway's own listener serves the browser-hosted Desktop when ``webapp.enabled`` is on.

With it on, the dashboard mount serves the Webapp bundle without the session token in the page, the
browser credential is the one kept in the gateway home (so it survives a restart), the host terminal
registry runs, and a browser holding that credential gets a socket ticket and opens a gateway session
on the same origin. The launcher's link comes over the control socket and spends once, on this host's
launch path and on that of macOS and the BSDs (taken here on any host), where the link is printed for
the user to open by hand; signing out
replaces the credential and closes the browser sockets it opened, while native clients stay
connected. Each browser socket ticket opens only its own socket. With it off, the listener is main's
private one: no Webapp surface, no ticket route, no credential file, no launch link.
"""
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _run(tmp_path, mode, platform="this platform"):
    root = Path(__file__).resolve().parents[2]
    home, user, dist = tmp_path / "state", tmp_path / "user", tmp_path / "dist-webapp"
    for path in (home, user, dist):
        path.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>WEBAPP_BUNDLE</body></html>", encoding="utf-8")
    port = _free_port()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1")
    result = subprocess.run(
        [sys.executable, str(root / "tests/gateway/fixtures/webapp_listener_peer.py"), mode, str(port), str(dist),
         platform],
        cwd=root, env=env, capture_output=True, text=True, timeout=120, check=False)
    receipt = home / "receipt.json"
    assert result.returncode == 0, result.stdout + result.stderr + (receipt.read_text() if receipt.exists() else "")
    return port, json.loads(receipt.read_text())


# Every check runs on this host's launch path and on that of macOS and the BSDs, where nothing tells who
# owns the browser's connection and the launch link is printed for the user to open by hand.
@pytest.fixture(scope="module", params=["this platform", "macos"])
def webapp_on(request, tmp_path_factory):
    return _run(tmp_path_factory.mktemp("webapp-on"), "on", request.param)


@pytest.fixture(scope="module")
def webapp_off(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("webapp-off"), "off")


def test_the_listener_binds_the_configured_port_and_native_clients_dial_it(webapp_on):
    port, report = webapp_on
    assert report["api_origin"] == f"http://127.0.0.1:{port}"


def test_the_listener_serves_the_webapp_bundle_without_the_credential(webapp_on):
    _, report = webapp_on
    assert (report["index_status"], report["index_is_webapp"]) == (200, True)
    assert report["index_carries_token"] is False
    assert report["surface"] == "webapp"


def test_the_browser_credential_survives_a_gateway_restart(webapp_on):
    _, report = webapp_on
    assert report["session_token_is_seeded"] is True


def test_the_host_terminal_runs_with_the_listener(webapp_on):
    _, report = webapp_on
    assert report["host_terminals"] is True
    assert report["host_terminals_after_stop"] is False


def test_a_browser_with_the_credential_opens_a_gateway_session(webapp_on):
    _, report = webapp_on
    assert report["ticket_status"] == 200
    assert report["browser_session"] is True


def test_the_launch_link_from_the_control_socket_hands_over_the_credential_once(webapp_on):
    port, report = webapp_on
    assert report["status"] == {"state": "serving", "url": f"http://127.0.0.1:{port}", "reason": None}
    assert (report["launch_ok"], report["launch_page"]) == (True, True)
    # Marked private (printed, never handed to a browser launcher) exactly where the platform cannot
    # check who redeems it.
    assert report["launch_private"] is report["launch_printed_for_the_user"]
    assert (report["handoff_status"], report["handoff_gives_credential"]) == (200, True)
    assert report["handoff_reuse_status"] == 403


def test_signing_out_cuts_off_the_old_browser_credential(webapp_on):
    _, report = webapp_on
    assert report["sign_out_ok"] is True
    assert report["old_credential_status"] == 401
    assert report["token_file_rotated"] is True


def test_each_browser_socket_ticket_opens_only_its_own_socket(webapp_on):
    _, report = webapp_on
    assert report["terminal_opens_with_its_ticket"] is True
    assert report["chat_opens_before_sign_out"] is True
    assert report["terminal_refuses_session_token"] == 403
    assert report["terminal_refuses_chat_ticket"] == 403
    assert report["chat_refuses_terminal_ticket"] == 403


def test_signing_out_closes_the_browser_sockets_and_keeps_native_clients(webapp_on):
    _, report = webapp_on
    assert report["chat_close_after_sign_out"] == 4401
    assert report["terminal_close_after_sign_out"] == 4401
    assert report["native_answers_after_sign_out"] is True


def test_without_the_webapp_the_listener_stays_mains_private_one(webapp_off):
    _, report = webapp_off
    assert report["index_is_webapp"] is False
    assert report["surface"] != "webapp"
    assert report["host_terminals"] is False
    assert report["token_file"] is False
    assert report["ticket_status"] in {403, 404}
    assert report["status"] == {"state": "off", "url": None, "reason": None}
