"""Signing out of a gateway's Webapp behind the sign-in gate signs the browsers out (a real gateway).

The gateway binds its Webapp on loopback with a public URL, which puts it behind the sign-in gate, and
the bundled password provider. A browser signs in, opens its chat socket and a host terminal, and a
native client is attached. ``webapp-sign-out`` (what ``hermes webapp --sign-out`` asks for) then ends
the browser's login: its sockets close with 4401, its requests get 401 and the page sends it to sign
in, tickets it minted open nothing, and its cookies replayed elsewhere get nothing, nor do the provider
tokens they held, presented on their own. The native client stays connected, and the browser can sign
in again.
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


@pytest.fixture(scope="module")
def signed_out(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    tmp = tmp_path_factory.mktemp("webapp-gated")
    home, user = tmp / "state", tmp / "user"
    for path in (home, user):
        path.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1")
    result = subprocess.run(
        [sys.executable, str(root / "tests/gateway/fixtures/webapp_gated_sign_out_peer.py"), str(_free_port())],
        cwd=root, env=env, capture_output=True, text=True, timeout=180, check=False)
    receipt = home / "receipt.json"
    assert result.returncode == 0, result.stdout + result.stderr + (receipt.read_text() if receipt.exists() else "")
    return json.loads(receipt.read_text())


def test_the_webapp_is_behind_the_sign_in_gate(signed_out):
    assert signed_out["gate_engaged"] is True and signed_out["surface"] == "webapp"
    assert "basic" in signed_out["providers"]
    assert signed_out["status"]["state"] == "serving"
    assert signed_out["before_sign_in"][0] == 401


def test_a_signed_in_browser_works_before_sign_out(signed_out):
    assert (signed_out["sign_in_status"], signed_out["sessions_before"], signed_out["me_before"]) == (200, 200, 200)
    assert signed_out["chat_open"] is True and signed_out["terminal_open"] is True


def test_signing_out_closes_the_browsers_sockets_and_keeps_native_clients(signed_out):
    assert signed_out["sign_out_ok"] is True
    assert signed_out["chat_close"] == 4401
    assert signed_out["terminal_close"] == 4401
    assert signed_out["native_answers"] is True
    assert signed_out["sign_out_result"]["sign_in_gate"] is True
    assert signed_out["sign_out_result"]["connections_closed"] == 2


def test_after_sign_out_the_browser_must_sign_in_again(signed_out):
    for probe in ("sessions_after", "me_after", "terminal_ticket_after", "replayed_cookies_after"):
        assert signed_out[probe][:2] in ([401, "unauthenticated"], [401, "session_expired"]), probe
    assert signed_out["page_after"][0] == 302 and signed_out["page_after"][2] == "/login"
    assert signed_out["old_terminal_ticket_after"] == 403


def test_after_sign_out_the_tokens_the_browsers_cookies_held_work_nowhere(signed_out):
    assert signed_out["stolen_access_token_after"][:2] == [401, "session_expired"]
    assert signed_out["stolen_refresh_token_after"][:2] == [401, "session_expired"]


def test_the_browser_can_sign_in_again(signed_out):
    assert (signed_out["sign_in_again_status"], signed_out["sessions_after_sign_in_again"]) == (200, 200)
