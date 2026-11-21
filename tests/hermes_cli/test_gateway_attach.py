"""`hermes gateway attach`: the public attach for a third-party local client (#142159).

The real CLI runs against a real control socket with a real ticket store: it prints the owner's
WebSocket URL and one single-use ticket bound to the caller's profile, or the bootstrap's verdict.
"""
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import pytest

from gateway.control_socket import GatewayControlServer
from gateway.runtime_bootstrap import TicketStore
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(request):
    # Short and symlink-free: the socket path must fit sun_path, and the server refuses to mint
    # for a home whose path crosses a symlink (macOS's /var).
    root = Path(tempfile.mkdtemp(prefix="h-")).resolve()
    request.addfinalizer(lambda: shutil.rmtree(root, ignore_errors=True))
    home = root / "home"
    home.mkdir(mode=0o700)
    return home


def _attach(home, *flags):
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "attach", "--json", *flags],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                          check=False)


def _serve(home, tickets):
    """A ready owner's control socket on its own loop; returns a stop callable."""
    identify = {"runtime_protocol": 1, "state": "ready", "served_profiles": [{"home": str(home), "profile_id": str(home)}],
                "capabilities": ["session-authority-v1"], "authority_epoch": 1, "instance_id": "owner",
                "api_origin": "http://127.0.0.1:8765", "supervisor": "none"}
    server = GatewayControlServer(home, verb_handlers={"identify": lambda: identify})
    server.ticket_store = tickets
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    assert asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)

    def stop():
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    return stop


@pytest.mark.platforms("linux", "macos")
def test_attach_prints_the_url_and_a_single_use_profile_ticket(home):
    tickets = TicketStore("owner", frozenset({str(home)}))
    stop = _serve(home, tickets)
    try:
        run = _attach(home)
        _check_attached(run, home, tickets)
    finally:
        stop()  # revokes every ticket, so redeem first


def _check_attached(run, home, tickets):
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    ticket = result["ticket"]
    assert result["state"] == "ready" and result["endpoint"]["profile_id"] == str(home)
    assert result["url"] == "ws://127.0.0.1:8765/api/ws"
    assert result["protocol"] == CANONICAL_GATEWAY_PROTOCOL
    assert result["protocols"] == [CANONICAL_GATEWAY_PROTOCOL, f"hermes-gateway-ticket.{ticket}"]
    # Only the calling process's stdout carries the credential.
    assert ticket not in run.stderr
    assert not [p for p in home.rglob("*") if p.is_file() and ticket in p.read_text(errors="ignore")]
    grant = tickets.redeem(ticket, profile_id=str(home), purpose="interactive")
    assert grant["profile_id"] == str(home) and grant["scope"] == "profile"
    with pytest.raises(PermissionError):
        tickets.redeem(ticket, profile_id=str(home), purpose="interactive")


def test_attach_without_an_owner_reports_the_bootstrap_verdict(home):
    run = _attach(home)
    assert run.returncode == 5, run.stderr
    result = json.loads(run.stdout)
    assert result["state"] == "absent" and result["endpoint"] is None
    assert result["message"] == "gateway absent: not ready" and "gateway absent: not ready" in run.stderr
    assert "ticket" not in result and "url" not in result


@pytest.mark.platforms("linux", "macos")
def test_a_ready_owner_that_refuses_the_ticket_is_one_bounded_verdict(home):
    """No ticket store: the owner answers identify (ready) but refuses session-ticket. A client that
    retries on this reply needs one JSON line, exit 7 and no credential, never a traceback."""
    stop = _serve(home, None)
    try:
        run = _attach(home)
    finally:
        stop()
    assert run.returncode == 7, run.stderr
    assert len(run.stdout.splitlines()) == 1 and "Traceback" not in run.stderr
    result = json.loads(run.stdout)
    assert (result["state"], result["reason_code"]) == ("inaccessible", "ticket_unavailable")
    assert result["message"].startswith("gateway inaccessible: ticket_unavailable")
    assert "ticket" not in result and "url" not in result
