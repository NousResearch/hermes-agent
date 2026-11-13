"""`hermes gateway tickets`: HTTP tickets for a client that reaches the gateway over SSH.

The real CLI runs as a subprocess, as an SSH exec would run it: `--probe` against nothing, and
`--serve` against a real control socket with a real ticket store, then against a real gateway
whose HTTP auth redeems the ticket.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import httpx
import pytest

from gateway.runtime_bootstrap import TicketStore
from tests.gateway.test_native_http_auth import daemon, headers
from tests.hermes_cli.test_gateway_attach import REPO, _serve


@pytest.fixture
def home(request):
    # Short and symlink-free, as the control socket needs (see test_gateway_attach's fixture).
    root = Path(tempfile.mkdtemp(prefix="h-")).resolve()
    request.addfinalizer(lambda: shutil.rmtree(root, ignore_errors=True))
    home = root / "home"
    home.mkdir(mode=0o700)
    return home


def _command(home, mode):
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}
    return [sys.executable, "-m", "hermes_cli.main", "gateway", "tickets", mode, "--json"], env


def _run(home, mode, stdin=""):
    argv, env = _command(home, mode)
    return subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
                          input=stdin, check=False)


@contextmanager
def _channel(home):
    """One open `--serve` channel: yields (hello, ask, done). ask(line) writes a line and reads its
    reply; done["stderr"] is the process's stderr once the block has closed stdin."""
    argv, env = _command(home, "--serve")
    with subprocess.Popen(argv, cwd=REPO, env=env, text=True, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        def ask(line):
            process.stdin.write(line + "\n")
            process.stdin.flush()
            return json.loads(process.stdout.readline())
        done = {}
        try:
            yield json.loads(process.stdout.readline()), ask, done
            # EOF on stdin ends the channel cleanly, with nothing more on stdout.
            rest, stderr = process.communicate(timeout=60)
            assert (process.returncode, rest) == (0, ""), stderr
            done["stderr"] = stderr
        finally:
            if process.poll() is None:
                process.kill()


def test_probe_names_the_protocol_and_purposes_without_a_gateway(home):
    run = _run(home, "--probe")
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"protocol": 1, "purposes": ["native-http"]}
    assert len(run.stdout.splitlines()) == 1
    # It started nothing: there is still no gateway for this home.
    assert json.loads(_run(home, "--serve").stdout)["error"]["code"] == "absent"


def test_serve_without_a_gateway_is_one_error_line_and_a_failing_exit(home):
    run = _run(home, "--serve", stdin='{"id":1,"purpose":"native-http"}\n')
    assert run.returncode == 5, run.stderr
    assert [json.loads(line) for line in run.stdout.splitlines()] == [
        {"type": "error", "error": {"code": "absent", "reason_code": None, "message": "gateway absent: not ready"}}]
    assert "Traceback" not in run.stderr


def test_a_mode_is_required(home):
    argv, env = _command(home, "--json")
    run = subprocess.run(argv[:-1], cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
                         stdin=subprocess.DEVNULL, check=False)
    assert run.returncode == 2 and run.stdout == ""


@pytest.mark.platforms("linux", "macos")
def test_serve_answers_each_request_line_with_one_single_use_http_ticket(home):
    tickets = TicketStore("owner", frozenset({str(home)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (hello, ask, done):
            assert (hello["type"], hello["protocol"]) == ("hello", 1)
            endpoint = hello["endpoint"]
            assert endpoint["profile_id"] == str(home) and endpoint["instance_id"] == "owner"
            assert endpoint["api_origin"] == "http://127.0.0.1:8765"
            assert endpoint["capabilities"] == ["session-authority-v1"]

            first, second = ask('{"id":1,"purpose":"native-http"}'), ask('{"id":"req-2","purpose":"native-http"}')
            assert {k: v for k, v in first.items() if k != "ticket"} == {"id": 1, "ok": True, "expires_in": 30}
            assert (second["id"], second["ok"], second["expires_in"]) == ("req-2", True, 30)
            assert first["ticket"] != second["ticket"]
            for reply in (first, second):
                # The WebSocket (purpose `interactive`) refuses it; the HTTP purpose takes it once.
                with pytest.raises(PermissionError):
                    tickets.redeem(reply["ticket"], profile_id=str(home), purpose="interactive")
                grant = tickets.redeem(reply["ticket"], profile_id=None, purpose="native-http")
                assert grant["profile_id"] == str(home) and grant["scope"] == "profile"
                with pytest.raises(PermissionError):
                    tickets.redeem(reply["ticket"], profile_id=None, purpose="native-http")
    finally:
        stop()
    for reply in (first, second):
        assert reply["ticket"] not in done["stderr"]
        assert not [p for p in home.rglob("*") if p.is_file() and reply["ticket"] in p.read_text(errors="ignore")]


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("line, request_id, code", [
    ("not json", None, "invalid_request"),
    ("", None, "invalid_request"),
    ('["native-http"]', None, "invalid_request"),
    ('{"purpose":"native-http"}', None, "invalid_request"),
    ('{"id":true,"purpose":"native-http"}', None, "invalid_request"),
    ('{"id":{"a":1},"purpose":"native-http"}', None, "invalid_request"),
    ('{"id":4}', 4, "unsupported_purpose"),
    ('{"id":5,"purpose":"interactive"}', 5, "unsupported_purpose"),
    ('{"id":6,"purpose":["native-http"]}', 6, "unsupported_purpose"),
])
def test_a_malformed_line_gets_an_error_reply_and_the_channel_stays_open(home, line, request_id, code):
    tickets = TicketStore("owner", frozenset({str(home)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (_hello, ask, done):
            refused = ask(line)
            assert (refused["id"], refused["ok"], refused["error"]["code"]) == (request_id, False, code)
            assert refused["error"]["message"] and "ticket" not in refused
            # Nothing was minted for it, and the next request is still answered.
            assert not tickets.outstanding()
            assert ask('{"id":7,"purpose":"native-http"}')["ok"] is True
    finally:
        stop()
    assert "Traceback" not in done["stderr"]


@pytest.mark.platforms("linux", "macos")
def test_a_gateway_that_refuses_the_ticket_is_an_error_reply_not_a_crash(home):
    """No ticket store: the owner answers identify (ready) but refuses session-ticket."""
    stop = _serve(home, None)
    try:
        with _channel(home) as (hello, ask, done):
            assert hello["type"] == "hello"
            for request_id in (1, 2):
                reply = ask(json.dumps({"id": request_id, "purpose": "native-http"}))
                assert (reply["id"], reply["ok"], reply["error"]["code"]) == (request_id, False, "ticket_unavailable")
                assert "ticket" not in reply
    finally:
        stop()
    assert "Traceback" not in done["stderr"]


@pytest.mark.platforms("linux")
def test_the_gateways_http_auth_accepts_a_served_ticket_exactly_once(tmp_path):
    with daemon(tmp_path) as (home, descriptor), _channel(home) as (hello, ask, _done):
        origin = hello["endpoint"]["api_origin"]
        assert origin == descriptor["api_origin"] and hello["endpoint"]["instance_id"] == descriptor["instance_id"]
        with httpx.Client(base_url=origin, trust_env=False) as client:
            assert client.get("/api/config").status_code == 401
            first, second = (ask(json.dumps({"id": n, "purpose": "native-http"}))["ticket"] for n in (1, 2))
            assert client.get("/api/config", headers=headers(first)).status_code == 200
            assert client.get("/api/config", headers=headers(first)).status_code == 401
            assert client.get("/api/config", headers=headers(second)).status_code == 200
