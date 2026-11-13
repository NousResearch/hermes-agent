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
import signal
import subprocess
import sys
import tempfile
import time

import httpx
import pytest

from gateway.runtime_bootstrap import TicketStore
from tests.gateway.test_native_http_auth import daemon, headers
from tests.gateway.test_normal_runtime_boot import control
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
    assert json.loads(run.stdout) == {
        "protocol": 1, "purposes": ["native-http", "native-profile-create", "native-profile-registry"],
        # This test process is not an elevated Windows session-0 login.
        "windows_service_session": False}
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
@pytest.mark.parametrize("line", [
    # Under the length limit; too deep for the JSON decoder on interpreters with a low nesting limit.
    "[" * 20_000 + "0" + "]" * 20_000,
    "[" * 100_000 + "0" + "]" * 100_000,  # too deep for every decoder, and over the length limit
    # Over the 64 KiB limit: refused although it is otherwise a valid request.
    json.dumps({"id": 1, "purpose": "native-http", "pad": "x" * 64 * 1024}),
    "x" * (3 * 64 * 1024 + 1),
], ids=["deeply-nested", "deeply-nested-and-oversized", "oversized-valid-request", "oversized-garbage"])
def test_a_deeply_nested_or_oversized_line_gets_one_error_reply_and_the_channel_stays_open(home, line):
    tickets = TicketStore("owner", frozenset({str(home)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (_hello, ask, done):
            refused = ask(line)
            assert (refused["id"], refused["ok"], refused["error"]["code"]) == (None, False, "invalid_request")
            assert "ticket" not in refused and not tickets.outstanding()
            # Exactly one reply for it: the next line read back is the next request's own reply.
            following = ask('{"id":7,"purpose":"native-http"}')
            assert (following["id"], following["ok"]) == (7, True)
    finally:
        stop()
    assert "Traceback" not in done["stderr"]


@pytest.mark.platforms("linux", "macos")
def test_a_request_line_of_exactly_the_length_limit_is_still_answered(home):
    tickets = TicketStore("owner", frozenset({str(home)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (_hello, ask, _done):
            request = '{"id":1,"purpose":"native-http","pad":""}'
            line = request.replace('""', json.dumps("x" * (64 * 1024 - len(request))))
            assert len(line) == 64 * 1024
            assert ask(line)["ok"] is True
            assert ask(line + " ")["error"]["code"] == "invalid_request"
            assert ask('{"id":7,"purpose":"native-http"}')["id"] == 7
    finally:
        stop()


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


def _named(home, name):
    """A named profile of the host whose root is *home*; returns its home."""
    profile = home / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}")
    return profile


@pytest.mark.platforms("linux", "macos")
def test_a_request_can_name_another_profile_the_gateway_serves(home):
    beta = _named(home, "beta")
    tickets = TicketStore("owner", frozenset({str(home), str(beta)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (hello, ask, _done):
            assert hello["endpoint"]["profile_id"] == str(home)
            for request, served in (({"profile": "beta"}, beta), ({"profile": " beta "}, beta),
                                    ({"profile": "default"}, home), ({}, home)):
                reply = ask(json.dumps({"id": 1, "purpose": "native-http", **request}))
                assert {k: v for k, v in reply.items() if k != "ticket"} == {"id": 1, "ok": True, "expires_in": 30}
                grant = tickets.redeem(reply["ticket"], profile_id=None, purpose="native-http")
                assert (grant["profile_id"], grant["scope"]) == (str(served), "profile")
    finally:
        stop()


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("purpose, capability", [
    ("native-profile-create", "profiles:create"), ("native-profile-registry", "profiles:registry")])
def test_serve_mints_the_profile_management_purposes_for_the_named_profile(home, purpose, capability):
    beta = _named(home, "beta")
    tickets = TicketStore("owner", frozenset({str(home), str(beta)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (_hello, ask, _done):
            for request, served in (({"profile": "beta"}, beta), ({}, home)):
                ticket = ask(json.dumps({"id": "m", "purpose": purpose, **request}))["ticket"]
                # Its own purpose only: neither the HTTP purpose nor the WebSocket's takes it.
                for other in ("native-http", "interactive"):
                    with pytest.raises(PermissionError):
                        tickets.redeem(ticket, profile_id=None, purpose=other)
                grant = tickets.redeem(ticket, profile_id=None, purpose=purpose)
                assert (grant["profile_id"], grant["scope"]) == (str(served), "profile")
                assert grant["capabilities"] == frozenset({capability})
    finally:
        stop()


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("profile, code", [
    ("ghost", "unknown_profile"),          # a valid name with no profile behind it
    ("unserved", "profile_not_served"),    # exists on disk; this gateway does not serve it
    ("../beta", "invalid_request"), ("", "invalid_request"), ("Beta", "invalid_request"),
    (7, "invalid_request"), (["beta"], "invalid_request"), (True, "invalid_request"),
])
def test_a_profile_the_gateway_cannot_ticket_is_an_error_reply_and_the_channel_stays_open(home, profile, code):
    beta = _named(home, "beta")
    _named(home, "unserved")
    tickets = TicketStore("owner", frozenset({str(home), str(beta)}))
    stop = _serve(home, tickets)
    try:
        with _channel(home) as (_hello, ask, done):
            for purpose in ("native-http", "native-profile-create", "native-profile-registry"):
                refused = ask(json.dumps({"id": 3, "purpose": purpose, "profile": profile}))
                assert (refused["id"], refused["ok"], refused["error"]["code"]) == (3, False, code)
                assert refused["error"]["message"] and "ticket" not in refused
            # Never the launch profile's ticket instead, and the next request is still answered.
            assert not tickets.outstanding()
            assert ask('{"id":4,"purpose":"native-http","profile":"beta"}')["ok"] is True
    finally:
        stop()
    assert "Traceback" not in done["stderr"]


@pytest.mark.platforms("linux", "macos")
def test_a_served_profiles_refused_ticket_is_not_reported_as_an_unserved_profile(home):
    """The owner lists beta as served but mints nothing (no ticket store)."""
    import asyncio
    import threading
    from gateway.control_socket import GatewayControlServer

    beta = _named(home, "beta")
    identify = {"runtime_protocol": 1, "state": "ready", "capabilities": ["session-authority-v1"],
                "served_profiles": [{"home": str(h), "profile_id": str(h)} for h in (home, beta)],
                "authority_epoch": 1, "instance_id": "owner", "api_origin": "http://127.0.0.1:8765",
                "supervisor": "none"}
    server = GatewayControlServer(home, verb_handlers={"identify": lambda: identify})
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    assert asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    try:
        with _channel(home) as (_hello, ask, done):
            reply = ask('{"id":1,"purpose":"native-http","profile":"beta"}')
            assert (reply["id"], reply["ok"], reply["error"]["code"]) == (1, False, "ticket_unavailable")
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    assert "Traceback" not in done["stderr"]


@contextmanager
def _multiplexing_daemon(tmp_path, names):
    """A real gateway that serves its root home and the named profiles; yields (home, descriptor)."""
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    model = {"model": {"provider": "custom", "default": "m", "base_url": "http://127.0.0.1:1/v1"},
             "auxiliary": {"title_generation": {"enabled": False}}}
    (home / "config.yaml").write_text(json.dumps({"gateway": {"multiplex_profiles": True}, **model}))
    for name in names:
        (home / "profiles" / name).mkdir(parents=True)
        (home / "profiles" / name / "config.yaml").write_text(json.dumps(model))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ", "TIRITH_ENABLED") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(REPO),
               PYTHONUNBUFFERED="1")
    with (tmp_path / "gateway.log").open("w+") as log:
        process = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=REPO, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline, descriptor = time.monotonic() + 60, {}
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, "identify")
                    if descriptor.get("state") == "ready" and len(descriptor.get("served_profiles", [])) == len(names) + 1:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.seek(0)
            assert descriptor.get("state") == "ready", (descriptor, log.read())
            yield home, descriptor
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def _settled(home):
    """Wait out the rescan a create/rename/delete starts, so the next ticket is not refused for it."""
    for _ in range(200):
        if control(home, "identify").get("state") == "ready":
            return
        time.sleep(.05)


@pytest.mark.platforms("linux")
def test_the_gateway_takes_a_named_profiles_ticket_on_that_profiles_routes_only(tmp_path):
    with _multiplexing_daemon(tmp_path, ("beta", "gamma")) as (home, descriptor), _channel(home) as (hello, ask, _done):
        assert hello["endpoint"]["instance_id"] == descriptor["instance_id"]

        def ticket(profile=None):
            reply = ask(json.dumps({"id": 1, "purpose": "native-http", **({"profile": profile} if profile else {})}))
            return headers(reply["ticket"])

        with httpx.Client(base_url=hello["endpoint"]["api_origin"], trust_env=False, timeout=60) as client:
            assert client.get("/api/config?profile=beta", headers=ticket("beta")).status_code == 200
            assert client.get("/api/profiles/beta/soul", headers=ticket("beta")).status_code == 200
            # Bound to beta: refused on the launch profile's routes and on a sibling's.
            for path in ("/api/config?profile=default", "/api/config?profile=gamma", "/api/profiles/gamma/soul"):
                refused = client.get(path, headers=ticket("beta"))
                assert (refused.status_code, refused.json()) == (403, {"detail": "profile_scope_mismatch"}), path
            # A request with no `profile` is the launch profile's ticket, as before: beta refuses it.
            assert client.get("/api/config", headers=ticket()).status_code == 200
            assert client.get("/api/config?profile=beta", headers=ticket()).status_code == 403
            assert client.get("/api/config?profile=default", headers=ticket("default")).status_code == 200


@pytest.mark.platforms("linux")
def test_the_gateway_takes_profile_management_tickets_on_their_own_routes_only(tmp_path):
    with _multiplexing_daemon(tmp_path, ("beta", "gamma")) as (home, _descriptor), _channel(home) as (hello, ask, _done):
        def ticket(purpose, profile=None):
            reply = ask(json.dumps({"id": 1, "purpose": purpose, **({"profile": profile} if profile else {})}))
            return headers(reply["ticket"])

        create, registry, made = "native-profile-create", "native-profile-registry", {"name": "made"}
        with httpx.Client(base_url=hello["endpoint"]["api_origin"], trust_env=False, timeout=60) as client:
            # An ordinary ticket creates, renames and deletes nothing, whichever profile it names.
            assert client.post("/api/profiles", json=made, headers=ticket("native-http")).status_code == 403
            for method, body in (("DELETE", None), ("PATCH", {"new_name": "renamed"})):
                refused = client.request(method, "/api/profiles/beta", json=body, headers=ticket("native-http", "beta"))
                assert refused.status_code == 403, (method, refused.text)
            # Each management ticket is for its own route: no read, and not the other's operation.
            for purpose, profile, method, path, body in (
                    (create, None, "GET", "/api/config", None),
                    (create, "beta", "DELETE", "/api/profiles/beta", None),
                    (create, "beta", "PATCH", "/api/profiles/beta", {"new_name": "renamed"}),
                    (registry, "beta", "GET", "/api/config?profile=beta", None),
                    (registry, "beta", "GET", "/api/profiles/beta/soul", None),
                    (registry, "beta", "POST", "/api/profiles", made)):
                refused = client.request(method, path, json=body, headers=ticket(purpose, profile))
                assert refused.status_code == 401, (purpose, method, path, refused.text)
            # A registry ticket is for the one profile it names.
            assert client.delete("/api/profiles/gamma", headers=ticket(registry, "beta")).status_code == 403
            assert client.delete("/api/profiles/gamma", headers=ticket(registry)).status_code == 403
            assert not (home / "profiles" / "made").exists()
            assert (home / "profiles" / "beta").is_dir() and (home / "profiles" / "gamma").is_dir()

            created = client.post("/api/profiles", json=made, headers=ticket(create))
            assert created.status_code == 200, created.text
            assert (home / "profiles" / "made").is_dir()
            _settled(home)
            renamed = client.patch("/api/profiles/gamma", json={"new_name": "delta"}, headers=ticket(registry, "gamma"))
            assert (renamed.status_code, renamed.json()["name"]) == (200, "delta"), renamed.text
            assert not (home / "profiles" / "gamma").exists() and (home / "profiles" / "delta").is_dir()
            _settled(home)
            assert client.delete("/api/profiles/beta", headers=ticket(registry, "beta")).status_code == 200
            assert not (home / "profiles" / "beta").exists()
            _settled(home)
            # The deleted profile is unknown from then on; the renamed one answers under its new name.
            gone = ask('{"id":9,"purpose":"native-http","profile":"beta"}')
            assert (gone["ok"], gone["error"]["code"]) == (False, "unknown_profile")
            assert client.get("/api/profiles/delta/soul", headers=ticket("native-http", "delta")).status_code == 200
