from __future__ import annotations

import dataclasses
import json
import os
import plistlib
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import pytest

from hermes_platform.resolver import CheckState, Effort
from hermes_platform.resolver.app import AppDef, AppResolver, Endpoint

TOKEN = "tok-3e1f9c-unique-fixture-value"


def _bundle(tmp_path, version="9.8.7"):
    app = tmp_path / "Applications" / "Thing.app"
    (app / "Contents").mkdir(parents=True)
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleShortVersionString": version}, fh)
    return app


def _server_json(tmp_path, *, url, pid=None):
    p = tmp_path / "server.json"
    p.write_text(json.dumps({"pid": os.getpid() if pid is None else pid, "http": url, "token": TOKEN}), encoding="utf-8")
    return p


def _resolver(tmp_path, exe, server_json=None):
    return AppResolver(AppDef(
        "thing", sys.platform, "executable", str(exe),
        liveness_kind="server_json" if server_json else "none",
        liveness_path=str(server_json) if server_json else "",
    ))




def test_locate_reports_expanded_path_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("THINGROOT", str(tmp_path))
    r = AppResolver(AppDef("thing", sys.platform, "executable", "$THINGROOT/bin/thing"))
    res = r.locate()
    assert res.kind == "missing"
    assert res.candidates[0].value == str(tmp_path / "bin" / "thing")


def test_bundle_presence_and_plist_version(tmp_path):
    app = _bundle(tmp_path)
    r = AppResolver(AppDef("thing", "darwin", "bundle", str(app), version_kind="plist"))
    res = r.locate()
    assert res.kind == "known_path" and res.command == (str(app),)
    insp = r.inspect(res)
    assert insp.version.state is CheckState.PRESENT and insp.version.value == "9.8.7"
    assert insp.signer.state is CheckState.NOT_CHECKED


def test_inspect_on_missing_is_not_checked(tmp_path):
    r = AppResolver(AppDef("thing", sys.platform, "executable", str(tmp_path / "none"), version_kind="plist"))
    insp = r.inspect(r.locate())
    assert insp.version.state is CheckState.NOT_CHECKED


def test_probe_without_liveness_source_is_all_not_checked(tmp_path):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    r = _resolver(tmp_path, exe)
    pr = r.probe(r.locate(), effort=Effort.NETWORK)
    assert pr.running.state is pr.answering.state is pr.endpoint.state is CheckState.NOT_CHECKED


def test_probe_local_reads_pid_and_endpoint_but_never_connects(tmp_path):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:1/mcp")
    r = _resolver(tmp_path, exe, sj)
    pr = r.probe(r.locate(), effort=Effort.LOCAL)
    assert pr.running.state is CheckState.PRESENT and pr.running.value is True
    assert pr.endpoint.value == "http://127.0.0.1:1/mcp"
    assert pr.answering.state is CheckState.NOT_CHECKED


def test_probe_is_not_a_boolean(tmp_path):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    pr = _resolver(tmp_path, exe).probe(_resolver(tmp_path, exe).locate(), effort=Effort.LOCAL)
    with pytest.raises(TypeError):
        bool(pr)


def test_token_never_appears_in_any_public_result(tmp_path):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:1/mcp")
    r = _resolver(tmp_path, exe, sj)
    res = r.locate()
    pr = r.probe(res, effort=Effort.LOCAL)
    for obj in (res, r.inspect(res), pr):
        assert TOKEN not in repr(obj)
        assert TOKEN not in json.dumps(dataclasses.asdict(obj), default=str)
        assert "token" not in {f.name for f in dataclasses.fields(obj)}


@pytest.mark.parametrize("url", [
    "http://[::1",
    "https://127.0.0.1:1234/mcp",
    "http://user:pw@127.0.0.1:1234/mcp",
    "http://example.com:1234/mcp",
    "http://127.0.0.1/mcp",
    "http://127.0.0.1:99999/mcp",
])
def test_non_loopback_or_malformed_endpoint_is_unavailable_and_never_contacted(tmp_path, url, monkeypatch):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url=url)
    r = _resolver(tmp_path, exe, sj)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no socket allowed"))
    pr = r.probe(r.locate(), effort=Effort.NETWORK)
    assert pr.endpoint.state is CheckState.UNAVAILABLE
    assert pr.answering.state is CheckState.NOT_CHECKED


@pytest.mark.parametrize("epath", [
    "@attacker.example/",      # userinfo rewrite: the bearer token would go to attacker.example
    "@169.254.169.254/x",      # metadata-endpoint variant of the same
    "@127.0.0.1:9/",           # same host:port, userinfo smuggled into netloc -- isolates the
                             # rejoined.username clause (hostname and port both still match)
    "mcp",                     # no leading slash: corrupts the authority, port parse fails
])
def test_manifest_endpoint_path_cannot_hijack_the_dialed_host(tmp_path, monkeypatch, epath):
    """A manifest-authored endpoint_path must never turn the probe into a token sender: the
    composed URL is re-parsed and must stay on the loopback host:port before any connect."""
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:9/mcp")
    r = AppResolver(AppDef(
        "thing", sys.platform, "executable", str(exe),
        liveness_kind="server_json", liveness_path=str(sj), endpoint_path=epath,
    ))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no socket allowed"))
    pr = r.probe(r.locate(), effort=Effort.NETWORK)
    assert pr.endpoint.state is CheckState.UNAVAILABLE
    assert pr.answering.state is CheckState.NOT_CHECKED
    # endpoint() exposes the same composed URL to the transport layer: it must refuse too.
    assert r.endpoint() is None


@pytest.mark.parametrize("epath", [
    "//evil.example/x",  # network-path-reference form: allowed by the charset, must not move authority
    "/a/../../b",        # '..' segments: authority can never move, compose must still dial loopback
    "/mcp?x=1",          # query tail: rejected at declaration parse, but a direct AppDef may carry it
])
def test_regex_admitted_endpoint_paths_stay_on_loopback(tmp_path, epath):
    """Paths the declaration charset admits (or a direct AppDef smuggles past it) may be odd
    but the composed URL's authority is machine-built: pin that it stays the loopback host."""
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:9/mcp")
    r = AppResolver(AppDef(
        "thing", sys.platform, "executable", str(exe),
        liveness_kind="server_json", liveness_path=str(sj), endpoint_path=epath,
    ))
    ep = r.endpoint()
    if ep is not None:  # '/mcp?x=1' stays loopback (rejected only at declaration parse)
        parts = urlsplit(ep.url)
        assert parts.hostname == "127.0.0.1" and parts.port == 9 and not parts.username


@pytest.mark.parametrize("endpoint", [
    "http://169.254.169.254/latest",   # non-loopback host
    "http://user:pw@127.0.0.1:9/mcp",  # loopback host but userinfo present
    "https://127.0.0.1:9/mcp",         # wrong scheme
    "http://127.0.0.1:0/mcp",          # port 0 is out of range; must not fall back to 80
])
def test_mcp_initialize_belt_refuses_non_loopback_url_directly(tmp_path, monkeypatch, endpoint):
    """The dial-site re-check is a second layer under _endpoint_observation: probe() can never
    reach it with a bad URL, so call it directly or the belt could silently rot."""
    from hermes_platform.resolver.app import _Session, _mcp_initialize
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no socket allowed"))
    obs = _mcp_initialize(_Session(pid=os.getpid(), url="", token=TOKEN), endpoint, 1.0)
    assert obs.state is CheckState.UNAVAILABLE  # policy refusal, not "server did not answer"


def test_ipv6_loopback_url_composes_with_brackets(tmp_path):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://[::1]:8080/ignored")
    r = _resolver(tmp_path, exe, sj)
    assert r.endpoint() == Endpoint("http://[::1]:8080/mcp", TOKEN)


@pytest.mark.parametrize("pid", [2**31, True, "1234"])
def test_pathological_pid_never_raises_and_is_not_alive(tmp_path, monkeypatch, pid):
    """os.kill overflows above int32 and bool is an int subclass (True probes PID 1); a
    malformed pid in the runtime file must degrade the probe, not crash it."""
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:9/mcp", pid=pid)
    r = _resolver(tmp_path, exe, sj)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no socket allowed"))
    pr = r.probe(r.locate(), effort=Effort.NETWORK)
    assert pr.running.state is CheckState.UNAVAILABLE
    assert pr.answering.state is CheckState.NOT_CHECKED
    assert r.endpoint() is None


def test_dead_pid_is_absent_and_skips_network(tmp_path, monkeypatch):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:1/mcp", pid=2**22 + 12345)
    r = _resolver(tmp_path, exe, sj)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no socket allowed"))
    pr = r.probe(r.locate(), effort=Effort.NETWORK)
    assert pr.running.state is CheckState.ABSENT
    assert pr.answering.state is CheckState.NOT_CHECKED


class _Handler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        _Handler.seen.append((self.path, self.headers.get("Authorization", "")))
        self.send_response(200 if body.get("method") == "initialize" else 400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"jsonrpc":"2.0","id":1,"result":{}}')

    def log_message(self, format, *args):  # noqa: A002
        pass


@pytest.fixture
def loopback_mcp():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.seen.clear()
    yield srv.server_address[1]
    srv.shutdown()


def test_endpoint_reads_server_json_every_call(tmp_path):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url="http://127.0.0.1:1111/ignored")
    resolver = _resolver(tmp_path, exe, sj)
    assert resolver.endpoint() == Endpoint("http://127.0.0.1:1111/mcp", TOKEN)
    sj.write_text(json.dumps({"pid": os.getpid(), "http": "http://127.0.0.1:2222/ignored", "token": "next"}))
    assert resolver.endpoint() == Endpoint("http://127.0.0.1:2222/mcp", "next")
    assert TOKEN not in repr(resolver.endpoint())


def test_network_probe_reads_server_json_every_call_and_answers(tmp_path, loopback_mcp):
    exe = tmp_path / "thing"
    exe.write_text("", encoding="utf-8")
    sj = _server_json(tmp_path, url=f"http://127.0.0.1:{loopback_mcp}/ignored")
    r = _resolver(tmp_path, exe, sj)
    res = r.locate()
    first = r.probe(res, effort=Effort.NETWORK)
    assert first.answering.state is CheckState.PRESENT and first.answering.value is True
    assert _Handler.seen[-1] == ("/mcp", f"Bearer {TOKEN}")
    # the vendor moves the port: the next probe must follow without any caller action
    sj.write_text(json.dumps({"pid": os.getpid(), "http": "http://127.0.0.1:1/x", "token": TOKEN}), encoding="utf-8")
    second = r.probe(res, effort=Effort.NETWORK)
    assert second.endpoint.value == "http://127.0.0.1:1/mcp"
    assert second.answering.state is CheckState.ABSENT


class _SlowDrip(BaseHTTPRequestHandler):
    """Fast 200 and headers, then one body byte every 0.3 s: each receive is under any per-socket timeout."""

    def do_POST(self):
        import time as _t
        self.send_response(200)
        self.send_header("Content-Length", "64")
        self.end_headers()
        for _ in range(64):
            try:
                self.wfile.write(b"x")
                self.wfile.flush()
            except OSError:
                return
            _t.sleep(0.3)

    def log_message(self, format, *args):  # noqa: A002
        pass


def test_network_probe_honors_one_absolute_deadline(tmp_path):
    import time as _t
    srv = HTTPServer(("127.0.0.1", 0), _SlowDrip)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        exe = tmp_path / "thing"
        exe.write_text("", encoding="utf-8")
        sj = _server_json(tmp_path, url=f"http://127.0.0.1:{srv.server_address[1]}/x")
        r = _resolver(tmp_path, exe, sj)
        t0 = _t.monotonic()
        pr = r.probe(r.locate(), effort=Effort.NETWORK, deadline_s=0.5)
        elapsed = _t.monotonic() - t0
        assert elapsed < 2.0, elapsed
        assert pr.answering.state is CheckState.PRESENT
    finally:
        srv.shutdown()
