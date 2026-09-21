"""A second `hermes serve` attaches to the host backend instead of binding a second port.

Attach is a PROOF, not a guess: exit 0 means the recorded owner answered on its recorded port and
serves what the caller asked for. Every other shape falls through to the bind or refuses loudly.
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from gateway import host_rendezvous as hr
from hermes_cli.main_dashboard import _attach_to_host_backend


def _args(**over):
    base = dict(host="127.0.0.1", port=9200, no_open=True, isolated=False, open_profile="")
    return SimpleNamespace(**{**base, **over})


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr("sys.argv", ["hermes", "serve"])
    return tmp_path


@pytest.fixture
def owner(host_dir):
    """A live backend answering the identity handshake as THIS pid, on a real ephemeral port."""
    serves_spa = {"value": True}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler API
            if self.path != hr.HOST_IDENTITY_PATH:
                self.send_error(404)
                return
            body = json.dumps({"ok": True, "pid": os.getpid(), "role": hr.ROLE_SERVE,
                               "servesSpa": serves_spa["value"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 — BaseHTTPRequestHandler API
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(port=server.server_port, serves_spa=serves_spa)
    finally:
        server.shutdown()
        server.server_close()


def _dead_port() -> int:
    spare = socket.socket()
    spare.bind(("127.0.0.1", 0))
    port = spare.getsockname()[1]
    spare.close()
    return port


def _publish(create_time, *, pid=None, port=9119, host="127.0.0.1"):
    record = hr.HostRecord(
        role=hr.ROLE_SERVE, pid=pid if pid is not None else os.getpid(), create_time=create_time,
        host=host, port=port, protocol_version=hr.HOST_PROTOCOL_VERSION,
        token_fingerprint="", profiles=("default", "alpha"),
        updated_at="2026-01-01T00:00:00+00:00")
    path = hr.record_path(hr.ROLE_SERVE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json()), encoding="utf-8")


def test_second_serve_attaches_to_the_live_host_backend(host_dir, owner, capsys):
    """A live record whose owner ANSWERS ends the second launch at exit 0 — it never binds."""
    _publish(hr.process_create_time(), port=owner.port)

    with pytest.raises(SystemExit) as exc:
        _attach_to_host_backend(_args(), headless_backend=True)

    assert exc.value.code == 0
    assert f"port {owner.port}" in capsys.readouterr().out


def test_stale_record_is_ignored_and_the_launch_proceeds(host_dir):
    """A record whose creation time does not match the live PID is a recycled PID, not a
    backend: the launch must fall through and bind, never attach."""
    _publish(1.0)

    assert _attach_to_host_backend(_args(), headless_backend=True) is None


def test_isolated_never_attaches(host_dir, owner):
    """`--isolated` is load-bearing for Desktop's SSH backend ownership proof."""
    _publish(hr.process_create_time(), port=owner.port)

    assert _attach_to_host_backend(_args(isolated=True), headless_backend=True) is None


def test_a_record_whose_owner_does_not_answer_falls_through_to_the_bind(host_dir):
    """The graceful-shutdown window: the PID is still alive and the record still on disk, but the
    socket is already closed. Exiting 0 here reported success with NOTHING listening."""
    _publish(hr.process_create_time(), port=_dead_port())

    assert _attach_to_host_backend(_args(), headless_backend=True) is None


def test_unprovable_liveness_still_has_to_answer(host_dir, monkeypatch):
    """Without psutil the liveness answer is ``None`` (unprovable). Treating that as "alive" made
    a record for a long-dead PID a permanent silent outage — every launch exited 0 forever."""
    monkeypatch.setattr("hermes_cli.process_identity._pid_alive_matches", lambda *_a, **_k: None)
    _publish(None, pid=2**22 - 1, port=_dead_port())

    assert _attach_to_host_backend(_args(), headless_backend=True) is None


@pytest.mark.parametrize(
    "argv,over",
    [(["hermes", "serve", "--port", "8899"], {"port": 8899}),
     (["hermes", "serve", "--host", "0.0.0.0"], {"host": "0.0.0.0"})],
    ids=["explicit-port", "explicit-host"],
)
def test_an_explicit_endpoint_the_owner_cannot_serve_is_refused(host_dir, owner, monkeypatch,
                                                                argv, over, capsys):
    """`--port 8899` answered with "use 127.0.0.1:<other>", or `--host 0.0.0.0` (LAN access)
    silently turned into a loopback attach, are both wrong answers: refuse, naming the owner."""
    monkeypatch.setattr("sys.argv", argv)
    _publish(hr.process_create_time(), port=owner.port)

    with pytest.raises(SystemExit) as exc:
        _attach_to_host_backend(_args(**over), headless_backend=True)

    assert exc.value.code == 1
    assert f"PID {os.getpid()}" in capsys.readouterr().out


def test_dashboard_is_never_routed_to_a_headless_backend(host_dir, owner, capsys):
    """`hermes serve` and `hermes dashboard` publish the same host role; only one mounts the SPA,
    so attaching a dashboard user to a headless backend opens a URL with no UI behind it."""
    owner.serves_spa["value"] = False
    _publish(hr.process_create_time(), port=owner.port)

    with pytest.raises(SystemExit) as exc:
        _attach_to_host_backend(_args(), headless_backend=False)

    assert exc.value.code == 1
    assert "no dashboard UI" in capsys.readouterr().out


def test_ipv6_owner_attach_prints_a_valid_bracketed_url(host_dir, monkeypatch, capsys):
    """The attach receipt is a browser URL, where a bare IPv6 literal is invalid."""
    record = hr.HostRecord(
        role=hr.ROLE_SERVE, pid=os.getpid(), create_time=hr.process_create_time(),
        host="::1", port=9119, protocol_version=hr.HOST_PROTOCOL_VERSION,
        token_fingerprint="", profiles=("default",),
        updated_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(
        "hermes_cli.main_dashboard._host_backend_attachment", lambda: record)
    monkeypatch.setattr(
        hr, "probe_owner",
        lambda _record: {"pid": os.getpid(), "role": hr.ROLE_SERVE, "servesSpa": True})

    with pytest.raises(SystemExit) as exc:
        _attach_to_host_backend(_args(host="::1"), headless_backend=True)

    assert exc.value.code == 0
    assert "http://[::1]:9119/?profile=default" in capsys.readouterr().out
