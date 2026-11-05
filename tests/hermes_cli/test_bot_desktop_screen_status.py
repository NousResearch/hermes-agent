"""``hermes computer-use screen status`` shows the lease the way every RPC surface does: the holder's viewer
id is a capability (whoever presents it co-drives or releases the lease), so only its short hash leaves the
gateway. Regression for #110006."""

import argparse
import hashlib
import json
import socket
import threading
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import computer_use_screen
from tools.bot_desktop import lease, runtime


def _parse(argv):
    parser = argparse.ArgumentParser()
    computer_use_screen.build_screen_parser(
        parser.add_subparsers(), lambda sub, help_text: sub.add_argument("--json", action="store_true"))
    return parser.parse_args(argv)


@pytest.fixture
def running_screen(monkeypatch):
    monkeypatch.setattr(runtime, "status", lambda profile=None: runtime.DesktopStatus(
        profile="default", supported=True, installed=True, missing=[], running=True, pid=4242, display=":20",
        socket="/x/rfb.sock", geometry="1440x900", install_command=None, browser=None))
    yield
    lease._reset_for_tests()


def test_status_never_prints_the_holders_raw_viewer_id(running_screen, capsys):
    from tui_gateway import methods_display

    secret = "secret-viewer-token-ABC123"
    lease.acquire(secret, reason="typing a password")
    rpc_view = methods_display._display_snapshot()["lease"]  # what display.status / display.lease emit
    assert secret not in json.dumps(rpc_view)

    assert computer_use_screen.SCREEN_ACTIONS["status"](_parse(["screen", "status", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert secret not in json.dumps(payload)
    assert payload["lease"] == rpc_view, "the CLI and the RPC surfaces emit one and the same redacted lease"
    assert payload["lease"]["viewer_hash"] == hashlib.sha256(secret.encode()).hexdigest()[:12]

    assert computer_use_screen.SCREEN_ACTIONS["status"](_parse(["screen", "status"])) == 0
    human = capsys.readouterr().out
    assert secret not in human and "control: human" in human


def _marker_socket_sandbox(tmp_path, monkeypatch):
    """A start marker naming a sandbox directory, as a sandbox-hosted screen records it."""
    from tools.bot_desktop import sandbox_host

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: {"dir": str(sandbox), "display": ":41"})
    monkeypatch.setattr(runtime, "geometry", lambda: "1440x900")
    return sandbox


class _RfbStub:
    """A one-shot RFB endpoint that records what the probe sends.

    The distinction matters: Xvnc counts a connect-and-drop as a failed security handshake and
    blacklists the client after a few of them, so a probe that hangs up before finishing the
    handshake turns every status call into a step toward a black screen.
    """

    def __init__(self, path):
        self.received = []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._server.settimeout(5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            try:
                conn.sendall(b"RFB 003.008\n")
                self.received.append(conn.recv(12))
                conn.sendall(b"\x01\x01")            # one security type: None
                self.received.append(conn.recv(1))
                conn.sendall(b"\x00\x00\x00\x00")    # SecurityResult: ok
                self.received.append(conn.recv(1))   # ClientInit
                conn.sendall(b"\x00" * 24)           # ServerInit; the probe reads it before closing
            except OSError:
                pass

    def close(self):
        self._server.close()
        self._thread.join(timeout=5)


def test_status_reads_running_from_the_socket_the_start_marker_names(tmp_path, monkeypatch):
    """A pane request owns no sandbox environment; the marker plus a live socket is enough."""
    sandbox = _marker_socket_sandbox(tmp_path, monkeypatch)
    stub = _RfbStub(sandbox / "rfb.sock")
    try:
        status = runtime._sandbox_status("default", SimpleNamespace(backend="apptainer"))
    finally:
        stub.close()
    assert status.running is True
    assert stub.received[:3] == [b"RFB 003.008\n", b"\x01", b"\x01"], (
        "the probe finished the RFB handshake instead of dropping the connection"
    )
    assert status.display == ":41"
    assert status.socket == str(sandbox / "rfb.sock")
    assert status.installed is True and status.blocker is None


def test_status_does_not_report_a_socket_file_no_screen_listens_on(tmp_path, monkeypatch):
    """A killed Xvnc leaves the socket file behind; status must fall through to the owned-env probe."""
    sandbox = _marker_socket_sandbox(tmp_path, monkeypatch)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sandbox / "rfb.sock"))
    stale.close()
    monkeypatch.setattr(runtime, "_owned_sandbox_env", lambda: None)
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: None)
    status = runtime._sandbox_status("default", SimpleNamespace(backend="apptainer"))
    assert status.running is False
    assert status.socket is None
