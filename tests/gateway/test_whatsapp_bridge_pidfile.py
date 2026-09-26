"""Regression tests: the WhatsApp stale-bridge cleanup must never kill a stranger.

The bridge records its PID in ``bridge.pid``. On the next start the gateway
SIGTERMs that PID to reap an orphaned bridge. The original code checked only
that the PID was *alive* — but once the bridge exits and is reaped the kernel
can recycle its number onto an unrelated process. Because the WhatsApp bridge
crash-loops, this cleanup ran constantly, and a recycled PID that had landed on
the user's browser main process got SIGTERMed, closing the browser at irregular
intervals (no crash, no coredump — a clean kill of a stranger).

These tests prove the identity guard: a PID is only signalled when it is still
our bridge (kernel start time matches). A recycled PID — or a legacy pidfile
with no start-time fingerprint — is left alone: a ``node`` + session-path
cmdline substring also matches log tails, editors, and greps that merely
mention the session (#116883), so it is not accepted as kill evidence.
"""

import subprocess
import sys
import time

import pytest

import os
import socket

from plugins.platforms.whatsapp.adapter import (
    _kill_stale_bridge_by_pidfile,
    _listener_pids_on_port,
    _write_bridge_pidfile,
)


def _spawn_sleeper(*extra_argv, seconds: float = 0.2) -> subprocess.Popen:
    """Spawn a real process living ``seconds``; optional extra argv shapes its cmdline."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})", *extra_argv]
    )


def _wait_dead(proc: subprocess.Popen, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return False




class TestIdentityGuard:
    def test_kills_when_start_time_matches(self, tmp_path):
        """A genuine bridge (recorded start time matches) IS reaped."""
        proc = _spawn_sleeper()
        try:
            _write_bridge_pidfile(tmp_path, proc.pid)
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert _wait_dead(proc), "the real bridge process should be killed"
            assert not (tmp_path / "bridge.pid").exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


    def test_legacy_pidfile_refuses_cmdline_only_kill(self, tmp_path):
        """Legacy pidfile (pid only): a node+session-looking cmdline is NOT trusted — fail closed (#116883).

        The same substring evidence matches a log tail / editor / grep that merely mentions the
        session path, so without a start-time fingerprint the guard must refuse the kill; the
        bridge-port scan reaps the orphan instead.
        """
        # Shape the cmdline to look like the node bridge for this session.
        proc = _spawn_sleeper("node", str(tmp_path), seconds=30)
        try:
            (tmp_path / "bridge.pid").write_text(str(proc.pid))  # legacy: pid only
            _kill_stale_bridge_by_pidfile(tmp_path)
            time.sleep(0.5)  # give a wrongly-delivered SIGTERM time to land and be observed
            assert proc.poll() is None, "a cmdline-only match must not be killed (fail closed)"
            assert not (tmp_path / "bridge.pid").exists()
        finally:
            proc.kill()
            proc.wait()


class TestKillPortProcess:
    """Freeing the bridge port must target only LISTENers, never clients.

    Root cause of the live Firefox kills: ``lsof -ti :PORT`` (and ``fuser
    PORT/tcp``) also returned *client* sockets whose connection merely involved
    the port number. The WhatsApp bridge uses port 3000 by default — a common
    local dev-server port — so a browser tab on ``localhost:3000`` was matched
    and SIGTERMed every time the (crash-looping) bridge restarted.
    """

    def test_listener_lookup_excludes_client_process(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.listen(5)
        # A separate process holding a *client* connection to that port.
        client = subprocess.Popen([
            sys.executable, "-c",
            "import socket,time; c=socket.create_connection(('127.0.0.1',%d)); time.sleep(0.2)" % port,
        ])
        try:
            conn, _ = srv.accept()  # establish the client connection
            pids = _listener_pids_on_port(port)
            if os.getpid() not in pids:
                pytest.skip("neither lsof nor ss detected the listener here")
            # The listener (this process) is found; the client process is NOT —
            # the LISTEN filter is what spares unrelated clients like a browser.
            assert client.pid not in pids
            conn.close()
        finally:
            client.kill()
            client.wait()
            srv.close()

