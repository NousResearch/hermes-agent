"""Unit tests for the extracted ``web_serve_bind`` slice.

These exercise the pure multi-host bind helpers in isolation — no FastAPI /
uvicorn import required — mirroring how the other ``web_*`` slices are tested.
The socket-bind tests use real loopback sockets on ephemeral ports (port 0) so
they never collide with a running dashboard and need no network fixture.
"""

import socket

import pytest

from hermes_cli.web_serve_bind import (
    all_hosts_loopback,
    any_host_requires_auth,
    close_server_sockets,
    create_server_sockets,
)


class TestAllHostsLoopback:
    def test_single_loopback(self):
        assert all_hosts_loopback(["127.0.0.1"]) is True
        assert all_hosts_loopback(["localhost"]) is True
        assert all_hosts_loopback(["::1"]) is True

    def test_dual_stack_loopback_is_still_loopback(self):
        # IPv4 + IPv6 loopback together is fully local.
        assert all_hosts_loopback(["127.0.0.1", "::1"]) is True

    def test_public_member_makes_it_not_loopback(self):
        assert all_hosts_loopback(["0.0.0.0"]) is False
        assert all_hosts_loopback(["::"]) is False
        assert all_hosts_loopback(["192.168.1.50"]) is False

    def test_mixed_loopback_public_is_not_loopback(self):
        # A single public member means the whole bind is remotely reachable.
        assert all_hosts_loopback(["127.0.0.1", "0.0.0.0"]) is False

    def test_case_and_whitespace_normalised(self):
        assert all_hosts_loopback([" Localhost ", "LOCALHOST"]) is True


class TestAnyHostRequiresAuth:
    def test_all_loopback_no_auth(self):
        # Inject a stub should_require_auth so this stays independent of
        # web_server's real gate logic.
        fake = lambda h, allow_public=False: h not in ("127.0.0.1", "localhost", "::1")
        assert any_host_requires_auth(["127.0.0.1"], should_require_auth=fake) is False
        assert (
            any_host_requires_auth(["127.0.0.1", "::1"], should_require_auth=fake)
            is False
        )

    def test_any_public_requires_auth(self):
        fake = lambda h, allow_public=False: h not in ("127.0.0.1", "localhost", "::1")
        assert (
            any_host_requires_auth(["127.0.0.1", "0.0.0.0"], should_require_auth=fake)
            is True
        )
        assert any_host_requires_auth(["10.0.0.5"], should_require_auth=fake) is True


class TestCreateServerSockets:
    def test_single_ipv4_ephemeral(self):
        socks = create_server_sockets(["127.0.0.1"], 0)
        try:
            assert len(socks) == 1
            assert socks[0].family == socket.AF_INET
            # Port 0 → OS assigned a real port.
            assert socks[0].getsockname()[1] > 0
        finally:
            close_server_sockets(socks)

    def test_dual_stack_shares_one_port(self):
        # Both listeners must end up on the SAME port even though we asked
        # for port 0 (ephemeral) — otherwise the browser couldn't reach both.
        try:
            socks = create_server_sockets(["127.0.0.1", "::1"], 0)
        except OSError as exc:
            pytest.skip(f"IPv6 loopback unavailable in this environment: {exc}")
        try:
            assert len(socks) == 2
            ports = {s.getsockname()[1] for s in socks}
            assert len(ports) == 1, f"dual-stack listeners diverged: {ports}"
            families = {s.family for s in socks}
            assert socket.AF_INET in families
            assert socket.AF_INET6 in families
        finally:
            close_server_sockets(socks)

    def test_ipv6_v6only_set(self):
        try:
            socks = create_server_sockets(["::1"], 0)
        except OSError as exc:
            pytest.skip(f"IPv6 unavailable: {exc}")
        try:
            v6 = [s for s in socks if s.family == socket.AF_INET6][0]
            assert v6.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
        finally:
            close_server_sockets(socks)

    def test_close_is_idempotent_safe(self):
        socks = create_server_sockets(["127.0.0.1"], 0)
        close_server_sockets(socks)
        # Double-close must not raise (best-effort teardown).
        close_server_sockets(socks)


# ---------------------------------------------------------------------------
# E2E: real start_server with a dual-loopback bind, both families reachable.
# Mirrors the subprocess-driven pattern in test_serve_port_in_use.py so we
# exercise the actual uvicorn startup(sockets=…) path, not just the helper.
# ---------------------------------------------------------------------------

import os
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ipv6_loopback_available() -> bool:
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.bind(("::1", 0))
        s.close()
        return True
    except OSError:
        return False


def _spawn_dual_stack_serve(port: int, tmp_path: Path) -> subprocess.Popen:
    home = tmp_path / "hermes_home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(HERMES_HOME=str(home), HERMES_SERVE_HEADLESS="1", PYTHONUNBUFFERED="1")
    for k in (
        "HERMES_DESKTOP",
        "HERMES_PARENT_PID",
        "HERMES_PARENT_START_MARKER",
        "HERMES_PARENT_NONCE",
    ):
        env.pop(k, None)
    code = (
        "from hermes_cli.web_server import start_server\n"
        f"start_server(hosts=['127.0.0.1', '::1'], port={port}, "
        "open_browser=False, headless=True)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="IPv6 loopback unavailable")
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX serve-runner path")
def test_dual_stack_serve_reachable_on_both_families(tmp_path):
    """A dual-loopback bind (127.0.0.1 + ::1) must accept connections on BOTH
    the IPv4 and IPv6 listener — the whole point of the multi-host feature."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    proc = _spawn_dual_stack_serve(port, tmp_path)
    lines: list[str] = []
    hit = threading.Event()

    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if "HERMES_BACKEND_READY" in line:
                hit.set()
                return

    threading.Thread(target=_pump, daemon=True).start()

    try:
        assert hit.wait(timeout=180), f"no READY sentinel:\n{''.join(lines)}"

        # Connect over IPv4 loopback.
        v4 = socket.create_connection(("127.0.0.1", port), timeout=5)
        v4.close()
        # Connect over IPv6 loopback.
        v6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        v6.settimeout(5)
        v6.connect(("::1", port))
        v6.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
