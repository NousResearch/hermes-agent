"""End-to-end smoke test for the iron-proxy egress integration.

Spins up the REAL iron-proxy binary (auto-installed if not present), routes
a curl request through it against a local fake upstream, and verifies that
the Authorization header was swapped from a proxy token to a real secret.

Gated on the network. Skipped by default in CI unless the user explicitly
opts in with --run-e2e or HERMES_RUN_E2E=1.  This is intentional — the test
downloads ~16MB and requires both `openssl` and `curl` to be present.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import pytest

from agent.proxy_sources import iron_proxy as ip


pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_RUN_E2E", "0") != "1",
    reason="E2E proxy test — set HERMES_RUN_E2E=1 to run (requires network + curl + openssl)",
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CaptureHandler(BaseHTTPRequestHandler):
    """Records the Authorization header of every incoming request."""

    captured_auth: Optional[str] = None  # class-level so tests can read it

    def do_GET(self):
        type(self).captured_auth = self.headers.get("Authorization")
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        return  # silence access log


def test_iron_proxy_swaps_authorization_header_end_to_end(hermes_home, monkeypatch):
    """Real binary, real CA, real curl. Verify the proxy swaps a proxy-token
    Authorization header for the real bearer value before forwarding."""

    if not __import__("shutil").which("curl"):
        pytest.skip("curl not available")
    if not __import__("shutil").which("openssl"):
        pytest.skip("openssl not available")

    # ----- fake upstream ----------------------------------------------------
    upstream_port = _free_port()
    server = HTTPServer(("127.0.0.1", upstream_port), _CaptureHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # ----- iron-proxy install + CA + config ---------------------------
        binary = ip.install_iron_proxy()
        assert binary.exists()
        ca_crt, ca_key = ip.ensure_ca_cert()
        assert ca_crt.exists()

        real_secret = "sk-real-upstream-value-deadbeef"
        monkeypatch.setenv("TEST_UPSTREAM_KEY", real_secret)
        proxy_token = ip.mint_proxy_token("test")

        mapping = ip.TokenMapping(
            proxy_token=proxy_token,
            real_env_name="TEST_UPSTREAM_KEY",
            upstream_hosts=("127.0.0.1",),
        )

        tunnel_port = _free_port()
        cfg = ip.build_proxy_config(
            mappings=[mapping],
            ca_cert=ca_crt,
            ca_key=ca_key,
            tunnel_port=tunnel_port,
            allowed_hosts=["127.0.0.1"],
            # Test target is on loopback — clear the default IMDS+loopback
            # deny list so iron-proxy will dial 127.0.0.1.
            upstream_deny_cidrs=[],
            # Hermetic: pin the bind to loopback.  Without this, Linux
            # hosts with docker0 present would bind the bridge gateway
            # (the production default) and the loopback curl below would
            # never reach the proxy.
            http_listen=[f"127.0.0.1:{tunnel_port}"],
        )
        ip.write_proxy_config(cfg)
        ip.write_mappings([mapping])

        # ----- start the proxy --------------------------------------------
        try:
            status = ip.start_proxy()
        except RuntimeError as exc:
            pytest.skip(f"iron-proxy could not start in this environment: {exc}")
        assert status.pid is not None

        # Wait up to 10s for the listener to come up.
        for _ in range(50):
            if ip._port_listening("127.0.0.1", tunnel_port):
                break
            time.sleep(0.2)
        else:
            pytest.fail("iron-proxy never started listening on the tunnel port")

        # ----- request through the proxy ----------------------------------
        # The fake upstream listens on plain HTTP (not HTTPS).  Plain-HTTP
        # absolute-form forwards are served by the http_listen listener on
        # tunnel_port + 1 (tunnel_port itself is the CONNECT/MITM listener
        # that HTTPS_PROXY traffic hits).  The secrets transform fires on
        # the plain forward too, swapping the Authorization header.
        result = subprocess.run(
            [
                "curl",
                "--silent",
                "--max-time", "10",
                "-x", f"http://127.0.0.1:{tunnel_port + 1}",
                "-H", f"Authorization: Bearer {proxy_token}",
                f"http://127.0.0.1:{upstream_port}/",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"curl failed: {result.stderr}"
        # Some iron-proxy versions return 200 with no body; only the swap matters.
        captured = _CaptureHandler.captured_auth
        assert captured is not None, "upstream never received the request"
        assert real_secret in captured, (
            f"Authorization header was not swapped — upstream saw: {captured!r}"
        )
        assert proxy_token not in captured, (
            f"Proxy token leaked through to upstream: {captured!r}"
        )

    finally:
        # ----- cleanup ------------------------------------------------------
        try:
            ip.stop_proxy()
        except Exception:
            pass
        server.shutdown()
        server.server_close()


class _CaptureXApiKeyHandler(BaseHTTPRequestHandler):
    """Records the x-api-key header of every incoming request."""

    captured_key: Optional[str] = None

    def do_GET(self):
        type(self).captured_key = self.headers.get("x-api-key")
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        return


def test_iron_proxy_swaps_x_api_key_header_end_to_end(hermes_home, monkeypatch):
    """Header-auth providers: the secrets transform must swap the proxy
    token out of a NON-Authorization header (x-api-key — the Anthropic
    native scheme) on the pinned binary."""

    if not __import__("shutil").which("curl"):
        pytest.skip("curl not available")
    if not __import__("shutil").which("openssl"):
        pytest.skip("openssl not available")

    upstream_port = _free_port()
    server = HTTPServer(("127.0.0.1", upstream_port), _CaptureXApiKeyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        binary = ip.install_iron_proxy()
        assert binary.exists()
        ca_crt, ca_key = ip.ensure_ca_cert()

        real_secret = "sk-ant-real-value-cafebabe"
        monkeypatch.setenv("TEST_XAPI_KEY", real_secret)
        proxy_token = ip.mint_proxy_token("anthropic")

        mapping = ip.TokenMapping(
            proxy_token=proxy_token,
            real_env_name="TEST_XAPI_KEY",
            upstream_hosts=("127.0.0.1",),
            match_headers=("x-api-key", "Authorization"),
        )

        tunnel_port = _free_port()
        cfg = ip.build_proxy_config(
            mappings=[mapping],
            ca_cert=ca_crt,
            ca_key=ca_key,
            tunnel_port=tunnel_port,
            allowed_hosts=["127.0.0.1"],
            upstream_deny_cidrs=[],
            http_listen=[f"127.0.0.1:{tunnel_port}"],
        )
        ip.write_proxy_config(cfg)
        ip.write_mappings([mapping])

        try:
            status = ip.start_proxy()
        except RuntimeError as exc:
            pytest.skip(f"iron-proxy could not start in this environment: {exc}")
        assert status.pid is not None

        for _ in range(50):
            if ip._port_listening("127.0.0.1", tunnel_port):
                break
            time.sleep(0.2)
        else:
            pytest.fail("iron-proxy never started listening on the tunnel port")

        result = subprocess.run(
            [
                "curl",
                "--silent",
                "--max-time", "10",
                "-x", f"http://127.0.0.1:{tunnel_port + 1}",
                "-H", f"x-api-key: {proxy_token}",
                f"http://127.0.0.1:{upstream_port}/",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"curl failed: {result.stderr}"
        captured = _CaptureXApiKeyHandler.captured_key
        assert captured is not None, "upstream never received the request"
        assert real_secret in captured, (
            f"x-api-key header was not swapped — upstream saw: {captured!r}"
        )
        assert proxy_token not in captured, (
            f"Proxy token leaked through to upstream: {captured!r}"
        )

    finally:
        try:
            ip.stop_proxy()
        except Exception:
            pass
        server.shutdown()
        server.server_close()


def test_iron_proxy_management_reload_end_to_end(hermes_home, monkeypatch):
    """Real binary: the management listener comes up, an authenticated
    POST /v1/reload succeeds after a config edit, and the edited ruleset
    takes effect WITHOUT a restart (same pid)."""

    if not __import__("shutil").which("curl"):
        pytest.skip("curl not available")
    if not __import__("shutil").which("openssl"):
        pytest.skip("openssl not available")

    upstream_port = _free_port()
    server = HTTPServer(("127.0.0.1", upstream_port), _CaptureHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        binary = ip.install_iron_proxy()
        assert binary.exists()
        ca_crt, ca_key = ip.ensure_ca_cert()

        real_secret = "sk-real-reload-value-0badf00d"
        monkeypatch.setenv("TEST_RELOAD_KEY", real_secret)
        token_v1 = ip.mint_proxy_token("v1")

        def _write_cfg(mapping):
            cfg = ip.build_proxy_config(
                mappings=[mapping],
                ca_cert=ca_crt,
                ca_key=ca_key,
                tunnel_port=tunnel_port,
                allowed_hosts=["127.0.0.1"],
                upstream_deny_cidrs=[],
                http_listen=[f"127.0.0.1:{tunnel_port}"],
            )
            ip.write_proxy_config(cfg)
            ip.write_mappings([mapping])

        tunnel_port = _free_port()
        _write_cfg(ip.TokenMapping(
            proxy_token=token_v1,
            real_env_name="TEST_RELOAD_KEY",
            upstream_hosts=("127.0.0.1",),
        ))

        try:
            status = ip.start_proxy()
        except RuntimeError as exc:
            pytest.skip(f"iron-proxy could not start in this environment: {exc}")
        pid_before = status.pid
        assert pid_before is not None

        for _ in range(50):
            if ip._port_listening("127.0.0.1", tunnel_port):
                break
            time.sleep(0.2)
        else:
            pytest.fail("iron-proxy never started listening")

        # Rotate the sandbox-visible token in the config, then hot-reload.
        token_v2 = ip.mint_proxy_token("v2")
        _write_cfg(ip.TokenMapping(
            proxy_token=token_v2,
            real_env_name="TEST_RELOAD_KEY",
            upstream_hosts=("127.0.0.1",),
        ))
        assert ip.reload_proxy() is True

        # Same daemon (no restart) ...
        assert ip.get_status().pid == pid_before

        # ... but the NEW token now swaps.
        _CaptureHandler.captured_auth = None
        result = subprocess.run(
            [
                "curl", "--silent", "--max-time", "10",
                "-x", f"http://127.0.0.1:{tunnel_port + 1}",
                "-H", f"Authorization: Bearer {token_v2}",
                f"http://127.0.0.1:{upstream_port}/",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"curl failed: {result.stderr}"
        captured = _CaptureHandler.captured_auth
        assert captured is not None and real_secret in captured, (
            f"post-reload token was not swapped — upstream saw: {captured!r}"
        )

    finally:
        try:
            ip.stop_proxy()
        except Exception:
            pass
        server.shutdown()
        server.server_close()
