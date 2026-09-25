"""E2E: credential swap must work through the CONNECT tunnel listener, not only plain forward.

Sandboxes reach iron-proxy via ``HTTPS_PROXY`` (a CONNECT tunnel on ``tunnel_listen``).  The
proxy evaluates a synthetic CONNECT request against the transform pipeline before any inner
request exists; that CONNECT carries no Authorization header.  A ``require: true`` secrets rule
that matches CONNECT therefore rejects every correctly tokened request before it is seen.

Hermetic: ``curl --proxytunnel`` forces a CONNECT tunnel to a local plain-HTTP upstream, which
exercises the same tunnel admission path as HTTPS without needing a trusted upstream cert.
Gated like the sibling E2E file (``HERMES_RUN_E2E=1``; needs curl + openssl + the binary).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def _free_port_block(n: int = 3) -> int:
    """First of ``n`` consecutive free ports (proxy derives http/management as +1/+2).

    Drawn from below the ephemeral range: the daemon's own ``:0`` listeners take sequential
    ephemeral ports on macOS and would land on a block picked from that range."""
    import random
    for _ in range(500):
        base = random.randrange(20000, 32000)
        socks = []
        try:
            for off in range(n):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", base + off))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port block")


class _Capture(BaseHTTPRequestHandler):
    auth: Optional[str] = None
    hits = 0

    def do_GET(self):
        type(self).auth = self.headers.get("Authorization")
        type(self).hits += 1
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        return


def _curl_via_tunnel(tunnel_port: int, upstream_port: int, token: Optional[str]) -> str:
    argv = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10",
            "--proxytunnel", "-x", f"http://127.0.0.1:{tunnel_port}"]
    if token:
        argv += ["-H", f"Authorization: Bearer {token}"]
    argv.append(f"http://127.0.0.1:{upstream_port}/")
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def test_tunnel_swaps_secret_and_still_requires_token(hermes_home, monkeypatch):
    """Tokened request through the CONNECT tunnel reaches upstream with the real secret;
    an untokened request to the same host is still refused (``require`` keeps working)."""
    if not (shutil.which("curl") and shutil.which("openssl")):
        pytest.skip("curl/openssl not available")

    _Capture.auth, _Capture.hits = None, 0
    upstream_port = _free_port_block(1)
    server = HTTPServer(("127.0.0.1", upstream_port), _Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ip.install_iron_proxy()
        ca_crt, ca_key = ip.ensure_ca_cert()
        real_secret = "tunnel-e2e-real-secret"
        monkeypatch.setenv("TUNNEL_E2E_KEY", real_secret)
        token = ip.mint_proxy_token("tunnel")
        mapping = ip.TokenMapping(proxy_token=token, real_env_name="TUNNEL_E2E_KEY",
                                  upstream_hosts=("127.0.0.1",))
        tunnel_port = _free_port_block()
        cfg = ip.build_proxy_config(mappings=[mapping], ca_cert=ca_crt, ca_key=ca_key,
                                    tunnel_port=tunnel_port, allowed_hosts=["127.0.0.1"],
                                    upstream_deny_cidrs=[], http_listen=[f"127.0.0.1:{tunnel_port}"])
        ip.write_proxy_config(cfg)
        ip.write_mappings([mapping])
        try:
            ip.start_proxy()
        except RuntimeError as exc:
            pytest.skip(f"iron-proxy could not start in this environment: {exc}")
        for _ in range(50):
            if ip._port_listening("127.0.0.1", tunnel_port):
                break
            time.sleep(0.2)
        else:
            pytest.fail("iron-proxy never started listening on the tunnel port")

        assert _curl_via_tunnel(tunnel_port, upstream_port, token) == "200"
        assert _Capture.auth == f"Bearer {real_secret}", f"upstream saw {_Capture.auth!r}"

        hits_before = _Capture.hits
        assert _curl_via_tunnel(tunnel_port, upstream_port, None) != "200"
        assert _Capture.hits == hits_before, "untokened request must not reach upstream"
    finally:
        try:
            ip.stop_proxy()
        except Exception:
            pass
        server.shutdown()
        server.server_close()
