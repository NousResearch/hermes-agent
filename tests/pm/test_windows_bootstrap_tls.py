"""Cold Windows acquisition uses OS trust without weakening integrity checks."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import threading

import pytest


@pytest.mark.platforms("windows")
def test_cold_isolated_bootstrap_acquires_pinned_tools_with_native_trust(tmp_path):
    """Real pinned archives and PM runtime, no app graph or trust-store edits."""
    source = Path(__file__).resolve().parents[2]
    repo = tmp_path / "source"
    repo.mkdir()
    for name in ("pm", "hermes_cli"):
        shutil.copytree(
            source / name, repo / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copy2(source / "hermes_constants.py", repo / "hermes_constants.py")
    home = tmp_path / "home"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "UV_", "HERMES_"))
    }
    env.update(
        HERMES_HOME=str(home),
        HERMES_RUNTIME_DIR=str(home / "tools"),
        UV_CACHE_DIR=str(tmp_path / "cache"),
    )
    # A secure OpenSSL context lacking its issuer reproduces the bootstrap
    # boundary. The native verifier and every actual request stay real.
    driver = """
import json, ssl, subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
ssl.create_default_context = lambda *a, **kw: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl._create_default_https_context = ssl.create_default_context
from pm.runtime import runtime_command, runtime_environment
from pm.lock import Facts
from pm.paths import store_root
assert 'truststore' not in sys.modules
command = runtime_command(Path(sys.argv[2]))
assert 'truststore' not in sys.modules, 'bootstrap imported application dependencies'
facts = Facts(store_root() / 'facts.json')
assert facts.get('python') and facts.get('uv')
child = subprocess.run(command, env=runtime_environment(), capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
assert child.returncode == 0, child.stdout + child.stderr
print(child.stdout)
"""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, truststore; print(json.dumps({'truststore': truststore.__file__}))",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-X", "utf8", "-c", driver, str(repo), str(probe)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(json.loads(result.stdout)["truststore"]).is_relative_to(home)
    assert not list(home.glob("installs/*/environments")), (
        "bootstrap selected application dependencies"
    )


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    "failure", ["hash", "certificate", "downgrade", "http", "pause"]
)
def test_native_bootstrap_rejects_unverified_downloads(tmp_path, failure):
    from datetime import datetime, timedelta, timezone
    from ipaddress import ip_address

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    from pm.downloader import (
        Download,
        DownloadPaused,
        DownloadTransportError,
        HashError,
        Source,
    )

    payload = b"synthetic pinned tool"
    started, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        requests = 0

        def do_GET(self):
            type(self).requests += 1
            if failure == "downgrade":
                self.send_response(302)
                self.send_header("Location", "http://example.invalid/tool")
            elif failure == "http":
                self.send_response(403)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if failure == "pause":
                started.set()
                release.wait(timeout=10)
            if failure in ("hash", "certificate"):
                self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if failure == "certificate":
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Untrusted bootstrap fixture")
        ])
        now = datetime.now(timezone.utc)
        cert = (
            x509
            .CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        certificate, private = tmp_path / "untrusted.pem", tmp_path / "key.pem"
        certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        private.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, private)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "tool.zip"
    destination.write_bytes(b"previous verified artifact")
    scheme = "https" if failure == "certificate" else "http"
    digest = "0" * 64 if failure == "hash" else hashlib.sha256(payload).hexdigest()
    try:
        job = Download(
            [
                Source(
                    f"{scheme}://127.0.0.1:{server.server_port}/tool",
                    destination,
                    digest,
                )
            ],
            native_tls=True,
            partials_dir=tmp_path / "partials",
        )

        def progress(*args):
            if failure == "pause" and started.is_set():
                job.pause()

        expected = (
            HashError
            if failure == "hash"
            else DownloadPaused
            if failure == "pause"
            else DownloadTransportError
        )
        with pytest.raises(expected) as rejected:
            job.run(progress=progress)
        if failure == "http":
            assert isinstance(rejected.value, DownloadTransportError)
            assert rejected.value.status == 403
        assert destination.read_bytes() == b"previous verified artifact"
        assert Handler.requests == (0 if failure == "certificate" else 1)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
