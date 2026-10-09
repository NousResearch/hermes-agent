"""The shell-staged uv can prepare TLS support before PM downloads Python."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
def test_staged_uv_prepares_pm_before_any_tool_download(tmp_path):
    from pm.packages import Uv
    from pm.store import current_target

    uv = shutil.which("uv")
    assert uv, "this bootstrap contract requires real uv"
    repo = Path(__file__).resolve().parents[2]
    stage = tmp_path / "source"
    for name in ("pm", "hermes_cli"):
        shutil.copytree(repo / name, stage / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "hermes_constants.py", stage / "hermes_constants.py")
    home = tmp_path / "home"
    store = home / "tools"
    target = current_target()
    # As in setup: uv has been verified/extracted, but PM has no installed facts.
    entry = store / Uv().store_entry("bootstrap-fixture", target)
    binary = Uv().binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    shutil.copy2(uv, binary)
    (stage / "pm" / "lock.json").write_text(json.dumps({"schema": 1, "packages": {
        name: {"version": "bootstrap-fixture", "artifacts": {target: {
            "url": f"https://must-not-fetch.invalid/{name}.tar.gz", "sha256": "a" * 64,
        }}} for name in ("uv", "python")
    }}))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "UV_"))}
    env.update(HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(store))
    code = """
import sys, subprocess
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pm.runtime import runtime_command, runtime_environment
result = subprocess.run(runtime_command(Path(sys.argv[2])), env=runtime_environment(),
                        capture_output=True, text=True)
assert result.returncode == 0, result.stdout + result.stderr
print(result.stdout)
"""
    probe = tmp_path / "probe.py"
    probe.write_text("import json, truststore; print(json.dumps({'tls': truststore.__file__}))")
    command = [sys.executable, "-I", "-S", "-c", code, str(stage), str(probe)]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(json.loads(result.stdout)["tls"]).is_relative_to(home)
    assert "Preparing the isolated Hermes runtime" in result.stderr
    assert "must-not-fetch.invalid" not in result.stderr
    warm = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert warm.returncode == 0, warm.stdout + warm.stderr
    assert "Preparing the isolated Hermes runtime" not in warm.stderr
    assert warm.stdout == result.stdout
    assert not (store / "facts.json").exists(), "preparing PM must not realize its tool closure"
    assert not list(store.glob("python-*"))
    assert not list(home.glob("installs/*/environments")), "no app environment during PM bootstrap"


@pytest.mark.platforms("linux")
def test_pm_cli_verifies_tls_with_platform_trust(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    import hashlib
    from http.server import ThreadingHTTPServer
    from ipaddress import ip_address
    import io
    import ssl
    import tarfile
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    from pm.runtime import prepare_runtime, runtime_environment
    from tests.pm._range_server import RangeHandler

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    uv = shutil.which("uv")
    assert uv
    python = prepare_runtime(Path(uv), Path(sys.executable), tmp_path / "runtime")
    source = Path(__file__).resolve().parents[2]
    repo = tmp_path / "source"
    for name in ("pm", "hermes_cli"):
        shutil.copytree(source / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source / "hermes_constants.py", repo / "hermes_constants.py")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PM test CA")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    bundle, private = tmp_path / "ca.pem", tmp_path / "server.key"
    bundle.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    payload = b"verified through platform trust"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("payload.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    body = stream.getvalue()

    class Handler(RangeHandler):
        payloads = {"/tool.tar.gz": body}

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(bundle, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (repo / "pm" / "lock.json").write_text(json.dumps({"schema": 1, "packages": {
        "tls-test": {"version": "1", "artifacts": {"any": {
            "url": f"https://127.0.0.1:{server.server_port}/tool.tar.gz",
            "sha256": hashlib.sha256(body).hexdigest(),
        }}},
    }}))
    # Inject missing OpenSSL defaults, not a fake platform. Only truststore's
    # real Linux CA discovery can find the test CA; urllib alone must reject it.
    package = repo / "pm" / "tls_fixture.py"
    package.write_text("from pm import Package, register\n@register\nclass Tool(Package):\n    name = 'tls-test'\n")
    driver = """
import runpy, ssl, sys
import truststore._openssl as platform_tls
defaults = ssl.get_default_verify_paths()
ssl.get_default_verify_paths = lambda: defaults._replace(cafile=None, capath=None)
platform_tls._CA_FILE_CANDIDATES = [sys.argv.pop(1)]
script = sys.argv.pop(1)
sys.path.insert(0, sys.argv.pop(1))
sys.argv[0] = script
# Register only the fixture package; the real entrypoint owns TLS activation.
module = runpy.run_path(script, run_name='tls_entrypoint_test')
import pm.tls_fixture
raise SystemExit(module['main']())
"""
    env = runtime_environment()
    env.pop("SSL_CERT_FILE", None)
    env.pop("SSL_CERT_DIR", None)
    env["NO_PROXY"] = "127.0.0.1"
    try:
        command = [str(python), "-I", "-B", "-c", driver, str(tmp_path / "missing-ca"),
                   str(repo / "pm" / "launch.py"), str(repo), "install", "tls-test"]
        rejected = subprocess.run(command, cwd=tmp_path, env=env,
                                  capture_output=True, text=True, timeout=60)
        assert rejected.returncode == 1, rejected.stdout + rejected.stderr
        assert "CERTIFICATE_VERIFY_FAILED" in rejected.stdout + rejected.stderr
        assert not list((home / "tools").glob("tls-test-*/payload.txt"))
        command[5] = str(bundle)
        result = subprocess.run(command, cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        installed = list((home / "tools").glob("tls-test-*/payload.txt"))
        assert len(installed) == 1, result.stdout + result.stderr
        assert installed[0].read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
def test_importing_launch_does_not_patch_ssl_context():
    import subprocess
    import sys

    from pm.paths import repo_root
    child = subprocess.run(
        [sys.executable, "-c", "import ssl; original = ssl.SSLContext; import pm.launch; assert ssl.SSLContext is original"],
        cwd=repo_root(), capture_output=True, text=True, timeout=30, check=False,
    )
    assert child.returncode == 0, child.stderr



def test_cold_downloader_trusts_a_stored_intermediate_without_its_root(tmp_path):
    """Windows can cache an intermediate (Let's Encrypt "Root YR") without its
    root; Python < 3.13 then refused every GitHub asset before PM existed."""
    from datetime import datetime, timedelta, timezone
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from ipaddress import ip_address
    import hashlib
    import ssl
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.now(timezone.utc)

    def issue(name, signer=None):
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        signer_key, signer_name = signer or (key, subject)
        builder = (x509.CertificateBuilder().subject_name(subject).issuer_name(signer_name)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
                   .add_extension(x509.BasicConstraints(ca=name != "leaf", path_length=None), critical=True))
        if name == "leaf":
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
        return key, subject, builder.sign(signer_key, hashes.SHA256())

    root_key, root_name, _ = issue("absent root")
    middle_key, middle_name, middle = issue("stored intermediate", (root_key, root_name))
    leaf_key, _, leaf = issue("leaf", (middle_key, middle_name))
    pem = lambda cert: cert.public_bytes(serialization.Encoding.PEM)
    stores = {"stored": pem(middle), "unrelated": pem(issue("unrelated CA")[2])}
    for name, body in stores.items():
        (tmp_path / f"{name}.pem").write_bytes(body)
    (tmp_path / "chain.pem").write_bytes(pem(leaf) + pem(middle))
    (tmp_path / "leaf.key").write_bytes(leaf_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    payload = b"pinned tool"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "chain.pem", tmp_path / "leaf.key")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # The cold update parent is the legacy venv's 3.11/3.12: drop the default
    # verify flags 3.13 added, on any host Python.
    driver = """
import ssl, sys
from pathlib import Path
legacy = ssl.create_default_context
def create(*args, **kwargs):
    context = legacy(*args, **kwargs)
    context.verify_flags &= ~(ssl.VERIFY_X509_PARTIAL_CHAIN | ssl.VERIFY_X509_STRICT)
    return context
ssl.create_default_context = ssl._create_default_https_context = create
sys.path.insert(0, sys.argv[1])
from pm.downloader import Download, Source
dest = Path(sys.argv[3])
Download([Source(sys.argv[2], dest, sys.argv[4])], partials_dir=dest.parent / "partials").run()
"""
    (tmp_path / "no-cert-dir").mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "SSL_"))}
    env.update(SSL_CERT_DIR=str(tmp_path / "no-cert-dir"), NO_PROXY="127.0.0.1")
    results = {}
    try:
        for name in stores:
            dest = tmp_path / name / "tool"
            results[name] = subprocess.run(
                [sys.executable, "-I", "-S", "-c", driver, str(Path(__file__).resolve().parents[2]),
                 f"https://127.0.0.1:{server.server_port}/tool", str(dest),
                 hashlib.sha256(payload).hexdigest()],
                env={**env, "SSL_CERT_FILE": str(tmp_path / f"{name}.pem")},
                capture_output=True, text=True, timeout=60)
    finally:
        server.shutdown()
        server.server_close()
    assert results["stored"].returncode == 0, results["stored"].stderr
    assert (tmp_path / "stored" / "tool").read_bytes() == payload
    assert "CERTIFICATE_VERIFY_FAILED" in results["unrelated"].stderr
    assert not (tmp_path / "unrelated" / "tool").exists()
