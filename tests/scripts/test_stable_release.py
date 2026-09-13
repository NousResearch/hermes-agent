"""Release gates and package transitions bind the intended immutable artifacts."""
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.releases.stable import (
    check_tag, plan_transitions, read_manifest, require_stable_identity,
    require_success, validate_candidates,
)

BASE = "https://releases.example"
ROOT = Path(__file__).resolve().parents[2]


def candidates(tag, commit, digest):
    packages = []
    for platform in ("windows", "macos"):
        for arch in ("x64", "arm64"):
            packages.append({
                "platform": platform, "arch": arch, "tag": tag, "commit": commit,
                "identity": "test.application",
                "version": f"{tag[1:]}.0" if platform == "windows" else tag[1:],
                **({"publisher": "CN=Test", "applicationId": "App"} if platform == "windows" else {"teamId": "ABCDEFGHIJ"}),
                "artifact": {"sha256": digest,
                             "url": f"{BASE}/releases/tag/{tag}/{arch}" + (".msixbundle" if platform == "windows" else ".zip")},
            })
    return {"schema": 1, "tag": tag, "commit": commit, "packages": packages}


def test_gate_requires_every_success_including_real_cli(tmp_path):
    required = ["ci", "docker", "acceptance", "publication"]
    success = {name: {"result": "success"} for name in required}
    require_success(success, required)
    with pytest.raises(ValueError, match="required-job list"):
        require_success(success, [])
    for name in required:
        for result in ("failure", "cancelled", "skipped", None):
            needs = copy.deepcopy(success)
            if result:
                needs[name]["result"] = result
            else:
                del needs[name]
            with pytest.raises(ValueError, match=name):
                require_success(needs, required)
    summary = tmp_path / "summary.md"
    env = {**os.environ, "RELEASE_NEEDS": json.dumps(success), "GITHUB_STEP_SUMMARY": str(summary), "PYTHONPATH": str(ROOT)}
    argv = [sys.executable, "-m", "scripts.releases.stable", "gate", *required]
    assert subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True).returncode == 0
    empty = subprocess.run(argv[:4], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert empty.returncode != 0
    assert "required-job list" in empty.stderr
    env["RELEASE_NEEDS"] = json.dumps({**success, "publication": {"result": "cancelled"}})
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode != 0
    assert "publication=cancelled" in result.stderr


def test_transitions_bind_all_arches_identity_version_and_archive():
    old = candidates("v1.2.3", "a" * 40, "1" * 64)
    new = candidates("v1.2.4", "b" * 40, "2" * 64)
    require_stable_identity(new["tag"], new["commit"], "refs/tags/v1.2.4")
    for tag, ref in [("v1.2.4", "refs/heads/main"), ("v1.2.4-canary.20260907143420", "refs/tags/v1.2.4-canary.20260907143420")]:
        with pytest.raises(ValueError):
            require_stable_identity(tag, new["commit"], ref)
    transitions = plan_transitions(old, new, BASE)
    assert {row["target"] for row in transitions} == {"windows-x64", "windows-arm64", "macos-x64", "macos-arm64"}
    assert all(row["transition"]["new"]["commit"] == new["commit"] for row in transitions)
    missing = copy.deepcopy(new)
    missing["packages"].pop()
    with pytest.raises(ValueError, match="both architectures"):
        plan_transitions(old, missing, BASE)
    with pytest.raises(ValueError, match="identity"):
        validate_candidates(new, new["tag"], old["commit"], BASE)
    for key, value in [("commit", old["commit"]), ("identity", "different"), ("publisher", "CN=Other"), ("version", "9.9.9.0")]:
        changed = copy.deepcopy(new)
        changed["packages"][0][key] = value
        with pytest.raises(ValueError):
            plan_transitions(old, changed, BASE)
    mutable = copy.deepcopy(new)
    mutable["packages"][0]["artifact"]["url"] = f"{BASE}/releases/win32/stable/current.msixbundle"
    with pytest.raises(ValueError, match="immutable"):
        plan_transitions(old, mutable, BASE)
    for suffix in ("../other.zip", "%2e%2e/other.zip", "%252e%252e/other.zip"):
        traversal = copy.deepcopy(new)
        traversal["packages"][0]["artifact"]["url"] = f"{BASE}/releases/tag/{new['tag']}/{suffix}"
        with pytest.raises(ValueError, match="path encoding"):
            plan_transitions(old, traversal, BASE)
    with pytest.raises(ValueError, match="increase"):
        plan_transitions(new, old, BASE)


def test_manifest_origin_checks_with_real_https(tmp_path):
    import datetime
    import ipaddress
    import ssl
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
            .not_valid_after(datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]), critical=False).sign(key, hashes.SHA256()))
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                         serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    requests = []
    data = b'{"schema":1}'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path in ("/same", "/cross"):
                self.send_response(302)
                host = "127.0.0.1" if self.path == "/same" else "localhost"
                self.send_header("Location", f"https://{host}:{self.server.server_port}/manifest")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_file, key_file)
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    client_context = ssl.create_default_context(cafile=str(cert_file))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                        urllib.request.HTTPSHandler(context=client_context)).open
    base = f"https://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        digest = hashlib.sha256(data).hexdigest()
        assert read_manifest(f"{base}/same", digest, expected_origin=base, opener=opener) == {"schema": 1}
        with pytest.raises(ValueError, match="origin"):
            read_manifest(f"{base}/cross", opener=opener)
        requests.clear()
        with pytest.raises(ValueError, match="origin"):
            read_manifest(f"https://localhost:{server.server_port}/manifest", expected_origin=base, opener=opener)
        assert requests == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_manifest_digest_and_tag_movement_fail_closed(tmp_path, monkeypatch):
    data = b'{"schema":1}'

    class Response(io.BytesIO):
        def geturl(self):
            return BASE + "/manifest.json"

    def opener(url, timeout):
        return Response(data)

    assert read_manifest(BASE, hashlib.sha256(data).hexdigest(), opener=opener) == {"schema": 1}
    with pytest.raises(ValueError, match="digest"):
        read_manifest(BASE, "f" * 64, opener=opener)
    commit = "a" * 40
    env = {"RELEASE_TAG": "v1.2.3", "GITHUB_SHA": commit, "GITHUB_REF": "refs/tags/v1.2.3"}

    def git(argv):
        if argv[1] == "ls-remote":
            return f"{'b' * 40}\trefs/tags/v1.2.3\n{commit}\trefs/tags/v1.2.3^{{}}"
        return commit if argv[1] == "rev-parse" else ""

    assert check_tag(env, git) == ("v1.2.3", commit)
    with pytest.raises(ValueError, match="moved"):
        check_tag(env, lambda argv: f"{'c' * 40}\trefs/tags/v1.2.3" if argv[1] == "ls-remote" else git(argv))

    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "fixture"], check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], check=True)
    (repo / "input").write_text("first", encoding="utf-8")
    subprocess.run(["git", "add", "input"], check=True)
    subprocess.run(["git", "commit", "-m", "first"], check=True, capture_output=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(["git", "tag", "v1.2.3"], check=True)
    subprocess.run(["git", "push", "origin", "main", "v1.2.3"], check=True, capture_output=True)
    env["GITHUB_SHA"] = actual
    assert check_tag(env) == ("v1.2.3", actual)
    subprocess.run(["git", "--git-dir", str(remote), "update-ref", "-d", "refs/tags/v1.2.3"], check=True)
    with pytest.raises(ValueError, match="moved"):
        check_tag(env)
