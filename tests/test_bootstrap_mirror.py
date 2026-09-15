"""Real bootstrap downloads preserve the pin across an upstream outage."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.requests.append(self.path)
            body = self.server.files.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.files, http.requests = {}, []
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield http, f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def fixture_bytes(server, mode, body):
    http, base = server
    if mode != "missing":
        http.files["/primary"] = b"bad bytes" if mode == "corrupt" else body
    http.files["/mirror"] = body
    return base + "/primary", base + "/mirror", hashlib.sha256(body).hexdigest()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("mode", ["primary", "missing", "corrupt", "both-missing"])
def test_windows_download_uses_only_verified_candidates(tmp_path, server, mode):
    body = b"exact archive bytes"
    primary, mirror, digest = fixture_bytes(server, mode, body)
    if mode == "both-missing":
        server[0].files.clear()
    destination = tmp_path / "out.zip"
    script = tmp_path / "driver.ps1"
    script.write_text(
        f". '{ROOT / 'scripts/install.ps1'}' -HermesHome '{tmp_path / 'home'}' -InstallDir '{tmp_path / 'repo'}'\n"
        f"Invoke-VerifiedDownload -Url '{primary}' -MirrorUrl '{mirror}' -Sha256 '{digest}' -OutFile '{destination}'\n",
        encoding="utf-8",
    )
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            capture_output=True, text=True, timeout=60)
    if mode in ("primary", "missing"):
        assert result.returncode == 0, result.stdout + result.stderr
        assert destination.read_bytes() == body
    else:
        assert result.returncode != 0
        assert primary in result.stdout + result.stderr
        if mode == "both-missing":
            assert mirror in result.stdout + result.stderr
    assert server[0].requests == (["/primary", "/mirror"] if mode in ("missing", "both-missing") else ["/primary"])


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("name", ["uv", "git"])
def test_windows_bootstrap_rejects_corrupt_bytes_before_extract(tmp_path, server, name):
    primary, mirror, digest = fixture_bytes(server, "corrupt", b"expected archive")
    script = tmp_path / "caller.ps1"
    script.write_text(
        f". '{ROOT / 'scripts/install.ps1'}' -HermesHome '{tmp_path / 'home'}'\n"
        f"$env:HERMES_RUNTIME_DIR = '{tmp_path / 'tools'}'\n"
        "function Get-Command { param($Name) return $null }\n"
        "$target = 'win32-' + (Get-WindowsArch)\n"
        f"$pin = @{{Url='{primary}'; MirrorUrl='{mirror}'; Sha256='{digest}'}}\n"
        + ("$script:UvPinFiles[$target] = $pin\nGet-Uv\n" if name == "uv" else "$script:GitPinFiles[$target] = $pin\nGet-PinnedGit\n"),
        encoding="utf-8",
    )
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "digest mismatch" in result.stdout + result.stderr
    assert server[0].requests == ["/primary"]
    assert not (tmp_path / "tools").exists()


def uv_archive():
    data = b"#!/bin/sh\n[ \"$1\" = --version ] && { printf 'fixture uv\\n'; exit 0; }; exit 73\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("uv/uv")
        member.mode, member.size = 0o755, len(data)
        archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", ["primary", "missing", "corrupt", "both-missing"])
def test_posix_download_keeps_the_pinned_hash(tmp_path, server, mode):
    primary, mirror, digest = fixture_bytes(server, mode, uv_archive())
    if mode == "both-missing":
        server[0].files.clear()
    script = f"""
source '{ROOT / 'scripts/install.sh'}'
command() {{ if [ "$*" = '-v uv' ]; then return 1; fi; builtin command "$@"; }}
uv_bootstrap_pin() {{ UV_PIN_VERSION=fixture; UV_PIN_URL='{primary}'; UV_PIN_MIRROR='{mirror}'; UV_PIN_SHA256='{digest}'; }}
ensure_uv
"""
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"), "HOME": str(tmp_path / "home"), "HERMES_RUNTIME_DIR": str(tmp_path / "tools")}
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == (mode in ("primary", "missing")), result.stdout + result.stderr
    if mode == "corrupt":
        assert "digest mismatch" in result.stderr
    if mode == "both-missing":
        assert primary in result.stderr and mirror in result.stderr
    assert server[0].requests == (["/primary", "/mirror"] if mode in ("missing", "both-missing") else ["/primary"])


@pytest.mark.platforms("windows")
def test_generated_windows_pins_match_the_shared_authority(tmp_path):
    from pm.artifact_mirror import mirror_url
    script = tmp_path / "pins.ps1"
    script.write_text(
        f". '{ROOT / 'scripts/install.ps1'}' -HermesHome '{tmp_path / 'home'}'\n"
        "@{uv=$script:UvPinFiles; git=$script:GitPinFiles} | ConvertTo-Json -Depth 5 -Compress\n",
        encoding="utf-8",
    )
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-File", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    lock = json.loads((ROOT / "pm/lock.json").read_text(encoding="utf-8"))["packages"]
    for name, targets in emitted.items():
        assert targets
        for target, pin in targets.items():
            authority = lock[name]["artifacts"][target]
            assert (pin["Url"], pin["Sha256"], pin["MirrorUrl"]) == (authority["url"], authority["sha256"], mirror_url(authority["sha256"]))


@pytest.mark.platforms("posix")
def test_dev_setup_reaches_the_same_mirror_without_python(tmp_path, server):
    from pm.store import current_target
    body = uv_archive()
    digest = hashlib.sha256(body).hexdigest()
    http, base = server
    http.files["/archive/" + digest] = body
    repo = tmp_path / "repo"
    (repo / "pm").mkdir(parents=True)
    shutil.copyfile(ROOT / "setup-hermes.sh", repo / "setup-hermes.sh")
    (repo / "pm/artifact-mirror.json").write_text(json.dumps({"origin": base, "prefix": "archive/"}, indent=2), encoding="utf-8")
    (repo / "pm/lock.json").write_text(json.dumps({"packages": {
        "uv": {"version": "fixture", "artifacts": {current_target(): {"url": base + "/missing.tar.gz", "sha256": digest}}},
        "python": {"version": "3.14.7"},
    }}, indent=2, sort_keys=True), encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"), "HOME": str(tmp_path / "home"), "HERMES_RUNTIME_DIR": str(tmp_path / "tools")}
    result = subprocess.run(["bash", str(repo / "setup-hermes.sh")], cwd=repo, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 73, result.stdout + result.stderr  # stop at the bootstrap interpreter boundary
    assert http.requests == ["/missing.tar.gz", "/archive/" + digest]
