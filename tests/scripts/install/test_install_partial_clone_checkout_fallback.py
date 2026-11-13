"""A fresh install whose blobless checkout is refused falls back to an unfiltered clone (#153258).

GitHub can answer the promisor lazy-fetch a ``--filter=blob:none`` checkout sends with HTTP 400 while
an ordinary clone of the same repository works. The fixture is a real smart-HTTP server
(``git http-backend``) that refuses exactly the upload-pack requests wanting a blob object.
"""
from __future__ import annotations

import gzip
import http.server
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"
POWERSHELL = next((c for c in ("pwsh", "powershell") if shutil.which(c)), None)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _wants(body: bytes) -> list[str]:
    wants, i = [], 0
    while i + 4 <= len(body):
        try:
            size = int(body[i:i + 4], 16)
        except ValueError:
            break
        if size < 4:
            i += 4
            continue
        line = body[i + 4:i + size].decode("latin-1").strip()
        if line.startswith("want "):
            wants.append(line.split()[1])
        i += size
    return wants


def _serve(project_root: Path, refused: list[str]) -> http.server.ThreadingHTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def _backend(self, body: bytes = b"") -> None:
            path, _, query = self.path.partition("?")
            env = dict(os.environ, GIT_PROJECT_ROOT=str(project_root), GIT_HTTP_EXPORT_ALL="1",
                       PATH_INFO=path, REQUEST_METHOD=self.command, QUERY_STRING=query,
                       CONTENT_TYPE=self.headers.get("Content-Type", ""), CONTENT_LENGTH=str(len(body)),
                       REMOTE_ADDR="127.0.0.1", GIT_PROTOCOL=self.headers.get("Git-Protocol", ""))
            if self.headers.get("Content-Encoding"):
                env["HTTP_CONTENT_ENCODING"] = self.headers["Content-Encoding"]
            out = subprocess.run(["git", "http-backend"], input=body, env=env, capture_output=True, check=False).stdout
            head, _, payload = out.partition(b"\r\n\r\n")
            status, headers = 200, []
            for line in head.decode("latin-1").split("\r\n"):
                key, _, value = line.partition(":")
                if key.lower() == "status":
                    status = int(value.split()[0])
                elif key:
                    headers.append((key, value.strip()))
            self.send_response(status)
            for key, value in headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._backend()

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            raw = gzip.decompress(body) if self.headers.get("Content-Encoding") == "gzip" else body
            wants = _wants(raw)
            kinds = subprocess.run(["git", "-C", str(project_root / "repo.git"), "cat-file", "--batch-check=%(objecttype)"],
                                   input="".join(f"{w}\n" for w in wants), capture_output=True, text=True,
                                   check=False).stdout.split() if wants else []
            if "blob" in kinds:
                refused.append(self.path)
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._backend(body)

        def log_message(self, format, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def blob_refusing_origin(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "main")
    (source / "README").write_text("release")
    _git(source, "add", "README")
    _git(source, "commit", "-qm", "release")
    _git(source, "tag", "v0.1.0")
    (source / "README").write_text("tip")
    _git(source, "commit", "-qam", "tip")
    served = tmp_path / "served"
    served.mkdir()
    subprocess.run(["git", "clone", "-q", "--bare", str(source), str(served / "repo.git")], check=True)
    _git(served / "repo.git", "config", "uploadpack.allowFilter", "true")
    _git(served / "repo.git", "config", "uploadpack.allowAnySHA1InWant", "true")
    refused: list[str] = []
    server = _serve(served, refused)
    yield f"http://127.0.0.1:{server.server_address[1]}/repo.git", served / "repo.git", refused
    server.shutdown()


def _stage(tmp_path: Path, url: str, prelude: str) -> subprocess.CompletedProcess:
    script = (f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest\nsleep() {{ :; }}\n"
              f"{prelude}\nstage_repository\n")
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=(tmp_path / "install").as_posix(), HERMES_REPO_URL=url,
               GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120, check=False)


_STABLE = "CHANNEL=stable\nlatest_stable_release() { printf v0.1.0; }"


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("pin", ["release", "commit"])
def test_refused_partial_checkout_publishes_an_unfiltered_clone(tmp_path, blob_refusing_origin, pin):
    url, origin, refused = blob_refusing_origin
    expected = _git(origin, "rev-parse", "v0.1.0^{commit}")
    prelude = _STABLE if pin == "release" else f"INSTALL_COMMIT={expected}"
    result = _stage(tmp_path, url, prelude)
    output = result.stdout + result.stderr
    assert refused, "the fixture never refused a blob fetch: the partial clone path was not exercised"
    assert result.returncode == 0, output
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == expected
    assert (install / "README").read_text() == "release"
    assert _git(install, "status", "--porcelain") == ""
    assert _git(install, "config", "--get-all", "remote.origin.fetch") == "+refs/heads/*:refs/remotes/origin/*"
    assert not list(tmp_path.glob(".hermes-clone-*"))


@pytest.mark.live_system_guard_bypass
def test_fallback_refuses_a_tag_that_moved_between_the_two_clones(tmp_path, blob_refusing_origin):
    """The unfiltered clone is published only when the release tag still names the commit it did."""
    url, origin, refused = blob_refusing_origin
    tip = _git(origin, "rev-parse", "main")
    move_tag = (f'git() {{ command git "$@" || return; case " $* " in *" --depth 1 --no-checkout "*) '
                f'command git -C {shlex.quote(origin.as_posix())} tag -f v0.1.0 {tip} >/dev/null 2>&1 ;; esac; }}')
    result = _stage(tmp_path, url, f"{_STABLE}\n{move_tag}")
    output = result.stdout + result.stderr
    assert refused
    assert result.returncode != 0
    assert "not publishing it" in output, output
    assert "could not check out the cloned tree; no checkout published: error: RPC failed; HTTP 400" in output, output
    assert not (tmp_path / "install").exists()
    assert not list(tmp_path.glob(".hermes-clone-*"))


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(POWERSHELL is None, reason="needs PowerShell")
def test_ps1_refused_partial_checkout_publishes_an_unfiltered_clone(tmp_path, blob_refusing_origin):
    url, origin, refused = blob_refusing_origin
    expected = _git(origin, "rev-parse", "v0.1.0^{commit}")
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_REPO_URL=url, GIT_TERMINAL_PROMPT="0")
    if os.name != "nt":
        # pwsh on POSIX cannot run the Windows-only pinned Git: seed its PM store slot with host git.
        version = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8-sig"))["packages"]["git"]["version"]
        cmd = tmp_path / "tools" / f"git-{version}-win32-x64" / "cmd"
        cmd.mkdir(parents=True)
        for name in ("git.exe", "git"):
            (cmd / name).symlink_to(shutil.which("git"))
        env.update(HERMES_RUNTIME_DIR=str(tmp_path / "tools"), PATH=str(cmd) + os.pathsep + env["PATH"])
    install = tmp_path / "install"
    result = subprocess.run([POWERSHELL, "-NoProfile", "-File", str(ROOT / "scripts" / "install.ps1"),
                             "-Stage", "repository", "-NonInteractive", "-Commit", expected,
                             "-InstallDir", str(install), "-HermesHome", str(tmp_path / "home")],
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=240, check=False)
    assert refused, "the fixture never refused a blob fetch: the partial clone path was not exercised"
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(install, "rev-parse", "HEAD") == expected
    assert (install / "README").read_text() == "release"
    assert _git(install, "status", "--porcelain") == ""
    assert not list(tmp_path.glob(".hermes-clone-*"))
