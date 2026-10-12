"""git stall detection in the installers: slow transfers abort so retries run.

Regression for #37246 (Ubuntu "early EOF / index-pack failed" at ~30 KiB/s)
and the stuck-clone reports: with no ``http.lowSpeedLimit``/``lowSpeedTime``,
a stalled ``git clone``/``fetch`` never exits non-zero, so the installer's
retry loop never runs and the install hangs forever.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
INSTALL_SH = ROOT / "scripts" / "install.sh"
INSTALL_PS1 = ROOT / "scripts" / "install.ps1"
POWERSHELL = next((c for c in ("pwsh", "powershell") if shutil.which(c)), None)

needs_bash_git = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash"
)

STALL_LIMIT = "http.lowSpeedLimit=1000"
STALL_TIME = "http.lowSpeedTime=120"
HTTP11 = "http.version=HTTP/1.1"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()


def _origin(path: Path, content: str = "one") -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "README").write_text(content)
    _git(path, "add", "README")
    _git(path, "commit", "-qm", content)
    return path


def _run_sh(tmp_path: Path, body: str, *, env_extra: dict | None = None,
            timeout: int = 60) -> subprocess.CompletedProcess:
    script = f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest\nsleep() {{ :; }}\n{body}\n"
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=(tmp_path / "install").as_posix())
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy, None)
    env["NO_PROXY"] = "*"
    env.update(env_extra or {})
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


def _stage_sh(tmp_path: Path, origin: Path, *, prelude: str = "",
              extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {"HERMES_REPO_URL": origin.as_posix(), **(extra_env or {})}
    return _run_sh(tmp_path, f"{prelude}\nstage_repository", env_extra=env)


def _calls(trace: Path, subcommand: str) -> list[list[str]]:
    return [
        line.split()
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines()
        if subcommand in line.split()
    ]


def _assert_stall_before(tokens: list[str], subcommand: str, *, http11: bool) -> None:
    assert STALL_LIMIT in tokens, tokens
    assert STALL_TIME in tokens, tokens
    assert tokens.index(subcommand) > tokens.index("-c"), tokens
    # `git -c ... clone`: never `git clone -c ...`, which persists into .git/config.
    assert tokens[tokens.index(subcommand) + 1] != "-c", tokens
    assert (HTTP11 in tokens) is http11, tokens


def _assert_no_persisted_stall_config(install: Path) -> None:
    config = (install / ".git" / "config").read_text()
    assert "lowSpeed" not in config, config
    assert "http.version" not in config, config


@needs_bash_git
def test_clone_attempts_carry_stall_opts_and_http11_fallback(tmp_path):
    origin = _origin(tmp_path / "origin")
    trace = tmp_path / "trace.log"
    stub = (
        "git() {\n"
        '    case " $* " in *" clone "*) '
        f'printf \'%s\\n\' "$*" >> {shlex.quote(trace.as_posix())}; '
        'case " $* " in *" --filter=blob:none --no-checkout "*) ;; *) return 1 ;; esac;;\n'
        "    esac\n"
        '    command git "$@"\n'
        "}"
    )
    result = _stage_sh(tmp_path, origin, prelude=stub)
    assert result.returncode == 0, result.stdout + result.stderr
    clones = _calls(trace, "clone")
    assert len(clones) == 4, [ " ".join(c) for c in clones ]
    _assert_stall_before(clones[0], "clone", http11=False)
    for attempt in clones[1:]:
        _assert_stall_before(attempt, "clone", http11=True)
    assert (tmp_path / "install" / "README").read_text() == "one"
    _assert_no_persisted_stall_config(tmp_path / "install")


@needs_bash_git
def test_update_fetch_carries_stall_opts(tmp_path):
    origin = _origin(tmp_path / "origin")
    assert _stage_sh(tmp_path, origin).returncode == 0
    (origin / "README").write_text("two")
    _git(origin, "commit", "-qam", "two")
    trace = tmp_path / "trace.log"
    tracer = f'git() {{ printf "%s\\n" "$*" >> {shlex.quote(trace.as_posix())}; command git "$@"; }}'
    result = _stage_sh(tmp_path, origin, prelude=tracer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "install" / "README").read_text() == "two"
    fetches = _calls(trace, "fetch")
    assert fetches, trace.read_text()
    for call in fetches:
        _assert_stall_before(call, "fetch", http11=False)


def _blackhole_port() -> tuple[socket.socket, threading.Event, int]:
    """A TCP server that accepts connections and then sends nothing: every
    transfer stalls at 0 B/s, so lowSpeed aborts it. No sleeps: each handler
    blocks in recv until the client aborts and closes."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    stop = threading.Event()

    def serve() -> None:
        server.settimeout(0.5)
        while not stop.is_set():
            conn = None
            # TimeoutError subclasses OSError, so one suppress covers
            # accept timeouts and closes.
            with contextlib.suppress(OSError):
                conn, _ = server.accept()
            if conn is None:
                continue
            with contextlib.suppress(OSError):
                _stall(conn)

    def _stall(conn: socket.socket) -> None:
        # Stall: read the request, then send nothing. Git's lowSpeed abort
        # closes the connection, which ends the second recv.
        conn.settimeout(60)
        conn.recv(65536)
        while conn.recv(65536):
            pass
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return server, stop, server.getsockname()[1]


@needs_bash_git
def test_stalled_clone_aborts_and_retries(tmp_path):
    """Against a server that stalls mid-transfer, the clone must fail fast
    (not hang) and the retry loop must fire: 3 attempts plus the deferred
    history clone. The stall window is shortened for the test only."""
    server, stop, port = _blackhole_port()
    try:
        trace = tmp_path / "trace.log"
        prelude = (
            "GIT_STALL_OPTS=(-c http.lowSpeedLimit=1000 -c http.lowSpeedTime=3)\n"
            "git() {\n"
            '    case " $* " in *" clone "*) '
            f'printf \'%s\\n\' "$*" >> {shlex.quote(trace.as_posix())};; esac\n'
            '    command git "$@"\n'
            "}"
        )
        start = time.monotonic()
        result = _run_sh(tmp_path, f"{prelude}\nstage_repository",
                         env_extra={"HERMES_REPO_URL": f"http://127.0.0.1:{port}/repo.git"},
                         timeout=120)
        elapsed = time.monotonic() - start
    finally:
        stop.set()
        server.close()
    assert result.returncode != 0, result.stdout + result.stderr
    assert "git clone failed" in result.stdout + result.stderr
    clones = _calls(trace, "clone")
    assert len(clones) >= 4, [ " ".join(c) for c in clones ]
    assert elapsed < 90, f"stall was not aborted promptly ({elapsed:.0f}s)"


def _seed_pm_git_store(tmp_path: Path, env: dict) -> None:
    if os.name == "nt":
        return
    # pwsh on POSIX cannot run the Windows-only pinned Git: seed its PM store
    # slot with host git (same pattern as test_install_clone_throttle_fallback).
    version = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8-sig"))["packages"]["git"]["version"]
    cmd = tmp_path / "tools" / f"git-{version}-win32-x64" / "cmd"
    cmd.mkdir(parents=True, exist_ok=True)
    host_git = shutil.which("git")
    assert host_git is not None
    for name in ("git.exe", "git"):
        link = cmd / name
        if not link.exists():
            link.symlink_to(host_git)
    env["HERMES_RUNTIME_DIR"] = str(tmp_path / "tools")
    env["PATH"] = str(cmd) + os.pathsep + env["PATH"]


_PS1_WRAPPER = """\
$ErrorActionPreference = 'Stop'
$traceFile = '__TRACE__'
$installPs1 = '__INSTALL_PS1__'
$realGit = (Get-Command git -CommandType Application | Select-Object -First 1).Source
$failDirect = '__FAIL_DIRECT__'
function git {
    $argv = @($args)
    Add-Content -LiteralPath $traceFile -Value ($argv -join ' ')
    if (($argv -contains 'clone') -and -not ($argv -contains '--no-checkout')) {
        if ($failDirect -eq '1') {
            $global:LASTEXITCODE = 1
            return
        }
    }
    & $realGit @argv
    $global:LASTEXITCODE = $LASTEXITCODE
}
. $installPs1
$Branch = 'main'
$InstallDir = '__INSTALL__'
$RepoUrl = '__REPO__'
$Commit = ''
$HermesHome = '__HOME__'
Stage-Repository
"""


def _run_ps1_wrapper(tmp_path: Path, install: Path, repo: str, *, fail_direct: bool,
                     trace: Path) -> subprocess.CompletedProcess:
    assert POWERSHELL is not None
    wrapper = tmp_path / f"wrapper-{int(fail_direct)}.ps1"
    wrapper.write_text(
        _PS1_WRAPPER.replace("__TRACE__", str(trace))
        .replace("__INSTALL_PS1__", str(INSTALL_PS1))
        .replace("__FAIL_DIRECT__", "1" if fail_direct else "0")
        .replace("__INSTALL__", str(install))
        .replace("__REPO__", repo)
        .replace("__HOME__", str(tmp_path / "home")),
        encoding="utf-8",
    )
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix())
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy, None)
    env["NO_PROXY"] = "*"
    _seed_pm_git_store(tmp_path, env)
    return subprocess.run([POWERSHELL, "-NoProfile", "-File", str(wrapper)], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=240, check=False)


@pytest.mark.skipif(POWERSHELL is None, reason="needs PowerShell")
def test_ps1_clone_attempts_carry_stall_opts_and_http11_fallback(tmp_path):
    origin = _origin(tmp_path / "origin")
    install = tmp_path / "install"
    trace = tmp_path / "trace.log"
    result = _run_ps1_wrapper(tmp_path, install, origin.as_posix(), fail_direct=True, trace=trace)
    assert result.returncode == 0, result.stdout + result.stderr
    clones = _calls(trace, "clone")
    assert len(clones) == 4, [" ".join(c) for c in clones]
    _assert_stall_before(clones[0], "clone", http11=False)
    for attempt in clones[1:]:
        _assert_stall_before(attempt, "clone", http11=True)
    assert (install / "README").read_text(encoding="utf-8-sig") == "one"
    _assert_no_persisted_stall_config(install)


@pytest.mark.skipif(POWERSHELL is None, reason="needs PowerShell")
def test_ps1_update_fetch_carries_stall_opts(tmp_path):
    origin = _origin(tmp_path / "origin")
    install = tmp_path / "install"
    trace = tmp_path / "trace.log"
    first = _run_ps1_wrapper(tmp_path, install, origin.as_posix(), fail_direct=False, trace=trace)
    assert first.returncode == 0, first.stdout + first.stderr
    (origin / "README").write_text("two")
    _git(origin, "commit", "-qam", "two")
    trace.unlink()
    result = _run_ps1_wrapper(tmp_path, install, origin.as_posix(), fail_direct=False, trace=trace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (install / "README").read_text(encoding="utf-8-sig") == "two"
    fetches = _calls(trace, "fetch")
    assert fetches, trace.read_text(encoding="utf-8-sig", errors="replace")
    for call in fetches:
        _assert_stall_before(call, "fetch", http11=False)
