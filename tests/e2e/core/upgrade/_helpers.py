"""Lane-private helpers for the upgrade / install-integrity and config round-trip suites.

Every Hermes process these suites spawn runs:

* with a HOME/HERMES_HOME under the test's tmp dir and an environment built from an
  allowlist (no inherited ``*_API_KEY`` / ``HERMES_*``), so only the fake provider is
  configured;
* inside a ``bwrap`` sandbox when bubblewrap is usable: its own PID namespace (the
  updater's process-table scans cannot see, let alone signal, any real gateway on the
  host), a tmpfs over ``/run/user/<uid>`` (no user systemd bus), the real
  ``~/.hermes`` bind-mounted read-only, and ``--die-with-parent`` so killing the
  sandbox kills every descendant (no orphans);
* with ``systemctl``/``launchctl``/``sudo``/``loginctl`` shims first on PATH that log
  their argv and fail, so a service-restart attempt is observable and never reaches a
  real supervisor.
"""

from __future__ import annotations

import os
import pwd
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

WORKTREE = Path(__file__).resolve().parents[4]
REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
UID = os.getuid()

_ENV_ALLOW = ("LANG", "LC_ALL", "TZ", "TERM", "SHELL", "USER", "LOGNAME", "TMPDIR", "SSL_CERT_FILE")
_SHIMMED = ("systemctl", "launchctl", "sudo", "loginctl", "journalctl")


def _bwrap_usable() -> bool:
    exe = shutil.which("bwrap")
    if not exe or sys.platform != "linux":
        return False
    try:
        r = subprocess.run(
            [exe, "--dev-bind", "/", "/", "--unshare-pid", "--proc", "/proc", "--die-with-parent", "true"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


BWRAP_OK = _bwrap_usable()


def sandbox_required_reason() -> str | None:
    """Why a test that runs the real updater must skip here, or None when it is safe.

    Without a PID-namespace sandbox the updater's all-profile gateway scan could reach a
    real gateway on a developer box; CI runners have none, so plain processes are fine
    there.
    """
    if BWRAP_OK:
        return None
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return None
    return "bubblewrap sandbox unavailable and not on CI; refusing to run the real updater next to a live install"


def write_shims(bin_dir: Path) -> Path:
    """Create failing, argv-logging shims for service managers; returns the call log path."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = bin_dir / "shim-calls.log"
    for name in _SHIMMED:
        p = bin_dir / name
        p.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "{name} $*" >> "{log}"\n'
            'echo "Failed to connect to bus: sandboxed test shim" >&2\n'
            "exit 1\n",
            encoding="utf-8",
        )
        p.chmod(0o755)
    return log


def isolated_env(
    root: Path,
    *,
    extra_path: Iterable[Path] = (),
    pythonpath: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Allowlisted environment with HOME/HERMES_HOME under ``root``."""
    home = root / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    shim_dir = root / "shims"
    write_shims(shim_dir)
    env = {k: os.environ[k] for k in _ENV_ALLOW if k in os.environ}
    env.setdefault("LANG", "C.UTF-8")
    env.update(
        HOME=str(home),
        HERMES_HOME=str(hermes_home),
        # The pytest ancestor marks this child as guarded; its HOME is already the sandbox.
        HERMES_STATE_DB_GUARD_BYPASS="1",
        XDG_RUNTIME_DIR=str(root / "run"),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_CACHE_HOME=str(home / ".cache"),
        DBUS_SESSION_BUS_ADDRESS="unix:path=/nonexistent/hermes-test-bus",
        NO_COLOR="1",
        TERM="dumb",
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="0",
        HERMES_DISABLE_LAZY_INSTALLS="1",
        TIRITH_ENABLED="false",
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=str(home / ".gitconfig"),
        GIT_AUTHOR_NAME="e2e", GIT_AUTHOR_EMAIL="e2e@example.invalid",
        GIT_COMMITTER_NAME="e2e", GIT_COMMITTER_EMAIL="e2e@example.invalid",
    )
    (root / "run").mkdir(parents=True, exist_ok=True)
    # Reuse the host uv cache (read/write, uv is concurrency-safe) so dependency syncs are
    # warm; never the real ~/.hermes.
    real_uv_cache = Path(os.environ.get("UV_CACHE_DIR") or REAL_HOME / ".cache" / "uv")
    if real_uv_cache.is_dir():
        env["UV_CACHE_DIR"] = str(real_uv_cache)
    base_path = os.environ.get("PATH", "/usr/bin:/bin")
    uv = shutil.which("uv") or (str(REAL_HOME / ".hermes" / "bin" / "uv") if (REAL_HOME / ".hermes" / "bin" / "uv").exists() else None)
    path_parts = [str(shim_dir), *[str(p) for p in extra_path]]
    if uv:
        path_parts.append(str(Path(uv).parent))
    env["PATH"] = os.pathsep.join(path_parts + [base_path])
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    if extra:
        env.update(extra)
    return env


def sandbox_argv(argv: Sequence[str], *, writable: Iterable[Path]) -> list[str]:
    """Wrap ``argv`` in the bwrap sandbox (no-op when bubblewrap is unusable)."""
    if not BWRAP_OK:
        return list(argv)
    # The child's allowlisted PATH may omit the Nix-provided bwrap (notably a login shell
    # with a clean distro PATH); resolve it in the parent before wrapping the command.
    cmd = [shutil.which("bwrap") or "bwrap", "--dev-bind", "/", "/"]
    real_hermes = REAL_HOME / ".hermes"
    if real_hermes.is_dir():
        cmd += ["--ro-bind", str(real_hermes), str(real_hermes)]
    for w in writable:
        w = Path(w)
        w.mkdir(parents=True, exist_ok=True)
        cmd += ["--bind", str(w), str(w)]
    run_user = Path(f"/run/user/{UID}")
    if run_user.is_dir():
        cmd += ["--tmpfs", str(run_user)]
    cmd += ["--unshare-pid", "--proc", "/proc", "--die-with-parent", "--"]
    return cmd + list(argv)


def run(
    argv: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path,
    writable: Iterable[Path],
    timeout: float = 300,
    input: str | None = None,
) -> subprocess.CompletedProcess:
    """Run one sandboxed process to completion; kills the whole sandbox on timeout."""
    proc = subprocess.Popen(
        sandbox_argv(argv, writable=writable),
        env=env, cwd=str(cwd), text=True,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, err = proc.communicate()
        raise AssertionError(
            f"{list(argv)} timed out after {timeout}s\nSTDOUT tail:\n{out[-4000:]}\nSTDERR tail:\n{err[-4000:]}"
        )
    return subprocess.CompletedProcess(list(argv), proc.returncode, out, err)


def kill_tree(proc: subprocess.Popen) -> None:
    """SIGKILL the process group we started (the sandbox's PID namespace dies with it)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def wait_for(predicate, *, timeout: float, interval: float = 0.1, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def describe(cp: subprocess.CompletedProcess, limit: int = 6000) -> str:
    return (
        f"argv={cp.args}\nrc={cp.returncode}\n--- stdout (tail) ---\n{(cp.stdout or '')[-limit:]}"
        f"\n--- stderr (tail) ---\n{(cp.stderr or '')[-limit:]}"
    )
