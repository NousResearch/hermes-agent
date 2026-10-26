"""macOS: start an unmanaged gateway as a one-off launchd job so it keeps Local Network access.

Local Network Privacy attributes a process to its responsible code. A gateway that Desktop, a
terminal or the updater starts as a plain child is attributed to that app, and once the app quits
every surviving descendant becomes responsible for itself: a bare venv Python, which macOS denies
with ``EHOSTUNREACH`` on any LAN connect (a model server on another machine). A job that launchd
spawns through ``/usr/bin/osascript`` is attributed to osascript, an exempt Apple platform binary,
for its whole life, whoever asked for it (#71206; ``gateway_launchd.launchd_program_arguments``).

Nothing is installed: the plist is loaded once and deleted, it is never in ~/Library/LaunchAgents
and has no KeepAlive, so the job never starts at login and launchd never restarts the gateway.
Stdlib only, because the restart watcher imports this from a bare interpreter.
"""
from __future__ import annotations

import os
import plistlib
import re
import subprocess
import time
import uuid
from pathlib import Path

# Never ``ai.hermes.*``: that prefix is what the gateway's restart route, its control-socket
# supervisor declaration and every fleet scan read as "an installed, KeepAlive service".
LABEL_PREFIX = "com.nousresearch.hermes.runtime."
_POLL_INTERVAL_S = 0.5


def _launchctl(*args: str, timeout: float = 10) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["launchctl", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None


class GatewayJob:
    """The ``poll()``/``pid`` view of a launched job that ``ensure_gateway_runtime`` reads."""

    def __init__(self, target: str):
        self.target = target
        self.pid: int | None = None
        self._status: int | None = None
        self._checked = 0.0

    def poll(self) -> int | None:
        if self._status is not None:
            return self._status
        if time.monotonic() - self._checked < _POLL_INTERVAL_S:
            return None
        self._checked = time.monotonic()
        done = _launchctl("print", self.target, timeout=5)
        if done is None:
            return None
        if done.returncode != 0:
            self._status = 1  # the job is gone without a recorded exit
            return self._status
        if match := re.search(r"^\s*pid = (\d+)", done.stdout, re.MULTILINE):
            self._record_gateway_pid(int(match.group(1)))
            return None
        if match := re.search(r"^\s*last exit code = (-?\d+)", done.stdout, re.MULTILINE):
            self._status = int(match.group(1))
        return self._status

    def _record_gateway_pid(self, wrapper_pid: int) -> None:
        # The job's own process is osascript; the gateway is its only child (the shell execs it).
        # Its pid is what gateway_state.json names, so a startup-failure report can match it.
        if self.pid not in (None, wrapper_pid):
            return
        try:
            out = subprocess.run(["pgrep", "-P", str(wrapper_pid)], capture_output=True, text=True, encoding="utf-8",
                                 timeout=5, check=False, stdin=subprocess.DEVNULL).stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            out = []
        self.pid = int(out[0]) if out else wrapper_pid


def _remove_finished_jobs() -> None:
    """Unload earlier jobs whose gateway has exited; a running one is never touched."""
    listed = _launchctl("list")
    for line in (listed.stdout.splitlines() if listed and listed.returncode == 0 else []):
        fields = line.split()
        if len(fields) == 3 and fields[0] == "-" and fields[2].startswith(LABEL_PREFIX):
            _launchctl("remove", fields[2])


def start_gateway_job(command: list[str], *, env: dict[str, str], cwd: str | Path,
                      log_path: str | Path) -> GatewayJob | None:
    """Run ``command`` as a one-off job in this user's GUI launchd domain. ``None`` when launchd
    cannot take it (no GUI session, e.g. over SSH): the caller starts it as a plain child instead."""
    from hermes_cli.gateway_launchd import launchd_program_arguments
    _remove_finished_jobs()
    label = f"{LABEL_PREFIX}{uuid.uuid4().hex[:12]}"
    target = f"gui/{os.getuid()}/{label}"  # windows-footgun: ok — macOS-only module
    log_path = Path(log_path)
    # launchd stamps XPC_SERVICE_NAME=<label> on the job; the gateway must still read as the
    # unsupervised process it is, or /restart would exit for a respawn no KeepAlive provides.
    wrapped = ["/usr/bin/env", "XPC_SERVICE_NAME=0", *command]
    plist = log_path.parent / f".{label}.plist"
    fd = os.open(plist, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # holds the launch env
    try:
        with os.fdopen(fd, "wb") as handle:
            plistlib.dump({
                "Label": label,
                "ProgramArguments": launchd_program_arguments(wrapped, log_path, log_path),
                "EnvironmentVariables": dict(env),
                "WorkingDirectory": str(cwd),
                "RunAtLoad": True,
            }, handle)
        done = _launchctl("bootstrap", f"gui/{os.getuid()}", str(plist))  # windows-footgun: ok — macOS-only
    finally:
        plist.unlink(missing_ok=True)
    return GatewayJob(target) if done is not None and done.returncode == 0 else None
